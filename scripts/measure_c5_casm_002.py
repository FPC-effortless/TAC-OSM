#!/usr/bin/env python3
"""TACOSM-C5-002: addressing and execution as separate measured quantities.

C5-001 measured one compound number whose selection rule depended on CASM-S
output being informative; it was not, and the run is void (see
``docs/TACOSM-C5-001-RESULT.md``). C5-002 separates the quantities, and adds a
representability gate that terminates the run as instrument-invalid *before*
the task stream rather than after it.

The load-bearing design constraint is that ``coverage_rate`` never reads a
CASM-S output. It is a set intersection between the retained indices and the
hidden acceptable set, so an execution failure cannot depress an addressing
number.

The control-plane repository stays torch-free. This script is the compute-side
entry point and requires a checkout of cdl-attention-experiment at the pinned
CASM-S commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tac_osm import Candidate, Computation, Outcome
from tac_osm.addressing import ContentAddressIndex
from tac_osm.benchmark_v1 import repeated_equality_population, task_from_population
from tac_osm.casm_adapter import casm_structure_from_program
from tac_osm.casm_runtime import CASM_COMMIT, CasmExecutionWork, CasmSRuntime
from tac_osm.contract import load_contract
from tac_osm.executor import relevance_program
from tac_osm.measurement import results as _results
from tac_osm.structured_verifier import RelationConstraintVerifier

EXPERIMENT_ID = "TACOSM-C5-002"
H_LEVELS = (8, 64, 256)
K_LEVELS = (2, 4)
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("exhaustive", "exact-indexed", "representation-addressed")
QUERIES = 100
STEPS = QUERIES
EVAL_STEPS = QUERIES
WARMUP = 3
DIM = 8
MARKED = 2

#: Bridge calibration, identical to C5-001 so the two experiments' CASM-S
#: states are comparable and the gate is not silently trained differently.
BRIDGE_TRAIN = 512
BRIDGE_VALIDATION = 128
BRIDGE_SEED = 20260929
TRAIN_EPOCHS = 100
TRAIN_LR = 2e-3

#: The gate runs before the task stream. Its pass condition is pre-registered
#: in ``docs/TACOSM-C5-002.md`` and is not tuned to a run: it asks whether the
#: frozen model, on this graph family, under the threshold the task stream
#: uses, separates the relation's satisfier from its non-satisfier.
GATE_STRUCTURES = 256
GATE_SEED = 20260930
#: The threshold is the one ``RelationConstraintVerifier`` uses and the one the
#: task stream's execution endpoint reads. The gate tests *separation* against
#: it — ``hi >= threshold > lo`` — rather than absolute output level, because
#: separation is what a within-query decision depends on.
GATE_THRESHOLD = 0.5
#: Minimum fraction of held-out pairs the gate must separate. Pre-registered:
#: a gate tuned to a run is not a gate.
GATE_MIN_ACCURACY = 0.5


# --------------------------------------------------------------------------- #
# The representability gate
# --------------------------------------------------------------------------- #


@dataclass
class GateResult:
    """The representability gate, as a record rather than a pass/fail flag.

    A failed gate is reported, not swallowed: it is the difference between an
    invalid run and an ambiguous one, and the record is what makes that
    difference visible to a reader who was not present for the run.
    """

    passed: bool
    accuracy: float
    n_pairs: int
    threshold: float
    min_accuracy: float
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "accuracy": self.accuracy,
            "n_pairs": self.n_pairs,
            "threshold": self.threshold,
            "min_accuracy": self.min_accuracy,
            "detail": self.detail,
        }


def gate_pairs(count: int, seed: int) -> tuple:
    """Held-out ``(reference, satisfying, violating, marks)`` tuples.

    The gate's structures come from a stream disjoint from the task stream's
    seeds, so the gate cannot be calibrated on the evaluation population. One
    pair is one satisfying descriptor and one violating descriptor against one
    reference, which is the smallest unit the separation question can be asked
    on.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        reference = tuple(rng.randrange(2) for _ in range(DIM))
        marks = tuple(sorted(rng.sample(range(DIM), MARKED)))
        # A descriptor agreeing with the reference on every marked position;
        # the unmarked positions are free and vary across pairs.
        satisfying = list(reference)
        for j in range(DIM):
            if j not in marks:
                satisfying[j] = rng.randrange(2)
        # The same descriptor with exactly one marked bit flipped, so the pair
        # differs on precisely the quantity the relation is defined over.
        violating = list(satisfying)
        flip = marks[rng.randrange(len(marks))]
        violating[flip] = 1 - violating[flip]
        out.append((reference, tuple(satisfying), tuple(violating), marks))
    return tuple(out)


def run_gate(runtime: CasmSRuntime) -> GateResult:
    """The pre-registered representability gate.

    For each held-out pair the frozen model must score the relation's
    satisfier at or above the task stream's threshold and the non-satisfier
    strictly below it. That is the *within-query separation* the execution
    endpoints depend on, and it is the requirement C5-001's calibration score
    did not test: a model can be 84% accurate in absolute terms and still rank
    candidates arbitrarily within one query.
    """
    pairs = gate_pairs(GATE_STRUCTURES, GATE_SEED)
    structures = []
    inputs = []
    for n, (reference, satisfying, violating, marks) in enumerate(pairs):
        for tag, descriptor in (("sat", satisfying), ("vio", violating)):
            program = relevance_program(reference, descriptor, marks, max_nodes=10)
            # Deterministic key: ``hash()`` on strings is salted per process,
            # which would make the gate's structure keys differ between two
            # runs of the same pre-registered design.
            key = f"gate-{n}-{tag}"
            structures.append(
                casm_structure_from_program(program, key=key, provenance="casm_s.gate")
            )
            inputs.append(tuple(float(x) for x in program.input_values))

    executions, _work = runtime.execute_many(tuple(structures), tuple(inputs))

    n_separated = 0
    for i in range(len(pairs)):
        hi = executions[2 * i].result.output
        lo = executions[2 * i + 1].result.output
        if hi >= GATE_THRESHOLD > lo:
            n_separated += 1
    accuracy = n_separated / len(pairs)
    return GateResult(
        passed=accuracy >= GATE_MIN_ACCURACY,
        accuracy=accuracy,
        n_pairs=len(pairs),
        threshold=GATE_THRESHOLD,
        min_accuracy=GATE_MIN_ACCURACY,
        detail={
            "structures": len(structures),
            "bridge_seed": BRIDGE_SEED,
            "gate_seed": GATE_SEED,
            "casm_commit": CASM_COMMIT,
        },
    )


# --------------------------------------------------------------------------- #
# Bridge calibration — identical to C5-001
# --------------------------------------------------------------------------- #


@dataclass
class BridgeExample:
    structure: object
    inputs: tuple[float, ...]
    target: float


def _relation_target(reference: Sequence[int], descriptor: Sequence[int],
                     marks: Sequence[int]) -> float:
    return float(all(descriptor[j] == reference[j] for j in marks))


def bridge_examples(count: int, seed: int) -> tuple[BridgeExample, ...]:
    """The same bridge calibration set C5-001 used.

    Identical by design: a different calibration would move the instrument and
    leave the gate's verdict unattributable — was it the model, the
    calibration, or the threshold?
    """
    rng = random.Random(seed)
    examples = []
    for i in range(count):
        reference = tuple(rng.randrange(2) for _ in range(DIM))
        descriptor = tuple(rng.randrange(2) for _ in range(DIM))
        marks = tuple(sorted(rng.sample(range(DIM), MARKED)))
        program = relevance_program(reference, descriptor, marks, max_nodes=10)
        structure = casm_structure_from_program(
            program, key=f"bridge-{seed}-{i}", provenance="casm_s.bridge_training"
        )
        examples.append(
            BridgeExample(
                structure=structure,
                inputs=tuple(float(x) for x in program.input_values),
                target=_relation_target(reference, descriptor, marks),
            )
        )
    return tuple(examples)


def _batch(runtime: CasmSRuntime, examples: Sequence[BridgeExample]):
    specs = [runtime._spec(e.structure) for e in examples]
    episodes = [runtime._episode_from_spec(s) for s in specs]
    width = max(len(spec.inputs) for spec in specs)
    rows = [[*e.inputs] + [0.0] * (width - len(e.inputs)) for e in examples]
    torch = runtime.torch
    x = torch.tensor(rows, dtype=torch.float32, device=runtime.device)
    y = torch.tensor([e.target for e in examples], dtype=torch.float32, device=runtime.device)
    return episodes, x, y


def train_bridge(runtime: CasmSRuntime) -> dict:
    """Bridge calibration, unchanged from C5-001.

    The calibration is reported and is *not* a capability result. The gate, not
    this score, is what decides whether the instrument can measure: the
    calibration is absolute-output accuracy while the decision downstream is a
    within-query separation, and conflating the two is the specific error that
    let C5-001 complete on an unrepresentative model.
    """
    train = bridge_examples(BRIDGE_TRAIN, BRIDGE_SEED)
    validation = bridge_examples(BRIDGE_VALIDATION, BRIDGE_SEED + 1)
    torch = runtime.torch
    model = runtime.model

    episodes, x, target = _batch(runtime, train)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR)
    history = []
    model.train()
    for _epoch in range(TRAIN_EPOCHS):
        pred, _gates, _meta = model(episodes, x)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(float(loss.detach().cpu().item()))

    model.eval()
    results, _ = runtime.execute_many(
        tuple(e.structure for e in validation),
        tuple(e.inputs for e in validation),
    )
    correct = sum(
        int((r.result.output >= GATE_THRESHOLD) == bool(e.target))
        for r, e in zip(results, validation)
    )
    return {
        "train_examples": len(train),
        "validation_examples": len(validation),
        "seed": BRIDGE_SEED,
        "epochs": TRAIN_EPOCHS,
        "learning_rate": TRAIN_LR,
        "final_train_loss": history[-1],
        "validation_accuracy": correct / len(validation),
        "history": history,
    }


def checkpoint_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Addressing
# --------------------------------------------------------------------------- #


def representation_retain(candidates: Sequence[Candidate], query_bits: Sequence[int],
                          k: int) -> tuple[int, ...]:
    """Fixed cosine retention: the non-learned addressing baseline.

    Identical to C5-001's baseline so the two experiments' addressing numbers
    are comparable, and so this arm stays registered as not a learned result.
    """
    q = tuple(float(x) for x in query_bits)
    nq = math.sqrt(sum(x * x for x in q))
    if nq == 0.0:
        scores = [0.0] * len(candidates)
    else:
        scores = []
        for c in candidates:
            v = tuple(float(x) for x in c.descriptor)
            nv = math.sqrt(sum(x * x for x in v))
            score = 0.0 if nv == 0.0 else sum(a * b for a, b in zip(q, v)) / (nq * nv)
            scores.append(score)
    return tuple(sorted(range(len(candidates)), key=lambda i: (-scores[i], i))[:k])


def _marks_for(task) -> list[int]:
    return [j for j, b in enumerate(task.query.context) if b == 1]


def _retained(task, index: ContentAddressIndex | None, h: int, k: int | None,
              arm: str) -> tuple[int, ...]:
    """Retained indices for one arm, without reading any model output.

    The exhaustive arm retains the full population, so its ``coverage_rate`` is
    1 by construction — the registered reading, because the reference arm has
    no addressing step to fail.
    """
    if arm == "exhaustive":
        return tuple(range(h))
    if arm == "exact-indexed":
        hit = index.lookup(task.public(), reference=task.reference_bits, k=k, relation="equality")
        return hit.candidate_indices
    if arm == "representation-addressed":
        return representation_retain(task.candidates, task.reference_bits, int(k))
    raise ValueError(f"unknown arm {arm}")


def _build_and_execute(runtime: CasmSRuntime, task, marks, reference, retained):
    """Compile and execute the retained structures for one query.

    The acceptable set is deliberately not a parameter and cannot reach the
    executor. Returns ``(executions, structures, work)``; the caller computes
    coverage from the retained indices it already holds.
    """
    structures = []
    inputs = []
    for i in retained:
        program = relevance_program(reference, task.candidates[i].descriptor, marks, max_nodes=10)
        structures.append(casm_structure_from_program(program, key=task.candidates[i].key, provenance="casm_s"))
        inputs.append(tuple(float(x) for x in program.input_values))
    return runtime.execute_many(tuple(structures), tuple(inputs))


# --------------------------------------------------------------------------- #
# The measured cell
# --------------------------------------------------------------------------- #


@dataclass
class CellResult:
    arm: str
    h: int
    k: int | None
    seed: int
    coverage_rate: float
    execution_accuracy_rate: float
    verification_rate: float
    structures_executed_per_query: float
    active_nodes_per_query: float
    candidate_edges_per_query: float
    gate_evaluations_per_query: float
    structural_operations_per_query: float
    node_outputs_per_query: float
    casm_forward_batches: int
    address_positions_per_query: float
    representation_candidates_scored_per_query: float
    wall_clock_seconds: float


class C5Benchmark:
    """The same static-population task stream C5-001 used.

    Kept identical so the two experiments' work endpoints are directly
    comparable: a difference in the accounting would otherwise be
    indistinguishable from a difference in the instrument.
    """

    _refs = (
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 1, 1, 0, 0, 1, 0, 1),
        (1, 0, 1, 1, 1, 0, 1, 0),
        (1, 1, 0, 1, 0, 1, 1, 0),
    )

    def __init__(self, seed: int, candidates: Sequence[Candidate]):
        self.seed = seed
        self.candidates = tuple(candidates)

    def task(self, step: int):
        reference = self._refs[(step + self.seed) % len(self._refs)]
        return task_from_population(
            candidates=self.candidates,
            reference_bits=reference,
            step=step,
            relation="equality",
            validity="multiple",
        )


def run_cell(runtime: CasmSRuntime, seed: int, h: int, k: int | None, arm: str) -> CellResult:
    population = repeated_equality_population(seed, dim=DIM, n_candidates=h, marked_positions=(0, 1))
    benchmark = C5Benchmark(seed, population)
    index = ContentAddressIndex.build(population, context=(1, 1, 0, 0, 0, 0, 0, 0)) if arm == "exact-indexed" else None
    verifier = RelationConstraintVerifier()

    covered = 0
    execution_correct = 0
    verified = 0
    work = CasmExecutionWork.zero()
    address_positions = 0
    representation_scored = 0
    forward_batches = 0

    # Fixed warmup is outside the measured query window.
    for step in range(WARMUP):
        task = benchmark.task(step)
        retained = _retained(task, index, h, k, arm)
        if not retained:
            raise RuntimeError(f"empty retained set in {arm}; no exhaustive fallback is permitted")
        _build_and_execute(runtime, task, _marks_for(task), task.reference_bits, retained)

    start = time.perf_counter()
    for q in range(QUERIES):
        step = q + WARMUP
        task = benchmark.task(step)
        reference = task.reference_bits
        marks = _marks_for(task)

        retained = _retained(task, index, h, k, arm)
        if not retained:
            raise RuntimeError(f"empty retained set in {arm}; no exhaustive fallback is permitted")
        if arm == "exact-indexed":
            hit = index.lookup(task.public(), reference=reference, k=k, relation="equality")
            address_positions += hit.inspected_positions
        elif arm == "representation-addressed":
            representation_scored += h

        executions, cell_work = _build_and_execute(
            runtime, task, marks, reference, retained
        )
        forward_batches += 1
        work = work + cell_work

        # --- the addressing endpoint ------------------------------------- #
        # Computed from the retained indices and the hidden acceptable set
        # alone. No model output, selection result or verification result is
        # read here, so an execution failure cannot depress an addressing
        # number. This is the constraint C5-001 violated by making
        # ``success_rate`` an argmax over model output.
        if any(i in task.acceptable_actions for i in retained):
            covered += 1

        # --- the execution endpoint -------------------------------------- #
        # Selection over the retained candidates only. It is not the addressing
        # decision — that is the retained set itself — and it does not feed
        # ``coverage_rate``. An uninformative output makes this selection
        # arbitrary, which now degrades only the execution endpoint and leaves
        # the addressing measurement intact. That is the separation C5-002
        # exists to make.
        local = max(range(len(executions)), key=lambda i: (executions[i].result.output, -i))
        selected_global = retained[local]
        selected = task.candidates[selected_global]
        result = executions[local].result

        program = relevance_program(reference, selected.descriptor, marks, max_nodes=10)
        structure = casm_structure_from_program(program, key=selected.key, provenance="casm_s")
        computation = Computation(
            structure=structure,
            action=selected_global,
            trace=(result.output,) + result.node_values,
        )
        outcome = Outcome(
            success=selected_global in task.acceptable_actions,
            value=1.0 if selected_global in task.acceptable_actions else 0.0,
            feedback="acceptable" if selected_global in task.acceptable_actions else "not_acceptable",
            detail=None,
        )
        evidence = verifier.verify_task(
            computation, outcome, candidate=selected, reference=reference,
            context=task.query.context, relation=task.relation,
        )
        if outcome.success:
            execution_correct += 1
        if evidence.valid:
            verified += 1

    elapsed = time.perf_counter() - start
    return CellResult(
        arm=arm, h=h, k=k, seed=seed,
        coverage_rate=covered / QUERIES,
        execution_accuracy_rate=execution_correct / QUERIES,
        verification_rate=verified / QUERIES,
        structures_executed_per_query=work.structures_executed / QUERIES,
        active_nodes_per_query=work.active_nodes / QUERIES,
        candidate_edges_per_query=work.candidate_edges / QUERIES,
        gate_evaluations_per_query=work.gate_evaluations / QUERIES,
        structural_operations_per_query=work.executed_structural_operations / QUERIES,
        node_outputs_per_query=work.node_outputs / QUERIES,
        casm_forward_batches=forward_batches,
        address_positions_per_query=address_positions / QUERIES,
        representation_candidates_scored_per_query=representation_scored / QUERIES,
        wall_clock_seconds=elapsed,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--casm-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--train-bridge", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--levels", default="8,64,256")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default=str(_results.results_dir_for(__file__) / "c5_002.json"))
    args = parser.parse_args()

    contract = load_contract(EXPERIMENT_ID)
    run_seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    run_levels = tuple(int(x) for x in args.levels.split(",") if x.strip())

    if args.smoke:
        contract.require_levels(run_levels, strict=False)
        contract.require_k_levels(K_LEVELS)
        contract.require_seeds(run_seeds, strict=False)
        contract.require_steps(args.steps)
        contract.require_eval_steps(args.eval_steps)
        contract.require_arms(ARMS)
        _results.report_smoke(
            contract, EXPERIMENT_ID,
            steps=args.steps, eval_steps=args.eval_steps,
            h_levels=run_levels, seeds=run_seeds, arms=ARMS,
        )
        print(f"machine-readable summary written to {args.output}")
        return

    contract.require_levels(run_levels)
    contract.require_k_levels(K_LEVELS)
    contract.require_seeds(run_seeds)
    contract.require_steps(args.steps)
    contract.require_eval_steps(args.eval_steps)
    contract.require_arms(ARMS)

    if args.train_bridge and args.checkpoint:
        raise SystemExit("--train-bridge and --checkpoint are mutually exclusive")
    if not args.train_bridge and not args.checkpoint:
        raise SystemExit("provide --checkpoint or --train-bridge for a confirmatory run")

    runtime = CasmSRuntime(
        casm_root=args.casm_root,
        checkpoint=args.checkpoint,
        max_nodes=10, dim=32, temperature=2.0, seed=BRIDGE_SEED, device=args.device,
    )

    calibration = None
    if args.train_bridge:
        calibration = train_bridge(runtime)
        if args.checkpoint_out:
            checkpoint = runtime.save_checkpoint(
                args.checkpoint_out,
                metadata={"experiment": EXPERIMENT_ID, "calibration": calibration},
            )
        else:
            checkpoint = None
    else:
        checkpoint = runtime.checkpoint

    # The gate runs before the task stream and terminates the run on failure,
    # so no capability number is emitted under an unrepresentative instrument.
    gate = run_gate(runtime)
    if not gate.passed:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment_id": EXPERIMENT_ID,
            "status": "INSTRUMENT_INVALID",
            "reason": (
                "the pre-registered representability gate failed on held-out "
                "structures; no capability number is reported. The failure is "
                "recorded against the instrument, not against CASM-S as a "
                "substrate."
            ),
            "casm_source": {"repository": "FPC-effortless/cdl-attention-experiment", "commit": CASM_COMMIT},
            "calibration": calibration,
            "gate": gate.to_dict(),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": checkpoint_sha256(Path(checkpoint)) if checkpoint else None,
            "config": {"h_levels": run_levels, "k_levels": K_LEVELS, "seeds": run_seeds,
                       "steps": args.steps, "eval_steps": args.eval_steps,
                       "queries": QUERIES, "warmup": WARMUP, "arms": ARMS},
            "cells": [],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("=" * 72)
        print("INSTRUMENT_INVALID: the representability gate failed.")
        print("=" * 72)
        print(f"gate accuracy {gate.accuracy:.4f} over {gate.n_pairs} held-out pairs")
        print(f"(threshold {gate.threshold}, minimum {gate.min_accuracy})")
        print("No capability number is reported. See docs/TACOSM-C5-002.md.")
        print(f"machine-readable summary written to {out}")
        return

    cells = []
    for h in run_levels:
        for seed in run_seeds:
            cells.append(run_cell(runtime, seed, h, None, "exhaustive"))
        for k in K_LEVELS:
            for seed in run_seeds:
                cells.append(run_cell(runtime, seed, h, k, "exact-indexed"))
                cells.append(run_cell(runtime, seed, h, k, "representation-addressed"))

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "casm_source": {"repository": "FPC-effortless/cdl-attention-experiment", "commit": CASM_COMMIT},
        "calibration": calibration,
        "gate": gate.to_dict(),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": checkpoint_sha256(Path(checkpoint)) if checkpoint else None,
        "config": {"h_levels": run_levels, "k_levels": K_LEVELS, "seeds": run_seeds,
                   "steps": args.steps, "eval_steps": args.eval_steps,
                   "queries": QUERIES, "warmup": WARMUP, "arms": ARMS},
        "cells": [cell.__dict__ for cell in cells],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"machine-readable summary written to {out}")
    print(f"gate: passed={gate.passed} accuracy={gate.accuracy:.4f} over {gate.n_pairs} pairs")
    for cell in cells:
        print(
            f"{cell.arm:>24} H={cell.h:>3} K={str(cell.k):>1} seed={cell.seed} "
            f"coverage={cell.coverage_rate:.4f} exec={cell.execution_accuracy_rate:.4f} "
            f"structures/query={cell.structures_executed_per_query:.2f} "
            f"edges/query={cell.candidate_edges_per_query:.2f}"
        )


if __name__ == "__main__":
    main()
