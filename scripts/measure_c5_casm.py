#!/usr/bin/env python3
"""TACOSM-C5-001: execute retained candidate structures with real CASM-S.

The control-plane repository stays torch-free. This script is the compute-side
entry point and requires a checkout of cdl-attention-experiment at the pinned
CASM-S commit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tac_osm import Candidate, Computation, Outcome, Structure
from tac_osm.addressing import ContentAddressIndex
from tac_osm.benchmark_v1 import repeated_equality_population, task_from_population
from tac_osm.casm_adapter import casm_structure_from_program
from tac_osm.casm_runtime import CASM_COMMIT, CasmExecutionWork, CasmSRuntime
from tac_osm.contract import load_contract
from tac_osm.measurement import results as _results
from tac_osm.executor import relevance_program
from tac_osm.structured_verifier import RelationConstraintVerifier

EXPERIMENT_ID = "TACOSM-C5-001"
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
BRIDGE_TRAIN = 512
BRIDGE_VALIDATION = 128
BRIDGE_SEED = 20260929
TRAIN_EPOCHS = 100
TRAIN_LR = 2e-3


@dataclass(frozen=True)
class BridgeExample:
    structure: Structure
    inputs: tuple[float, ...]
    target: float


def relation_target(reference: Sequence[int], descriptor: Sequence[int], marks: Sequence[int]) -> float:
    return float(all(descriptor[j] == reference[j] for j in marks))


def bridge_examples(count: int, seed: int) -> tuple[BridgeExample, ...]:
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
                target=relation_target(reference, descriptor, marks),
            )
        )
    return tuple(examples)


def train_bridge(runtime: CasmSRuntime) -> dict:
    train = bridge_examples(BRIDGE_TRAIN, BRIDGE_SEED)
    validation = bridge_examples(BRIDGE_VALIDATION, BRIDGE_SEED + 1)
    torch = runtime.torch
    model = runtime.model

    def batch(examples):
        specs = [runtime._spec(e.structure) for e in examples]
        episodes = [runtime._episode_from_spec(s) for s in specs]
        width = max(len(s.inputs) for s in specs)
        rows = [[*e.inputs] + [0.0] * (width - len(e.inputs)) for e in examples]
        x = torch.tensor(rows, dtype=torch.float32, device=runtime.device)
        y = torch.tensor([e.target for e in examples], dtype=torch.float32, device=runtime.device)
        return episodes, x, y

    episodes, x, target = batch(train)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR)
    history = []
    model.train()
    for epoch in range(TRAIN_EPOCHS):
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
        int((r.result.output >= 0.5) == bool(e.target))
        for r, e in zip(results, validation)
    )
    validation_accuracy = correct / len(validation)
    return {
        "train_examples": len(train),
        "validation_examples": len(validation),
        "seed": BRIDGE_SEED,
        "epochs": TRAIN_EPOCHS,
        "learning_rate": TRAIN_LR,
        "final_train_loss": history[-1],
        "validation_accuracy": validation_accuracy,
        "history": history,
    }


def checkpoint_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def representation_retain(candidates: Sequence[Candidate], query_bits: Sequence[int], k: int) -> tuple[int, ...]:
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


@dataclass
class CellResult:
    arm: str
    h: int
    k: int | None
    seed: int
    success_rate: float
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
    def __init__(self, seed: int, candidates: Sequence[Candidate]):
        self.seed = seed
        self.candidates = tuple(candidates)
        self._refs = (
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 1, 1, 0, 0, 1, 0, 1),
            (1, 0, 1, 1, 1, 0, 1, 0),
            (1, 1, 0, 1, 0, 1, 1, 0),
        )

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
    successes = 0
    verified = 0
    work = CasmExecutionWork.zero()
    address_positions = 0
    representation_scored = 0

    # Fixed warmup is outside the measured query window.
    for step in range(WARMUP):
        task = benchmark.task(step)
        reference = task.reference_bits
        marks = [j for j, b in enumerate(task.query.context) if b == 1]
        retained = tuple(range(h))
        if arm == "exact-indexed":
            hit = index.lookup(task.public(), reference=reference, k=k, relation="equality")
            retained = hit.candidate_indices
        elif arm == "representation-addressed":
            retained = representation_retain(task.candidates, reference, int(k))
        structures = []
        inputs = []
        for i in retained:
            program = relevance_program(reference, task.candidates[i].descriptor, marks, max_nodes=10)
            structures.append(casm_structure_from_program(program, key=task.candidates[i].key, provenance="casm_s"))
            inputs.append(tuple(float(x) for x in program.input_values))
        runtime.execute_many(tuple(structures), tuple(inputs))

    start = time.perf_counter()
    forward_batches = 0
    for q in range(QUERIES):
        step = q + WARMUP
        task = benchmark.task(step)
        reference = task.reference_bits
        marks = [j for j, b in enumerate(task.query.context) if b == 1]

        if arm == "exhaustive":
            retained = tuple(range(h))
        elif arm == "exact-indexed":
            hit = index.lookup(task.public(), reference=reference, k=k, relation="equality")
            retained = hit.candidate_indices
            address_positions += hit.inspected_positions
        elif arm == "representation-addressed":
            retained = representation_retain(task.candidates, reference, int(k))
            representation_scored += h
        else:
            raise ValueError(f"unknown arm {arm}")

        if not retained:
            raise RuntimeError(f"empty retained set in {arm}; no exhaustive fallback is permitted")

        structures = []
        inputs = []
        for i in retained:
            program = relevance_program(reference, task.candidates[i].descriptor, marks, max_nodes=10)
            structures.append(casm_structure_from_program(program, key=task.candidates[i].key, provenance="casm_s"))
            inputs.append(tuple(float(x) for x in program.input_values))

        executions, cell_work = runtime.execute_many(tuple(structures), tuple(inputs))
        forward_batches += 1
        work = work + cell_work

        local = max(range(len(executions)), key=lambda i: (executions[i].result.output, -i))
        selected_global = retained[local]
        selected = task.candidates[selected_global]
        result = executions[local].result
        computation = Computation(
            structure=structures[local],
            action=selected_global,
            trace=(result.output,) + result.node_values,
        )
        outcome = C5Benchmark_observe(task, selected_global)
        evidence = verifier.verify_task(
            computation, outcome, candidate=selected, reference=reference,
            context=task.query.context, relation=task.relation
        )
        successes += int(outcome.success)
        verified += int(evidence.valid)

    elapsed = time.perf_counter() - start
    return CellResult(
        arm=arm, h=h, k=k, seed=seed,
        success_rate=successes / QUERIES,
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


def C5Benchmark_observe(task, action: int) -> Outcome:
    success = action in task.acceptable_actions
    return Outcome(success=success, value=1.0 if success else 0.0, feedback="acceptable" if success else "not_acceptable", detail=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--casm-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--train-bridge", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--levels", default="8,64,256")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default=str(_results.results_dir_for(__file__) / "c5_001.json"))
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
            checkpoint = runtime.save_checkpoint(args.checkpoint_out, metadata={"experiment": EXPERIMENT_ID, "calibration": calibration})
        else:
            checkpoint = None
    else:
        checkpoint = runtime.checkpoint

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
        "casm_source": {"repository": "FPC-effortless/cdl-attention-experiment", "commit": CASM_COMMIT},
        "calibration": calibration,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": checkpoint_sha256(Path(checkpoint)) if checkpoint else None,
        "config": {"h_levels": run_levels, "k_levels": K_LEVELS, "seeds": run_seeds, "steps": args.steps, "eval_steps": args.eval_steps, "queries": QUERIES, "warmup": WARMUP, "arms": ARMS},
        "cells": [cell.__dict__ for cell in cells],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"machine-readable summary written to {out}")
    for cell in cells:
        print(
            f"{cell.arm:>24} H={cell.h:>3} K={str(cell.k):>1} seed={cell.seed} "
            f"success={cell.success_rate:.4f} structures/query={cell.structures_executed_per_query:.2f} "
            f"edges/query={cell.candidate_edges_per_query:.2f}"
        )


if __name__ == "__main__":
    main()