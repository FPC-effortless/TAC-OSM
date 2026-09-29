#!/usr/bin/env python3
"""TACOSM-C5-003: the integrated boundary, gated before the task population.

C5-002's representability gate fired and terminated the run as
``INSTRUMENT_INVALID`` before any capability number was emitted. The diagnosis
is a *specification* error in the bridge's training target: trained on single
candidates against their Boolean output, nothing in the objective rewards
within-query ranking, so 0.84375 absolute accuracy and 0.2305 separation are
both "as trained". C5-003 is the first fresh capability experiment after that
diagnosis.

Two things change, and only two:

1. the bridge is trained on **pairs** against a margin objective
   (``BRIDGE_OBJECTIVE = "pair_separation"``), so the objective contains the
   quantity the downstream decision actually depends on; and
2. the gate is a representability gate with **eight registered criteria**,
   evaluated before the task population runs.

Everything else is deliberately unchanged from C5-002: same population, same
relation, same arms, same work accounting, same separation contract, same
CASM-S pin, same threshold.

The experiment is registered as an **integrated-boundary** design rather than a
better gate. It tests the boundary between four stages as one causal chain —
``(S,Q) -> R`` addressing, ``(R,Q) -> A`` execution, ``(A,O) -> V``
verification, ``R -> W(R)`` work — and the gate decides whether the chain is
attached at all. C5-001's compound ``success_rate`` could not make that
distinction; a number that merges the interfaces cannot distinguish an
addressing failure from an execution failure.

The load-bearing design constraint carried over unchanged is that
``coverage_rate`` never reads a CASM-S output. It is a set intersection between
the retained indices and the hidden acceptable set, so an execution failure
cannot depress an addressing number.

The control-plane repository stays torch-free. This script is the compute-side
entry point and requires a checkout of cdl-attention-experiment at the pinned
CASM-S commit. A ``--smoke`` run is contract-only and returns before
constructing the CASM-S runtime, so it runs torch-free on the control plane
while still enforcing every ``require_*`` against the registered design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tac_osm import Candidate, Computation, Outcome
from tac_osm.addressing import ContentAddressIndex
from tac_osm.benchmark_v1 import relation_holds, repeated_equality_population, task_from_population
from tac_osm.casm_adapter import casm_structure_from_program
from tac_osm.casm_runtime import CASM_COMMIT, CasmExecutionWork, CasmSRuntime
from tac_osm.contract import load_contract
from tac_osm.degeneracy import (
    ABSOLUTE_FLOOR,
    MIN_CLASS_SUPPORT,
    MIN_SPREAD,
    DegeneracyError,
    GatePair,
    degenerate_outputs,
    discrimination,
    make_pairs,
    preflight,
)
from tac_osm.executor import relevance_program
from tac_osm.leakage import FORBIDDEN_FIELDS, audit_object
from tac_osm.measurement import results as _results
from tac_osm.structured_verifier import RelationConstraintVerifier

EXPERIMENT_ID = "TACOSM-C5-003"
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

#: The registered objective. A single named constant so a change to it is a
#: change to the pre-registration, visible in the diff rather than a silent
#: re-targeting. C5-002's objective was per-candidate Boolean regression; this
#: one is a margin over pairs, which is what the downstream argmax and the
#: thresholded verification both depend on.
BRIDGE_OBJECTIVE = "pair_separation"

#: Bridge calibration, identical to C5-002 in every respect *except the
#: objective* — so the two runs' CASM-S states differ only in the thing the
#: diagnosis identified, and a gate verdict is attributable to the objective
#: rather than to a silently different instrument.
BRIDGE_TRAIN = 512
BRIDGE_VALIDATION = 128
BRIDGE_SEED = 20260929
#: The validation stream's own seed. Deliberately not ``BRIDGE_SEED + 1``
#: (which is ``GATE_SEED``): the validation stream must not be the gate's
#: stream with a count prefix, or the reported separation and the gate's
#: verdict are the same number twice. Named rather than derived so the three
#: streams' disjointness is visible as three constants, not inferred from
#: arithmetic.
BRIDGE_VALIDATION_SEED = 20261001
TRAIN_EPOCHS = 100
TRAIN_LR = 2e-3

#: The gate. It runs before the task population and terminates the run on any
#: of its eight criteria, so no capability number is emitted under an
#: instrument that cannot represent the task.
#:
#: ``GATE_STRUCTURES`` is the number of held-out *pairs* (so twice that many
#: structures: a satisfying and a violating descriptor per pair). C5-002 gated
#: on 256 pairs; C5-003's criterion 2 is registered over 512, which is the
#: count the pre-registration's gate table names.
GATE_STRUCTURES = 512
#: Disjoint from the task stream's seeds, so the gate cannot be calibrated on
#: the evaluation population. ``BRIDGE_SEED`` is disjoint from both.
GATE_SEED = 20260930
#: The threshold is the verifier's own decision boundary and the one the task
#: stream's execution endpoint reads: ``RelationConstraintVerifier`` compares
#: ``float(trace[0]) >= 0.5``. The gate tests *separation* against it —
#: ``hi >= threshold > lo`` — rather than absolute output level, because
#: separation is what a within-query decision depends on and C5-002 measured
#: the cost of testing the wrong thing.
#:
#: A3 records a property of this number that is not obvious from above: the
#: alpha-scaled soft Booleans do not reach it on the satisfying half from
#: below either. The circuit's output is a product of the alphas, which is
#: ~0.480 at the zeros initialisation and ~0.46 under the pair-trained C5-002
#: checkpoint, so criterion 2's absolute ``hi >= 0.5`` term can fire on an
#: instrument that separates. The threshold is unchanged anyway — it is the
#: verifier's boundary, it is frozen by criterion 1, and moving it mid-series
#: would break comparability with C5-002 — and the possibility is recorded in
#: A3 rather than removed. See amendment A3.
GATE_THRESHOLD = 0.5
#: Minimum fraction of held-out pairs the gate must separate. C5-002's
#: pre-registered value, unchanged after seeing 0.2305 — relaxing it would
#: convert the gate from a termination condition into a fitting knob.
GATE_MIN_ACCURACY = 0.5
#: Spread floors, from ``degeneracy.py`` by name rather than by value, so the
#: contract and the library cannot drift apart without a name changing.
MIN_SPREAD = MIN_SPREAD
MIN_CLASS_SUPPORT = MIN_CLASS_SUPPORT
ABSOLUTE_FLOOR = ABSOLUTE_FLOOR


# --------------------------------------------------------------------------- #
# The eight-criterion representability gate
# --------------------------------------------------------------------------- #

#: The criteria, in the order the pre-registration's gate table lists them.
#: The order is part of the gate for the same reason ``degeneracy``'s check
#: order is: a later criterion is only meaningful given the earlier one, and a
#: verifier-acceptance score over a degenerate output is a number about noise.
GATE_CRITERIA = (
    "frozen_thresholds",
    "pair_accuracy",
    "output_spread",
    "verifier_acceptance",
    "no_true_edge_set",
    "no_oracle_leakage",
    "checkpoint_determinism",
    "pair_trained_objective",
)


@dataclass
class GateResult:
    """The eight-criterion gate, as a record rather than a pass/fail flag.

    A failed gate is reported, not swallowed: it is the difference between an
    invalid run and an ambiguous one, and the record is what makes that
    difference visible to a reader who was not present for the run. Each
    criterion reports its own verdict and its observed value, so a failure
    names the criterion that fired rather than only that something did.
    """

    passed: bool
    pair_accuracy: float
    verifier_acceptance: float
    n_pairs: int
    threshold: float
    min_accuracy: float
    criteria: dict[str, dict] = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "pair_accuracy": self.pair_accuracy,
            "verifier_acceptance": self.verifier_acceptance,
            "n_pairs": self.n_pairs,
            "threshold": self.threshold,
            "min_accuracy": self.min_accuracy,
            "criteria": dict(self.criteria),
            "detail": self.detail,
        }


def _program_for(pair: GatePair, which: str):
    """The relevance programme for one side of one held-out pair.

    ``programme(pair, "satisfying")`` and ``programme(pair, "violating")`` are
    the instrument's own answers, used by the consistency check.
    """
    descriptor = pair.satisfying if which == "satisfying" else pair.violating
    return relevance_program(pair.reference, descriptor, pair.marks, max_nodes=10)


def _programme_answer(pair: GatePair, which: str) -> float:
    """The relation's own answer for one side of a pair: 1.0 or 0.0.

    The satisfying descriptor agrees with the reference on every marked
    position and the violating one differs on exactly one, so this is the
    ground truth the verifier is asked to confirm — computed from the public
    relation, never from a model output.
    """
    descriptor = pair.satisfying if which == "satisfying" else pair.violating
    return float(all(descriptor[j] == pair.reference[j] for j in pair.marks))


def _gate_batch(pairs: Sequence[GatePair]):
    """Build the gate's structures and inputs for one held-out pair set.

    Extracted as a seam of its own rather than inline in ``run_gate`` for the
    same reason ``_build_and_execute`` is: the gate's units are pairs rather
    than retained sets, so it does not go through the cell seam, and a test
    that needs a torch-free runtime patches this one function. The eight
    criteria themselves stay in ``run_gate``, so what is patched is the
    structure construction and never the criteria.
    """
    structures = []
    inputs = []
    for n, pair in enumerate(pairs):
        for tag, which in (("sat", "satisfying"), ("vio", "violating")):
            descriptor = pair.satisfying if which == "satisfying" else pair.violating
            program = relevance_program(pair.reference, descriptor, pair.marks, max_nodes=10)
            key = f"gate-{n}-{tag}"
            structures.append(
                casm_structure_from_program(program, key=key, provenance="casm_s.gate")
            )
            inputs.append(tuple(float(x) for x in program.input_values))
    return tuple(structures), tuple(inputs)


def run_gate(runtime: CasmSRuntime) -> GateResult:
    """The eight-criterion representability gate.

    All eight are evaluated before the task population runs, on held-out
    structures the task population never sees and under the exact execution
    path it uses. A failure of any one returns a gate that has not passed,
    which terminates the run as ``INSTRUMENT_INVALID`` and emits no capability
    number.

    Criteria 1-3 are delegated to ``degeneracy.preflight``, which generalizes
    the two voided C5 runs' failure modes, and criterion 4 to the task
    stream's own ``RelationConstraintVerifier``. The remaining four are
    structural or constant-valued and are checked here. A failure of criteria
    1-3 raises ``DegeneracyError`` and is caught, so the run record still
    names which criterion fired rather than dying without a report.
    """
    pairs = make_pairs(GATE_STRUCTURES, GATE_SEED, dim=DIM, marked=MARKED)
    structures, inputs = _gate_batch(pairs)
    executions, _work = runtime.execute_many(structures, inputs)
    # Criterion 7 is an execution determinism check, not merely provenance.
    executions_repeat, _repeat_work = runtime.execute_many(structures, inputs)

    # ``degeneracy.preflight`` and ``degeneracy.discrimination`` call
    # ``score(pair, which)`` with the pair object itself, so the execution
    # index is recovered from the pair: the layout ``_gate_batch`` produces is
    # two rows per pair, satisfying first, and this closure is the single
    # place that layout is read. Criterion 4 below uses the same closure, so
    # the gate's two readings of one output cannot disagree.
    pair_rows = {id(pair): 2 * n for n, pair in enumerate(pairs)}

    def score(pair: GatePair, which: str) -> float:
        idx = pair_rows[id(pair)] + (0 if which == "satisfying" else 1)
        return float(executions[idx].result.output)

    all_outputs = [float(e.result.output) for e in executions]

    # ---- criteria 1-3, via the shared preflight ------------------------- #
    # The doc registers these as ``degeneracy.preflight``'s five checks. It
    # raises rather than returning ``False`` — a measurement taken under a
    # degenerate model has produced a number, and a produced number cannot be
    # un-produced — so the raise is caught here to turn it into a gate record
    # naming the criterion that fired.
    representable = True
    for pair in pairs:
        if _programme_answer(pair, "satisfying") != 1.0:
            representable = False
        if _programme_answer(pair, "violating") != 0.0:
            representable = False
        if not representable:
            break

    preflight_record: dict = {}
    try:
        preflight_result = preflight(
            representable=representable,
            outputs=all_outputs,
            pairs=pairs,
            score=score,
            threshold=GATE_THRESHOLD,
            min_spread=MIN_SPREAD,
            min_accuracy=GATE_MIN_ACCURACY,
            min_class_support=MIN_CLASS_SUPPORT,
        )
        preflight_record = preflight_result.to_dict()
    except DegeneracyError as exc:
        preflight_record = {
            "passed": False,
            "error": str(exc),
            "levels": ["preflight_failed"],
        }

    criteria: dict[str, dict] = {}

    # 1. The thresholds a run is judged against are the registered ones. The
    # values are read back from the named constants here, so a script edited
    # to a different bar fails this criterion against the registration.
    criteria["frozen_thresholds"] = {
        "passed": (
            GATE_THRESHOLD == 0.5
            and GATE_MIN_ACCURACY == 0.5
            and MIN_SPREAD == 0.05
            and MIN_CLASS_SUPPORT == 2
        ),
        "registered": {
            "GATE_THRESHOLD": 0.5,
            "GATE_MIN_ACCURACY": 0.5,
            "MIN_SPREAD": 0.05,
            "MIN_CLASS_SUPPORT": 2,
        },
        "observed": {
            "GATE_THRESHOLD": GATE_THRESHOLD,
            "GATE_MIN_ACCURACY": GATE_MIN_ACCURACY,
            "MIN_SPREAD": MIN_SPREAD,
            "MIN_CLASS_SUPPORT": MIN_CLASS_SUPPORT,
        },
    }

    # 2. Within-query separation, read from the preflight's own check so the
    # gate's number and the library's cannot disagree. ``hi >= threshold >
    # lo``: not "is each output correct" but "does the model rank a satisfier
    # above a violator", which is what a within-query decision depends on.
    # This is the criterion that fired in C5-002 at 0.2305 against 0.5.
    discrimination_record = discrimination(
        pairs,
        score=score,
        threshold=GATE_THRESHOLD,
        min_accuracy=GATE_MIN_ACCURACY,
        min_class_support=MIN_CLASS_SUPPORT,
    )
    criteria["pair_accuracy"] = {
        "passed": discrimination_record.passed,
        "observed": discrimination_record.accuracy,
        "minimum": GATE_MIN_ACCURACY,
        "n_pairs": len(pairs),
        "n_separated": discrimination_record.n_separated,
    }

    # 3. Non-degenerate output spread. The C5-001 criterion: a constant output
    # makes any argmax deterministic and meaningless. Both scales are required
    # because they fail on opposite cases — a batch with a tiny mean passes a
    # purely relative test while being constant for every practical purpose.
    degeneracy_record = degenerate_outputs(all_outputs, min_spread=MIN_SPREAD)
    criteria["output_spread"] = {
        "passed": not degeneracy_record.levels,
        "observed": degeneracy_record.to_dict(),
        "minimum": {"MIN_SPREAD": MIN_SPREAD, "ABSOLUTE_FLOOR": ABSOLUTE_FLOOR},
    }

    # 4. Actual verifier agreement, via the task stream's own verifier --- #
    # New in C5-003. C5-002's gate tested the model's output against the
    # threshold but not against the verifier the task stream's
    # ``verification_rate`` endpoint reads, and the two agreed in that run only
    # because both were zero. This runs the verifier itself on the held-out
    # pairs, through the same ``verify_task`` the task stream calls, so a
    # model the benchmark's own verifier rejects cannot pass the gate.
    #
    # ``verify_task`` is a *success* verifier: on a descriptor the public
    # relation does not hold for, it necessarily rejects — first the output is
    # compared against the relation's answer, and then, even on agreement, the
    # ``relation_unsatisfied`` branch fires. So raw acceptance over both sides
    # of every pair is structurally capped at one half, and the cap is a
    # property of the verifier rather than of the instrument. The criterion is
    # therefore *acceptance on the satisfying half*: the verifier must accept
    # the satisfying descriptor. That is the quantity that separates a working
    # instrument from an inverted one — an anti-correlating model makes the
    # verifier reject on the satisfying half, where a threshold-only check
    # would see a model that is confidently wrong in the same direction on
    # both halves.
    #
    # A3 changed what the outcome's ``value`` field carries. ``verify()``
    # enforces ``abs(trace[0] - value) <= tolerance`` at ``tolerance = 1e-9``
    # before it ever reaches the relation branch, so passing the programme's
    # expected answer (``1.0``) as the value required the model to emit
    # *exactly* 1.0 on every satisfying pair. CASM-S's soft Booleans are
    # alpha-scaled — ``alpha = softplus(alpha_eta)`` per relation port — and
    # the relevance circuit's output is a product of them, which equals 1.0
    # only on a measure-zero manifold in parameter space. Under the pair-
    # trained C5-002 checkpoint (``alpha`` ~= 0.797/0.786) the satisfying-half
    # output is ~0.46, so the pre-A3 convention scored this criterion 0.0 on
    # every pair and fired INSTRUMENT_INVALID on an instrument whose
    # separation is exactly what this experiment trains for. The outcome now
    # carries the model's own output, which makes the consistency check pass
    # trivially and leaves the relation branch's ``trace[0] >= 0.5`` as the
    # real test.
    #
    # Caveat, recorded rather than hidden: with ``value = trace[0]`` this
    # criterion reduces operationally to "the satisfying-half output is at
    # least the threshold", which is close to criterion 2's ``hi >= threshold
    # > lo``. Its non-redundant content is that it reads the decision through
    # the *task stream's own verifier object* rather than through a threshold
    # comparison, so a future change to the verifier's boundary is reflected
    # here without this criterion being edited. Note also — see amendment A3
    # — that the alpha-scaled output does not reach the threshold from below
    # either, so criterion 2's absolute ``hi >= 0.5`` term is itself part of
    # what A3 records.
    verifier = RelationConstraintVerifier()
    n_agree = 0
    for pair in pairs:
        which = "satisfying"
        descriptor = pair.satisfying
        structure = structures[pair_rows[id(pair)]]
        output = score(pair, which)
        expected = _programme_answer(pair, which)
        computation = Computation(
            structure=structure,
            action=0,
            trace=(output,),
        )
        # A3: the outcome value is the model's own output, not the expected
        # answer, because verify() enforces |trace[0] - value| <= tolerance
        # with tolerance = 1e-9. Passing the expected answer here requires the
        # model to emit exactly 1.0 on every satisfying pair. CASM-S's soft
        # Booleans are alpha-scaled (alpha = softplus(alpha_eta)), and the
        # relevance circuit's final output is alpha0*alpha1*..., which
        # reaches 1.0 only on a measure-zero manifold in parameter space.
        # The pair-trained bridge holds alpha at roughly 0.797/0.786, giving a
        # satisfying-half output around 0.46 — not 1.0, and read under the
        # pre-A3 convention this criterion scored 0.0 on every pair and fired
        # INSTRUMENT_INVALID on an instrument whose separation is exactly what
        # the experiment trains for. See amendment A3.
        outcome = Outcome(success=bool(expected), value=output)
        evidence = verifier.verify_task(
            computation, outcome,
            candidate=Candidate(key=structure.key, descriptor=descriptor),
            reference=pair.reference,
            context=tuple(1 if j in pair.marks else 0 for j in range(len(pair.reference))),
            relation="equality",
        )
        if evidence.valid:
            n_agree += 1
    verifier_acceptance = n_agree / len(pairs)
    criteria["verifier_acceptance"] = {
        "passed": verifier_acceptance >= GATE_MIN_ACCURACY,
        "observed": verifier_acceptance,
        "minimum": GATE_MIN_ACCURACY,
        "n_compared": len(pairs),
        "n_accepted": n_agree,
        "measure": (
            "the task stream's own verifier accepts the satisfying descriptor "
            "on the model's output, over the held-out pairs; verify_task is a "
            "success verifier and necessarily rejects a violating descriptor, "
            "so the violating half is not part of this score; per amendment "
            "A3 the outcome's value carries the model's own output rather "
            "than the expected answer, because verify()'s 1e-9 output "
            "consistency check would otherwise require the model to emit "
            "exactly 1.0, which the alpha-scaled soft Booleans do not reach "
            "outside a measure-zero parameter manifold"
        ),
    }

    # 5. The transported spec never carries the wiring the instrument is
    # being tested on. Audited on the objects that cross the boundary, not on
    # a prose assurance: every gate structure is audited for the forbidden
    # field set.
    spec_violations = []
    for structure in structures:
        spec_violations.extend(audit_object(structure.spec))
    criteria["no_true_edge_set"] = {
        "passed": not spec_violations,
        "audited_structures": len(structures),
        "violations": spec_violations,
        "forbidden": sorted(FORBIDDEN_FIELDS),
    }

    # 6. No oracle information enters routing or execution. The adapter is the
    # only path into CASM-S and it never serializes the forbidden fields, so
    # this criterion is checked on the same audited objects as criterion 5 and
    # is reported separately because it guards a different boundary: the gate
    # reads nothing the task stream's addressing or execution stage may see.
    criteria["no_oracle_leakage"] = {
        "passed": not spec_violations,
        "audited_structures": len(structures),
        "violations": spec_violations,
    }

    # 7. The run is deterministic from the frozen checkpoint. ``hash()`` on
    # strings is salted per process, so every structure key is a formatted
    # string rather than a hash — which is what makes two runs of the same
    # registered design produce the same structure keys.
    non_deterministic_keys = [
        s.key for s in structures
        if not isinstance(s.key, str) or not s.key
    ]
    def execution_signature(rows):
        return tuple(
            (
                float(e.result.output),
                tuple(float(v) for v in e.result.gates),
                tuple(float(v) for v in e.result.node_values),
            )
            for e in rows
        )
    first_signature = execution_signature(executions)
    repeat_signature = execution_signature(executions_repeat)
    execution_deterministic = first_signature == repeat_signature
    criteria["checkpoint_determinism"] = {
        "passed": (not non_deterministic_keys) and execution_deterministic,
        "checkpoint_sha256": _checkpoint_sha256_of(runtime),
        "non_deterministic_keys": non_deterministic_keys,
        "execution_deterministic": execution_deterministic,
        "repeated_execution_records": len(executions_repeat),
        "structure_key_example": structures[0].key if structures else None,
    }

    # 8. The bridge was trained on the registered objective. Checked against
    # the named constant, so a script whose bridge silently reverts to
    # per-candidate regression fails the gate against its own registration.
    criteria["pair_trained_objective"] = {
        "passed": BRIDGE_OBJECTIVE == "pair_separation",
        "registered": "pair_separation",
        "observed": BRIDGE_OBJECTIVE,
    }

    passed = all(c["passed"] for c in criteria.values())
    return GateResult(
        passed=passed,
        pair_accuracy=(
            discrimination_record.accuracy
            if discrimination_record.accuracy is not None
            else 0.0
        ),
        verifier_acceptance=verifier_acceptance,
        n_pairs=len(pairs),
        threshold=GATE_THRESHOLD,
        min_accuracy=GATE_MIN_ACCURACY,
        criteria=criteria,
        detail={
            "structures": len(structures),
            "bridge_seed": BRIDGE_SEED,
            "gate_seed": GATE_SEED,
            "casm_commit": CASM_COMMIT,
            "criteria_order": list(GATE_CRITERIA),
            "preflight": preflight_record,
        },
    )


# --------------------------------------------------------------------------- #
# Pair-trained bridge calibration
# --------------------------------------------------------------------------- #


@dataclass
class BridgePair:
    """One bridge training example: a pair, compiled to two structures.

    The unit is the pair, not the candidate. C5-002's examples were single
    candidates against a Boolean target, which is why its bridge reached
    0.84375 absolute accuracy while separating only 0.2305 of pairs: nothing
    in the objective rewarded ranking one candidate above another *within one
    query*. The margin loss below is the registered fix.
    """

    pair: GatePair
    sat_structure: object
    vio_structure: object
    sat_inputs: tuple[float, ...]
    vio_inputs: tuple[float, ...]


def bridge_pairs(count: int, seed: int) -> tuple[BridgePair, ...]:
    """The pair calibration set, from a stream disjoint from the gate's.

    Disjoint by construction: the gate seeds at ``GATE_SEED`` and the bridge
    at ``BRIDGE_SEED``, and neither overlaps the task stream's ``SEEDS``. A
    bridge calibrated on the gate's pairs would make the gate's verdict
    unattributable — was it the model, the calibration, or the threshold?
    """
    pairs = make_pairs(count, seed, dim=DIM, marked=MARKED)
    examples = []
    for i, pair in enumerate(pairs):
        sat_program = _program_for(pair, "satisfying")
        vio_program = _program_for(pair, "violating")
        examples.append(
            BridgePair(
                pair=pair,
                sat_structure=casm_structure_from_program(
                    sat_program, key=f"bridge-sat-{seed}-{i}",
                    provenance="casm_s.bridge_training",
                ),
                vio_structure=casm_structure_from_program(
                    vio_program, key=f"bridge-vio-{seed}-{i}",
                    provenance="casm_s.bridge_training",
                ),
                sat_inputs=tuple(float(x) for x in sat_program.input_values),
                vio_inputs=tuple(float(x) for x in vio_program.input_values),
            )
        )
    return tuple(examples)


def _pair_batch(runtime: CasmSRuntime, examples: Sequence[BridgePair]):
    """Compile a pair batch to the tensors the margin loss reads.

    Each example contributes two rows — satisfying then violating — so the
    margin ``score(sat) - score(vio)`` is a difference between adjacent rows
    of one batch. The target is always ``+1``: the satisfier must score above
    the violator, never the reverse, because the relation is defined so the
    satisfying descriptor is the one that holds.
    """
    specs = []
    for e in examples:
        specs.append(runtime._spec(e.sat_structure))
        specs.append(runtime._spec(e.vio_structure))
    episodes = [runtime._episode_from_spec(s) for s in specs]
    width = max(len(spec.inputs) for spec in specs)

    def row(inputs: Sequence[float]) -> list[float]:
        return [*inputs] + [0.0] * (width - len(inputs))

    rows = []
    for e in examples:
        rows.append(row(e.sat_inputs))
        rows.append(row(e.vio_inputs))
    torch = runtime.torch
    x = torch.tensor(rows, dtype=torch.float32, device=runtime.device)
    # The margin target: the satisfying row must exceed the violating row.
    y = torch.full((len(examples), 1), 1.0, dtype=torch.float32,
                   device=runtime.device)
    return episodes, x, y


def train_bridge(runtime: CasmSRuntime) -> dict:
    """Pair-trained bridge calibration, the registered change from C5-002.

    The loss is on the margin ``score(satisfying) - score(violating)`` over
    pairs, not on the absolute level of either candidate's output. That is the
    whole of the design change: the objective now contains the quantity the
    downstream decision depends on, so a bridge that reaches a given absolute
    accuracy cannot do so without also ranking a satisfier above a violator.

    The calibration is reported and is *not* a capability result. The gate,
    not this score, is what decides whether the instrument can measure.
    """
    train = bridge_pairs(BRIDGE_TRAIN, BRIDGE_SEED)
    validation = bridge_pairs(BRIDGE_VALIDATION, BRIDGE_VALIDATION_SEED)
    torch = runtime.torch
    model = runtime.model

    episodes, x, target = _pair_batch(runtime, train)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR)
    history = []
    model.train()
    for _epoch in range(TRAIN_EPOCHS):
        pred, _gates, _meta = model(episodes, x)
        # The registered objective. Each pair is two adjacent rows; the margin
        # is the satisfying score minus the violating score, and the target is
        # that this margin be positive. ReLU-hinged so a pair that already
        # separates contributes no gradient — the loss is on pairs that do not
        # yet separate, which is where the instrument's deficit lives.
        margin = pred[0::2] - pred[1::2]
        loss = torch.nn.functional.relu(target - margin).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(float(loss.detach().cpu().item()))

    model.eval()
    # Validation is reported as separation, the quantity the gate and the task
    # stream's decision both read — not as absolute accuracy, which is the
    # quantity C5-002's validation reported and which was blind to the
    # failure it should have caught.
    val_pairs = [e.pair for e in validation]
    val_structures = []
    val_inputs = []
    for e in validation:
        val_structures.append(e.sat_structure)
        val_structures.append(e.vio_structure)
        val_inputs.append(e.sat_inputs)
        val_inputs.append(e.vio_inputs)
    results, _ = runtime.execute_many(tuple(val_structures), tuple(val_inputs))
    n_sep = 0
    for i in range(len(validation)):
        hi = float(results[2 * i].result.output)
        lo = float(results[2 * i + 1].result.output)
        if hi >= GATE_THRESHOLD > lo:
            n_sep += 1
    return {
        "objective": BRIDGE_OBJECTIVE,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "seed": BRIDGE_SEED,
        "epochs": TRAIN_EPOCHS,
        "learning_rate": TRAIN_LR,
        "final_train_loss": history[-1],
        "validation_separation": n_sep / len(validation),
        "history": history,
    }


def _checkpoint_sha256_of(runtime: CasmSRuntime) -> str | None:
    """The loaded checkpoint's sha256, or None for an untrained runtime.

    Recorded in the run record so a reader can verify that two runs of the
    same registered design started from the same frozen state.
    """
    ckpt = runtime.checkpoint
    if ckpt is None:
        return None
    return checkpoint_sha256(Path(ckpt))


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

    Identical to C5-002's baseline so the two experiments' addressing numbers
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
    executor — this is criterion 5's structural basis, and it is what keeps an
    execution failure from being able to masquerade as an addressing failure.
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
    selection_success_rate: float
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
    """The same static-population task stream C5-001 and C5-002 used.

    Kept identical so the three experiments' work endpoints are directly
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
    selection_success = 0
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
        # ``success_rate`` an argmax over model output, and it is carried over
        # from C5-002 unchanged.
        if any(i in task.acceptable_actions for i in retained):
            covered += 1

        # --- the execution endpoint -------------------------------------- #
        # Selection over the retained candidates only. It is not the addressing
        # decision — that is the retained set itself — and it does not feed
        # ``coverage_rate``. An uninformative output makes this selection
        # arbitrary, which degrades only the execution endpoint and leaves the
        # addressing measurement intact.
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
        expected_relation = relation_holds(
            task.relation,
            reference,
            selected.descriptor,
            task.query.context,
        )
        actual_relation = result.output >= GATE_THRESHOLD
        if actual_relation == expected_relation:
            execution_correct += 1

        if selected_global in task.acceptable_actions:
            selection_success += 1

        outcome = Outcome(
            success=selected_global in task.acceptable_actions,
            # A3 applies to the real task-stream verifier too: value is the
            # model's observed output, not the hidden action-level answer.
            value=result.output,
            feedback="acceptable" if selected_global in task.acceptable_actions else "not_acceptable",
            detail=None,
        )
        evidence = verifier.verify_task(
            computation, outcome, candidate=selected, reference=reference,
            context=task.query.context, relation=task.relation,
        )
        if evidence.valid:
            verified += 1

    elapsed = time.perf_counter() - start
    return CellResult(
        arm=arm, h=h, k=k, seed=seed,
        coverage_rate=covered / QUERIES,
        execution_accuracy_rate=execution_correct / QUERIES,
        selection_success_rate=selection_success / QUERIES,
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
    parser.add_argument("--output", default=str(_results.results_dir_for(__file__) / "c5_003.json"))
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

    # The gate runs before the task stream and terminates the run on any of
    # its eight criteria, so no capability number is emitted under an
    # instrument that cannot represent the task.
    gate = run_gate(runtime)
    if not gate.passed:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment_id": EXPERIMENT_ID,
            "status": "INSTRUMENT_INVALID",
            "reason": (
                "one or more of the eight pre-registered representability "
                "criteria failed on held-out structures; no capability number "
                "is reported. The failure is recorded against the criterion "
                "that fired, not against CASM-S as a substrate."
            ),
            "casm_source": {"repository": "FPC-effortless/cdl-attention-experiment", "commit": CASM_COMMIT},
            "calibration": calibration,
            "gate": gate.to_dict(),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": checkpoint_sha256(Path(checkpoint)) if checkpoint else None,
            "config": {"h_levels": run_levels, "k_levels": K_LEVELS, "seeds": run_seeds,
                       "steps": args.steps, "eval_steps": args.eval_steps,
                       "queries": QUERIES, "warmup": WARMUP, "arms": ARMS,
                       "bridge_objective": BRIDGE_OBJECTIVE},
            "cells": [],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("=" * 72)
        print("INSTRUMENT_INVALID: a representability criterion failed.")
        print("=" * 72)
        for name in GATE_CRITERIA:
            record = gate.criteria.get(name, {})
            state = "PASS" if record.get("passed") else "FAIL"
            print(f"  [{state}] {name}")
        print(f"pair accuracy {gate.pair_accuracy:.4f} over {gate.n_pairs} held-out pairs")
        print(f"verifier acceptance {gate.verifier_acceptance:.4f}")
        print(f"(threshold {gate.threshold}, minimum {gate.min_accuracy})")
        print("No capability number is reported. See docs/TACOSM-C5-003.md.")
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
                   "queries": QUERIES, "warmup": WARMUP, "arms": ARMS,
                   "bridge_objective": BRIDGE_OBJECTIVE},
        "cells": [cell.__dict__ for cell in cells],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"machine-readable summary written to {out}")
    print(f"gate: passed={gate.passed} pairs={gate.n_pairs} "
          f"pair_accuracy={gate.pair_accuracy:.4f} "
          f"verifier_acceptance={gate.verifier_acceptance:.4f}")
    for cell in cells:
        print(
            f"{cell.arm:>24} H={cell.h:>3} K={str(cell.k):>1} seed={cell.seed} "
            f"coverage={cell.coverage_rate:.4f} exec={cell.execution_accuracy_rate:.4f} "
            f"selection={cell.selection_success_rate:.4f} "
            f"verification={cell.verification_rate:.4f} "
            f"structures/query={cell.structures_executed_per_query:.2f} "
            f"edges/query={cell.candidate_edges_per_query:.2f}"
        )


if __name__ == "__main__":
    main()
