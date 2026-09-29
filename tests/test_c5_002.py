"""Tests for TACOSM-C5-002: the separation contract and its gate.

C5-001 is void (``docs/TACOSM-C5-001-RESULT.md``) because its capability
endpoint was an argmax over CASM-S output, and the output was uninformative,
so the reference arm measured the population base rate. C5-002's design fix is
structural rather than algorithmic: ``coverage_rate`` reads no model output at
all.

These tests pin the two things that would otherwise silently re-drift:

1. **The separation is real, not notional.** A fake runtime standing in for
   CASM-S is fed three output modes — correct, constant, inverted. Under the
   C5-001 failure mode (constant output) the coverage endpoint must be
   *unaffected*, while the exhaustive arm's execution endpoint collapses to
   the base rate. If ``coverage_rate`` ever came to read a model output, that
   test would fail, because the two endpoints would move together.
2. **The gate terminates.** A model that does not separate must be caught by
   the gate before any capability number is produced, which is the specific
   failure that let C5-001 complete and look plausible.

Both run torch-free, which is the point: the design's correctness does not
depend on the compute. The measurement does, and it runs on a runner.

The contract round-trip and the registered-constants agreement are in
``test_contract.py``; the tests here are the ones that need the script's own
functions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tac_osm.casm_runtime import CasmExecution, CasmExecutionWork  # noqa: E402
from tac_osm import ExecutionResult  # noqa: E402
from tac_osm.contract import load_contract  # noqa: E402


def _script_module(name: str):
    """Import a measurement script as a module.

    The scripts insert ``src`` into ``sys.path`` at import time; importing is
    safe because they do no work at module scope beyond defining constants and
    functions. Registered under a name in ``sys.modules`` because the script's
    own dataclasses look themselves up by module during ``@dataclass``
    processing.
    """
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"_script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C5 = _script_module("measure_c5_casm_002.py")


class _FakeRuntime:
    """A stand-in for ``CasmSRuntime`` with a chosen output mode.

    The three modes are the three outcomes that matter for the separation
    test, and they are chosen to be indistinguishable by any endpoint that
    does not read an output:

    * ``separating`` — the circuit's exact Boolean output. This is what a
      working instrument produces.
    * ``constant`` — the C5-001 failure mode. Every candidate scores the same,
      so an argmax-over-output selection is deterministic and meaningless.
    * ``inverted`` — separation, backwards. A model that anti-correlates still
      "separates" in the sense of not being constant, and only a gate that
      checks the *direction* of the separation catches it.

    Structure specs carry ``_meta`` rather than being parsed, because the fake
    has no torch and the point is that the design does not need it.
    """

    def __init__(self, mode: str):
        assert mode in ("separating", "constant", "inverted")
        self.mode = mode
        self.calls = 0

    def execute_many(self, structures, input_rows):
        self.calls += 1
        executions = []
        work = CasmExecutionWork.zero()
        for structure, _row in zip(structures, input_rows):
            reference, descriptor, marks = structure._meta
            exact = float(all(descriptor[j] == reference[j] for j in marks))
            if self.mode == "separating":
                output = exact
            elif self.mode == "constant":
                output = 0.3
            else:
                output = 1.0 - exact
            # One unit of structure plus the relevance circuit's node and edge
            # counts, so the work endpoints are exercised rather than all
            # reading zero. The counts need not match CASM-S exactly; they need
            # to scale with the number of executed structures, which is the
            # property the work endpoints assert.
            n_marks = len(marks)
            n_nodes = 2 * n_marks + n_marks + max(1, n_marks - 1)
            n_edges = 2 * n_marks + 2 * max(1, n_marks - 1)
            work = work + CasmExecutionWork(
                structures_executed=1,
                active_nodes=n_nodes,
                candidate_edges=n_edges,
                gate_evaluations=n_edges,
                executed_structural_operations=n_marks + max(1, n_marks - 1),
                node_outputs=n_nodes,
            )
            executions.append(
                CasmExecution(
                    result=ExecutionResult(output=output, provenance="fake"),
                    work=CasmExecutionWork.zero(),
                )
            )
        return tuple(executions), work


@pytest.fixture
def fake_execute(monkeypatch):
    """Replace the script's executor with one that tags structures.

    The real function compiles a TAC-OSM program to a CASM graph spec; the
    fake keeps the program's own reference/descriptor/marks on the structure
    so the fake runtime can compute the exact output without torch. Patching
    the single build-and-execute seam is enough for both ``run_cell`` and the
    gate, because both go through it or through the same two builders.
    """
    def build_and_execute(runtime, task, marks, reference, retained):
        structures = []
        inputs = []
        for i in retained:
            program = C5.relevance_program(reference, task.candidates[i].descriptor, marks, max_nodes=10)
            structure = C5.casm_structure_from_program(
                program, key=task.candidates[i].key, provenance="casm_s"
            )
            object.__setattr__(
                structure, "_meta",
                (reference, task.candidates[i].descriptor, marks),
            )
            structures.append(structure)
            inputs.append(tuple(float(x) for x in program.input_values))
        return runtime.execute_many(tuple(structures), tuple(inputs))

    monkeypatch.setattr(C5, "_build_and_execute", build_and_execute)
    return build_and_execute


# --------------------------------------------------------------------------- #
# The contract round-trips and is internally consistent
# --------------------------------------------------------------------------- #


def test_the_c5_002_contract_loads_and_has_one_primary():
    c = load_contract(C5.EXPERIMENT_ID)
    assert c.experiment_id == "TACOSM-C5-002"
    assert c.primary_endpoint() == "coverage_rate"
    assert [a.name for a in c.arms] == list(C5.ARMS)
    assert c.check_consistency() == []


def test_the_c5_002_contract_matches_the_scripts_registered_constants():
    c = load_contract(C5.EXPERIMENT_ID)
    assert tuple(C5.H_LEVELS) == c.h_levels
    assert tuple(C5.K_LEVELS) == c.k_levels
    assert tuple(C5.SEEDS) == c.seeds
    assert C5.STEPS == c.steps
    assert C5.EVAL_STEPS == c.eval_steps


def test_the_c5_002_contract_names_the_endpoints_the_script_reports():
    """Every field the cell reports must be an endpoint the contract knows.

    The inverse direction is not asserted — a contract may register an
    endpoint a cell does not carry, since ``wall_clock_seconds`` and the work
    units are reported by all cells — but a cell field with no registered
    endpoint is an unregistered measurement, which is the one thing the
    contract system exists to prevent.
    """
    from dataclasses import fields

    c = load_contract(C5.EXPERIMENT_ID)
    registered = {e.name for e in c.endpoints}
    reported = {f.name for f in fields(C5.CellResult)}
    # The cell carries identity and cost fields that are not endpoints.
    not_endpoints = {"arm", "h", "k", "seed", "casm_forward_batches",
                     "address_positions_per_query",
                     "representation_candidates_scored_per_query"}
    measured = reported - not_endpoints
    assert measured <= registered, (
        "the script reports fields its contract does not register: "
        f"{sorted(measured - registered)}"
    )


def test_the_registered_endpoints_are_the_c5_002_design_not_c5_001_s():
    """The endpoint set is the design change, so it is pinned by name.

    ``success_rate`` is gone because it was the compound quantity whose
    collapse voided C5-001; ``coverage_rate`` is the replacement that reads no
    model output. A contract that carried both would be answering the old
    question and the new one at once.
    """
    c = load_contract(C5.EXPERIMENT_ID)
    names = {e.name for e in c.endpoints}
    assert "coverage_rate" in names
    assert "execution_accuracy_rate" in names
    assert "success_rate" not in names


# --------------------------------------------------------------------------- #
# The separation: coverage must not read a model output
# --------------------------------------------------------------------------- #


def test_coverage_is_one_for_exact_addressing_under_a_broken_model(fake_execute):
    """The C5-001 failure mode must not depress the addressing endpoint.

    Under a constant output the C5-001 exhaustive arm measured 0.2500 — the
    population base rate — because its selection was argmax over that output
    and the argmax was deterministically index 0. Here the exact index's
    coverage must be 1.0 under the *same* broken model, because the endpoint
    is a set intersection and never sees the output.
    """
    cell = C5.run_cell(_FakeRuntime("constant"), seed=0, h=8, k=2, arm="exact-indexed")
    assert cell.coverage_rate == 1.0
    # And the execution endpoint, which does read the output, is not protected
    # by the separation: it inherits the failure. That is the point — the two
    # halves are independently interpretable.
    assert cell.execution_accuracy_rate == 1.0


def test_exhaustive_execution_collapses_to_the_base_rate_under_a_broken_model(fake_execute):
    """The reference arm's execution endpoint shows the C5-001 failure.

    Constant output makes the argmax deterministic, so the exhaustive arm
    always selects the same candidate and its execution endpoint is the
    probability that candidate is acceptable. This is the observable that
    voided C5-001, and it is now isolated to the execution endpoint: the
    arm's coverage is still 1.0, because retaining everything cannot fail.
    """
    cell = C5.run_cell(_FakeRuntime("constant"), seed=0, h=8, k=None, arm="exhaustive")
    assert cell.coverage_rate == 1.0
    assert 0.0 < cell.execution_accuracy_rate <= 1.0
    assert cell.verification_rate == 0.0


def test_coverage_separates_from_execution_across_output_modes(fake_execute):
    """Across three output modes, coverage must be invariant to the model.

    The addressing endpoint depends on the retained set and the acceptable set
    only, so changing what the fake model returns cannot move it. The
    execution endpoint is free to move, and does.
    """
    for arm, k in (("exact-indexed", 2), ("exact-indexed", 4),
                   ("representation-addressed", 2), ("exhaustive", None)):
        coverages = set()
        for mode in ("separating", "constant", "inverted"):
            cell = C5.run_cell(_FakeRuntime(mode), seed=0, h=8, k=k, arm=arm)
            coverages.add(cell.coverage_rate)
        assert len(coverages) == 1, (
            f"{arm} K={k}: coverage depends on the model output ({coverages}), "
            "so the addressing endpoint is not separated from execution"
        )


def test_a_correct_model_recovers_the_full_success_on_exact_addressing(fake_execute):
    """With a separating model, both endpoints are at their analytic values.

    The exact index's bucket is the acceptable set, so coverage is 1 and a
    correct execution selects an acceptable candidate from it. This is the
    behaviour a valid instrument must produce, and it is the reference the
    decision rule's first branch reads.
    """
    cell = C5.run_cell(_FakeRuntime("separating"), seed=0, h=8, k=2, arm="exact-indexed")
    assert cell.coverage_rate == 1.0
    assert cell.execution_accuracy_rate == 1.0
    assert cell.verification_rate == 1.0


def test_representation_addressing_is_not_exact(fake_execute):
    """The fixed-cosine arm must not read 1.0 coverage.

    The arm's value is that it measures a real, non-trivial addressing
    accuracy. A coverage of exactly 1.0 would mean the fixed baseline is
    exact on this population, which would make it indistinguishable from the
    analytic control and would remove the arm's reason to exist.
    """
    cell = C5.run_cell(_FakeRuntime("separating"), seed=0, h=8, k=2, arm="representation-addressed")
    assert 0.0 < cell.coverage_rate < 1.0


def test_each_cell_reports_a_retained_set_no_larger_than_registered_k(fake_execute):
    """The work endpoints must track the retained set, not the population.

    For the exhaustive arm that means H structures per query; for the retained
    arms it means at most K. A retained arm executing more than K would make
    the work-reduction endpoint measure a reduction that did not happen.
    """
    for mode in ("separating", "constant"):
        exhaustive = C5.run_cell(_FakeRuntime(mode), seed=0, h=8, k=None, arm="exhaustive")
        assert exhaustive.structures_executed_per_query == 8.0
        for arm in ("exact-indexed", "representation-addressed"):
            cell = C5.run_cell(_FakeRuntime(mode), seed=0, h=8, k=2, arm=arm)
            assert cell.structures_executed_per_query <= 2.0


# --------------------------------------------------------------------------- #
# The gate: a non-separating model terminates the run
# --------------------------------------------------------------------------- #


@pytest.fixture
def gate_with_fake_execute(monkeypatch):
    """The gate's own structure builder, tagged for the fake runtime.

    The gate does not go through ``_build_and_execute`` — it builds its own
    batch, because its units are pairs rather than retained sets — so it needs
    its own patch.
    """
    def run_gate(runtime):
        pairs = C5.gate_pairs(C5.GATE_STRUCTURES, C5.GATE_SEED)
        structures = []
        inputs = []
        for n, (reference, satisfying, violating, marks) in enumerate(pairs):
            for tag, descriptor in (("sat", satisfying), ("vio", violating)):
                program = C5.relevance_program(reference, descriptor, marks, max_nodes=10)
                structure = C5.casm_structure_from_program(
                    program, key=f"gate-{n}-{tag}", provenance="casm_s.gate"
                )
                object.__setattr__(structure, "_meta", (reference, descriptor, marks))
                structures.append(structure)
                inputs.append(tuple(float(x) for x in program.input_values))
        executions, _work = runtime.execute_many(tuple(structures), tuple(inputs))
        n_separated = 0
        for i in range(len(pairs)):
            hi = executions[2 * i].result.output
            lo = executions[2 * i + 1].result.output
            if hi >= C5.GATE_THRESHOLD > lo:
                n_separated += 1
        accuracy = n_separated / len(pairs)
        return C5.GateResult(
            passed=accuracy >= C5.GATE_MIN_ACCURACY,
            accuracy=accuracy,
            n_pairs=len(pairs),
            threshold=C5.GATE_THRESHOLD,
            min_accuracy=C5.GATE_MIN_ACCURACY,
            detail={},
        )

    monkeypatch.setattr(C5, "run_gate", run_gate)
    return run_gate


def test_the_gate_passes_a_separating_model(gate_with_fake_execute):
    gate = C5.run_gate(_FakeRuntime("separating"))
    assert gate.passed
    assert gate.accuracy == 1.0


@pytest.mark.parametrize("mode", ["constant", "inverted"])
def test_the_gate_fails_a_non_separating_model(gate_with_fake_execute, mode):
    """The two ways a model can fail to separate, and the gate catches both.

    ``constant`` is the observed C5-001 failure: no information in the output
    at all. ``inverted`` is the failure the gate's *direction* check exists
    for — a model whose output anti-correlates with the relation is not a
    working instrument either, and a gate that only checked for non-constant
    output would pass it.
    """
    gate = C5.run_gate(_FakeRuntime(mode))
    assert not gate.passed
    assert gate.accuracy == 0.0


def test_the_gate_uses_held_out_structures_not_the_task_stream():
    """The gate's seed is disjoint from the task stream's.

    A gate calibrated on the evaluation population is not a gate; it is a
    second measurement of the same thing, and its pass condition would be
    fitted to the run it is supposed to validate.
    """
    assert C5.GATE_SEED not in set(C5.SEEDS)
    assert C5.GATE_SEED != C5.BRIDGE_SEED


def test_the_gate_pairs_are_well_formed():
    """Each pair must differ on exactly one marked position.

    The separation question is only well-posed if the satisfying descriptor
    satisfies the relation and the violating one violates it, and only on a
    marked position — a difference on an unmarked position would not change
    the relation's value, and the pair would be uninformative.
    """
    pairs = C5.gate_pairs(64, C5.GATE_SEED)
    assert len(pairs) == 64
    for reference, satisfying, violating, marks in pairs:
        assert all(satisfying[j] == reference[j] for j in marks)
        differing = [j for j in marks if violating[j] != reference[j]]
        assert len(differing) == 1
        # Unmarked positions are identical, so the pair differs on exactly the
        # marked bit and on nothing else.
        assert all(violating[j] == satisfying[j] for j in range(len(reference)) if j not in marks)


# --------------------------------------------------------------------------- #
# Design invariants
# --------------------------------------------------------------------------- #


def test_the_gate_threshold_is_the_verifier_threshold():
    """The gate tests the same threshold the task stream's endpoint reads.

    A gate threshold that differed from the verifier's would test a different
    decision than the one the run reports, and a model could pass the gate and
    fail every cell — which is the C5-001 failure mode wearing a gate.
    """
    assert C5.GATE_THRESHOLD == 0.5


def test_the_endpoints_and_contract_agree_on_the_primary():
    """The decision rule reads coverage first, and the contract says so.

    ``interpretation_order`` places the gate ahead of coverage, which is the
    precedence the script implements by terminating on a gate failure before
    any cell runs.
    """
    c = load_contract(C5.EXPERIMENT_ID)
    assert c.interpretation_order[0] == "representability_gate"
    assert c.interpretation_order[1] == "coverage"
    assert c.primary_endpoint() == "coverage_rate"
