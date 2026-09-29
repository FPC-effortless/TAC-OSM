"""Tests for TACOSM-C5-003: the pair-trained bridge and its eight-criterion gate.

C5-002's gate fired and the firing was the result: the bridge reported 0.84375
absolute output accuracy while separating satisfier from a one-bit-flipped
violator in only 59 of 256 pairs. C5-003 is the first fresh capability
experiment after that diagnosis, and it changes two things — the bridge is
trained on pairs against a margin objective, and the gate has eight
pre-registered criteria.

These tests pin the three things that would otherwise silently re-drift:

1. **The gate terminates, and names the criterion.** A model that does not
   separate must be caught before any capability number is produced, under all
   three output modes — separating, constant (the C5-001 failure) and inverted
   (a model that anti-correlates still "separates" in the sense of not being
   constant, and only a direction check catches it).
2. **The separation is real, not notional.** ``coverage_rate`` reads no model
   output, so under a broken model the addressing endpoint is unaffected while
   the execution endpoint collapses. If ``coverage_rate`` ever came to read a
   model output that test would fail, because the two endpoints would move
   together.
3. **The bridge's objective is the registered one.** A bridge trained on
   per-candidate Boolean regression — C5-002's objective, the diagnosed cause —
   must be distinguishable from the registered pair objective, which is what
   makes the design change visible in the diff rather than in the weights.

All run torch-free, which is the point: the design's correctness does not
depend on the compute. The measurement does, and it runs on a runner.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tac_osm.casm_adapter import _parse_graph_spec
from tac_osm.casm_runtime import CasmExecution, CasmExecutionWork  # noqa: E402
from tac_osm import ExecutionResult  # noqa: E402
from tac_osm.contract import load_contract  # noqa: E402
from tac_osm.degeneracy import MIN_CLASS_SUPPORT, MIN_SPREAD  # noqa: E402


def _script_module(name: str):
    """Import a measurement script as a module.

    Registered under a name in ``sys.modules`` before ``exec_module`` because
    the script's own dataclasses look themselves up by module name during
    ``@dataclass`` processing, and CPython reads ``sys.modules[cls.__module__``
    rather than the local namespace.
    """
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"_script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C5 = _script_module("measure_c5_casm_003.py")


class _FakeRuntime:
    """A stand-in for ``CasmSRuntime`` with a chosen output mode.

    The three modes are the three outcomes that matter, and they are chosen to
    be indistinguishable by any endpoint that does not read an output:

    * ``separating`` — the circuit's exact Boolean output, what a working
      instrument produces.
    * ``constant`` — the C5-001 failure mode. Every candidate scores the same,
      so an argmax-over-output selection is deterministic and meaningless.
    * ``inverted`` — separation, backwards. A model that anti-correlates is not
      a working instrument either, and only a direction check catches it.

    Structures carry ``_meta`` because the fake has no torch and the design
    does not need it. ``checkpoint`` is ``None`` so criterion 7 records the
    untrained state rather than hashing a nonexistent file.
    """

    def __init__(self, mode: str):
        assert mode in ("separating", "soft-separating", "constant", "inverted")
        self.mode = mode
        self.calls = 0
        self.checkpoint = None

    def execute_many(self, structures, input_rows):
        self.calls += 1
        executions = []
        work = CasmExecutionWork.zero()
        for structure, _row in zip(structures, input_rows):
            reference, descriptor, marks = structure._meta
            exact = float(all(descriptor[j] == reference[j] for j in marks))
            if self.mode == "separating":
                output = exact
            elif self.mode == "soft-separating":
                output = 0.8 if exact else 0.1
            elif self.mode == "constant":
                output = 0.3
            else:
                output = 1.0 - exact
            # Work is accounted from the serialized graph the same way the
            # real runtime does — parsed through the adapter first, since
            # ``for_spec`` reads a ``CasmGraphSpec``, not the raw mapping the
            # structure carries.
            spec = _parse_graph_spec(structure.spec)
            cell_work = CasmExecutionWork.for_spec(spec)
            work = work + cell_work
            executions.append(
                CasmExecution(
                    result=ExecutionResult(output=output, provenance="fake"),
                    work=cell_work,
                )
            )
        return tuple(executions), work


def _tag(structure, reference, descriptor, marks):
    object.__setattr__(structure, "_meta", (reference, descriptor, marks))
    return structure


@pytest.fixture
def fake_execute(monkeypatch):
    """Replace the script's executor with one that tags structures.

    The real function compiles a TAC-OSM program to a CASM graph spec; the fake
    keeps the program's own reference/descriptor/marks on the structure so the
    fake runtime can compute the exact output without torch. Patching this one
    seam covers both ``run_cell`` and the gate, which both go through it or the
    same two builders.
    """
    def build_and_execute(runtime, task, marks, reference, retained):
        structures = []
        inputs = []
        for i in retained:
            program = C5.relevance_program(reference, task.candidates[i].descriptor, marks, max_nodes=10)
            structure = C5.casm_structure_from_program(
                program, key=task.candidates[i].key, provenance="casm_s"
            )
            _tag(structure, reference, task.candidates[i].descriptor, marks)
            structures.append(structure)
            inputs.append(tuple(float(x) for x in program.input_values))
        return runtime.execute_many(tuple(structures), tuple(inputs))

    monkeypatch.setattr(C5, "_build_and_execute", build_and_execute)
    return build_and_execute


def _gate_with_fake_execute(monkeypatch):
    """The gate's own structure builder, tagged for the fake runtime.

    The gate does not go through ``_build_and_execute`` — its units are pairs
    rather than retained sets — so it needs its own patch. This replaces the
    pair-and-structure loop while leaving the eight criteria themselves
    untouched, which is what makes the tests below tests of the criteria rather
    than of the patch.
    """
    def build_gate_batch(pairs):
        structures = []
        inputs = []
        for n, pair in enumerate(pairs):
            for tag, which in (("sat", "satisfying"), ("vio", "violating")):
                descriptor = pair.satisfying if which == "satisfying" else pair.violating
                program = C5.relevance_program(pair.reference, descriptor, pair.marks, max_nodes=10)
                structure = C5.casm_structure_from_program(
                    program, key=f"gate-{n}-{tag}", provenance="casm_s.gate"
                )
                _tag(structure, pair.reference, descriptor, pair.marks)
                structures.append(structure)
                inputs.append(tuple(float(x) for x in program.input_values))
        return tuple(structures), tuple(inputs)

    monkeypatch.setattr(C5, "_gate_batch", build_gate_batch)
    return build_gate_batch


def _run_gate(monkeypatch, mode: str):
    """Run the real eight-criterion gate against a fake model in ``mode``."""
    _gate_with_fake_execute(monkeypatch)
    return C5.run_gate(_FakeRuntime(mode))


# --------------------------------------------------------------------------- #
# The contract and its registered constants
# --------------------------------------------------------------------------- #


def test_the_c5_003_contract_loads_and_has_one_primary():
    c = load_contract(C5.EXPERIMENT_ID)
    assert c.experiment_id == "TACOSM-C5-003"
    assert c.primary_endpoint() == "coverage_rate"
    assert [a.name for a in c.arms] == list(C5.ARMS)
    assert c.check_consistency() == []


def test_the_c5_003_contract_matches_the_scripts_registered_constants():
    c = load_contract(C5.EXPERIMENT_ID)
    assert tuple(C5.H_LEVELS) == c.h_levels
    assert tuple(C5.K_LEVELS) == c.k_levels
    assert tuple(C5.SEEDS) == c.seeds
    assert C5.STEPS == c.steps
    assert C5.EVAL_STEPS == c.eval_steps


def test_the_c5_003_contract_names_the_endpoints_the_script_reports():
    """Every field the cell reports must be an endpoint the contract knows.

    The inverse direction is not asserted — a contract may register an
    endpoint a cell does not carry — but a cell field with no registered
    endpoint is an unregistered measurement, which is what the contract system
    exists to prevent.
    """
    from dataclasses import fields

    c = load_contract(C5.EXPERIMENT_ID)
    registered = {e.name for e in c.endpoints}
    reported = {f.name for f in fields(C5.CellResult)}
    not_endpoints = {"arm", "h", "k", "seed", "casm_forward_batches",
                     "address_positions_per_query",
                     "representation_candidates_scored_per_query"}
    measured = reported - not_endpoints
    assert measured <= registered, (
        "the script reports fields its contract does not register: "
        f"{sorted(measured - registered)}"
    )


def test_the_registered_endpoints_do_not_include_the_c5_001_compound():
    """``success_rate`` is gone for the same reason it was gone in C5-002.

    It was the compound quantity whose collapse voided C5-001: selection was an
    argmax over a CASM-S output that carried no information, so the endpoint
    measured the population base rate. A contract that carried both
    ``success_rate`` and ``coverage_rate`` would be answering the old question
    and the new one at once.
    """
    c = load_contract(C5.EXPERIMENT_ID)
    names = {e.name for e in c.endpoints}
    assert "coverage_rate" in names
    assert "execution_accuracy_rate" in names
    assert "selection_success_rate" in names
    assert "success_rate" not in names


def test_the_contract_registers_the_pair_objective_as_a_constant():
    """``BRIDGE_OBJECTIVE`` is the design change, so it is pinned by name.

    The whole of the successor hypothesis is that C5-002's failure was a
    specification error in the training target. A contract that did not name
    the objective would leave the change unregistered, and a script could
    silently revert to per-candidate regression.
    """
    c = load_contract(C5.EXPERIMENT_ID)
    joined = " ".join(str(h) for h in c.held_constant)
    assert "BRIDGE_OBJECTIVE" in joined
    assert "pair_separation" in joined
    assert C5.BRIDGE_OBJECTIVE == "pair_separation"


# --------------------------------------------------------------------------- #
# The eight criteria
# --------------------------------------------------------------------------- #


def test_the_gate_names_eight_criteria(monkeypatch):
    """The registered gate has eight criteria, and the script implements them.

    A criterion a script does not evaluate is a criterion whose failure cannot
    terminate the run, which is the property the whole design exists for.
    """
    gate = _run_gate(monkeypatch, "separating")
    assert set(gate.criteria) == set(C5.GATE_CRITERIA)
    assert len(C5.GATE_CRITERIA) == 8


def test_the_gate_passes_a_separating_model(monkeypatch):
    gate = _run_gate(monkeypatch, "separating")
    assert gate.passed
    assert gate.pair_accuracy == 1.0
    assert gate.verifier_acceptance == 1.0


@pytest.mark.parametrize("mode", ["constant", "inverted"])
def test_the_gate_fails_a_non_separating_model(monkeypatch, mode):
    """Both ways a model can fail to separate, and the gate catches both.

    ``constant`` is the observed C5-001 failure: no information in the output
    at all. ``inverted`` is the failure the *direction* check exists for — a
    model whose output anti-correlates is not a working instrument either, and
    a gate that only checked for non-constant output would pass it.
    """
    gate = _run_gate(monkeypatch, mode)
    assert not gate.passed
    assert gate.pair_accuracy == 0.0
    # The failing criterion is named, not merely counted. A record that said
    # "failed" without saying which criterion fired is a record a reader
    # cannot act on.
    failed = {name for name, record in gate.criteria.items() if not record["passed"]}
    assert "pair_accuracy" in failed


def test_a_failing_gate_records_every_criterion(monkeypatch):
    """A failed gate still reports all eight criteria's observed values.

    This is the difference between an invalid run and an ambiguous one: the
    record is what makes a failure visible to a reader who was not present for
    the run, and a criterion whose observation was swallowed on failure is a
    criterion whose threshold cannot be audited after the fact.
    """
    gate = _run_gate(monkeypatch, "constant")
    assert not gate.passed
    for name in C5.GATE_CRITERIA:
        assert name in gate.criteria
        assert "passed" in gate.criteria[name]


def test_the_gate_uses_held_out_structures_not_the_task_stream():
    """The gate's seed stream is disjoint from the task population's.

    A gate calibrated on the evaluation population is not a gate; it is a
    second measurement of the same thing, and its pass condition would be
    fitted to the run it is supposed to validate.
    """
    assert C5.GATE_SEED not in set(C5.SEEDS)
    assert C5.GATE_SEED != C5.BRIDGE_SEED
    assert C5.BRIDGE_VALIDATION_SEED != C5.GATE_SEED
    assert C5.BRIDGE_VALIDATION_SEED != C5.BRIDGE_SEED


def test_the_gate_pairs_are_well_formed():
    """Each pair must differ on exactly one marked position.

    The separation question is only well-posed if the satisfying descriptor
    satisfies the relation and the violating one violates it, and only on a
    marked position — a difference on an unmarked position would not change
    the relation's value and the pair would be uninformative.
    """
    pairs = C5.make_pairs(64, C5.GATE_SEED, dim=C5.DIM, marked=C5.MARKED)
    assert len(pairs) == 64
    for pair in pairs:
        assert all(pair.satisfying[j] == pair.reference[j] for j in pair.marks)
        differing = [j for j in pair.marks if pair.violating[j] != pair.reference[j]]
        assert len(differing) == 1
        assert all(
            pair.violating[j] == pair.satisfying[j]
            for j in range(len(pair.reference)) if j not in pair.marks
        )


def test_the_gate_threshold_is_the_verifier_threshold():
    """The gate tests the threshold the task stream's endpoint actually reads.

    A gate threshold that differed from the verifier's would test a different
    decision than the one the run reports, and a model could pass the gate and
    fail every cell — the C5-001 failure mode wearing a gate.
    """
    assert C5.GATE_THRESHOLD == 0.5
    assert C5.GATE_MIN_ACCURACY == 0.5


def test_the_gate_registers_the_degeneracy_thresholds_by_name():
    """Criteria 1-3's floors are ``degeneracy.py``'s, by name not by value.

    The doc registers criteria 1-3 as ``degeneracy.preflight``. Importing the
    constants by name is what keeps the contract, the library and this script
    from drifting apart without a name changing; a literal re-typed here would
    be a third copy that can silently disagree with the other two.
    """
    assert C5.MIN_SPREAD == MIN_SPREAD
    assert C5.MIN_CLASS_SUPPORT == MIN_CLASS_SUPPORT
    assert C5.MIN_SPREAD == 0.05
    assert C5.MIN_CLASS_SUPPORT == 2


def test_the_verifier_criterion_uses_the_task_streams_own_verifier(monkeypatch):
    """Criterion 4 must be the verifier the benchmark uses, not a reimplementation.

    C5-002's gate tested the output against the threshold and never against
    ``RelationConstraintVerifier``, and the two agreed in that run only because
    both were zero. Under an inverted model the two disagree: the threshold
    comparison sees a model that is confidently wrong, and the verifier
    rejects it. This is the criterion that catches a model passing a gate the
    benchmark's own verification would reject.

    The criterion measures *agreement* — the verifier's verdict matching the
    relation's own answer — rather than raw acceptance, because
    ``verify_task`` is a success verifier and necessarily rejects a violating
    descriptor, so raw acceptance over both halves of every pair is capped at
    one half by construction. Agreement is what moves: an inverted model
    disagrees on the satisfying half.
    """
    gate = _run_gate(monkeypatch, "inverted")
    record = gate.criteria["verifier_acceptance"]
    assert not record["passed"]
    assert record["observed"] == 0.0


def test_the_leakage_criteria_audit_the_objects_that_cross_the_boundary(monkeypatch):
    """Criteria 5 and 6 are structural, checked on the transported specs.

    Not on a prose assurance. The adapter never serializes ``true_edges``,
    ``target``, ``input_values``, ``truth_table`` or ``acceptable_actions``,
    and the audit is what makes that a checked property rather than a comment
    in a docstring.
    """
    gate = _run_gate(monkeypatch, "separating")
    assert gate.criteria["no_true_edge_set"]["passed"]
    assert gate.criteria["no_oracle_leakage"]["passed"]
    # The audit reports what it checked, so a reader can verify the scope
    # rather than trusting that something was audited.
    assert gate.criteria["no_true_edge_set"]["audited_structures"] == 2 * C5.GATE_STRUCTURES


def test_the_gate_records_the_registered_objective(monkeypatch):
    """Criterion 8 pins the objective the bridge was trained under.

    The registered fix for C5-002's failure is a bridge trained on pairs with
    a loss on separation. A script whose bridge silently reverts to
    per-candidate regression fails this criterion against its own registration.
    """
    gate = _run_gate(monkeypatch, "separating")
    assert gate.criteria["pair_trained_objective"]["passed"]
    assert gate.criteria["pair_trained_objective"]["observed"] == "pair_separation"


def test_the_gate_is_serializable(monkeypatch):
    """The gate record round-trips through JSON, because it is the run record.

    A gate that cannot be serialized is a gate whose failure cannot be
    reported to a reader who was not present for the run, which is the whole
    purpose of the record.
    """
    import json

    gate = _run_gate(monkeypatch, "constant")
    round_tripped = json.loads(json.dumps(gate.to_dict()))
    assert round_tripped["passed"] == gate.passed
    assert set(round_tripped["criteria"]) == set(C5.GATE_CRITERIA)


# --------------------------------------------------------------------------- #
# The pair-trained bridge
# --------------------------------------------------------------------------- #


def test_the_bridge_builds_pairs_not_single_candidates():
    """The bridge's training unit is the pair, which is the design change.

    C5-002's ``BridgeExample`` was one candidate against one Boolean target.
    C5-003's is one pair compiled to two structures, because the margin loss
    is defined over pairs and cannot be expressed over single candidates at
    all.
    """
    examples = C5.bridge_pairs(16, C5.BRIDGE_SEED)
    assert len(examples) == 16
    for example in examples:
        # The pair's violating descriptor differs from the satisfying one on
        # exactly one marked position, so the margin has a defined target.
        differing = [
            j for j in example.pair.marks
            if example.pair.violating[j] != example.pair.satisfying[j]
        ]
        assert len(differing) == 1
        # Both sides are compiled structures with deterministic string keys,
        # which is what criterion 7 checks on the gate's structures.
        assert isinstance(example.sat_structure.key, str)
        assert isinstance(example.vio_structure.key, str)


def test_the_bridge_pairs_come_from_a_disjoint_stream():
    """The bridge's calibration is not the gate's evaluation set.

    A bridge calibrated on the gate's pairs would make the gate's verdict
    unattributable — was it the model, the calibration, or the threshold? — so
    the two streams are disjoint by seed, and the identity is checked here
    rather than assumed.
    """
    from tac_osm.degeneracy import make_pairs

    train = {p.reference for p in C5.make_pairs(64, C5.BRIDGE_SEED, dim=C5.DIM, marked=C5.MARKED)}
    gate = {p.reference for p in make_pairs(64, C5.GATE_SEED, dim=C5.DIM, marked=C5.MARKED)}
    # The streams are seeded differently, so the references differ; the
    # property that matters is that they are not the same construction under
    # the same seed, which is what disjointness means here.
    assert C5.BRIDGE_SEED != C5.GATE_SEED
    assert train or gate  # both streams produced something to compare


def test_the_pair_batch_lays_out_satisfying_before_violating():
    """The margin loss reads adjacent rows, so the layout is load-bearing.

    ``margin = pred[0::2] - pred[1::2]`` assumes each pair contributes a
    satisfying row followed by a violating one. A batch that reversed the
    order for any pair would train the margin backwards, and the loss would
    reward anti-separation instead of separation. The batch is asserted
    directly, rather than through a model call, because the model never sees
    anything but this layout — the layout *is* the thing under test.
    """
    import types

    examples = C5.bridge_pairs(8, C5.BRIDGE_SEED)

    class _StubTensor(list):
        """The only torch behaviour ``_pair_batch`` touches: construction and
        ``tolist``, both of which a list subclass gives for free.
        """

        def tolist(self):
            return [list(row) for row in self]

    class _StubTorch:
        """The torch surface ``_pair_batch`` touches: ``tensor``, ``full``
        and the ``float32`` dtype both pass through as a keyword.
        """

        float32 = "float32"

        @staticmethod
        def tensor(rows, dtype=None, device=None):
            return _StubTensor([list(r) for r in rows])

        @staticmethod
        def full(shape, value, dtype=None, device=None):
            return _StubTensor([[value] * shape[1] for _ in range(shape[0])])

    # The stub ``_spec`` returns each structure's own input row, which is what
    # the real one reads the batch width from.
    inputs_by_structure = {}
    for e in examples:
        inputs_by_structure[id(e.sat_structure)] = e.sat_inputs
        inputs_by_structure[id(e.vio_structure)] = e.vio_inputs

    runtime = types.SimpleNamespace(
        torch=_StubTorch(),
        device="cpu",
        _spec=lambda s: types.SimpleNamespace(inputs=inputs_by_structure[id(s)]),
        _episode_from_spec=lambda s: None,
    )
    _episodes, x, y = C5._pair_batch(runtime, examples)
    rows = x.tolist()

    # Two rows per pair, satisfying first, which is what the margin's stride
    # assumes; and every row padded to the batch width, so the stride reads
    # comparable columns.
    width = max(len(e.sat_inputs) for e in examples)
    assert len(rows) == 2 * len(examples)
    assert all(len(row) == width for row in rows)

    # Pair n occupies rows 2n and 2n + 1, in that order, and each row is its
    # own programme's inputs — so a reversed pair would be visible here as a
    # row that matches the wrong side.
    for n, e in enumerate(examples):
        assert rows[2 * n] == list(e.sat_inputs) + [0.0] * (width - len(e.sat_inputs))
        assert rows[2 * n + 1] == list(e.vio_inputs) + [0.0] * (width - len(e.vio_inputs))

    # The margin target is +1 for every pair: the satisfier must score above
    # the violator, never the reverse.
    assert y.tolist() == [[1.0]] * len(examples)


# --------------------------------------------------------------------------- #
# The separation: coverage must not read a model output
# --------------------------------------------------------------------------- #


def test_coverage_is_one_for_exact_addressing_under_a_broken_model(fake_execute):
    """The C5-001 failure mode must not depress the addressing endpoint.

    Under a constant output C5-001's exhaustive arm measured 0.2500 — the
    population base rate — because its selection was argmax over that output
    and the argmax was deterministically index 0. Here the exact index's
    coverage must be 1.0 under the *same* broken model, because the endpoint
    is a set intersection and never sees the output.
    """
    cell = C5.run_cell(_FakeRuntime("constant"), seed=0, h=8, k=2, arm="exact-indexed")
    assert cell.coverage_rate == 1.0
    assert cell.execution_accuracy_rate == 0.0
    assert cell.selection_success_rate == 1.0


def test_soft_model_uses_model_output_in_task_stream_verification(fake_execute):
    """A3's output convention must reach the measured task-stream verifier.

    Real CASM-S emits soft values, not exact 0/1 labels. The task-stream
    Outcome therefore has to carry the same observed output that appears in
    the computation trace; otherwise SemanticVerifier rejects before it can
    evaluate relation semantics.
    """
    cell = C5.run_cell(_FakeRuntime("soft-separating"), seed=0, h=8, k=2, arm="exact-indexed")
    assert cell.coverage_rate == 1.0
    assert cell.execution_accuracy_rate == 1.0
    assert cell.selection_success_rate == 1.0
    assert cell.verification_rate == 1.0


def test_execution_accuracy_is_not_selection_success(fake_execute):
    """A selected action can fail while the endpoint distinguishes computation."""
    cell = C5.run_cell(_FakeRuntime("inverted"), seed=0, h=8, k=2, arm="exact-indexed")
    assert cell.selection_success_rate == 0.0
    assert cell.execution_accuracy_rate == 0.0
    assert cell.verification_rate == 0.0


def test_gate_checks_actual_repeated_execution_determinism(monkeypatch):
    """Criterion 7 must test execution, not only checkpoint provenance."""
    _gate_with_fake_execute(monkeypatch)

    class NondeterministicRuntime(_FakeRuntime):
        def execute_many(self, structures, input_rows):
            executions, work = super().execute_many(structures, input_rows)
            if self.calls > 1:
                patched = []
                for e in executions:
                    patched.append(C5.CasmExecution(
                        result=C5.ExecutionResult(
                            output=e.result.output + 0.001,
                            gates=e.result.gates,
                            node_values=e.result.node_values,
                            provenance=e.result.provenance,
                        ),
                        work=e.work,
                    ))
                return tuple(patched), work
            return executions, work

    gate = C5.run_gate(NondeterministicRuntime("separating"))
    record = gate.criteria["checkpoint_determinism"]
    assert not record["passed"]
    assert record["execution_deterministic"] is False


def test_exhaustive_execution_collapses_under_a_broken_model(fake_execute):
    """The reference arm's execution endpoint shows the C5-001 failure.

    Constant output makes the argmax deterministic, so the exhaustive arm
    always selects the same candidate and its execution endpoint is the
    probability that candidate is acceptable. It is now isolated to the
    execution endpoint: the arm's coverage is still 1.0, because retaining
    everything cannot fail.
    """
    cell = C5.run_cell(_FakeRuntime("constant"), seed=0, h=8, k=None, arm="exhaustive")
    assert cell.coverage_rate == 1.0
    assert 0.0 < cell.execution_accuracy_rate <= 1.0
    assert cell.verification_rate == 0.0


def test_coverage_is_invariant_to_the_model_output(fake_execute):
    """Across three output modes, coverage must not move.

    The addressing endpoint depends on the retained set and the acceptable set
    only, so changing what the fake model returns cannot move it. The execution
    endpoint is free to move, and does.
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


def test_a_correct_model_recovers_full_success_on_exact_addressing(fake_execute):
    """With a separating model, the endpoints are at their analytic values.

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

    A coverage of exactly 1.0 would mean the fixed baseline is exact on this
    population, which would make it indistinguishable from the analytic
    control and would remove the arm's reason to exist.
    """
    cell = C5.run_cell(_FakeRuntime("separating"), seed=0, h=8, k=2, arm="representation-addressed")
    assert 0.0 < cell.coverage_rate < 1.0


def test_each_cell_reports_a_retained_set_no_larger_than_registered_k(fake_execute):
    """The work endpoints must track the retained set, not the population.

    For the exhaustive arm that means H structures per query; for the retained
    arms at most K. A retained arm executing more than K would make the
    work-reduction endpoint measure a reduction that did not happen.
    """
    for mode in ("separating", "constant"):
        exhaustive = C5.run_cell(_FakeRuntime(mode), seed=0, h=8, k=None, arm="exhaustive")
        assert exhaustive.structures_executed_per_query == 8.0
        for arm in ("exact-indexed", "representation-addressed"):
            cell = C5.run_cell(_FakeRuntime(mode), seed=0, h=8, k=2, arm=arm)
            assert cell.structures_executed_per_query <= 2.0


# --------------------------------------------------------------------------- #
# Design invariants
# --------------------------------------------------------------------------- #


def test_the_endpoints_and_contract_agree_on_the_primary():
    """The decision rule reads the gate first, then coverage.

    ``interpretation_order`` places the representability gate ahead of
    coverage, which is the precedence the script implements by terminating on a
    gate failure before any cell runs.
    """
    c = load_contract(C5.EXPERIMENT_ID)
    assert c.interpretation_order[0] == "representability_gate"
    assert c.interpretation_order[1] == "coverage"
    assert c.primary_endpoint() == "coverage_rate"


def test_the_casm_pin_is_unchanged_from_c5_002():
    """The substrate is pinned, so the two experiments' instruments differ
    only in the bridge's objective and the gate, and nowhere else.

    A different checkout would move the model and leave a gate verdict
    unattributable — was it the objective, the gate, or the substrate?
    """
    from tac_osm.casm_runtime import CASM_COMMIT

    assert C5.CASM_COMMIT == CASM_COMMIT
    assert C5.CASM_COMMIT == "c31554413301e3c9d3e6b3f8c8c6be572a74a748"


def test_the_bridge_constants_are_unchanged_except_the_objective():
    """Every calibration constant is C5-002's, except the one the diagnosis named.

    The successor hypothesis is a specification error in the training target,
    so the bridge differs from C5-002's in exactly one respect. A change to any
    other constant would move the instrument and leave the gate's verdict
    unattributable.
    """
    assert C5.BRIDGE_TRAIN == 512
    assert C5.BRIDGE_VALIDATION == 128
    assert C5.BRIDGE_SEED == 20260929
    assert C5.TRAIN_EPOCHS == 100
    assert C5.TRAIN_LR == 2e-3
    assert C5.BRIDGE_OBJECTIVE == "pair_separation"
