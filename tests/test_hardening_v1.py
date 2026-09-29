"""Gates for the hardened v1 runtime path.

These tests are architectural guards, not capability evidence. They ensure that
the temporal boundary, retrieval boundary, verifier/repair boundary, and
trajectory/cost artifacts are real before an expensive experiment is allowed.
"""

from __future__ import annotations

import pytest

from tac_osm import Candidate, Computation, Outcome, Query, RoutingDecision, Structure
from tac_osm.addressing import ContentAddressIndex
from tac_osm.benchmark_v1 import generate_task, relation_holds
from tac_osm.hardened import HardenedLoop, RelationExecutor, TemporalBenchmark
from tac_osm.structured_verifier import BoundedExecutableRepair, SemanticVerifier
from tac_osm.temporal import TemporalPersistentState


class PublicRelationRouter:
    """A deterministic test router using only query/state-visible information."""

    def __call__(self, query, state, candidates):
        bits_part = query.text.partition("\t")[0]
        bits = (
            tuple(int(x) for x in bits_part.split())
            if bits_part.strip()
            else ()
        )
        if not bits:
            read = state.read(query)
            bits = tuple(read.values[0]) if read.values else ()
        scores = tuple(
            float(
                sum(
                    int(c.descriptor[j] == bits[j])
                    for j in range(min(len(c.descriptor), len(bits), len(query.context)))
                    if query.context[j]
                )
            )
            for c in candidates
        )
        selected = max(range(len(candidates)), key=lambda i: (scores[i], -i))
        return RoutingDecision(
            selected=selected,
            scores=scores,
            provenance="test_public_router",
        )


def test_temporal_write_is_not_readable_before_the_declared_boundary():
    state = TemporalPersistentState()
    state.stage_world_write(
        __import__("tac_osm").StateUpdate(
            key="k",
            value=(1, 0, 1, 0),
            step=0,
        ),
        delay=4,
    )
    query = Query(text="\tk", context=(1, 1, 1, 1), step=0)
    assert state.read(query).values == ()
    state.advance_to(3)
    assert state.read(query).values == ()
    state.advance_to(4)
    assert state.read(query).values == ((1, 0, 1, 0),)
    assert state.write_step("k") == 0


@pytest.mark.parametrize("relation", ("equality", "xor_parity", "majority", "any_match"))
@pytest.mark.parametrize("validity", ("unique", "multiple", "none"))
def test_generalized_generator_controls_validity_without_gold_leaking_into_query(
    relation, validity
):
    task = generate_task(
        17,
        dim=8,
        n_candidates=32,
        relation=relation,
        validity=validity,
        state_address="addr",
    )
    observed_valid = {
        i
        for i, c in enumerate(task.candidates)
        if relation_holds(
            relation,
            task.reference_bits,
            c.descriptor,
            task.query.context,
        )
    }
    expected_size = {"unique": 1, "multiple": 2, "none": 0}[validity]
    assert observed_valid == set(task.acceptable_actions)
    assert len(observed_valid) == expected_size
    assert "addr" in task.public().text
    assert str(task.acceptable_actions) not in task.public().text


def test_content_address_query_cost_does_not_scan_history():
    reference = (1, 0, 1, 0, 1, 0, 1, 0)
    query = Query(
        text="1 0 1 0 1 0 1 0",
        context=(1, 1, 0, 0, 0, 0, 0, 0),
        step=10,
    )
    for h in (64, 256, 1024):
        task = generate_task(
            h,
            dim=8,
            n_candidates=h,
            relation="equality",
            validity="unique",
            reference_bits=reference,
        )
        index = ContentAddressIndex.build(task.candidates, context=query.context)
        hit = index.lookup(
            query,
            reference=reference,
            k=4,
            relation="equality",
        )
        assert hit.inspected_positions == 2
        assert hit.bucket_size >= 1
        assert len(hit.candidate_indices) <= 4


def test_index_rejects_relations_it_does_not_implement():
    task = generate_task(3, relation="xor_parity", validity="unique")
    index = ContentAddressIndex.build(task.candidates, context=task.query.context)
    with pytest.raises(ValueError, match="supports"):
        index.lookup(
            task.query,
            reference=task.reference_bits,
            k=4,
            relation="xor_parity",
        )


def test_indexed_loop_passes_only_retained_candidates_to_router():
    benchmark = TemporalBenchmark(
        seed=5,
        dim=8,
        n_candidates=64,
        relation="equality",
        validity="unique",
    )
    benchmark.schedule_lookup_probe(delay=4)
    seen = []

    def router(query, state, candidates):
        seen.append(len(candidates))
        return PublicRelationRouter()(query, state, candidates)

    loop = HardenedLoop(
        router=router,
        benchmark=benchmark,
        index=ContentAddressIndex(),
        index_k=4,
    )
    trajectory = loop.run(5)
    row = trajectory.steps[-1]
    assert row.retrieval["total_candidates"] == 64
    assert row.retrieval["retained_count"] <= 4
    assert len(seen) == 5
    assert all(size <= 4 for size in seen)
    assert loop.costs.candidates_routed == sum(
        step.retrieval["retained_count"] for step in trajectory.steps
    )


def test_index_miss_fails_closed_instead_of_silently_falling_back_to_full_history():
    benchmark = TemporalBenchmark(
        seed=6,
        dim=8,
        n_candidates=16,
        relation="equality",
        validity="none",
    )
    benchmark.schedule_task(0)
    loop = HardenedLoop(
        router=PublicRelationRouter(),
        benchmark=benchmark,
        index=ContentAddressIndex(),
        index_k=4,
    )
    with pytest.raises(RuntimeError, match="retrieval returned an empty set"):
        loop.step(0)


def test_temporal_probe_has_intervening_decision_boundaries():
    benchmark = TemporalBenchmark(seed=9, n_candidates=16)
    read_task = benchmark.schedule_lookup_probe(write_step=0, delay=4)
    assert read_task.query.step == 4
    benchmark.next_task(0)
    query = read_task.public()
    assert benchmark.state.read(query).values == ()
    benchmark.next_task(1)
    benchmark.next_task(2)
    benchmark.next_task(3)
    assert benchmark.state.read(query).values == ()
    benchmark.next_task(4)
    assert benchmark.state.read(query).values == (read_task.reference_bits,)


def test_learning_write_is_delayed_to_the_next_boundary():
    benchmark = TemporalBenchmark(seed=11, n_candidates=16)
    benchmark.schedule_task(0)
    loop = HardenedLoop(
        router=PublicRelationRouter(),
        benchmark=benchmark,
        index=None,
    )
    loop.step(0)
    assert any(
        item.update.key.startswith("experience:episode-0:0")
        for item in benchmark.state.pending()
    )
    benchmark.next_task(1)
    assert not any(
        item.update.key.startswith("experience:episode-0:0")
        for item in benchmark.state.pending()
    )
    assert benchmark.state.available("experience:episode-0:0")


def test_verifier_reports_observation_mismatch_as_repairable_evidence():
    verifier = SemanticVerifier()
    computation = Computation(
        structure=Structure(key="c0"),
        action=0,
        trace=(1.0, 1.0),
    )
    evidence = verifier.verify(
        computation,
        Outcome(success=False, value=0.0),
    )
    assert not evidence.valid
    assert evidence.failed_constraint == "output_observation_consistency"
    assert evidence.counterexample
    assert evidence.repair_target == "computation.output"


def test_bounded_repair_reexecutes_and_records_a_patch_without_gold():
    candidates = ("bad", "good", "unused")
    repair = BoundedExecutableRepair(max_attempts=2)
    verifier = SemanticVerifier()

    def compute(candidate):
        output = 1.0
        return Computation(
            structure=Structure(key=str(candidate)),
            action=0,
            trace=(output,),
        )

    def verify(comp):
        outcome = Outcome(success=(comp.structure.key == "good"), value=1.0 if comp.structure.key == "good" else 0.0)
        return verifier.verify(comp, outcome)

    result = repair.repair(
        candidates,
        compute=compute,
        verify=verify,
        selected_index=0,
    )
    assert result.passed
    assert result.attempts == 2
    assert result.selected_index == 1
    assert result.patch == "replace_candidate:0->1"


def test_trajectory_is_first_class_and_contiguous():
    benchmark = TemporalBenchmark(seed=12, n_candidates=8)
    benchmark.schedule_task(0)
    benchmark.schedule_task(1)
    loop = HardenedLoop(
        router=PublicRelationRouter(),
        benchmark=benchmark,
        index=None,
    )
    trajectory = loop.run(2)
    assert len(trajectory) == 2
    assert trajectory.steps[0].step == 0
    assert trajectory.steps[1].step == 1
    payload = trajectory.to_dict()
    assert payload["episode_id"] == "episode-0"
    assert len(payload["steps"]) == 2
    assert "verification" in payload["steps"][0]
    assert "learning" in payload["steps"][0]
    assert "available_addresses" in payload["steps"][0]["state_before"]
    assert "pending_writes" in payload["steps"][0]["state_after"]


def test_runtime_costs_keep_index_build_and_query_cost_separate():
    benchmark = TemporalBenchmark(seed=13, n_candidates=32)
    benchmark.schedule_task(0)
    loop = HardenedLoop(
        router=PublicRelationRouter(),
        benchmark=benchmark,
        index=ContentAddressIndex(),
        index_k=4,
    )
    row = loop.step(0)
    costs = loop.costs.to_dict()
    assert costs["index_build_candidates"] == 32
    assert costs["address_query_positions"] == 2
    assert costs["candidates_routed"] <= 4
    assert costs["executor_invocations"] == 1
    assert costs["verifier_checks"] == 1
    assert "wall_clock_seconds" in costs
    assert row.retrieval["retained_count"] == costs["candidates_routed"]


def test_relation_executor_is_explicitly_a_synthetic_control_not_a_casm_claim():
    task = generate_task(14, relation="equality", validity="unique")
    candidate = task.candidates[next(iter(task.acceptable_actions))]
    computation = RelationExecutor().execute(
        candidate,
        task.reference_bits,
        task.query.context,
        task.relation,
    )
    assert computation.structure.provenance == "hardened_v1.synthetic_executor"
    assert computation.structure.spec["kind"] == "synthetic_relation_control"


def test_representation_router_is_separate_from_feature_construction():
    from tac_osm.representation_router import RepresentationSimilarityRouter

    def qenc(query, state):
        bits = query.text.partition("\t")[0]
        return tuple(float(x) for x in bits.split()) if bits.strip() else (0.0, 0.0)

    def cenc(candidate, state):
        return tuple(float(x) for x in candidate.descriptor)

    task = generate_task(
        21,
        dim=8,
        n_candidates=8,
        relation="equality",
        validity="unique",
    )
    router = RepresentationSimilarityRouter(qenc, cenc)
    decision = router.route(task.public(), TemporalPersistentState(), task.candidates)
    assert decision.provenance == "representation_similarity_r1"
    assert len(decision.scores) == len(task.candidates)


def test_casm_adapter_keeps_external_dependency_outside_tac_osm():
    """The CASM-S boundary is structural, ordered, torch-free, and fail-closed."""
    import inspect
    import json
    from dataclasses import dataclass
    from enum import Enum

    from tac_osm import ExecutionResult, Structure
    from tac_osm import casm_adapter

    class FakeOp(str, Enum):
        INPUT = "INPUT"
        NOT = "NOT"
        AND = "AND"
        OR = "OR"
        XOR = "XOR"

    @dataclass(frozen=True)
    class FakeNode:
        index: int
        op: FakeOp
        depth: int
        slot: int

        @property
        def arity(self):
            return {
                FakeOp.INPUT: 0,
                FakeOp.NOT: 1,
                FakeOp.AND: 2,
                FakeOp.OR: 2,
                FakeOp.XOR: 2,
            }[self.op]

    @dataclass(frozen=True)
    class FakeEdge:
        src: int
        dst: int
        port: int

    @dataclass(frozen=True)
    class FakeEpisode:
        nodes: tuple
        true_edges: tuple
        candidate_edges: tuple
        inputs: tuple
        output: int
        input_values: tuple
        target: int
        truth_table: dict
        active_count: int

    edges = (
        FakeEdge(0, 2, 0),
        FakeEdge(1, 2, 0),
        FakeEdge(0, 2, 1),
        FakeEdge(1, 2, 1),
    )
    episode = FakeEpisode(
        nodes=(
            FakeNode(0, FakeOp.INPUT, 0, 0),
            FakeNode(1, FakeOp.INPUT, 0, 1),
            FakeNode(2, FakeOp.XOR, 1, 2),
            FakeNode(3, FakeOp.INPUT, 0, 3),
        ),
        true_edges=(edges[0], edges[3]),
        candidate_edges=edges,
        inputs=(0, 1),
        output=2,
        input_values=(1, 0),
        target=3,
        truth_table={(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
        active_count=3,
    )

    spec = casm_adapter.casm_graph_spec_from_episode(episode)

    # CASM-S Node/Edge field names and exact candidate-edge ordering are
    # preserved. The explicit edge index freezes alignment with the gate
    # tensor produced by the source model.
    assert spec["schema"] == casm_adapter.CASM_SCHEMA
    assert spec["kind"] == casm_adapter.CASM_KIND
    assert spec["active_count"] == 3
    assert spec["inputs"] == [0, 1]
    assert spec["output"] == 2
    assert [node["index"] for node in spec["nodes"]] == [0, 1, 2, 3]
    assert [node["op"] for node in spec["nodes"]] == [
        "INPUT", "INPUT", "XOR", "INPUT"
    ]
    assert [edge["index"] for edge in spec["candidate_edges"]] == [0, 1, 2, 3]
    assert [
        (edge["src"], edge["dst"], edge["port"])
        for edge in spec["candidate_edges"]
    ] == [
        (0, 2, 0),
        (1, 2, 0),
        (0, 2, 1),
        (1, 2, 1),
    ]

    # Hidden/oracle fields never cross the boundary, and the payload is plain
    # JSON data rather than a live CASM/Torch object.
    encoded = json.dumps(spec)
    assert json.loads(encoded) == spec
    for forbidden in (
        "true_edges",
        "true_edge_set",
        "target",
        "input_values",
        "truth_table",
        "acceptable_actions",
    ):
        assert forbidden not in spec
        assert forbidden not in encoded

    structure = casm_adapter.casm_structure_from_episode(
        episode, key="candidate-7", provenance="casm_s.test"
    )
    assert isinstance(structure, Structure)
    assert structure.spec == spec
    assert structure.provenance == "casm_s.test"

    from tac_osm.executor import relevance_program

    program = relevance_program((0, 1, 0, 1), (0, 0, 0, 1), (0, 1), max_nodes=10)
    compiled = casm_adapter.casm_graph_spec_from_program(program)
    assert compiled["active_count"] >= program.active_count
    assert len(compiled["candidate_edges"]) > len(program.candidate_edges)
    assert {node["op"] for node in compiled["nodes"]} <= {"INPUT", "NOT", "AND", "OR", "XOR"}
    assert "true_edges" not in compiled
    assert [edge["index"] for edge in compiled["candidate_edges"]] == list(
        range(len(compiled["candidate_edges"]))
    )

    observed = {}

    def fake_casm(casm_spec, inputs):
        observed["spec"] = casm_spec
        observed["inputs"] = tuple(inputs)

        assert all(
            edge["index"] == i
            for i, edge in enumerate(casm_spec["candidate_edges"])
        )

        # Same positional gate convention as CASM-S: one gate for every
        # candidate edge, indexed by its position in candidate_edges.
        gates = [1.0, 0.0, 0.0, 1.0]
        values = [0.0] * casm_spec["active_count"]
        for input_position, node_index in enumerate(casm_spec["inputs"]):
            values[node_index] = float(inputs[input_position])

        selected_by_port = {}
        for edge, gate in zip(casm_spec["candidate_edges"], gates):
            if gate > 0.5:
                selected_by_port[edge["port"]] = edge["src"]

        a = values[selected_by_port[0]]
        b = values[selected_by_port[1]]
        values[2] = float(a != b)

        return {
            "output": values[casm_spec["output"]],
            "gates": gates,
            "node_values": values,
            "provenance": "fake_casm_s",
        }

    result = casm_adapter.CasmExecutorAdapter(fake_casm).execute(
        structure, (1.0, 0.0)
    )
    assert observed["spec"] == spec
    assert observed["inputs"] == (1.0, 0.0)
    assert isinstance(result, ExecutionResult)
    assert result.output == 1.0
    assert result.gates == (1.0, 0.0, 0.0, 1.0)
    assert result.node_values == (1.0, 0.0, 1.0)
    assert result.provenance == "fake_casm_s"

    # Tensor-like values are normalized by duck typing; the adapter has no
    # tensor-library import and therefore remains runnable on Termux.
    class Scalar:
        def item(self):
            return 1.0

    class Vector:
        def tolist(self):
            return [1.0, 0.0, 1.0]

    tensor_result = casm_adapter.CasmExecutorAdapter(
        lambda _spec, _inputs: {
            "output": Scalar(),
            "gates": [1.0, 0.0, 0.0, 1.0],
            "node_values": Vector(),
        }
    ).execute(structure, (1.0, 0.0))
    assert tensor_result.output == 1.0
    assert tensor_result.node_values == (1.0, 0.0, 1.0)

    with pytest.raises(ValueError, match="gates length"):
        casm_adapter.CasmExecutorAdapter(
            lambda _spec, _inputs: {
                "output": 1.0,
                "gates": [1.0, 0.0],
                "node_values": [1.0, 0.0, 1.0],
            }
        ).execute(structure, (1.0, 0.0))

    with pytest.raises(ValueError, match="node_values length"):
        casm_adapter.CasmExecutorAdapter(
            lambda _spec, _inputs: {
                "output": 1.0,
                "gates": [1.0, 0.0, 0.0, 1.0],
                "node_values": [1.0],
            }
        ).execute(structure, (1.0, 0.0))

    with pytest.raises(ValueError, match="inputs length"):
        casm_adapter.CasmExecutorAdapter(fake_casm).execute(
            structure, (1.0,)
        )

    tampered = dict(spec)
    tampered["candidate_edges"] = [
        dict(edge) for edge in spec["candidate_edges"]
    ]
    tampered["candidate_edges"][2]["index"] = 7

    calls = []
    with pytest.raises(ValueError, match="preserve list order"):
        casm_adapter.CasmExecutorAdapter(
            lambda bad_spec, bad_inputs: calls.append((bad_spec, bad_inputs))
        ).execute(
            Structure(key="tampered", spec=tampered), (1.0, 0.0)
        )
    assert calls == []

    typed = casm_adapter.CasmGraphSpec.from_episode(episode)
    typed_seen = {}

    def typed_fake(casm_spec, inputs):
        typed_seen["spec"] = casm_spec
        typed_seen["inputs"] = tuple(inputs)
        return {
            "output": 1.0,
            "gates": [1.0, 0.0, 0.0, 1.0],
            "node_values": [1.0, 0.0, 1.0],
        }

    typed_result = casm_adapter.CasmExecutorAdapter(typed_fake).execute(
        Structure(key="typed", spec=typed), (1.0, 0.0)
    )
    assert typed_result.output == 1.0
    assert typed_seen["spec"] == typed.to_dict()

    source = inspect.getsource(casm_adapter)
    assert "import torch" not in source
    assert "from torch" not in source

def test_relation_constraint_verifier_rejects_semantically_invalid_action_without_gold():
    from tac_osm.structured_verifier import RelationConstraintVerifier

    task = generate_task(
        31,
        dim=8,
        n_candidates=8,
        relation="equality",
        validity="unique",
    )
    bad = next(i for i in range(len(task.candidates)) if i not in task.acceptable_actions)
    candidate = task.candidates[bad]
    computation = RelationExecutor().execute(
        candidate,
        task.reference_bits,
        task.query.context,
        task.relation,
    )
    outcome = TemporalBenchmark.observe(task, bad)
    evidence = RelationConstraintVerifier().verify_task(
        computation,
        outcome,
        candidate=candidate,
        reference=task.reference_bits,
        context=task.query.context,
        relation=task.relation,
    )
    assert not evidence.valid
    assert evidence.failed_constraint == "relation_unsatisfied"
    assert evidence.repair_target == f"candidate:{candidate.key}"


def test_static_population_preserves_actions_across_queries():
    from tac_osm.benchmark_v1 import repeated_equality_population, task_from_population

    population = repeated_equality_population(41, dim=8, n_candidates=64)
    first = task_from_population(
        candidates=population,
        reference_bits=(0, 0, 1, 0, 1, 0, 1, 0),
        step=0,
        validity="multiple",
    )
    second = task_from_population(
        candidates=population,
        reference_bits=(1, 1, 0, 1, 0, 1, 0, 1),
        step=1,
        validity="multiple",
    )
    assert first.candidates == second.candidates
    assert first.acceptable_actions
    assert second.acceptable_actions
    assert first.acceptable_actions != second.acceptable_actions


def test_hardened_index_never_receives_hidden_truth_after_reset():
    from tac_osm.addressing import ContentAddressIndex
    from tac_osm.hardened import HardenedLoop, TemporalBenchmark

    benchmark = TemporalBenchmark(seed=51, n_candidates=16)
    task = benchmark.schedule_lookup_probe(delay=1, key="hidden-test")
    captured = []

    class RecordingIndex(ContentAddressIndex):
        @classmethod
        def build(cls, candidates, *, context):
            base = ContentAddressIndex.build(candidates, context=context)
            return cls(_buckets=base._buckets, built_candidates=base.built_candidates)

        def lookup(self, query, *, reference, k=None, relation="equality"):
            from tac_osm.addressing import AddressHit
            captured.append(tuple(reference))
            return AddressHit(
                candidate_indices=(0,),
                inspected_positions=len(reference),
                bucket_size=1,
            )

    loop = HardenedLoop(
        router=lambda query, state, candidates: PublicRelationRouter()(query, state, candidates),
        benchmark=benchmark,
        index=RecordingIndex(),
        index_k=4,
        repair=None,
        learning_enabled=False,
        before_read=lambda task, step, state: state.clear() if step == 1 else None,
    )
    loop.step(0)
    loop.step(1)
    assert captured
    assert captured[-1] == ()
    assert tuple(task.reference_bits) != captured[-1]


def test_temporal_interventions_are_explicit_and_non_silent():
    state = TemporalPersistentState()
    state.write(__import__("tac_osm").StateUpdate(key="a", value=(0, 1, 0, 1), step=0))
    state.write(__import__("tac_osm").StateUpdate(key="b", value=(1, 0, 1, 0), step=0))
    state.corrupt("a", position=1)
    assert state.read(Query(text="\ta", step=0)).values == ((0, 0, 0, 1),)
    state.swap_values("a", "b")
    assert state.read(Query(text="\ta", step=0)).values == ((1, 0, 1, 0),)
    state.clear()
    assert state.addresses() == ()

def test_registered_measurement_scripts_execute_as_declared_smoke_tests():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    commands = [
        [sys.executable, "scripts/measure_temporal_persistence.py", "--smoke", "--steps", "3", "--eval-steps", "2", "--seeds", "0", "--levels", "1"],
        [sys.executable, "scripts/measure_selective_scaling.py", "--smoke", "--steps", "2", "--eval-steps", "2", "--seeds", "0", "--levels", "8"],
        [sys.executable, "scripts/measure_c5_casm.py", "--smoke",
 "--steps", "100", "--eval-steps", "100",
 "--seeds", "0,1,2,3,4", "--levels", "8,64,256"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        assert "SMOKE TEST" in result.stdout
        assert "machine-readable summary written" in result.stdout

def test_static_population_query_uses_an_actual_tab_delimiter():
    from tac_osm.benchmark_v1 import repeated_equality_population, task_from_population

    task = task_from_population(
        candidates=repeated_equality_population(61, dim=8, n_candidates=64),
        reference_bits=(0, 1, 0, 1, 0, 1, 0, 1),
        step=0,
        state_address="address",
        validity="multiple",
    )
    assert "\t" in task.query.text
    assert "\\t" not in task.query.text
    assert task.query.text.endswith("\taddress")