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
    from tac_osm.casm_adapter import CasmExecutorAdapter
    from tac_osm import ExecutionResult, Structure

    adapter = CasmExecutorAdapter(
        lambda spec, inputs: {
            "output": sum(inputs),
            "node_values": tuple(inputs),
            "provenance": "test_casm",
        }
    )
    result = adapter.execute(Structure(key="demo", spec={"op": "sum"}), (1.0, 2.0))
    assert isinstance(result, ExecutionResult)
    assert result.output == 3.0
    assert result.node_values == (1.0, 2.0)
    assert result.provenance == "test_casm"


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
        def lookup(self, query, *, reference, k=None, relation="equality"):
            captured.append(tuple(reference))
            return super().lookup(query, reference=reference, k=k, relation=relation)

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