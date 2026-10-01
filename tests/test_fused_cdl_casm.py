from __future__ import annotations

import random

from tac_osm.environment import (
    build_lookup_task,
    build_replay_task,
    build_relational_task,
    satisfies_relation,
)
from tac_osm.executor import ExecutorConfig, StructuralExecutor, relevance_program
from tac_osm.execution_feedback import ExecutionGroundTruthVerifier
from tac_osm.fused_cdl_casm import (
    CDLConfig,
    CDLStudentRouter,
    CasmComputationSelector,
)
from tac_osm.model import ModelConfig, TacOsmModel
from tac_osm.state import PersistentStore, StateConfig


def test_cdl_student_route_has_no_target_or_outcome_inputs() -> None:
    router = CDLStudentRouter(CDLConfig(input_dim=8, latent_dim=16, seed=0))
    state = PersistentStore(StateConfig(n_slots=16, seed=0))
    task = build_relational_task(1, dim=8, n_candidates=8)
    decision = router.route(task.public(), state, task.candidates)
    assert 0 <= decision.selected < len(task.candidates)
    assert len(decision.scores) == 8


def test_cdl_analytic_witness_separates_relation() -> None:
    router = CDLStudentRouter(CDLConfig(input_dim=8, latent_dim=16, seed=0))
    router.set_analytic_relation()
    state = PersistentStore(StateConfig(n_slots=16, seed=0))
    for seed in range(30):
        task = build_relational_task(seed, dim=8, n_candidates=8)
        decision = router.route(task.public(), state, task.candidates)
        assert decision.selected == task.target_action


def test_casm_exact_relation_matches_environment() -> None:
    executor = StructuralExecutor(
        ExecutorConfig(type="learned", dim=8, max_nodes=10, seed=0)
    )
    for seed in range(30):
        task = build_relational_task(seed, dim=8, n_candidates=8)
        marks = [i for i, bit in enumerate(task.query.context) if bit]
        ref = tuple(int(x) for x in task.query.text.partition("\t")[0].split())
        selector = CasmComputationSelector(max_nodes=10)
        structure = selector.select(
            reference=ref,
            candidate=task.candidates[task.target_action],
            marks=marks,
            step_index=0,
        )
        result = executor.execute(structure, ())
        expected = satisfies_relation(ref, task.query.context, task.candidates[task.target_action].descriptor)
        assert bool(round(result.output)) is expected
        assert selector.last_selection is not None
        assert selector.last_selection.active_nodes == result.node_values.__len__()


def test_verifier_learning_localizes_lookup_target_post_hoc() -> None:
    state = PersistentStore(StateConfig(n_slots=32, seed=0))
    task = build_lookup_task(5, state, dim=8, n_candidates=8)
    router = CDLStudentRouter(CDLConfig(input_dim=8, latent_dim=16, seed=0))
    decision = router.route(task.public(), state, task.candidates)
    before = [list(row) for row in router.wq]
    executor = StructuralExecutor(ExecutorConfig(max_nodes=10, dim=8, seed=0))
    selector = CasmComputationSelector(max_nodes=10)
    read = state.read(task.query)
    ref = tuple(read.values[0])
    marks = [i for i, bit in enumerate(task.query.context) if bit]
    structure = selector.select(
        reference=ref,
        candidate=task.candidates[(decision.selected + 1) % len(task.candidates)],
        marks=marks,
        step_index=0,
    )
    result = executor.execute(structure, ())
    from tac_osm import Computation
    from tac_osm import Outcome

    wrong = (decision.selected + 1) % len(task.candidates)
    outcome = Outcome(
        success=(wrong == task.target_action),
        value=float(wrong == task.target_action),
        detail=task.detail,
    )
    verification = ExecutionGroundTruthVerifier().verify(
        Computation(structure=structure, action=wrong, trace=(result.output,)),
        outcome,
    )
    router.learn_from_verifier(
        query=task.public(),
        state=state,
        candidates=task.candidates,
        selected=wrong,
        outcome=outcome,
        verification=verification,
        scores=decision.scores,
    )
    assert router.updates >= 1
    assert router.wq != before


def test_full_loop_preserves_world_fact_and_writes_experience_namespace() -> None:
    seed = 7
    state = PersistentStore(StateConfig(n_slots=64, seed=seed))
    from tac_osm.environment import WorldConfig, WorldEnvironment

    env = WorldEnvironment(
        WorldConfig(
            dim=8,
            n_candidates=8,
            noise=0.10,
            families=("state_lookup",),
            seed=seed,
        )
    )
    router = CDLStudentRouter(CDLConfig(input_dim=8, latent_dim=16, seed=seed))
    model = TacOsmModel(
        state=state,
        router=router,
        executor=StructuralExecutor(ExecutorConfig(max_nodes=10, dim=8, seed=seed)),
        verifier=ExecutionGroundTruthVerifier(),
        repair=None,
        environment=env,
        computation_selector=CasmComputationSelector(max_nodes=10),
        config=ModelConfig(n_steps=1, learn=True, seed=seed),
    )
    step = model.step(0)
    assert step.verification.passed == step.outcome.success
    address = step.query.text.partition("\t")[2]
    snapshot = {slot.key: slot.value for slot in state.snapshot() if slot.used}
    assert address in snapshot
    assert snapshot[address]
    assert any(key.startswith("experience:state_lookup:") for key in snapshot)
