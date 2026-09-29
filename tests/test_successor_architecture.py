"""Pre-capability gates for the TAC-OSM successor architecture."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm import Candidate, Query, StateUpdate, Structure
from tac_osm.environment import build_lookup_task, build_replay_task, build_relational_task
from tac_osm.executor import relevance_program
from tac_osm.explicit_executor import ExplicitGraphExecutor, ExplicitExecutorConfig
from tac_osm.packed import PackedGraphBatch
from tac_osm.state import PersistentStore, StateConfig
from tac_osm.state_addressing import StateAddressor
from tac_osm.energy_router import EnergyRouterConfig, RepresentationEnergyRouter


def test_explicit_executor_uses_true_edges_not_candidate_substrate():
    program = relevance_program((1, 0, 1, 0), (1, 1, 1, 0), [0, 2], max_nodes=16)
    result = ExplicitGraphExecutor().execute(
        Structure(key="p", spec=program),
        [],
    )
    assert result.output == 1.0
    assert sum(result.gates) == len(program.true_edges)
    assert sum(result.gates) < len(program.candidate_edges)


def test_exact_mode_is_distinct_from_soft_mode():
    # One marked position disagrees. Exact semantics must return 0, while the
    # explicit soft ablation retains a non-zero analog value.
    program = relevance_program((1, 0, 1, 0), (0, 1, 1, 0), [0, 2], max_nodes=16)
    exact = ExplicitGraphExecutor(ExplicitExecutorConfig(mode="exact")).execute(
        Structure(key="p", spec=program), []
    )
    soft = ExplicitGraphExecutor(
        ExplicitExecutorConfig(mode="soft", alpha=(0.7, 0.7))
    ).execute(Structure(key="p", spec=program), [])
    assert exact.output == 0.0
    assert 0.0 < soft.output < 1.0


def test_addressor_exposes_only_addressed_value():
    store = PersistentStore(StateConfig(seed=0, n_slots=8))
    store.write(StateUpdate(key="k1", value=(1, 0, 1), task_key="k1"))
    store.write(StateUpdate(key="k2", value=(0, 1, 0), task_key="k2"))
    memory = StateAddressor().address(
        Query(text="\tk2", context=(1, 1, 0)),
        store,
    )
    assert memory.found
    assert memory.key == "k2"
    assert memory.value == (0, 1, 0)
    assert memory.pool_size >= 2


def test_energy_router_excludes_candidate_action_index():
    task = build_relational_task(7, dim=8, n_candidates=8)
    store = PersistentStore(StateConfig(seed=0, n_slots=8))
    router = RepresentationEnergyRouter(EnergyRouterConfig(seed=1))
    first = router.route(task.query, store, task.candidates)
    swapped = tuple(
        Candidate(
            key=c.key,
            descriptor=c.descriptor,
            action=(99 - i),
            provenance=c.provenance,
        )
        for i, c in enumerate(task.candidates)
    )
    second = router.route(task.query, store, swapped)
    assert first.scores == second.scores


def test_energy_router_learns_from_success_only():
    task = build_relational_task(9, dim=8, n_candidates=8)
    store = PersistentStore(StateConfig(seed=0, n_slots=8))
    router = RepresentationEnergyRouter(EnergyRouterConfig(seed=2))
    before = router.updates
    loss_fail = router.learn_from_outcome(
        task.query, store, task.candidates, task.target_action, success=False
    )
    assert loss_fail == 0.0
    assert router.updates == before
    loss_ok = router.learn_from_outcome(
        task.query, store, task.candidates, task.target_action, success=True
    )
    assert loss_ok >= 0.0
    assert router.updates == before + 1


def test_packed_batch_counts_active_work():
    programs = [
        relevance_program((1, 0, 1, 0), (1, 1, 1, 0), [0, 2], max_nodes=16),
        relevance_program((0, 1, 0, 1), (0, 0, 0, 1), [1, 3], max_nodes=16),
    ]
    batch = PackedGraphBatch.from_programs(programs)
    batch.validate()
    assert batch.n_graphs == 2
    assert batch.total_active_nodes == sum(p.active_count for p in programs)
    assert batch.total_active_edges == sum(len(p.true_edges) for p in programs)
    assert batch.total_candidate_edges == sum(len(p.candidate_edges) for p in programs)
def test_successor_loop_uses_new_router_and_executor():
    from tac_osm.successor_builder import SuccessorConfig, build_successor

    model = build_successor(
        SuccessorConfig(n_steps=3, seed=11, learn=True, executor_mode="exact")
    )
    episode = model.run()

    assert episode.n == 3
    assert model.router.__class__.__name__ == "RepresentationEnergyRouter"
    assert model.executor.__class__.__name__ == "ExplicitGraphExecutor"
    for step in episode.steps:
        spec = step.computation.structure.spec
        assert spec.true_edges
        assert spec.true_edge_set <= {
            (edge.src, edge.dst, edge.port) for edge in spec.candidate_edges
        }
    assert model.router.updates >= 0


def test_successor_router_reports_route_time_diagnostics():
    task = build_relational_task(12, dim=8, n_candidates=8)
    store = PersistentStore(StateConfig(seed=0, n_slots=8))
    router = RepresentationEnergyRouter(
        EnergyRouterConfig(input_dim=8, latent_dim=16, seed=4)
    )

    decision = router.route(task.query, store, task.candidates)
    diag = router.last_diagnostics

    assert diag is not None
    assert diag.candidate_count == 8
    assert diag.candidates_scored == 8
    assert diag.candidate_coverage == 1.0
    assert 1 <= diag.selected_rank <= 8
    assert diag.selected_energy == -decision.scores[decision.selected]

def test_successor_config_freezes_single_execution_budget():
    from tac_osm.successor_builder import SuccessorConfig

    assert SuccessorConfig(top_k=1).top_k == 1
    try:
        SuccessorConfig(top_k=2)
    except ValueError as exc:
        assert "multi-candidate execution" in str(exc)
    else:
        raise AssertionError("top_k > 1 must not bypass the v1 execution contract")

def test_successor_representation_has_constructive_shared_witness():
    store = PersistentStore(StateConfig(seed=0, n_slots=64))
    router = RepresentationEnergyRouter(
        EnergyRouterConfig(input_dim=8, latent_dim=8, seed=5)
    )
    router.set_analytic_relation()

    builders = (
        lambda i: build_relational_task(1000 + i, dim=8, n_candidates=8),
        lambda i: build_lookup_task(2000 + i, store, dim=8, n_candidates=8),
        lambda i: build_replay_task(3000 + i, store, dim=8, n_candidates=8),
    )

    for build in builders:
        for i in range(16):
            task = build(i)
            decision = router.route(task.query, store, task.candidates)
            assert decision.selected == task.target_action
            assert router.last_diagnostics is not None
            assert router.last_diagnostics.selected_rank == 1
            assert router.last_diagnostics.hard_negative_margin >= 2.0

