"""Pre-capability gates for the TAC-OSM successor architecture."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm import Candidate, Query, StateUpdate, Structure
from tac_osm.environment import build_relational_task
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
    program = relevance_program((1, 0, 1, 0), (1, 1, 1, 0), [0, 2], max_nodes=16)
    exact = ExplicitGraphExecutor(ExplicitExecutorConfig(mode="exact")).execute(
        Structure(key="p", spec=program), []
    )
    soft = ExplicitGraphExecutor(
        ExplicitExecutorConfig(mode="soft", alpha=(0.7, 0.7))
    ).execute(Structure(key="p", spec=program), [])
    assert exact.output in (0.0, 1.0)
    assert soft.output != exact.output


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
