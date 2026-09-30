"""Tests for TACOSM-REP-002 explicit-program identifiability."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm import Candidate, StateUpdate, Structure
from tac_osm.energy_router import EnergyRouterConfig, RepresentationEnergyRouter
from tac_osm.explicit_executor import ExplicitGraphExecutor
from tac_osm.leakage import audit_router_inputs
from tac_osm.state import PersistentStore, StateConfig
from tac_osm.topology_router import ExplicitProgramEnergyRouter
from tac_osm.topology_tasks import (
    TOPOLOGY_EDGE_UNIVERSE,
    build_topology_task,
    program_for_candidate,
)


def test_same_descriptor_can_carry_distinct_executable_programs():
    task = build_topology_task(3)
    assert len({c.descriptor for c in task.candidates}) == 1
    assert len({c.executable_edges for c in task.candidates}) == len(task.candidates)


def test_legacy_descriptor_router_collides_on_distinct_programs():
    task = build_topology_task(5)
    store = PersistentStore(StateConfig(seed=0, n_slots=8))
    router = RepresentationEnergyRouter(EnergyRouterConfig(seed=2))
    scores = router.score(task.query, store, task.candidates)
    assert len(set(scores)) == 1


def test_explicit_router_distinguishes_candidate_programs():
    task = build_topology_task(7)
    router = ExplicitProgramEnergyRouter()
    observations = [router.candidate_observation(c) for c in task.candidates]
    assert len(set(observations)) == len(task.candidates)


def test_action_index_permutation_does_not_change_explicit_observation():
    task = build_topology_task(11)
    router = ExplicitProgramEnergyRouter()
    before = [router.candidate_observation(c) for c in task.candidates]
    swapped = tuple(
        Candidate(
            key=c.key,
            descriptor=c.descriptor,
            action=(99 - i),
            provenance=c.provenance,
            executable_edges=c.executable_edges,
        )
        for i, c in enumerate(task.candidates)
    )
    after = [router.candidate_observation(c) for c in swapped]
    assert before == after


def test_candidate_topology_passes_router_leakage_audit():
    task = build_topology_task(13)
    audit_router_inputs(query=task.public(), candidates=task.candidates)
    assert all(c.executable_edges for c in task.candidates)


def test_explicit_program_executes_declared_true_edges():
    task = build_topology_task(17)
    candidate = task.candidates[0]
    program = program_for_candidate(candidate, input_values=task.input_values)
    execution = ExplicitGraphExecutor().execute_with_work(
        Structure(key=candidate.key, spec=program),
        [],
    )
    expected = tuple(
        edge in candidate.executable_edges for edge in TOPOLOGY_EDGE_UNIVERSE
    )
    assert execution.work.active_edges == 2
    assert tuple(bool(g) for g in execution.result.gates) == expected


def test_explicit_router_analytic_witness_is_exact():
    for seed in range(8):
        task = build_topology_task(100 + seed)
        router = ExplicitProgramEnergyRouter()
        router.set_analytic_relation()
        decision = router.route(
            task.query,
            PersistentStore(StateConfig(seed=seed)),
            task.candidates,
        )
        assert decision.selected == task.target_action
        diag = router.last_diagnostics
        assert diag is not None
        assert diag.selected_rank == 1
        assert diag.hard_negative_margin == 4.0


def test_topology_task_programs_share_fixed_substrate():
    task = build_topology_task(23)
    programs = [
        program_for_candidate(c, input_values=task.input_values)
        for c in task.candidates
    ]
    assert len({tuple((e.src, e.dst, e.port) for e in p.candidate_edges) for p in programs}) == 1
    assert all(p.true_edge_set <= {
        (e.src, e.dst, e.port) for e in p.candidate_edges
    } for p in programs)
