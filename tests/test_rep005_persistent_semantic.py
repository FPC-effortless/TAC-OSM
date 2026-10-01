"""Tests for TACOSM-REP-005 persistent semantic routing."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm import Candidate, Query, StateUpdate
from tac_osm.persistent_semantic import build_persistent_semantic_task
from tac_osm.persistent_semantic_router import PersistentSemanticGraphRouter
from tac_osm.graph_program_router import GraphProgramRouterConfig
from tac_osm.state_addressing import StateAddressor
from tac_osm.temporal import TemporalPersistentState


def test_query_contains_address_only():
    task = build_persistent_semantic_task(3, write_step=7, delay=1)
    assert task.query.text == "\tgoal:semantic"
    assert task.target_signature == tuple(task.write.value)


def test_memory_is_unavailable_before_temporal_boundary():
    task = build_persistent_semantic_task(5, write_step=0, delay=2)
    state = TemporalPersistentState()
    state.stage_world_write(task.write, delay=task.delay)
    state.advance_to(0)
    memory = StateAddressor().address(task.query, state)
    assert not memory.found


def test_memory_becomes_available_at_declared_read_step():
    task = build_persistent_semantic_task(5, write_step=0, delay=2)
    state = TemporalPersistentState()
    state.stage_world_write(task.write, delay=task.delay)
    state.advance_to(task.read_step)
    memory = StateAddressor().address(task.query, state)
    assert memory.found
    assert memory.value == task.target_signature


def test_persistent_router_reads_state_not_query_payload():
    task = build_persistent_semantic_task(7, write_step=0, delay=1)
    state = TemporalPersistentState()
    state.stage_world_write(task.write, delay=task.delay)
    state.advance_to(task.read_step)
    router = PersistentSemanticGraphRouter(GraphProgramRouterConfig(seed=0))
    decision = router.route(task.query, state, task.candidates)
    assert len(decision.scores) == len(task.candidates)


def test_missing_state_fails_closed():
    task = build_persistent_semantic_task(11, write_step=0, delay=1)
    state = TemporalPersistentState()
    router = PersistentSemanticGraphRouter(GraphProgramRouterConfig(seed=0))
    try:
        router.route(task.query, state, task.candidates)
    except ValueError as exc:
        assert "unavailable" in str(exc)
    else:
        raise AssertionError("router accepted absent persistent semantic state")


def test_state_address_is_public_but_value_is_not_in_query():
    task = build_persistent_semantic_task(13, write_step=0, delay=1)
    assert task.query.text.partition("\t")[2] == task.write.key
    assert " ".join(str(x) for x in task.target_signature) not in task.query.text


def test_action_permutation_does_not_change_scores():
    task = build_persistent_semantic_task(17, write_step=0, delay=1)
    state = TemporalPersistentState()
    state.stage_world_write(task.write, delay=1)
    state.advance_to(1)
    router = PersistentSemanticGraphRouter(GraphProgramRouterConfig(seed=2))
    before = router.state_score(task.query, state, task.candidates)
    swapped = tuple(
        Candidate(
            key=c.key,
            descriptor=c.descriptor,
            action=100-i,
            provenance=c.provenance,
            executable_edges=c.executable_edges,
        )
        for i, c in enumerate(task.candidates)
    )
    after = router.state_score(task.query, state, swapped)
    assert before == after


def test_task_has_single_target_semantic_class():
    task = build_persistent_semantic_task(23, write_step=0, delay=1)
    from tac_osm.semantic_topology import semantic_signature
    matches = [semantic_signature(c.executable_edges) == task.target_signature for c in task.candidates]
    assert matches.count(True) == 1