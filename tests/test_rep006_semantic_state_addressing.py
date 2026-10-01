"""Tests for TACOSM-STATE-REP-006 semantic addressing."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm import Query
from tac_osm.semantic_state_addressor import SemanticStateAddressor
from tac_osm.semantic_state_tasks import (
    STATE_ADDRESS_COUNT,
    build_semantic_state_task,
)
from tac_osm.temporal import TemporalPersistentState


def test_addresses_are_opaque_and_target_address_is_absent_from_query():
    task = build_semantic_state_task(3, write_step=0, n_states=STATE_ADDRESS_COUNT)
    assert task.target_address not in task.query.text
    assert all(address.startswith("mem-") for address in (u.key for u in task.state_updates))


def test_target_semantic_value_is_unique_in_state_pool():
    task = build_semantic_state_task(5, write_step=0, n_states=STATE_ADDRESS_COUNT)
    matches = [u.value == task.target_signature for u in task.state_updates]
    assert matches.count(True) == 1


def test_all_state_items_cross_the_same_temporal_boundary():
    task = build_semantic_state_task(7, write_step=0, delay=2, n_states=STATE_ADDRESS_COUNT)
    state = TemporalPersistentState()
    for update in task.state_updates:
        state.stage_world_write(update, delay=task.delay)
    state.advance_to(task.read_step)
    assert len(state.addresses()) == STATE_ADDRESS_COUNT


def test_query_has_semantic_content_but_no_address():
    task = build_semantic_state_task(11, write_step=0)
    assert len(task.query.text.split()) == 5
    assert "	" not in task.query.text


def test_semantic_addressor_identity_is_exact():
    for seed in range(5):
        task = build_semantic_state_task(seed + 20, write_step=0)
        state = TemporalPersistentState()
        task.stage(state)
        addressor = SemanticStateAddressor()
        addressor.set_identity()
        decision = addressor.select(
            task.query, state, target_address=task.target_address
        )
        assert decision.target_rank == 1
        assert decision.selected_address == task.target_address


def test_missing_semantic_state_fails_closed():
    task = build_semantic_state_task(31, write_step=0)
    state = TemporalPersistentState()
    addressor = SemanticStateAddressor()
    try:
        addressor.select(task.query, state)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty semantic state pool was accepted")


def test_action_metadata_is_not_used_in_state_addressing():
    task = build_semantic_state_task(37, write_step=0)
    state = TemporalPersistentState()
    task.stage(state)
    addressor = SemanticStateAddressor()
    before = addressor.select(task.query, state).scores

    renamed = tuple(
        type(update)(
            key=update.key,
            value=update.value,
            task_key=update.task_key,
            success_score=update.success_score,
            step=update.step,
        )
        for update in task.state_updates
    )
    state2 = TemporalPersistentState()
    for update in renamed:
        state2.stage_world_write(update, delay=task.delay)
    state2.advance_to(task.read_step)
    after = addressor.select(task.query, state2).scores
    assert before == after
