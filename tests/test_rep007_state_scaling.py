"""Tests for TACOSM-STATE-REP-007 state population scaling."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.semantic_state_addressor import SemanticStateAddressor
from tac_osm.semantic_state_tasks import build_semantic_state_task
from tac_osm.temporal import TemporalPersistentState


def _prepare(task):
    state = TemporalPersistentState()
    if task.write_step:
        state.advance_to(task.write_step)
    task.stage(state)
    return state


def test_population_levels_produce_requested_state_pool_size():
    for m in (2, 4, 8, 16, 32):
        task = build_semantic_state_task(10 + m, write_step=0, n_states=m)
        state = _prepare(task)
        assert len(state.addresses()) == m


def test_target_semantic_value_is_unique_at_all_registered_levels():
    for m in (2, 4, 8, 16, 32):
        task = build_semantic_state_task(100 + m, write_step=0, n_states=m)
        assert sum(u.value == task.target_signature for u in task.state_updates) == 1


def test_identity_addressor_is_exact_at_all_registered_levels():
    for m in (2, 4, 8, 16, 32):
        for seed in range(3):
            task = build_semantic_state_task(
                1000 + m * 11 + seed, write_step=0, n_states=m
            )
            state = _prepare(task)
            addressor = SemanticStateAddressor()
            addressor.set_identity()
            decision = addressor.select(
                task.query, state, target_address=task.target_address
            )
            assert decision.target_rank == 1
            assert decision.selected_address == task.target_address


def test_address_cost_is_linear_in_state_population():
    addressor = SemanticStateAddressor()
    for m in (2, 4, 8, 16, 32):
        task = build_semantic_state_task(2000 + m, write_step=0, n_states=m)
        state = _prepare(task)
        decision = addressor.select(task.query, state, target_address=task.target_address)
        assert decision.total_macs == 40 + 48 * m


def test_opaque_addresses_do_not_appear_in_query():
    for m in (2, 4, 8, 16, 32):
        task = build_semantic_state_task(3000 + m, write_step=0, n_states=m)
        assert all(address not in task.query.text for address in state_addresses(task))


def state_addresses(task):
    return [u.key for u in task.state_updates]
