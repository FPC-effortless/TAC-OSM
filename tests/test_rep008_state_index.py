"""Tests for the REP-008 exact semantic state-index ceiling."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.exact_state_index import ExactSemanticStateIndex
from tac_osm.semantic_state_tasks import build_semantic_state_task
from tac_osm.temporal import TemporalPersistentState


def prepare(task):
    state = TemporalPersistentState()
    if task.write_step:
        state.advance_to(task.write_step)
    task.stage(state)
    return state


def test_exact_index_retrieves_target_without_scanning_at_query_time():
    task = build_semantic_state_task(3, write_step=0, n_states=32)
    state = prepare(task)
    index = ExactSemanticStateIndex()
    build = index.build(state)
    lookup = index.lookup(task.query)
    assert lookup.found
    assert lookup.address == task.target_address
    assert lookup.query_operations == 1
    assert build.build_reads == 32


def test_exact_index_reports_duplicate_bucket_without_hiding_it():
    task = build_semantic_state_task(5, write_step=0, n_states=32)
    state = prepare(task)
    index = ExactSemanticStateIndex()
    index.build(state)
    lookup = index.lookup(task.query)
    assert lookup.found
    assert lookup.address == task.target_address
    assert lookup.bucket_size == 1


def test_unbuilt_index_fails_closed():
    task = build_semantic_state_task(7, write_step=0, n_states=8)
    index = ExactSemanticStateIndex()
    try:
        index.lookup(task.query)
    except RuntimeError as exc:
        assert "not been built" in str(exc)
    else:
        raise AssertionError("unbuilt exact index accepted a query")


def test_index_uses_opaque_addresses_as_values_not_query_inputs():
    task = build_semantic_state_task(11, write_step=0, n_states=16)
    state = prepare(task)
    index = ExactSemanticStateIndex()
    index.build(state)
    assert task.target_address not in task.query.text
    lookup = index.lookup(task.query)
    assert lookup.address == task.target_address
