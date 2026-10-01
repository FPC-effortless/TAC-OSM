"""Tests for the direct C5 execution-subset control."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.selective_execution import (
    MATCH_MARKS,
    WORK_UNITS_PER_CANDIDATE,
    SelectiveProgramExecutor,
    build_index,
    build_population,
    build_task,
    retrieve,
)


def test_relevant_set_is_exactly_four_at_all_registered_H():
    for h in (64, 128, 256):
        task = build_task(3, h, 0)
        assert len(task.relevant_indices) == 4


def test_full_execution_work_scales_with_H():
    executor = SelectiveProgramExecutor()
    for h in (64, 128, 256):
        task = build_task(5, h, 1)
        _, work, invocations = executor.execute_population(task.candidates, task.query)
        assert invocations == h
        assert work == h * WORK_UNITS_PER_CANDIDATE


def test_selective_execution_work_scales_with_R_not_H():
    executor = SelectiveProgramExecutor()
    for h in (64, 128, 256):
        task = build_task(7, h, 2)
        relevant = tuple(task.candidates[i] for i in task.relevant_indices)
        output, work, invocations = executor.execute_population(relevant, task.query)
        assert invocations == 4
        assert work == 4 * WORK_UNITS_PER_CANDIDATE
        assert output == task.expected_output


def test_exact_index_retains_all_four_relevant_candidates():
    for h in (64, 128, 256):
        task = build_task(11, h, 3)
        index = build_index(task)
        retained = retrieve(task, index, 4)
        assert set(retained) == set(task.relevant_indices)


def test_index_is_built_once_and_lookup_does_not_rebuild():
    task = build_task(13, 256, 4)
    index = build_index(task)
    first = retrieve(task, index, 4)
    second = retrieve(task, index, 4)
    assert first == second
    assert index.built_candidates == 256


def test_query_changes_reuse_same_candidate_universe():
    candidates = build_population(17, 256)
    task_a = build_task(17, 256, 5)
    task_b = build_task(17, 256, 6)
    assert task_a.candidates == candidates
    assert task_b.candidates == candidates
    assert task_a.query.text != task_b.query.text


def test_executor_work_is_positive_and_fixed_per_candidate():
    task = build_task(19, 64, 7)
    executor = SelectiveProgramExecutor()
    _, work1, calls1 = executor.execute_population(task.candidates[:1], task.query)
    _, work4, calls4 = executor.execute_population(task.candidates[:4], task.query)
    assert work1 == WORK_UNITS_PER_CANDIDATE
    assert work4 == 4 * WORK_UNITS_PER_CANDIDATE
    assert calls1 == 1
    assert calls4 == 4


def test_selective_output_matches_exhaustive_output():
    executor = SelectiveProgramExecutor()
    for h in (64, 128, 256):
        task = build_task(23, h, 8)
        full_output, _, _ = executor.execute_population(task.candidates, task.query)
        relevant = tuple(task.candidates[i] for i in task.relevant_indices)
        selective_output, _, _ = executor.execute_population(relevant, task.query)
        assert full_output == selective_output == task.expected_output