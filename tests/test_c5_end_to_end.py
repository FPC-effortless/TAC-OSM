"""Tests for TACOSM-C5-END-TO-END-001."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.c5_end_to_end import (
    EndToEndExecutor,
    build_candidate_index,
    build_population,
    build_state,
    build_state_index,
    build_task,
    full_state_scan,
    prepare_state,
)
from tac_osm.hamming_state_index import pool_for_addresses


def test_state_pool_has_64_unique_semantic_codes():
    updates = build_state(3)
    assert len(updates) == 64
    assert len({u.value for u in updates}) == 64


def test_candidate_population_has_exactly_four_programs_per_code():
    for h in (64, 128, 256):
        population = build_population(5, h)
        counts = {}
        for candidate in population:
            counts[candidate.descriptor] = counts.get(candidate.descriptor, 0) + 1
        assert all(count == 4 for count in counts.values())
        assert len(population) == h


def test_each_task_has_exactly_four_relevant_programs():
    for h in (64, 128, 256):
        task = build_task(7, h, 1)
        assert len(task.relevant_indices) == 4


def test_noisy_state_query_is_one_bit_from_target():
    task = build_task(11, 128, 2)
    query = tuple(int(x) for x in task.query.text.split())
    assert sum(a != b for a, b in zip(query, task.target_value)) == 1
    assert task.target_address not in task.query.text


def test_full_state_scan_recovers_target():
    task = build_task(13, 256, 3)
    state = prepare_state(task)
    address, value, operations = full_state_scan(task, state)
    assert address == task.target_address
    assert value == task.target_value
    assert operations == 64 * 10


def test_hamming_state_index_recovers_target():
    task = build_task(17, 256, 4)
    state = prepare_state(task)
    index = build_state_index(state)
    hit = index.lookup(task.query)
    assert hit.addresses == (task.target_address,)
    assert hit.query_operations == 56


def test_candidate_index_recovers_all_four_relevant_programs():
    task = build_task(19, 256, 5)
    index = build_candidate_index(task.candidates)
    reference = task.target_value
    hit = index.lookup(
        task.query,
        reference=reference,
        k=4,
        relation="equality",
    )
    assert set(hit.candidate_indices) == set(task.relevant_indices)
    assert hit.bucket_size == 4


def test_full_and_selective_execution_produce_identical_output():
    executor = EndToEndExecutor()
    for h in (64, 128, 256):
        task = build_task(23, h, 6)
        full_output, full_work, full_calls = executor.execute_population(task.candidates)
        relevant = tuple(task.candidates[i] for i in sorted(task.relevant_indices))
        selective_output, selective_work, selective_calls = executor.execute_population(
            relevant
        )
        assert full_output == selective_output == task.expected_output
        assert full_calls == h
        assert selective_calls == 4
        assert full_work == h * 12
        assert selective_work == 4 * 12


def test_indexed_shortlist_is_fixed_at_four_as_H_grows():
    for h in (64, 128, 256):
        task = build_task(29, h, 7)
        candidate_index = build_candidate_index(task.candidates)
        hit = candidate_index.lookup(
            task.query,
            reference=task.target_value,
            k=4,
            relation="equality",
        )
        assert len(hit.candidate_indices) == 4
        assert len(hit.candidate_indices) < h


def test_full_and_indexed_state_paths_both_recover_same_state():
    for h in (64, 128, 256):
        task = build_task(31, h, 8)
        state = prepare_state(task)
        full_address, full_value, _ = full_state_scan(task, state)
        index = build_state_index(state)
        hit = index.lookup(task.query)
        assert full_address == hit.addresses[0] == task.target_address
        assert full_value == task.target_value
