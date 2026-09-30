"""Tests for REP-009 noisy selective state index."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.hamming_state_index import HammingStateIndex, pool_for_addresses
from tac_osm.noisy_state_tasks import CODEBOOK, STATE_BITS, build_pool, build_query_for_target, prepare_state
from tac_osm.semantic_state_addressor import SemanticAddressingConfig, SemanticStateAddressor


def test_codebook_has_64_codes_with_minimum_distance_three():
    assert len(CODEBOOK) == 64
    for i, left in enumerate(CODEBOOK):
        for right in CODEBOOK[i + 1:]:
            assert sum(a != b for a, b in zip(left, right)) >= 3


def test_noisy_query_differs_from_target_by_exactly_one_bit():
    pool = build_pool(3, n_states=16)
    task = build_query_for_target(pool, 5, step=1, flip=2)
    query_bits = tuple(int(x) for x in task.query.text.split())
    assert sum(a != b for a, b in zip(query_bits, task.target_value)) == 1
    assert task.target_address not in task.query.text


def test_hamming_radius_two_index_contains_target():
    pool = build_pool(5, n_states=64)
    state = prepare_state(pool)
    index = HammingStateIndex(bits=STATE_BITS, radius=2, shortlist_k=4)
    index.build(state)
    for target in (0, 7, 19, 63):
        query = build_query_for_target(pool, target, step=1, flip=target % STATE_BITS).query
        lookup = index.lookup(query)
        assert pool.updates[target].key in lookup.addresses
        assert lookup.shortlist_size <= 4
        assert lookup.query_operations == 56


def test_index_shortlist_is_smaller_than_full_pool_on_registered_pool():
    pool = build_pool(9, n_states=64)
    state = prepare_state(pool)
    index = HammingStateIndex(bits=STATE_BITS, radius=2, shortlist_k=4)
    index.build(state)
    query = build_query_for_target(pool, 11, step=1, flip=3).query
    lookup = index.lookup(query)
    assert lookup.shortlist_size <= 4
    assert lookup.shortlist_size < len(pool.updates)


def test_full_and_indexed_scorer_use_same_weights():
    pool = build_pool(13, n_states=32)
    state = prepare_state(pool)
    addressor = SemanticStateAddressor(SemanticAddressingConfig(input_dim=STATE_BITS, latent_dim=8, seed=2))
    query = build_query_for_target(pool, 4, step=1, flip=1).query
    full = addressor.select(query, state, target_address=pool.updates[4].key)
    index = HammingStateIndex(bits=STATE_BITS, radius=2, shortlist_k=4)
    index.build(state)
    lookup = index.lookup(query)
    shortlist = pool_for_addresses(state, lookup.addresses)
    indexed = addressor.select_from_pool(query, shortlist, target_address=pool.updates[4].key)
    assert len(full.scores) == 32
    assert len(indexed.scores) == lookup.shortlist_size
    assert full.total_macs == 80 + 88 * 32
    assert indexed.total_macs == 80 + 88 * lookup.shortlist_size


def test_analytic_identity_is_exact_after_one_bit_noise():
    addressor = SemanticStateAddressor(SemanticAddressingConfig(input_dim=STATE_BITS, latent_dim=STATE_BITS, seed=4))
    addressor.set_identity()
    pool = build_pool(17, n_states=64)
    state = prepare_state(pool)
    for target in (0, 8, 31, 63):
        query = build_query_for_target(pool, target, step=1, flip=(target + 1) % STATE_BITS).query
        decision = addressor.select(query, state, target_address=pool.updates[target].key)
        assert decision.selected_address == pool.updates[target].key
        assert decision.target_rank == 1