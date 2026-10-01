"""Structural tests for TACOSM-C5-PROTOTYPE-SELECTIVE-001."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.c5_end_to_end import build_task, prepare_state
from tac_osm.learned_prototype_state import (
    LearnedPrototypeStateIndex,
    PrototypeStateIndexConfig,
)
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from tac_osm.noisy_state_tasks import CODEBOOK


def _model(seed=0):
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10,
            latent_dim=16,
            learning_rate=0.02,
            margin=0.25,
            epochs=2,
            bucket_bits=8,
            probe_radius=1,
            shortlist_k=1,
            seed=seed,
        )
    )


def test_registered_prototype_shape_and_capacity():
    cfg = PrototypeStateIndexConfig(
        prototype_count=16,
        bucket_capacity=4,
        kmeans_iterations=8,
    )
    assert cfg.prototype_count * cfg.bucket_capacity == 64
    assert cfg.kmeans_iterations == 8


def test_prototype_build_requires_enough_training_codes():
    model = _model()
    selective = LearnedPrototypeStateIndex(model)
    task = build_task(0, 64, 0)
    state = prepare_state(task)
    try:
        selective.build(state, CODEBOOK[:8])
    except ValueError as exc:
        assert "cover all prototypes" in str(exc)
    else:
        raise AssertionError("insufficient prototype training codes must fail")


def test_prototype_build_assigns_all_sixty_four_states_with_capacity_four():
    model = _model(1)
    model.train_with_negative_coverage(
        CODEBOOK[16:40], negative_count=8, aggregation="mean"
    )
    selective = LearnedPrototypeStateIndex(model)
    task = build_task(1, 64, 0)
    state = prepare_state(task)
    diagnostics = selective.build(state, CODEBOOK[16:64])
    assert diagnostics.state_items == 64
    assert diagnostics.max_bucket_size <= 4
    assert diagnostics.min_bucket_size >= 0


def test_registered_prototype_build_mac_ledger():
    model = _model()
    selective = LearnedPrototypeStateIndex(model)
    task = build_task(2, 64, 0)
    state = prepare_state(task)
    diagnostics = selective.build(state, CODEBOOK[16:64])
    assert diagnostics.train_embedding_macs == 7680
    assert diagnostics.kmeans_macs == 98304
    assert diagnostics.state_embedding_macs == 10240
    assert diagnostics.state_assignment_macs == 16384
    assert diagnostics.total_build_macs == 132608


def test_prototype_lookup_is_bounded_to_four_states():
    model = _model(3)
    model.train_with_negative_coverage(
        CODEBOOK[16:40], negative_count=8, aggregation="mean"
    )
    selective = LearnedPrototypeStateIndex(model)
    task = build_task(3, 64, 1)
    state = prepare_state(task)
    selective.build(state, CODEBOOK[16:64])
    hit = selective.lookup(task.query)
    assert len(hit.candidate_addresses) <= 4
    assert hit.state_rerank_macs <= 64


def test_prototype_query_score_cost_is_two_hundred_fifty_six_macs():
    model = _model()
    selective = LearnedPrototypeStateIndex(model)
    task = build_task(4, 64, 0)
    state = prepare_state(task)
    selective.build(state, CODEBOOK[16:64])
    hit = selective.lookup(task.query)
    assert hit.prototype_score_macs == 16 * 16


def test_train_eval_codebooks_are_disjoint():
    assert set(CODEBOOK[16:64]).isdisjoint(CODEBOOK[:16])


def test_state_pool_is_sixty_four_items():
    task = build_task(5, 64, 0)
    state = prepare_state(task)
    assert len(state.addresses()) == 64
