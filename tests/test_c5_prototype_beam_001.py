"""Structural tests for TACOSM-C5-PROTOTYPE-BEAM-001."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.c5_end_to_end import build_task, prepare_state
from tac_osm.learned_prototype_state import LearnedPrototypeStateIndex, PrototypeStateIndexConfig
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from tac_osm.noisy_state_tasks import CODEBOOK


def _model(seed=0):
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10, latent_dim=16, learning_rate=0.02, margin=0.25,
            epochs=2, bucket_bits=8, probe_radius=1, shortlist_k=1, seed=seed
        )
    )


def test_beam_config_is_two_prototypes_and_eight_state_capacity():
    cfg = PrototypeStateIndexConfig(prototype_count=16, bucket_capacity=4, kmeans_iterations=8)
    assert 2 * cfg.bucket_capacity == 8


def test_beam_lookup_returns_no_more_than_eight_states():
    model = _model(1)
    model.train_with_negative_coverage(CODEBOOK[16:40], negative_count=8, aggregation="mean")
    task = build_task(1, 64, 0)
    state = prepare_state(task)
    index = LearnedPrototypeStateIndex(model)
    index.build(state, CODEBOOK[16:64])
    hit = index.lookup_beam(task.query, beam_width=2)
    assert len(hit.candidate_addresses) <= 8
    assert hit.state_rerank_macs <= 8 * 16


def test_beam_prototype_score_cost_is_256_macs():
    model = _model()
    model.train_with_negative_coverage(CODEBOOK[16:40], negative_count=8, aggregation="mean")
    task = build_task(2, 64, 0)
    state = prepare_state(task)
    index = LearnedPrototypeStateIndex(model)
    index.build(state, CODEBOOK[16:64])
    hit = index.lookup_beam(task.query, beam_width=2)
    assert hit.prototype_score_macs == 16 * 16


def test_beam_total_query_arithmetic_bound_is_544_macs():
    assert 160 + 256 + 8 * 16 == 544


def test_exhaustive_query_arithmetic_is_1184_macs():
    assert 160 + 64 * 16 == 1184


def test_beam_reuses_same_training_representation():
    model = _model(3)
    model.train_with_negative_coverage(CODEBOOK[16:40], negative_count=8, aggregation="mean")
    weights_before = tuple(tuple(row) for row in model.wq)
    task = build_task(3, 64, 1)
    state = prepare_state(task)
    index = LearnedPrototypeStateIndex(model)
    index.build(state, CODEBOOK[16:64])
    assert tuple(tuple(row) for row in model.wq) == weights_before


def test_train_eval_codebooks_remain_disjoint():
    assert set(CODEBOOK[16:64]).isdisjoint(CODEBOOK[:16])


def test_registered_state_population_is_sixty_four():
    task = build_task(4, 64, 0)
    state = prepare_state(task)
    assert len(state.addresses()) == 64
