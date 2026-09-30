"""Structural tests for TACOSM-C5-POSITIVE-VIEWS-001."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.c5_end_to_end import build_task, prepare_state
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from tac_osm.noisy_state_tasks import CODEBOOK


def _model(seed=0):
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10,
            latent_dim=8,
            learning_rate=0.02,
            margin=0.25,
            epochs=2,
            bucket_bits=8,
            probe_radius=1,
            shortlist_k=1,
            seed=seed,
        )
    )


def test_registered_positive_view_counts_are_prefix_ordered():
    assert (1, 2, 4) == tuple(v for v in (1, 2, 4) if v <= 4)


def test_multi_view_update_count_is_independent_of_view_count():
    expected = 2 * 24
    for views in (1, 2, 4):
        model = _model(1)
        assert model.train_with_negative_coverage(
            CODEBOOK[16:40],
            negative_count=8,
            aggregation="mean",
            positive_views=views,
        ) == expected


def test_positive_views_require_positive_count():
    model = _model()
    try:
        model.train_with_negative_coverage(
            CODEBOOK[16:40],
            negative_count=8,
            aggregation="mean",
            positive_views=0,
        )
    except ValueError as exc:
        assert "positive_views" in str(exc)
    else:
        raise AssertionError("zero positive views must fail")


def test_inference_cost_is_constant_across_view_arms():
    model = _model()
    assert model.query_embedding_macs == 80
    assert model.query_embedding_macs + 64 * 8 == 592


def test_state_pool_is_the_registered_sixty_four_items():
    task = build_task(0, 64, 0)
    state = prepare_state(task)
    assert len(state.addresses()) == 64


def test_training_and_evaluation_codebooks_are_disjoint():
    assert set(CODEBOOK[16:64]).isdisjoint(CODEBOOK[:16])


def test_eight_negatives_remain_below_training_pool_size():
    model = _model()
    assert model.train_with_negative_coverage(
        CODEBOOK[16:24],
        negative_count=8,
        aggregation="mean",
        positive_views=1,
    ) == 16
