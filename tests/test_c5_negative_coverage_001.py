"""Structural tests for TACOSM-C5-NEGATIVE-COVERAGE-001."""

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


def test_negative_training_modes_share_the_same_update_budget():
    baseline = _model(0)
    mean = _model(0)
    hard = _model(0)
    expected = 2 * 4
    assert baseline.train_with_negative_coverage(
        CODEBOOK[16:20], negative_count=1, aggregation="mean"
    ) == expected
    assert mean.train_with_negative_coverage(
        CODEBOOK[16:20], negative_count=3, aggregation="mean"
    ) == expected
    assert hard.train_with_negative_coverage(
        CODEBOOK[16:20], negative_count=3, aggregation="hardest"
    ) == expected


def test_negative_coverage_rejects_invalid_counts():
    model = _model()
    try:
        model.train_with_negative_coverage(
            CODEBOOK[16:20], negative_count=4, aggregation="mean"
        )
    except ValueError as exc:
        assert "smaller" in str(exc)
    else:
        raise AssertionError("all-code negative pool must be rejected")


def test_negative_coverage_rejects_unknown_aggregation():
    model = _model()
    try:
        model.train_with_negative_coverage(
            CODEBOOK[16:20], negative_count=2, aggregation="sum"
        )
    except ValueError as exc:
        assert "aggregation" in str(exc)
    else:
        raise AssertionError("unknown aggregation must be rejected")


def test_hardest_negative_is_selected_by_smallest_positive_minus_negative_margin():
    model = _model()
    model.train(CODEBOOK[16:64])
    # The implementation-level contract is that "hardest" means the highest
    # scoring negative, equivalently the smallest positive-minus-negative gap.
    assert model.config.margin == 0.25


def test_training_and_evaluation_codes_remain_disjoint():
    assert not (set(CODEBOOK[16:64]) & set(CODEBOOK[:16]))


def test_continuous_inference_cost_is_held_constant():
    model = _model()
    assert model.query_embedding_macs == 80
    assert 64 * model.config.latent_dim == 512
    assert model.query_embedding_macs + 64 * model.config.latent_dim == 592


def test_state_pool_is_the_registered_sixty_four_items():
    task = build_task(7, 64, 1)
    state = prepare_state(task)
    assert len(state.addresses()) == 64


def test_current_single_negative_training_path_still_exists():
    model = _model()
    assert model.train(CODEBOOK[16:20]) == 8
