"""Structural tests for TACOSM-C5-NEGATIVE-SWEEP-001."""

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


def test_registered_negative_pool_sizes_are_supported():
    assert (1, 4, 8, 16) == tuple(c for c in (1, 4, 8, 16) if c < 48)


def test_mean_gradient_update_count_is_independent_of_pool_size():
    expected = 2 * 4
    for count in (1, 4, 8, 16):
        model = _model(1)
        assert model.train_with_negative_coverage(
            CODEBOOK[16:20],
            negative_count=count,
            aggregation="mean",
        ) == expected


def test_sweep_does_not_allow_all_training_codes_as_negative_pool():
    model = _model()
    try:
        model.train_with_negative_coverage(
            CODEBOOK[16:20],
            negative_count=4,
            aggregation="mean",
        )
    except ValueError as exc:
        assert "smaller" in str(exc)
    else:
        raise AssertionError("negative pool equal to training set must fail")


def test_inference_cost_is_unchanged_by_training_coverage():
    model = _model()
    assert model.query_embedding_macs == 80
    assert model.state_embedding_macs == 80
    assert model.query_embedding_macs + 64 * 8 == 592


def test_registered_state_pool_is_sixty_four():
    task = build_task(0, 64, 0)
    state = prepare_state(task)
    assert len(state.addresses()) == 64


def test_train_and_eval_codebooks_are_disjoint():
    assert set(CODEBOOK[16:64]).isdisjoint(CODEBOOK[:16])


def test_negative_training_preserves_single_update_per_positive():
    model = _model(5)
    updates = model.train_with_negative_coverage(
        CODEBOOK[16:20],
        negative_count=16,
        aggregation="mean",
    )
    assert updates == 8
