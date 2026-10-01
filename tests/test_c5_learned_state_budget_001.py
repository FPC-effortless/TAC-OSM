"""Structural tests for TACOSM-C5-LEARNED-STATE-BUDGET-001."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.c5_end_to_end import build_task, prepare_state
from tac_osm.learned_state_index import (
    LearnedSemanticStateIndex,
    LearnedStateIndexConfig,
)
from tac_osm.noisy_state_tasks import CODEBOOK


def _model(seed: int = 0, epochs: int = 2) -> LearnedSemanticStateIndex:
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10,
            latent_dim=8,
            learning_rate=0.02,
            margin=0.25,
            epochs=epochs,
            bucket_bits=8,
            probe_radius=1,
            shortlist_k=1,
            seed=seed,
        )
    )


def test_registered_train_eval_code_split_is_disjoint():
    assert not (set(CODEBOOK[16:64]) & set(CODEBOOK[:16]))
    assert len(CODEBOOK[16:64]) == 48
    assert len(CODEBOOK[:16]) == 16


def test_continuous_query_cost_is_five_hundred_ninety_two_macs():
    model = _model()
    assert model.query_embedding_macs == 80
    assert 64 * model.config.latent_dim == 512
    assert model.query_embedding_macs + 64 * model.config.latent_dim == 592


def test_state_build_cost_is_five_thousand_one_hundred_twenty_macs():
    model = _model()
    assert model.state_embedding_macs == 80
    assert 64 * model.state_embedding_macs == 5120


def test_training_pairs_equal_epochs_times_training_codes():
    model = _model(1, epochs=3)
    assert model.train(CODEBOOK[16:20]) == 12


def test_h_anchor_task_preserves_sixty_four_state_items():
    task = build_task(3, 64, 1)
    state = prepare_state(task)
    assert len(state.addresses()) == 64


def test_h_anchor_query_is_one_bit_noisy():
    task = build_task(4, 64, 7)
    query_bits = tuple(int(x) for x in task.query.text.split())
    assert sum(a != b for a, b in zip(query_bits, task.target_value)) == 1


def test_distinct_budget_models_are_constructible():
    low = _model(5, epochs=32)
    high = _model(5, epochs=512)
    assert low.config.epochs == 32
    assert high.config.epochs == 512
    assert low.query_embedding_macs == high.query_embedding_macs == 80
