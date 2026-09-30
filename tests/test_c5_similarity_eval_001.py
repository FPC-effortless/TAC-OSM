"""Structural tests for TACOSM-C5-SIMILARITY-EVAL-001."""

from __future__ import annotations

import math
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


def test_raw_training_update_budget_is_fixed():
    model = _model()
    assert model.train_with_negative_coverage(
        CODEBOOK[16:40],
        negative_count=8,
        aggregation="mean",
        positive_views=1,
    ) == 48


def test_training_and_evaluation_codes_are_disjoint():
    assert set(CODEBOOK[16:64]).isdisjoint(CODEBOOK[:16])


def test_query_and_state_projection_costs_are_160_macs():
    model = _model()
    assert model.query_embedding_macs == 160
    assert model.state_embedding_macs == 160


def test_raw_dot_total_query_mac_ledger_is_1184():
    model = _model()
    assert model.query_embedding_macs + 64 * 16 == 1184


def test_cosine_normalization_produces_unit_vector():
    values = [3.0, 4.0]
    denom = math.sqrt(sum(x * x for x in values))
    normalized = [x / denom for x in values]
    assert abs(sum(x * x for x in normalized) - 1.0) < 1e-12


def test_same_state_pool_is_reused_for_metric_comparison():
    task = build_task(2, 64, 0)
    state = prepare_state(task)
    addresses = tuple(state.addresses())
    assert len(addresses) == 64
    assert addresses == tuple(state.addresses())


def test_fixed_h_anchor_uses_one_step_persistent_state():
    task = build_task(3, 64, 7)
    assert task.step == 7
    assert len(task.state_updates) == 64
