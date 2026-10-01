"""Structural tests for TACOSM-C5-LEARNED-STATE-DIAG-001."""

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
            input_dim=10, latent_dim=8, epochs=2,
            bucket_bits=8, probe_radius=1, shortlist_k=1, seed=seed
        )
    )


def test_train_and_eval_code_splits_are_disjoint():
    assert not (set(CODEBOOK[:16]) & set(CODEBOOK[16:64]))
    assert len(CODEBOOK[:16]) == 16
    assert len(CODEBOOK[16:64]) == 48


def test_model_query_and_state_embedding_costs_are_eighty_macs():
    m = _model()
    assert m.query_embedding_macs == 80
    assert m.state_embedding_macs == 80


def test_continuous_dot_product_cost_is_five_hundred_ninety_two():
    m = _model()
    task = build_task(1, 64, 1)
    assert m.query_embedding_macs + 64 * m.config.latent_dim == 592
    assert len(task.state_updates) == 64


def test_training_pair_count_is_epochs_times_codes():
    m = _model()
    assert m.train(CODEBOOK[16:20]) == 2 * 4


def test_binary_radius_one_has_nine_probes():
    m = _model()
    m.train(CODEBOOK[16:64])
    task = build_task(2, 64, 1)
    state = prepare_state(task)
    m.build(state)
    assert m.lookup(task.query).query_probes == 9


def test_state_pool_contains_all_sixty_four_items():
    m = _model()
    m.train(CODEBOOK[16:64])
    task = build_task(3, 64, 2)
    state = prepare_state(task)
    build = m.build(state)
    assert build.state_items == 64
    assert build.build_encodes == 64


def test_continuous_and_binary_use_same_encoder_state():
    m = _model(4)
    m.train(CODEBOOK[16:64])
    task = build_task(4, 64, 3)
    state = prepare_state(task)
    emb = m.encode_state(task.target_value)
    m.build(state)
    assert isinstance(emb[0], float)
    assert len(m.lookup(task.query).addresses) <= 1


def test_wrong_query_width_is_rejected():
    m = _model()
    from tac_osm import Query
    bad = Query(text="0 1 0", context=(), step=1)
    try:
        m.encode_query(bad)
    except ValueError as exc:
        assert "width" in str(exc)
    else:
        raise AssertionError("wrong-width query must be rejected")
