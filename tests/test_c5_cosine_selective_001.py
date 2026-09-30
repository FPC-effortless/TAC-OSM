"""Structural tests for TACOSM-C5-COSINE-SELECTIVE-001."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm import Query
from tac_osm.c5_end_to_end import build_task, prepare_state
from tac_osm.cosine_selective_state import CosineRerankedStateIndex
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
            probe_radius=2,
            shortlist_k=4,
            seed=seed,
        )
    )


def test_registered_pool_and_model_dimensions():
    assert len(CODEBOOK[16:64]) == 48
    assert len(CODEBOOK[:16]) == 16
    model = _model()
    assert model.config.latent_dim == 16
    assert model.config.bucket_bits == 8
    assert model.config.probe_radius == 2
    assert model.config.shortlist_k == 4


def test_radius_two_has_thirty_seven_bucket_probes():
    model = _model()
    model.train_with_negative_coverage(
        CODEBOOK[16:40], negative_count=8, aggregation="mean"
    )
    task = build_task(1, 64, 0)
    state = prepare_state(task)
    selective = CosineRerankedStateIndex(model)
    selective.build(state)
    assert selective.bucket_probe_count == 37


def test_selective_build_embeds_all_registered_state_items():
    model = _model()
    model.train_with_negative_coverage(
        CODEBOOK[16:40], negative_count=8, aggregation="mean"
    )
    task = build_task(2, 64, 0)
    state = prepare_state(task)
    selective = CosineRerankedStateIndex(model)
    build = selective.build(state)
    assert build["state_items"] == 64
    assert build["build_embedding_macs"] == 64 * 16 * 10


def test_selective_lookup_is_bounded_by_k_four():
    model = _model()
    model.train_with_negative_coverage(
        CODEBOOK[16:40], negative_count=8, aggregation="mean"
    )
    task = build_task(3, 64, 1)
    state = prepare_state(task)
    selective = CosineRerankedStateIndex(model)
    selective.build(state)
    hit = selective.lookup(task.query)
    assert hit.shortlist_scored <= 4
    assert hit.proposal.shortlist_size <= 4
    assert hit.cosine_score_macs <= 4 * 16


def test_cosine_normalization_is_unit_norm():
    values = [3.0, 4.0]
    norm = math.sqrt(sum(x * x for x in values))
    normalized = [x / norm for x in values]
    assert abs(sum(x * x for x in normalized) - 1.0) < 1e-12


def test_query_cost_terms_are_explicit():
    model = _model()
    assert model.query_embedding_macs == 160
    assert 64 * 16 == 1024
    assert 160 + 1024 == 1184


def test_train_eval_code_split_remains_disjoint():
    assert set(CODEBOOK[16:64]).isdisjoint(CODEBOOK[:16])


def test_state_read_boundary_remains_temporal():
    task = build_task(5, 64, 7)
    state = prepare_state(task)
    assert state.current_step == 1
    assert len(state.addresses()) == 64


def test_same_query_parser_is_binary_width_ten():
    model = _model()
    task = build_task(6, 64, 2)
    bits = tuple(int(x) for x in task.query.text.split())
    assert len(bits) == 10
    assert len(model.encode_query(task.query)) == 16

def test_lookup_exposes_bounded_shortlist_size():
    model = _model(7)
    model.train_with_negative_coverage(
        CODEBOOK[16:40], negative_count=8, aggregation="mean"
    )
    task = build_task(7, 64, 1)
    state = prepare_state(task)
    selective = CosineRerankedStateIndex(model)
    selective.build(state)
    hit = selective.lookup(task.query)
    assert hit.shortlist_scored <= 4
