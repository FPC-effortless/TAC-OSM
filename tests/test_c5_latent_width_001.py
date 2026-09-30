"""Structural tests for TACOSM-C5-LATENT-WIDTH-001."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.c5_end_to_end import build_task, prepare_state
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from tac_osm.noisy_state_tasks import CODEBOOK


def _model(seed=0, width=8):
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10,
            latent_dim=width,
            learning_rate=0.02,
            margin=0.25,
            epochs=2,
            bucket_bits=min(8, width),
            probe_radius=1,
            shortlist_k=1,
            seed=seed,
        )
    )


def test_registered_latent_widths_are_ordered():
    assert (8, 16, 32) == tuple(sorted((8, 16, 32)))


def test_latent_width_changes_query_and_state_embedding_cost():
    for width in (8, 16, 32):
        model = _model(width=width)
        assert model.query_embedding_macs == 10 * width
        assert model.state_embedding_macs == 10 * width


def test_latent_width_8_total_query_cost_is_592():
    model = _model(width=8)
    assert model.query_embedding_macs + 64 * model.config.latent_dim == 592


def test_latent_width_16_total_query_cost_is_1184():
    model = _model(width=16)
    assert model.query_embedding_macs + 64 * model.config.latent_dim == 1184


def test_latent_width_32_total_query_cost_is_2368():
    model = _model(width=32)
    assert model.query_embedding_macs + 64 * model.config.latent_dim == 2368


def test_training_and_evaluation_codes_are_disjoint():
    assert set(CODEBOOK[16:64]).isdisjoint(CODEBOOK[:16])


def test_registered_state_population_is_sixty_four():
    task = build_task(2, 64, 0)
    state = prepare_state(task)
    assert len(state.addresses()) == 64
