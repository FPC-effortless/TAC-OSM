"""Structural tests for TACOSM-C5-LEARNED-STATE-001."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm import Query
from tac_osm.c5_end_to_end import build_task, prepare_state
from tac_osm.learned_state_index import (
    LearnedSemanticStateIndex,
    LearnedStateIndexConfig,
)
from tac_osm.noisy_state_tasks import CODEBOOK


def _make_index(seed: int = 0) -> LearnedSemanticStateIndex:
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10,
            latent_dim=8,
            bucket_bits=8,
            probe_radius=1,
            shortlist_k=1,
            epochs=2,
            seed=seed,
        )
    )


def test_registered_code_split_is_disjoint():
    assert not (set(CODEBOOK[:16]) & set(CODEBOOK[16:64]))
    assert len(CODEBOOK[:16]) == 16
    assert len(CODEBOOK[16:64]) == 48


def test_config_exposes_registered_costs():
    index = _make_index()
    assert index.query_embedding_macs == 80
    assert index.state_embedding_macs == 80


def test_training_requires_multiple_codes():
    index = _make_index()
    try:
        index.train(CODEBOOK[:1])
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("single-code training must be rejected")


def test_training_does_not_use_state_addresses():
    index = _make_index()
    pairs = index.train(CODEBOOK[16:20])
    assert pairs == 2 * 4


def test_bucket_code_is_binary_width_eight():
    index = _make_index()
    embedding = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    assert index.bucket_code(embedding) == int("10101010", 2)


def test_radius_one_probe_count_is_nine():
    index = _make_index()
    index.train(CODEBOOK[16:20])
    task = build_task(3, 64, 1)
    state = prepare_state(task)
    index.build(state)
    hit = index.lookup(task.query)
    assert hit.query_probes == 9


def test_build_reads_all_state_items_once():
    index = _make_index()
    index.train(CODEBOOK[16:20])
    task = build_task(5, 64, 2)
    state = prepare_state(task)
    build = index.build(state)
    assert build.state_items == 64
    assert build.build_encodes == 64
    assert build.unique_buckets >= 1


def test_lookup_shortlist_is_bounded():
    index = _make_index()
    index.train(CODEBOOK[16:64])
    task = build_task(7, 128, 3)
    state = prepare_state(task)
    index.build(state)
    hit = index.lookup(task.query)
    assert len(hit.addresses) <= 1
    assert hit.shortlist_size <= 1


def test_query_parser_rejects_wrong_width():
    index = _make_index()
    bad = Query(text="0 1 0", context=(1,), step=1)
    try:
        index.encode_query(bad)
    except ValueError as exc:
        assert "width" in str(exc)
    else:
        raise AssertionError("wrong-width query must be rejected")


def test_lookup_requires_build():
    index = _make_index()
    task = build_task(9, 256, 4)
    try:
        index.lookup(task.query)
    except RuntimeError as exc:
        assert "built" in str(exc)
    else:
        raise AssertionError("lookup before build must fail")
