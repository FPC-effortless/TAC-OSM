from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CASM_ROOT = ROOT / "third_party" / "cdl-attention-experiment"
if not CASM_ROOT.exists():
    pytest.skip("pinned external generator is not checked out", allow_module_level=True)

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_graph_casm_selective_010 as g010
from tac_osm.behavioral_compatibility_router import (
    BehavioralCompatibilityRouter,
    BehavioralRouterConfig,
)
from tac_osm.behavioral_inverted_index import BehavioralInvertedIndex
from tac_osm.indexed_behavioral_reranker import CachedBehavioralReranker


def _rows(ep, n=4):
    return tuple(
        (tuple(k), int(ep.truth_table[k]))
        for k in g010.INPUT_ROWS[:n]
    )


def test_cached_reranker_matches_012_scores():
    model = BehavioralCompatibilityRouter(BehavioralRouterConfig(seed=31))
    candidates = g010.generate(13011, 8)
    features = torch.tensor(
        [g010.features(ep, True) for ep in candidates], dtype=torch.float32
    )
    rows = _rows(candidates[0])
    row_tensor = torch.tensor(
        [[float(x) for x in (*bits, y)] for bits, y in rows],
        dtype=torch.float32,
    )
    cached = CachedBehavioralReranker(model, features)
    expected = model.compatibility_scores(features, row_tensor).tolist()
    observed = cached.score(range(len(candidates)), row_tensor)
    assert observed == pytest.approx(expected, rel=0.0, abs=1e-6)


def test_cached_reranker_candidate_order_invariant():
    model = BehavioralCompatibilityRouter(BehavioralRouterConfig(seed=32))
    candidates = g010.generate(13012, 16)
    features = torch.tensor(
        [g010.features(ep, True) for ep in candidates], dtype=torch.float32
    )
    rows = _rows(candidates[0])
    row_tensor = torch.tensor(
        [[float(x) for x in (*bits, y)] for bits, y in rows],
        dtype=torch.float32,
    )
    cached = CachedBehavioralReranker(model, features)
    a = cached.score([0, 3, 7, 11], row_tensor)
    b = cached.score([11, 7, 3, 0], row_tensor)
    by_id_a = {i: s for i, s in zip([0, 3, 7, 11], a)}
    by_id_b = {i: s for i, s in zip([11, 7, 3, 0], b)}
    assert by_id_a == pytest.approx(by_id_b, rel=0.0, abs=1e-6)


def test_exact_index_reuses_one_library_and_masks_prefix():
    candidates = g010.generate(13013, 64)
    index = BehavioralInvertedIndex(candidates)
    support = _rows(candidates[17])
    full, _ = index.retrieve(support, population_size=64)
    prefix, _ = index.retrieve(support, population_size=32)
    assert set(prefix).issubset(set(full))
    assert all(i < 32 for i in prefix)
