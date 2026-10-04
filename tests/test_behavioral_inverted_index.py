from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASM_ROOT = ROOT / "third_party" / "cdl-attention-experiment"
if not CASM_ROOT.exists():
    pytest.skip("pinned external generator is not checked out", allow_module_level=True)

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_graph_casm_selective_010 as g010
from tac_osm.behavioral_inverted_index import BehavioralInvertedIndex


def test_index_retrieves_exact_support_consistent_candidates():
    candidates = g010.generate(13001, 32)
    support = tuple(
        (tuple(bits), int(candidates[0].truth_table[bits]))
        for bits in tuple(__import__('itertools').product((0, 1), repeat=g010.INPUT_COUNT))[:4]
    )
    index = BehavioralInvertedIndex(candidates)
    got, cost = index.retrieve(support)
    expected = [
        i for i, ep in enumerate(candidates)
        if all(int(ep.truth_table[tuple(bits)]) == int(y) for bits, y in support)
    ]
    assert got == expected
    assert cost.support_rows == 4
    assert cost.result_candidates == len(expected)
    assert cost.posting_lookups == 4


def test_nested_population_mask_does_not_expose_larger_library():
    candidates = g010.generate(13003, 64)
    support = tuple(
        (tuple(bits), int(candidates[0].truth_table[bits]))
        for bits in tuple(__import__('itertools').product((0, 1), repeat=g010.INPUT_COUNT))[:4]
    )
    index = BehavioralInvertedIndex(candidates)
    got, _ = index.retrieve(support, population_size=32)
    assert all(i < 32 for i in got)


def test_candidate_order_changes_ids_but_not_behavioral_identity():
    candidates = g010.generate(13004, 32)
    support = tuple(
        (tuple(bits), int(candidates[0].truth_table[bits]))
        for bits in tuple(__import__('itertools').product((0, 1), repeat=g010.INPUT_COUNT))[:4]
    )
    index1 = BehavioralInvertedIndex(candidates)
    got1, _ = index1.retrieve(support)
    permutation = list(reversed(range(len(candidates))))
    permuted = [candidates[i] for i in permutation]
    index2 = BehavioralInvertedIndex(permuted)
    got2, _ = index2.retrieve(support)
    keys1 = {g010.structure_key(candidates[i]) for i in got1}
    keys2 = {g010.structure_key(permuted[i]) for i in got2}
    assert keys1 == keys2
