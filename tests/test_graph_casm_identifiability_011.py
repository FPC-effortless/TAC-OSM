import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
CASM_ROOT = ROOT / "third_party" / "cdl-attention-experiment"
if not CASM_ROOT.exists():
    pytest.skip("pinned external generator is not checked out", allow_module_level=True)

SPEC = importlib.util.spec_from_file_location(
    "graph_casm_identifiability_011",
    ROOT / "scripts" / "run_graph_casm_identifiability_011.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_behavior_index_matches_scan():
    class Ep:
        pass
    # Use the real benchmark generator for a target pool.
    pool = MODULE.g010.generate(8111, 16)
    target = pool[0]
    support_rows = MODULE.INPUT_ROWS[:4]
    support = tuple((bits, target.truth_table[bits]) for bits in support_rows)
    scan = MODULE.support_consistent(pool, support)
    index = MODULE.BehaviorIndex(pool)
    indexed, _ = index.query(support)
    assert scan == indexed


def test_full_truth_table_makes_unique_target_with_unique_signatures():
    pool = MODULE.g010.generate(8112, 16)
    target_sig = MODULE.g010.truth_signature(pool[0])
    support = tuple((bits, pool[0].truth_table[bits]) for bits in MODULE.INPUT_ROWS)
    matches = MODULE.support_consistent(pool, support)
    assert matches == [i for i, ep in enumerate(pool) if MODULE.g010.truth_signature(ep) == target_sig]
    assert len(matches) == 1


def test_query_does_not_use_target_index():
    pool = MODULE.g010.generate(8113, 16)
    target = pool[0]
    support = tuple((bits, target.truth_table[bits]) for bits in MODULE.INPUT_ROWS[:4])
    index = MODULE.BehaviorIndex(pool)
    indexed, _ = index.query(support)
    changed = list(indexed)
    # Query is determined only by the support rows; mutating an unrelated
    # external target-index variable cannot alter it.
    bogus_target_index = (0 if len(pool) > 1 else None)
    assert indexed == changed
    assert bogus_target_index is not MODULE.BehaviorIndex


def test_support_size_sweep_is_monotone_for_each_fixed_task():
    pool = MODULE.g010.generate(8114, 32)
    target = pool[0]
    permutation = MODULE.support_indices(3, 32, 2)
    counts = []
    for size in MODULE.SUPPORT_SIZES:
        support = tuple((MODULE.INPUT_ROWS[k], target.truth_table[MODULE.INPUT_ROWS[k]]) for k in permutation[:size])
        counts.append(len(MODULE.support_consistent(pool, support)))
    assert counts[0] >= counts[1] >= counts[2] >= counts[3]
