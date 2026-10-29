"""Constructed strict source-boundary and integrity-gap tests."""
import torch
from scripts.langdecode066_source_boundary import HIGH_CLOCKS, MAX_CLOCK, check, run


def test_unchanged_capped_cache_source_exact_dense_and_high_clocks():
    row=check(torch.float32)
    assert row["dense_clock_tests"]==65536
    assert row["checked_positions"]==65536+len(HIGH_CLOCKS)
    assert row["large_clock_state_transitions"]>=10
    assert row["max_abs_original_position_error"]==0
    assert row["bitwise_original_position_and_state"]


def test_fp64_and_scoped_local_gradient_match():
    row=check(torch.float64)
    assert row["position_linear_weight_gradient_bitwise"]
    assert row["max_abs_original_position_error"]==0


def test_mutable_cache_integrity_issue_must_remain_exposed_until_hardened():
    report=run()
    assert report["status"]=="NUMERICAL_RANGE_PASS_ENGINE_CACHE_INTEGRITY_OPEN"
    assert report["full_source_ieee_ad_proof"]=="OPEN"
    # This negative constructed control flags an actual oversight in
    # experimental 064; do not silently claim it is protected.
    assert all(row["original_guard_detected_cache_mutation"] is False
               for row in report["rows"])
    assert all(row["integrity_status"]=="OPEN_MUTABLE_ENGINE_CACHE_UNTRACKED"
               for row in report["rows"])
