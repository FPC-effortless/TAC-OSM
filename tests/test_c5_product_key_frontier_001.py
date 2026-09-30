"""Contract-level checks for the C5 product-key frontier experiment."""

from __future__ import annotations

from scripts.run_c5_product_key_frontier_001 import (
    ARM_NAMES,
    ARM_SPECS,
    CAPABILITY_FLOOR,
    FACTOR_BEAMS_BY_SIZE,
    safe_ratio,
)


def test_frontier_arm_grid_is_valid():
    assert len(ARM_SPECS) == 9
    assert len(ARM_NAMES) == len(ARM_SPECS)
    assert all(beam <= factor_size for factor_size, beam in ARM_SPECS)
    assert FACTOR_BEAMS_BY_SIZE[16] == (4, 6, 8, 12, 16)
    assert FACTOR_BEAMS_BY_SIZE[32] == (4, 6, 8, 12)


def test_capability_floor_is_registered():
    assert CAPABILITY_FLOOR == 0.90


def test_safe_ratio_handles_zero():
    assert safe_ratio(1.0, 0.0) != safe_ratio(1.0, 1.0)


def test_safe_ratio_basic():
    assert safe_ratio(0.9, 1.0) == 0.9
