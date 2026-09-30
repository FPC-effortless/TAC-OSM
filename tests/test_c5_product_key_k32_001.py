"""Structural tests for TACOSM-C5-PRODUCT-KEY-K32-001."""

from __future__ import annotations

from pathlib import Path

import scripts.run_c5_product_key_k32_001 as protocol


def test_k32_protocol_pins_only_the_new_budget():
    assert protocol.K == 32
    assert protocol.FACTOR_BEAM == 6
    assert protocol.EXHAUSTIVE_QUERY_MACS == 1184


def test_k32_measurement_script_exists_at_repo_boundary():
    path = Path(protocol.__file__).resolve()
    assert path.name == "run_c5_product_key_k32_001.py"
