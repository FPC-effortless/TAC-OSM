"""Fail-closed checks for the Integration-001 independent validator."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "validate_plm_integration_001.py"
CONTRACT = MODULE.parents[1] / "contracts" / "TACOSM-PLM-INTEGRATION-001.json"
spec = importlib.util.spec_from_file_location("integration_validator", MODULE)
assert spec and spec.loader
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def baseline():
    return {
        "experiment_id": "TACOSM-PLM-INTEGRATION-001",
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "source_sha256": "a" * 64,
        "instrumentation": {
            "no_full_scan": True, "weights_changed": True,
            "nonzero_write_gradients": True, "key_family_disjoint": True,
            "no_oracle_input": True, "actual_write_counts_verified": True,
        },
        "episodes": [],
    }


def test_no_records_void():
    assert validator.validate(baseline(), json.loads(CONTRACT.read_text()))[
        "decision"] == "VOID"


def test_contract_tampering_void():
    record = baseline()
    record["contract_sha256"] = "fake"
    assert validator.validate(record, json.loads(CONTRACT.read_text()))[
        "decision"] == "VOID"


def test_missing_instrumentation_void():
    record = baseline()
    record["instrumentation"]["no_full_scan"] = False
    result = validator.validate(record, json.loads(CONTRACT.read_text()))
    assert result["decision"] == "VOID"
    assert any("no_full_scan" in x for x in result["problems"])


def test_incomplete_episode_void():
    record = baseline()
    record["episodes"] = [{"seed": 101, "H": 32, "D": 1,
                           "K": 8, "interference_strength": 0}]
    result = validator.validate(record, json.loads(CONTRACT.read_text()))
    assert result["decision"] == "VOID"


def test_registered_grid_and_controls_frozen():
    contract = json.loads(CONTRACT.read_text())
    assert contract["status"] == "pre-registered"
    assert contract["evaluation"]["seeds"] == [101, 211, 307, 401, 503]
    assert contract["evaluation"]["history_lengths"] == [32, 128, 512]
    assert contract["evaluation"]["intervening_writes"] == [1, 4, 16, 32]
    assert contract["evaluation"]["retrieval_budgets"] == [4, 8, 16]
    assert "paired_swap" in contract["arms"]["memory"]
    assert "exhaustive_exact" in contract["arms"]["execution"]
