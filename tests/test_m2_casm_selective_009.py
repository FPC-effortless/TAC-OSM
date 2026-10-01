from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.casm_s_adapter import CASMSAdapter, WorkAccounting


def test_execution_work_accounting_contract():
    w = WorkAccounting(17, 17, 6)
    assert w.total == 40


def test_m2_contract_is_preregistered():
    import json
    p = ROOT / "contracts" / "TACOSM-M2-CASM-SELECTIVE-009.json"
    data = json.loads(p.read_text())
    assert data["status"] == "preregistered"
    assert data["external_executor"]["commit"] == "c31554413301e3c9d3e6b3f8c8c6be572a74a748"
    assert data["leakage_rules"]
    assert data["primary_endpoint"]["capability_floor"] == 0.8
