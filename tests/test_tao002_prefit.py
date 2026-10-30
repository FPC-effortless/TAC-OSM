"""TAO002 source/graph preflight tests. Genuine source reads, never training."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research/mathematical-engineering/langmem086"))

import torch
import pytest

from scripts import tao002_prefit
from tac_osm.optimizer_tao import TAO


def test_tao002_plan_has_all_arms_and_eligible_actual_model():
    report = tao002_prefit.verify_original_graph()
    assert report["parameter_count"] == 100521
    assert report["eligible_ffn_shape"] == [94, 160]
    rows = report["routing"]
    assert {r["arm"] for r in rows} == set(tao002_prefit.ARMS)
    assert all(r["orthogonal_parameter_names"] == ["ffn_up.weight"]
               for r in rows if r["arm"] != "adamw")
    assert rows[0]["orthogonal_parameter_count"] == 0
    assert all(r["orthogonal_parameter_count"] == 15040
               for r in rows if r["arm"] != "adamw")


def test_tao002_pinned_real_fit_schedule_matches_historical_and_seeds():
    report = tao002_prefit.audit_fit_samples()
    assert report["original_FIT_sha256"] == tao002_prefit.SOURCE_SHA
    assert report["fit_owner_count"] == 14
    assert len(report["frozen_sample_schedules"]) == 5
    for s in report["frozen_sample_schedules"]:
        assert s["targets"] == 163840


def test_tao002_prefit_never_steps_or_backprops(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("no model-training operation permitted during preflight")
    monkeypatch.setattr(torch.optim.AdamW, "step", denied)
    monkeypatch.setattr(TAO, "step", denied)
    monkeypatch.setattr(torch.Tensor, "backward", denied)
    report = tao002_prefit.prefit()
    assert report["fit_performed"] is False
    assert report["optimizer_steps"] == 0
    assert report["dev_bytes_accessed"] is False
    assert report["protected_TEST_accessed"] is False
    assert report["full_matched_comparison_gate"] != "PASS"


def test_tao002_committed_receipt_matches_reconstructed_original_preflight():
    import json
    frozen_path = ROOT / "research/mathematical-engineering/tao002/ORIGINAL_PREFIT_RECEIPT.json"
    frozen = json.loads(frozen_path.read_text())
    calculated = tao002_prefit.prefit()
    assert frozen["fit_performed"] is False
    assert frozen["optimizer_steps"] == 0
    for key in calculated:
        assert frozen[key] == calculated[key]
    assert frozen["source_ci_run_url"].endswith("/41009870520")
