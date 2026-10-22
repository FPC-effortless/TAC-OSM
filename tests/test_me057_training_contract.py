import json
from pathlib import Path

import numpy as np

from scripts.me057_train_fused_anchor_robot import build, schedule
from scripts.validate_research_parity import check_manifest


def identified():
    return {"delta_scale": np.full(14, .02, np.float32),
            "state_mean": np.zeros(14, np.float32),
            "state_scale": np.ones(14, np.float32),
            "action_mean": np.zeros(14, np.float32),
            "action_scale": np.ones(14, np.float32),
            "servo_gain": np.full(14, .1, np.float32),
            "servo_step_limit": np.full(14, .03, np.float32)}


def test_registered_models_remain_within_one_percent():
    counts = {name: sum(p.numel() for p in build(name, identified()).parameters())
              for name in ("native_tac", "gru_world")}
    assert counts == {"native_tac": 38268, "gru_world": 38295}


def test_metadata_only_schedule_is_deterministic():
    members = [(0, 300), (1, 310)]
    left = schedule(members, 57001)
    right = schedule(members, 57001)
    assert left == right
    assert len(left[0]) == 600


def test_me057_registration_and_resource_receipt_pass_preflight():
    plan = json.loads(Path(
        "research/mathematical-engineering/me057/FAIR_COMPARISON.json").read_text())
    assert check_manifest(plan, "preflight")["status"] == "PASS"
    receipt = json.loads(Path(
        "research/me057-fused-anchor-preflight.json").read_text())
    assert receipt["strict_parameter_gate_pass"]
    assert receipt["strict_flop_gate_pass"]
    assert receipt["strict_5pct_gate_pass"]
    assert receipt["model_fits_started"] == 0
