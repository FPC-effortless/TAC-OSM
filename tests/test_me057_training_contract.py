import json
from pathlib import Path

import numpy as np
import torch

from scripts.me057_profile_fused_anchor import fused_objective, gru_objective
from scripts.me057_train_fused_anchor_robot import (
    SEEDS, build, paired_seed_wins, schedule)
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


def test_seed_win_count_is_json_native():
    episodes = ["7"]
    scores = {}
    for seed in SEEDS:
        scores[f"native_tac-s{seed}"] = {"7": {"physical_MAE": np.float64(.1)}}
        scores[f"gru_world-s{seed}"] = {"7": {"physical_MAE": np.float64(.2)}}
    wins = paired_seed_wins(scores, episodes)
    assert type(wins) is int
    assert json.loads(json.dumps({"wins": wins})) == {"wins": 5}


def test_me057_original_receipt_is_preserved_but_does_not_authorize_a_new_fit():
    plan = json.loads(Path(
        "research/mathematical-engineering/me057/FAIR_COMPARISON.json").read_text())
    import pytest
    from scripts.validate_research_parity import ParityError
    with pytest.raises(ParityError,match="100k target"):
        check_manifest(plan,"preflight")
    receipt = json.loads(Path(
        "research/me057-fused-anchor-preflight.json").read_text())
    assert receipt["strict_parameter_gate_pass"]
    assert receipt["strict_flop_gate_pass"]
    assert receipt["cpu_runtime_is_efficiency_only"]
    assert receipt["model_fits_started"] == 0


def test_completed_result_preserves_failed_gate_and_checkpoint_hashes():
    result = json.loads(Path(
        "research/me057-fused-anchor-completed-results.json").read_text())
    assert result["test_summary"]["registered_accuracy_gate_pass"] is False
    assert result["test_summary"]["TAC_seed_wins"] == 1
    assert len(result["checkpoints"]) == 10
    assert len({row["sha256"] for row in result["checkpoints"]}) == 10
    assert result["claims"] == {
        "predictive_superiority": False,
        "decision_intelligence": False,
        "policy_value": False,
        "autonomous_control": False,
    }


def test_registered_objectives_train_every_counted_parameter():
    states = torch.randn(2, 273, 14)
    actions = torch.randn(2, 272, 14)
    human = torch.zeros(2, 273, 1, dtype=torch.bool)
    source = (states, actions, human, ((0, 0), (1, 0)))
    for name in ("native_tac", "gru_world"):
        candidate = build(name, identified())
        methods = (candidate._burnin_training, candidate._observe_training,
                   candidate._predict_training)
        if name == "native_tac":
            methods = (methods[0], candidate._burnin_boundary_training,
                       methods[1], methods[2],
                       candidate._dense_transition_auxiliary_training)
            objective = fused_objective(candidate, source, methods)
        else:
            objective = gru_objective(candidate, source, methods)
        objective.backward()
        missing = [parameter_name for parameter_name, parameter
                   in candidate.named_parameters() if parameter.grad is None]
        assert missing == []
