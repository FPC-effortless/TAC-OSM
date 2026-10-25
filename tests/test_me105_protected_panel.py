import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import me105_protected_panel as panel
from scripts.me102_reconstruction_panel import initialize as initialize_me102


def test_fixed_panel_contract_and_trainable_count():
    assert panel.SEEDS == (105001, 105002, 105003, 105004, 105005)
    assert tuple(panel.RECIPES) == ("likelihood", "balanced", "point_guard")
    assert panel.STEPS == 1536
    assert panel.BATCH == 16
    model = panel.ProtectedSensorWorld()
    params = list(model.parameters())
    assert sum(p.numel() for p in params) == 100349
    assert len(params) == 33
    assert all(p.requires_grad for p in params)
    assert torch.count_nonzero(model.width_head.weight) == 0
    assert torch.count_nonzero(model.width_head.bias) == 0
    assert torch.count_nonzero(model.median_head.weight) == 0
    assert torch.count_nonzero(model.median_head.bias) == 0


def test_optimizer_groups_cover_all_parameters_with_fixed_lrs():
    model = panel.ProtectedSensorWorld()
    groups = panel.optimizer_parameter_groups(model)
    params = [parameter for group in groups for parameter in group["params"]]
    assert len(params) == 33
    assert len({id(parameter) for parameter in params}) == 33
    assert [group["lr"] for group in groups] == [0.000003, 0.0003]
    assert len(groups[1]["params"]) == 4


def test_parent_load_allows_only_zero_initialized_width_head():
    # The old full model is instantiated without loading or training a checkpoint.
    dummy = {"phase": "fit", "mean": np.zeros(6), "std": np.ones(6),
             "raw": np.zeros((1, 25, 6)), "observed": np.ones((1, 25, 6), dtype=bool),
             "gap": [], "general": []}
    parent = initialize_me102(dummy, "strong_reconstruction", 7)
    model = panel.ProtectedSensorWorld()
    loaded = panel.load_parent(model, {"model": parent.state_dict()})
    assert loaded.missing_keys == ["width_head.weight", "width_head.bias", "median_head.weight", "median_head.bias"]
    assert loaded.unexpected_keys == []
    for name, value in parent.state_dict().items():
        assert torch.equal(model.state_dict()[name], value), name


def test_plan_is_deterministic_and_half_gap_half_general():
    d = {"gap": np.asarray([[i, i + 1] for i in range(20)]),
         "general": np.asarray([[100 + i, i + 1] for i in range(20)])}
    first = panel.make_plan(105001, d)
    second = panel.make_plan(105001, d)
    assert first.shape == (1536, 16, 2)
    assert np.array_equal(first, second)
    assert np.all(first[:, :8, 0] < 20)
    assert np.all(first[:, 8:, 0] >= 100)


def test_recipe_coefficients_share_the_fixed_objective_weights():
    assert panel.RECIPES == {
        "likelihood": {"point": 0.1, "interval": 0.0},
        "balanced": {"point": 0.5, "interval": 0.25},
        "point_guard": {"point": 1.0, "interval": 0.25},
    }
