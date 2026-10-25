import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import me106_priority_panel as panel
from scripts import me105_protected_panel as me105
from scripts.me102_reconstruction_panel import initialize as initialize_me102
from tac_osm.variants.point_priority_sensor_world import PointPrioritySensorWorld


def test_fixed_panel_contract_and_trainable_count():
    assert panel.SEEDS == (106001, 106002, 106003, 106004, 106005)
    assert panel.PARENT_SEEDS == (102001, 102002, 102003, 102004, 102005)
    assert tuple(panel.RECIPES) == ("balanced", "point_guard", "point_strong")
    assert panel.RECIPES == {
        "balanced": {"point": 0.5, "interval": 0.25},
        "point_guard": {"point": 1.0, "interval": 0.25},
        "point_strong": {"point": 2.0, "interval": 0.25},
    }
    assert panel.STEPS == 1536
    assert panel.BATCH == 16
    assert panel.CONTEXTS == (8, 64, 256)
    model = PointPrioritySensorWorld()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 100349
    assert sum(p.numel() for p in model.parameters()) == 100349
    assert len(list(model.parameters())) == 33
    assert all(p.requires_grad for p in model.parameters())


def test_me105_runner_and_checkpoints_are_not_mutated():
    assert me105.SEEDS == (105001, 105002, 105003, 105004, 105005)
    assert me105.RECIPES == {
        "likelihood": {"point": 0.1, "interval": 0.0},
        "balanced": {"point": 0.5, "interval": 0.25},
        "point_guard": {"point": 1.0, "interval": 0.25},
    }


def test_optimizer_groups_cover_all_parameters_with_fixed_lrs():
    model = PointPrioritySensorWorld()
    groups = panel.optimizer_parameter_groups(model)
    params = [parameter for group in groups for parameter in group["params"]]
    assert len(params) == 33
    assert len({id(parameter) for parameter in params}) == 33
    assert [group["lr"] for group in groups] == [0.000003, 0.0003]
    assert len(groups[1]["params"]) == 4


def test_parent_load_accepts_only_new_zero_initialized_heads():
    dummy = {"phase": "fit", "mean": np.zeros(6), "std": np.ones(6),
             "raw": np.zeros((1, 25, 6)), "observed": np.ones((1, 25, 6), dtype=bool),
             "gap": [], "general": []}
    parent = initialize_me102(dummy, "strong_reconstruction", 7)
    model = PointPrioritySensorWorld()
    loaded = panel.load_parent(model, {"model": parent.state_dict()})
    assert loaded.missing_keys == ["width_head.weight", "width_head.bias", "median_head.weight", "median_head.bias"]
    assert loaded.unexpected_keys == []
    for name, value in parent.state_dict().items():
        assert torch.equal(model.state_dict()[name], value), name


def test_plan_is_deterministic_and_half_gap_half_general():
    d = {"gap": np.asarray([[i, i + 1] for i in range(20)]),
         "general": np.asarray([[100 + i, i + 1] for i in range(20)])}
    first = panel.make_plan(106001, d)
    second = panel.make_plan(106001, d)
    assert first.shape == (1536, 16, 2)
    assert np.array_equal(first, second)
    assert np.all(first[:, :8, 0] < 20)
    assert np.all(first[:, 8:, 0] >= 100)


def test_gradient_role_api_is_the_only_fit_assignment_path():
    source = Path(panel.__file__).read_text()
    assert ".backward(" not in source
    assert "assign_priority_gradients(model, parts" in source
    assert "point_strong" in source


def test_gradient_view_preserves_forecast_values_exactly():
    model = PointPrioritySensorWorld()
    location = torch.tensor([[0.5, -0.2]], dtype=torch.float32)
    left = torch.tensor([[0.4, 0.3]], dtype=torch.float32)
    right = torch.tensor([[0.6, 0.8]], dtype=torch.float32)
    correction = torch.tensor([[0.17, -0.08]], dtype=torch.float32)
    from tac_osm.variants.median_split_sensor_world import MedianSplitForecast
    original = MedianSplitForecast(location, left, right)
    view = model.uncertainty_forecast(original, {"median_correction": correction})
    assert torch.equal(view.log_location, original.log_location)
    assert torch.equal(view.left_scale, original.left_scale)
    assert torch.equal(view.right_scale, original.right_scale)


def test_projection_removes_only_opposing_auxiliary_gradient():
    from scripts.me106_gradient_roles import assign_priority_gradients

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

    model = Tiny()
    parts = {
        "point": -2 * model.weight,
        "nll": model.weight,
        "location_mse": model.weight * 0,
        "reconstruction_nll": model.weight * 0,
        "interval": model.weight * 0,
    }
    receipt = assign_priority_gradients(model, parts, 1.0, 0.25)
    assert torch.equal(model.weight.grad, torch.tensor(-2.0))
    assert receipt["auxiliary_dot_before"] < 0
    assert abs(float(receipt["auxiliary_dot_after"])) < 1e-12
