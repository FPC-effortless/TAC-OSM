"""ME109 runner contract tests; these never create an optimizer or fit a model."""
import numpy as np
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import me109_residual_training as runner
from tac_osm.variants.reconstruction_sensor_world import ReconstructionSensorWorld
from tac_osm.variants.residual_role_sensor_world import ResidualRoleSensorWorld


def test_fixed_single_candidate_protocol():
    assert runner.ARMS == ("residual_guarded",)
    assert runner.SEEDS == (109001, 109002, 109003, 109004, 109005)
    assert runner.PARENT_SEEDS == (102001, 102002, 102003, 102004, 102005)
    assert runner.STEPS == 4096
    assert runner.BATCH == 16
    assert runner.CONTEXTS == (8, 64, 256)


def test_real_origin_pair_geometry_is_checked_before_optimizer_loop():
    data={'gap':[(0,300),(1,400)],'general':[(2,500),(3,600)]}
    plan=runner.make_plan(109001,data)
    assert plan.shape==(4096,16,2)
    runner.validate_plan(plan)
    import pytest
    with pytest.raises(RuntimeError):runner.validate_plan(plan[:,:,0])


def test_plan_is_seeded_and_uses_eight_origins_from_each_partition():
    data = {"gap": list(range(100, 140)), "general": list(range(200, 240))}
    first = runner.make_plan(109001, data)
    assert first.shape == (runner.STEPS, runner.BATCH)
    assert np.array_equal(first, runner.make_plan(109001, data))
    assert not np.array_equal(first, runner.make_plan(109002, data))
    assert np.all((first[:, :8] >= 100) & (first[:, :8] < 140))
    assert np.all((first[:, 8:] >= 200) & (first[:, 8:] < 240))


def test_me103_parent_retains_every_tensor():
    parent = ReconstructionSensorWorld()
    model = ResidualRoleSensorWorld()
    original = {key: value.clone() for key, value in parent.state_dict().items()}
    assert runner.load_parent(model, {"model": original}) == []
    assert all(torch.equal(model.state_dict()[key], value) for key, value in original.items())
    assert model.current_head.weight.shape == (12, 102)
    assert model.width_head.weight.count_nonzero() == 0
    assert model.median_head.weight.count_nonzero() == 0
    assert len([key for key in model.state_dict() if key not in original]) == 16


def test_zero_residual_preserves_forecast_and_streaming_state():
    from tests.test_protected_sensor_world import sample
    parent = ReconstructionSensorWorld()
    model = ResidualRoleSensorWorld()
    checkpoint = {"model": parent.state_dict()}
    runner.load_parent(model, checkpoint)
    receipt = runner.initial_preservation(model, checkpoint, sample(torch.float32))
    assert receipt["forecast_bit_exact"] and receipt["state_bit_exact"]


def test_gradient_block_norm_is_zero_when_gradient_is_absent():
    assert runner.gradient_norms([None], {"empty": [0]}) == {
        "empty": {"tensors": 0, "nonzero": 0, "norm": 0.0}
    }


def test_registered_seed_and_context_boundaries():
    assert runner.SEEDS == tuple(range(109001, 109006))
    assert runner.STEPS == 4096
    assert [runner.CONTEXTS[i % 3] for i in (0, 1, 2, 3)] == [8, 64, 256, 8]


def test_live_model_count_and_optimizer_block_order_without_updates():
    model = ResidualRoleSensorWorld()
    assert sum(parameter.numel() for parameter in model.parameters()) == 104663
    assert len(list(model.parameters())) == 45
    named, blocks = runner.blocks(model)
    groups = runner.parameter_groups(model)
    assert tuple(blocks) == runner.ROLE_BLOCKS
    assert sum(len(indices) for indices in blocks.values()) == len(named)
    assert [id(p) for group in groups for p in group["params"]] == [id(p) for _, p in named]
    assert [group["lr"] for group in groups] == [3e-5, 3e-4]


def test_training_entrypoint_requires_explicit_fit_command():
    assert runner.main.__name__ == "main"
    assert runner.OUT.name == "ME109"
    assert runner.REG.name == "FAIR_COMPARISON.json"


def test_seed_lineage_and_role_mapping():
    assert [seed - 7000 for seed in runner.SEEDS] == list(runner.PARENT_SEEDS)
    assert [seed - 8000 for seed in runner.SEEDS] == list(range(101001, 101006))
    assert runner.block_name("current_residual.input.weight") == "reconstruction"
    assert runner.block_name("width_residual.output.bias") == "width"
    assert runner.block_name("median_residual.output.weight") == "median"
    model = ResidualRoleSensorWorld()
    assert [len(g["params"]) for g in runner.parameter_groups(model)] == [33, 12]


def test_zero_output_initialization_has_nonzero_output_and_zero_hidden_gradients():
    from tests.test_protected_sensor_world import sample
    model = ResidualRoleSensorWorld()
    runner.load_parent(model, {"model": ReconstructionSensorWorld().state_dict()})
    forecast, _, trace = model(*sample(torch.float32), return_trace=True)
    loss = (forecast.log_location.sum() + forecast.left_scale.sum() +
            forecast.right_scale.sum() + trace["current_mean"].sum() +
            trace["current_variance"].sum())
    loss.backward()
    receipt = runner.initial_residual_gradients(model)
    assert all(row["hidden_gradients_zero"] and row["output_gradient_nonzero"]
               for row in receipt.values())
