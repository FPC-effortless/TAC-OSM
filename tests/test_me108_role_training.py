"""ME108 runner contract tests; these never create an optimizer or fit a model."""
import numpy as np
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import me108_role_training as runner
from tac_osm.variants.reconstruction_sensor_world import ReconstructionSensorWorld
from tac_osm.variants.role_specialized_sensor_world import RoleSpecializedSensorWorld


def test_fixed_single_candidate_protocol():
    assert runner.ARMS == ("role_guarded",)
    assert runner.SEEDS == (108001, 108002, 108003, 108004, 108005)
    assert runner.PARENT_SEEDS == (102001, 102002, 102003, 102004, 102005)
    assert runner.STEPS == 4096
    assert runner.BATCH == 16
    assert runner.CONTEXTS == (8, 64, 256)


def test_real_origin_pair_geometry_is_checked_before_optimizer_loop():
    data={'gap':[(0,300),(1,400)],'general':[(2,500),(3,600)]}
    plan=runner.make_plan(108001,data)
    assert plan.shape==(4096,16,2)
    runner.validate_plan(plan)
    import pytest
    with pytest.raises(RuntimeError):runner.validate_plan(plan[:,:,0])


def test_plan_is_seeded_and_uses_eight_origins_from_each_partition():
    data = {"gap": list(range(100, 140)), "general": list(range(200, 240))}
    first = runner.make_plan(108001, data)
    assert first.shape == (runner.STEPS, runner.BATCH)
    assert np.array_equal(first, runner.make_plan(108001, data))
    assert not np.array_equal(first, runner.make_plan(108002, data))
    assert np.all((first[:, :8] >= 100) & (first[:, :8] < 140))
    assert np.all((first[:, 8:] >= 200) & (first[:, 8:] < 240))


def test_me103_reconstruction_parent_drops_only_its_old_linear_head():
    parent = ReconstructionSensorWorld()
    model = RoleSpecializedSensorWorld()
    dropped = runner.load_parent(model, {"model": parent.state_dict()})
    assert dropped == ["current_head.bias", "current_head.weight"]
    assert model.current_head.input.weight.shape == (16, 102)
    assert model.width_head.input.weight.shape == (16, 102)
    assert model.median_head.input.weight.shape == (16, 102)


def test_gradient_block_norm_is_zero_when_gradient_is_absent():
    assert runner.gradient_norms([None], {"empty": [0]}) == {
        "empty": {"tensors": 0, "nonzero": 0, "norm": 0.0}
    }


def test_registered_seed_and_context_boundaries():
    assert runner.SEEDS == tuple(range(108001, 108006))
    assert runner.STEPS == 4096
    assert [runner.CONTEXTS[i % 3] for i in (0, 1, 2, 3)] == [8, 64, 256, 8]


def test_live_model_count_and_optimizer_block_order_without_updates():
    model = RoleSpecializedSensorWorld()
    assert sum(parameter.numel() for parameter in model.parameters()) == 102713
    assert len(list(model.parameters())) == 39
    named, blocks = runner.blocks(model)
    groups = runner.parameter_groups(model)
    assert tuple(blocks) == runner.ROLE_BLOCKS
    assert sum(len(indices) for indices in blocks.values()) == len(named)
    assert [id(p) for group in groups for p in group["params"]] == [id(p) for _, p in named]
    assert [group["lr"] for group in groups] == [3e-5, 3e-4]


def test_training_entrypoint_requires_explicit_fit_command():
    assert runner.main.__name__ == "main"
    assert runner.OUT.name == "ME108"
    assert runner.REG.name == "FAIR_COMPARISON.json"
