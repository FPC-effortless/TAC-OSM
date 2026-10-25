"""No-fit integrity checks for the fixed ME103 continuation contract."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from scripts import me100_sensor_data as data
from scripts.me103_continuation_panel import (
    DOC, PARENT, RECIPES, SEEDS, STEPS, TOTAL, next_plan, verify_parent,
)
from scripts.me102_reconstruction_panel import initialize
from tac_osm.parameter_budget import budgeted_optimizer


def test_continuation_plan_is_exact_suffix_of_me102_default_rng_stream():
    fit = data.load("fit")
    stored = np.load((DOC.parent / "me102" / "FIT_PLANS.npz"))
    for seed in SEEDS:
        full = np.concatenate((stored[str(seed)], next_plan(seed, fit)), axis=0)
        rng = np.random.default_rng(seed)
        expected = np.asarray([
            [fit['gap'][i] for i in rng.integers(len(fit['gap']), size=8)] +
            [fit['general'][i] for i in rng.integers(len(fit['general']), size=8)]
            for _ in range(TOTAL)
        ], dtype=np.int64)
        assert full.shape == (TOTAL, 16, 2)
        np.testing.assert_array_equal(full, expected)
        np.testing.assert_array_equal(full[:STEPS], stored[str(seed)])
        np.testing.assert_array_equal(full[STEPS:], next_plan(seed, fit))


def test_parent_optimizer_moments_restore_without_inheriting_guard_step_count(tmp_path):
    """The resumed optimizer keeps ME102 moments while ME103 guard starts fresh."""
    recipe, seed = "light_floor", SEEDS[0]
    _, parent, row = verify_parent(recipe, seed)
    model = initialize(data.load('fit'), recipe, seed)
    model.load_state_dict(parent['model'])
    opt = budgeted_optimizer(model, torch.optim.AdamW, lr=.0006, weight_decay=.001,
                             budget_manifest=None,
                             budget_receipt=tmp_path/'no-fit-receipt.json',
                             budget_seed=seed, budget_architecture=recipe)
    # Optimizer state loading must leave _tac_parameter_budget_guard intact.
    guard = opt._tac_parameter_budget_guard
    opt.load_state_dict(parent['optimizer'])
    assert opt._tac_parameter_budget_guard is guard
    assert guard.steps == 0
    assert len(opt.state) == len(parent['optimizer']['state'])
    for current, saved in zip(opt.state.values(), parent['optimizer']['state'].values()):
        assert int(current['step']) == STEPS
        assert torch.equal(current['exp_avg'], saved['exp_avg'])
        assert torch.equal(current['exp_avg_sq'], saved['exp_avg_sq'])
    assert int(min(v['step'] for v in opt.state.values())) == STEPS
    assert int(max(v['step'] for v in opt.state.values())) == STEPS
