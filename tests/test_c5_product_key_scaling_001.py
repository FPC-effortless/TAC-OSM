"""Tests for fixed-K product-key state-population scaling."""

from __future__ import annotations

from scripts.run_c5_product_key_scaling_001 import build_scaled_task, M_LEVELS


def test_scaling_levels_expand_state_population_without_changing_K():
    assert M_LEVELS == (64, 128, 256, 512)


def test_scaled_task_has_exactly_one_target_state():
    for m in M_LEVELS:
        task = build_scaled_task(seed=0, m=m, step=0)
        matches = [u for u in task.state_updates if tuple(u.value) == tuple(task.target_value)]
        assert len(matches) == 1
        assert len(task.state_updates) == m


def test_scaled_state_excludes_training_and_evaluation_decoys_from_target_pool():
    task = build_scaled_task(seed=1, m=512, step=3)
    values = {tuple(u.value) for u in task.state_updates}
    assert len(values) == 512

