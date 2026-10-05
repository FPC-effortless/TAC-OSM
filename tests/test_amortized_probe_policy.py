import pytest
import torch

from tac_osm.amortized_probe_policy import (
    AmortizedProbePolicy,
    BitBeliefStats,
    budget_teacher_score,
    budget_utility_from_counts,
    choose_greedy_budget_action,
    make_action_sketch_matrix,
)


def test_budget_utility_from_counts():
    assert budget_utility_from_counts([8, 8], 8) == 1.0
    assert budget_utility_from_counts([16], 8) == 0.5


def test_budget_teacher_prefers_sub_budget_partition_when_costs_equal():
    evidence = {
        0: [(0,), (0,), (1,), (1,)],
        1: [(0,), (0,), (0,), (1,)],
    }
    score = choose_greedy_budget_action(
        evidence, budget=2, acquisition_cost_by_action={0: 1, 1: 1}
    )
    assert score.action == 0
    assert score.budget_utility == 1.0


def test_bit_belief_stats_add_remove_round_trip():
    stats = BitBeliefStats()
    sigs = [(0,1,0,1,0,1), (1,1,0,0,0,1)]
    for sig in sigs:
        stats.add(sig)
    before = stats.features(action_row=(0,1,0,1), acquisition_cost=2, budget=8)
    stats.remove(sigs[0])
    stats.add(sigs[0])
    after = stats.features(action_row=(0,1,0,1), acquisition_cost=2, budget=8)
    assert before == after


def test_action_sketch_has_fixed_width():
    sigs = {0: [(0,0,0,0,0,0), (1,1,0,0,0,1)]}
    rows = {0: (0,1,0,1)}
    x, actions = make_action_sketch_matrix(
        sigs, rows, {0: 1.0}, budget=8
    )
    assert x.ndim == 2 and x.shape[0] == 1
    assert actions == (0,)


def test_amortized_policy_shape_and_choice():
    model = AmortizedProbePolicy(feature_dim=30, hidden_dim=16)
    x = torch.zeros((4, 30))
    y = model(x)
    assert y.shape == (4,)
    assert model.choose(x) in range(4)


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        budget_utility_from_counts([], 8)
    with pytest.raises(ValueError):
        BitBeliefStats().remove((0,0,0,0,0,0))
    with pytest.raises(ValueError):
        AmortizedProbePolicy(0)
