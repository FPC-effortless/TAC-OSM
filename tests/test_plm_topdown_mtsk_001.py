from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm.mtsk_topdown import (  # noqa: E402
    BENCHMARK_HASH,
    GROUND_TRUTH_ALPHAS,
    MLPPolicy,
    PersistentTemporalState,
    evaluate,
    representability_witness,
    featurize,
    make_balanced_pairs,
    target_action,
)


def test_generator_is_balanced_and_paired():
    examples = make_balanced_pairs(123, 64, 30)
    assert len(examples) == 60
    for i in range(0, len(examples), 2):
        a, b = examples[i], examples[i + 1]
        assert a.current_observation == b.current_observation == 0.0
        assert a.label != b.label
        assert np.array_equal(b.history, -a.history)


def test_ground_truth_uses_three_nearby_timescales():
    assert tuple(np.round(GROUND_TRUTH_ALPHAS, 6)) == (0.45, 0.88, 0.985)
    assert len(BENCHMARK_HASH) == 64


def test_state_ablation_dimensions_are_matched():
    for arm in ("no_state", "single_timescale", "two_timescale", "mtsk"):
        state = PersistentTemporalState(arm)
        value = state.process(np.ones(16))
        assert value.shape == (3,)


def test_policy_parameter_count_is_identical():
    assert MLPPolicy(0, hidden=12).parameter_count == 86
    assert MLPPolicy(1, hidden=12).parameter_count == 86


def test_no_state_is_current_input_only_on_decision():
    examples = make_balanced_pairs(77, 64, 20)
    policy = MLPPolicy(3)
    result = evaluate(policy, examples, "no_state")
    assert result.state_footprint == 0
    assert result.action_gap == 0.0


def test_mtsk_representability_witness_passes():
    witness = representability_witness(seed=12345, pairs_per_family=100)
    assert min(witness.values()) >= 0.90


def test_mtsk_features_contain_multiple_temporal_coordinates():
    examples = make_balanced_pairs(99, 256, 12)
    X = featurize(examples, "mtsk")
    assert X.shape == (24, 4)
    assert np.std(X[:, 1]) > 0
    assert np.std(X[:, 2]) > 0
    assert np.std(X[:, 3]) > 0


def test_reset_intervention_preserves_computation_accounting():
    examples = make_balanced_pairs(88, 64, 20)
    policy = MLPPolicy(4)
    X = np.asarray([[0.0, 1.0, -0.5, 0.2] for _ in examples])
    y = np.asarray([x.label for x in examples])
    policy.fit(X, y, 10, 0.05)
    normal = evaluate(policy, examples, "mtsk")
    reset = evaluate(policy, examples, "mtsk", intervention="reset")
    assert normal.evaluation_work_per_episode == reset.evaluation_work_per_episode


def test_environment_outcome_is_exact_action_vs_label():
    from tac_osm.mtsk_topdown import environment_outcome
    assert environment_outcome(1, 1) is True
    assert environment_outcome(1, 0) is False


def test_target_action_changes_under_paired_history_negation():
    examples = make_balanced_pairs(321, 32, 12)
    for i in range(0, len(examples), 2):
        pair_type = ((0, 2), (0, 1), (1, 2))[(i // 2) % 3]
        assert target_action(examples[i].history, pair_type) != target_action(
            examples[i + 1].history, pair_type
        )
