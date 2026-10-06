from __future__ import annotations

import numpy as np

from tac_osm.timescale_transfer import (
    ARMS,
    ARM_ALPHAS,
    DISTRIBUTED_EIGHT_ALPHAS,
    PAIR_TYPES_TEST,
    PAIR_TYPES_TRAIN,
    POLICY_HIDDEN,
    TEST_FILTER_ALPHAS,
    TRAIN_FILTER_ALPHAS,
    FixedTemporalState,
    _features_for_state,
    make_pairs,
    make_policy,
    parameter_count,
    representability_witness,
)


def test_registered_filter_sets_are_disjoint():
    assert set(np.round(TRAIN_FILTER_ALPHAS, 12)).isdisjoint(
        set(np.round(TEST_FILTER_ALPHAS, 12))
    )


def test_pair_types_cover_hidden_comparisons():
    assert len(PAIR_TYPES_TRAIN) == 21
    assert len(PAIR_TYPES_TEST) == 21
    assert PAIR_TYPES_TRAIN[0] == (0, 1)
    assert PAIR_TYPES_TEST[-1] == (5, 6)


def test_negation_pairs_have_opposite_labels_and_same_current_observation():
    examples = make_pairs(123, 32, 10, alphas=TRAIN_FILTER_ALPHAS, pair_types=PAIR_TYPES_TRAIN)
    for i in range(0, len(examples), 2):
        a, b = examples[i], examples[i + 1]
        assert a.current_observation == b.current_observation == 0.0
        assert a.label != b.label
        assert np.array_equal(b.history, -a.history)


def test_parameter_count_is_matched_for_all_arms():
    assert all(parameter_count(arm) == 86 for arm in ARMS)


def test_policy_input_width_matches_state_arm():
    for arm in ARMS:
        p = make_policy(1, arm)
        assert p.parameter_count == 86
        assert p.input_dim == (9 if len(ARM_ALPHAS[arm]) == 8 else 4)
        assert p.hidden == POLICY_HIDDEN[arm]


def test_feature_shapes_are_registered():
    examples = make_pairs(5, 16, 4, alphas=TRAIN_FILTER_ALPHAS, pair_types=PAIR_TYPES_TRAIN)
    assert _features_for_state(examples, "three_timescale").shape == (8, 4)
    assert _features_for_state(examples, "distributed_eight").shape == (8, 9)
    assert FixedTemporalState("distributed_eight").footprint == 8


def test_representability_witness_clears_registered_gate():
    witness = representability_witness(128)
    assert min(witness.values()) > 0.99
    assert DISTRIBUTED_EIGHT_ALPHAS.shape == (8,)


def test_standard_policy_can_train_and_produce_binary_actions():
    examples = make_pairs(9, 32, 20, alphas=TRAIN_FILTER_ALPHAS, pair_types=PAIR_TYPES_TRAIN)
    X = _features_for_state(examples, "three_timescale")
    y = np.asarray([e.label for e in examples], dtype=np.int64)
    policy = make_policy(9, "three_timescale")
    policy.fit(X, y, steps=5, learning_rate=0.05)
    pred = policy.predict(X)
    assert set(pred.tolist()).issubset({0, 1})
