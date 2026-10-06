import numpy as np

from tac_osm.memory_isolation import (
    ARMS,
    GROUND_TRUTH_ALPHAS,
    PersistentTemporalState,
    STATE_ALPHAS,
    featurize,
    make_pairs,
    representability_sign_accuracy,
)


def test_fixed_ground_truth_relation():
    assert GROUND_TRUTH_ALPHAS == (0.45, 0.985)


def test_state_arm_structure():
    assert ARMS == ("no_state", "single_timescale", "two_timescale", "mtsk")
    assert STATE_ALPHAS["no_state"] == (None, None, None)
    assert STATE_ALPHAS["single_timescale"] == (0.90, None, None)
    assert STATE_ALPHAS["two_timescale"] == (0.50, 0.98, None)


def test_negation_pairs_are_balanced():
    examples = make_pairs(11, 32, 10)
    for first, second in zip(examples[::2], examples[1::2]):
        assert first.current_observation == second.current_observation == 0.0
        assert first.label != second.label
        assert np.array_equal(second.history, -first.history)


def test_pair_family_is_not_an_input():
    examples = make_pairs(12, 16, 4)
    mutated = [
        type(examples[0])(
            history=examples[0].history.copy(),
            label=0,
            pair_id=999999,
            current_observation=examples[0].current_observation,
        )
    ]
    for arm in ARMS:
        first = featurize([examples[0]], arm)
        second = featurize(mutated, arm)
        assert np.array_equal(first, second)


def test_no_state_features_are_zero():
    examples = make_pairs(13, 16, 4)
    assert np.allclose(featurize(examples, "no_state")[:, 1:], 0.0)


def test_single_and_two_state_padding_is_zero():
    examples = make_pairs(14, 16, 4)
    one = featurize(examples, "single_timescale")
    two = featurize(examples, "two_timescale")
    assert np.allclose(one[:, 2:], 0.0)
    assert np.allclose(two[:, 3:], 0.0)

    state = PersistentTemporalState("mtsk")
    state.reset()
    for value in examples[0].history:
        state.update(float(value))
    assert np.allclose(state.values, featurize([examples[0]], "mtsk")[0, 1:])


def test_mtsk_representation_sign_is_above_gate():
    assert representability_sign_accuracy() >= 0.90
