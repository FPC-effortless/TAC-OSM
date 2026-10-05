from tac_osm.structured_action_probe import (
    ProbeAction,
    candidate_count_lower_bound,
    choose_best_action,
    deterministic_information_gain_bits,
    expected_remaining_candidates,
    score_action,
    signature_partition,
)


def test_partition_and_information_gain():
    evidence = ("a", "a", "b", "b")
    assert signature_partition(evidence) == {"a": 2, "b": 2}
    assert deterministic_information_gain_bits(evidence) == 1.0
    assert expected_remaining_candidates(evidence) == 2.0


def test_information_gain_is_zero_for_constant_channel():
    evidence = (0, 0, 0, 0)
    assert deterministic_information_gain_bits(evidence) == 0.0
    assert expected_remaining_candidates(evidence) == 4.0


def test_collision_bound_is_m_over_q():
    assert candidate_count_lower_bound(512, 16) == 32.0
    assert candidate_count_lower_bound(512, 64) == 8.0


def test_choose_best_action_uses_information_per_work():
    a = ProbeAction("scalar_row", (0,))
    b = ProbeAction("activation_trace", (0,))
    sa = score_action(a, (0, 0, 1, 1), expected_cost=1.0)
    sb = score_action(b, (0, 1, 2, 3), expected_cost=3.0)
    best = choose_best_action((sa, sb))
    assert best.action == a
    assert sa.information_per_work > sb.information_per_work


def test_empty_and_invalid_inputs_fail_closed():
    try:
        expected_remaining_candidates(())
    except ValueError:
        pass
    else:
        raise AssertionError("empty evidence must fail")

    try:
        candidate_count_lower_bound(4, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero alphabet must fail")

    try:
        choose_best_action(())
    except ValueError:
        pass
    else:
        raise AssertionError("empty score set must fail")
