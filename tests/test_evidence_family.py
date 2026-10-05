from tac_osm.evidence_family import (
    budget_success,
    effective_information_bits,
    expected_remaining,
    information_bits,
    information_per_work,
    partition_counts,
)


def test_balanced_binary_partition():
    evidence = (0, 0, 1, 1)
    assert partition_counts(evidence) == {0: 2, 1: 2}
    assert information_bits(evidence) == 1.0
    assert expected_remaining(evidence) == 2.0
    assert effective_information_bits(evidence) == 1.0
    assert information_per_work(evidence, 1.0) == 1.0


def test_budget_success_is_post_selection():
    candidates = (("00",), ("11",), ("00",), ("01",))
    assert budget_success(candidates, 2, ("00",), 1) == 0.0
    assert budget_success(candidates, 0, ("00",), 1) == 1.0


def test_budget_success_respects_budget():
    candidates = tuple((i % 2,) for i in range(10))
    assert budget_success(candidates, 8, (0,), 3) == 0.0
    assert budget_success(candidates, 2, (0,), 3) == 1.0


def test_invalid_inputs_fail_closed():
    try:
        partition_counts(())
    except ValueError:
        pass
    else:
        raise AssertionError("empty evidence must fail")
