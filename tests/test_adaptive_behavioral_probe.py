from dataclasses import dataclass

from tac_osm.adaptive_behavioral_probe import AdaptiveBehavioralProbeSelector


@dataclass
class EP:
    truth_table: dict[int, int]


def test_selects_balanced_row():
    candidates = [
        EP({0: 0, 1: 0, 2: 0}),
        EP({0: 0, 1: 1, 2: 0}),
        EP({0: 1, 1: 0, 2: 0}),
        EP({0: 1, 1: 1, 2: 0}),
    ]
    d = AdaptiveBehavioralProbeSelector().choose(
        candidates, (0, 1, 2, 3), (0, 1, 2)
    )
    assert d.row_index == 0
    assert (d.partition_zero, d.partition_one) == (2, 2)
    assert d.worst_case_remaining == 2


def test_lowest_row_breaks_equal_balance():
    candidates = [EP({0: 0, 1: 1}), EP({0: 1, 1: 0})]
    d = AdaptiveBehavioralProbeSelector().choose(
        candidates, (0, 1), (1, 0)
    )
    assert d.row_index == 0


def test_filter_is_exact():
    candidates = [EP({0: 0}), EP({0: 1}), EP({0: 0})]
    selector = AdaptiveBehavioralProbeSelector()
    assert selector.filter_compatible(candidates, (0, 1, 2), 0, 0) == (0, 2)
    assert selector.filter_compatible(candidates, (0, 1, 2), 0, 1) == (1,)


def test_empty_inputs_fail_closed():
    selector = AdaptiveBehavioralProbeSelector()
    candidates = [EP({0: 0})]
    try:
        selector.choose(candidates, (), (0,))
    except ValueError:
        pass
    else:
        raise AssertionError("empty compatible set must fail")


@dataclass
class TupleKeyEP:
    truth_table: dict[tuple[int, int], int]


def test_tuple_key_truth_tables_use_explicit_row_keys():
    candidates = [
        TupleKeyEP({(0, 0): 0, (1, 1): 0}),
        TupleKeyEP({(0, 0): 0, (1, 1): 1}),
        TupleKeyEP({(0, 0): 1, (1, 1): 0}),
        TupleKeyEP({(0, 0): 1, (1, 1): 1}),
    ]
    rows = ((0, 0), (1, 1))
    selector = AdaptiveBehavioralProbeSelector()
    d = selector.choose(candidates, (0, 1, 2, 3), (0, 1), row_keys=rows)
    assert d.row_index == 0
    assert (d.partition_zero, d.partition_one) == (2, 2)
    assert selector.filter_compatible(
        candidates, (0, 1, 2, 3), 1, 1, row_keys=rows
    ) == (1, 3)
