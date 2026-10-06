import math
from tac_osm.evidence_index import ExactEvidenceIndex, PrefixEvidenceHistogramIndex


def test_index_matches_balanced_partition():
    idx = ExactEvidenceIndex.build(
        {
            0: ("a", "a", "b", "b"),
            1: ("x", "y", "x", "y"),
        },
        evidence_width=1,
    )
    score, work = idx.score_action(
        0,
        idx.full_bitmap(4),
        population_size=4,
        expected_cost=1.0,
    )
    assert score.information_gain_bits == 1.0
    assert score.expected_remaining_candidates == 2.0
    assert work == 2


def test_index_filters_to_prefix_population():
    idx = ExactEvidenceIndex.build(
        {
            0: ("a", "a", "b", "b", "b", "b"),
            1: ("x", "x", "x", "y", "y", "y"),
        },
        evidence_width=1,
    )
    score, _ = idx.score_action(
        0,
        idx.full_bitmap(4),
        population_size=4,
        expected_cost=1.0,
    )
    assert score.expected_remaining_candidates == 2.0


def test_choose_best_is_deterministic():
    idx = ExactEvidenceIndex.build(
        {
            0: ("a", "a", "b", "b"),
            1: ("z", "z", "z", "z"),
        },
        evidence_width=1,
    )
    score, _ = idx.choose_best(
        idx.full_bitmap(4),
        population_size=4,
        expected_cost_by_action={0: 1.0, 1: 1.0},
    )
    assert score.action == 0


def test_invalid_inputs_fail_closed():
    try:
        ExactEvidenceIndex.build({}, evidence_width=1)
    except ValueError:
        pass
    else:
        raise AssertionError("empty index must fail")

    try:
        ExactEvidenceIndex.build({0: (0, 1)}, evidence_width=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero evidence width must fail")


def test_choose_best_reports_work_for_all_actions():
    idx = ExactEvidenceIndex.build(
        {
            0: ("a", "a", "b", "b", "c", "c", "d", "d"),
            1: ("x", "y", "x", "y", "x", "y", "x", "y"),
        },
        evidence_width=1,
    )
    score, work = idx.choose_best(
        idx.full_bitmap(8),
        population_size=8,
        expected_cost_by_action={0: 1.0, 1: 1.0},
    )
    assert score.action == 0
    assert work == 6


def test_prefix_histogram_index_matches_exhaustive_partition_score():
    evidence = (0, 0, 1, 1, 1, 0)
    costs = (2.0, 2.0, 4.0, 4.0, 4.0, 4.0)
    idx = PrefixEvidenceHistogramIndex.build(
        {0: evidence},
        {0: costs},
        evidence_width=1,
    )
    score, reads = idx.score_action(0, population_size=6)

    counts = {0: 3, 1: 3}
    entropy = -sum((n / 6) * math.log2(n / 6) for n in counts.values())
    expected_remaining = sum(n * n for n in counts.values()) / 6
    expected_cost = sum(costs) / 6

    assert np.isclose(score.information_gain_bits, entropy)
    assert np.isclose(score.expected_remaining_candidates, expected_remaining)
    assert np.isclose(score.expected_cost, expected_cost)
    assert reads == 2


def test_prefix_histogram_query_does_not_scan_candidate_records():
    evidence = ("a", "a", "b", "b", "c", "c", "d", "d")
    idx = PrefixEvidenceHistogramIndex.build(
        {0: evidence, 1: tuple(reversed(evidence))},
        {0: (1.0,) * 8, 1: (1.0,) * 8},
        evidence_width=2,
    )
    score, reads = idx.choose_best(population_size=8)
    assert score.action == 0
    assert reads <= 2 * (2 ** 2)
    assert idx.build_candidate_records == 16


def test_prefix_histogram_rejects_mismatched_evidence_and_cost_domains():
    try:
        PrefixEvidenceHistogramIndex.build(
            {0: ("a", "b")},
            {1: (1.0, 1.0)},
            evidence_width=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched action domains must fail closed")
