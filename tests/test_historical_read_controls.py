"""Constructed fixtures: engineering checks, no real-source model evidence."""
import pytest

from tac_osm.historical_read_controls import WriteIdentity, select_intervening_writes


def select(writes, **overrides):
    args = dict(target=WriteIdentity("station-A", 100), query_hour=116,
                count=4, seed=14014)
    return select_intervening_writes(writes, **(args | overrides))


def trace():
    return [WriteIdentity("station-A", h) for h in range(100, 117)]


def test_owner_and_prediction_time_boundary():
    baseline = select(trace())
    poisoned = trace() + [WriteIdentity("station-B", h) for h in range(90, 130)]
    poisoned += [WriteIdentity("station-A", h) for h in (99, 117, 1000)]
    assert select(poisoned) == baseline
    assert len(set(baseline["selected_hours"])) == 4
    assert all(100 < h < 116 for h in baseline["selected_hours"])
    assert baseline["scientific_qualification"] is False


def test_label_and_input_order_poison_cannot_select_opposite_pair():
    # Labels and paired cohort positions exist only outside the selector.
    records = [(w, i % 2, float(i)) for i, w in enumerate(trace())]
    before = select([w for w, label, score in records])
    poisoned = [(w, 1 - label, -1e20) for w, label, score in reversed(records)]
    assert select([w for w, label, score in poisoned]) == before
    assert select(trace(), count=2)["selected_hours"] == before["selected_hours"][:2]


def test_missing_controls_and_duplicate_identity_fail_closed():
    with pytest.raises(ValueError, match="insufficient"):
        select(trace(), count=16)
    with pytest.raises(ValueError, match="duplicate"):
        select(trace() + trace()[:1])
    with pytest.raises(ValueError, match="target missing"):
        select(trace()[1:])


@pytest.mark.parametrize("overrides", [dict(query_hour=100), dict(query_hour=True),
    dict(count=0), dict(count=True), dict(seed=-1), dict(seed=True)])
def test_invalid_registration_is_rejected(overrides):
    with pytest.raises(ValueError):
        select(trace(), **overrides)


def test_trace_change_has_a_different_replay_identity():
    before = select(trace())
    after = select(trace()[0:8])
    assert before["selection_sha256"] != after["selection_sha256"]
