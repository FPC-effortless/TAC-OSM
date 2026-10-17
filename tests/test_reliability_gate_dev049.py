import math

import pytest

from tac_osm.development.reliability_gate import (
    outcome_totals, select_threshold, services_for, zero_event_upper_bound,
)


def test_safe_endpoint_services_even_exact_opposite_probabilities():
    assert services_for([[0, 1], [1, 0]], -1.) == (0, 1)


def test_empirical_gate_pays_for_services_instead_of_hiding_failures():
    p = [[[.1, 0], [.8, 0]], [[.9, 0], [.05, 0]]]
    y = [[[1, 0], [1, 0]], [[0, 0], [0, 0]]]
    cost_t, _ = select_threshold(p, y, zero_failures=False)
    safe_t, _ = select_threshold(p, y, zero_failures=True)
    assert outcome_totals(p, y, safe_t)["failures"] == 0
    assert outcome_totals(p, y, safe_t)["services"] >= outcome_totals(p, y, cost_t)["services"]


def test_no_safe_action_is_not_silently_called_certified():
    with pytest.raises(ValueError, match="no empirically feasible"):
        select_threshold([[[.4, .4]]], [[[1, 1]]], zero_failures=True)


@pytest.mark.parametrize("p", [[[float("nan"), 0]], [[-1, 0]], [[.1]], []])
def test_bad_predictions_rejected(p):
    with pytest.raises(ValueError):
        services_for(p, .25)


def test_zero_failures_has_nonzero_risk_bound():
    assert math.isclose(zero_event_upper_bound(144), 1 - .05 ** (1 / 144))
    assert zero_event_upper_bound(144) > .02
