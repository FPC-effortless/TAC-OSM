import pytest
from tac_osm.trace_capability import budget_capped_utility


def test_budget_utility_rewards_sub_budget_buckets():
    assert budget_capped_utility((0,) * 8 + (1,) * 8, 8) == 1.0
    assert budget_capped_utility((0,) * 16, 8) == 0.5


def test_budget_utility_is_bounded():
    assert budget_capped_utility((0,1,2,3), 8) == 1.0


def test_budget_utility_rejects_invalid():
    with pytest.raises(ValueError):
        budget_capped_utility((), 8)
    with pytest.raises(ValueError):
        budget_capped_utility((0,1), 0)
