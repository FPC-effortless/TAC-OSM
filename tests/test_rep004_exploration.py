"""Tests for TACOSM-REP-004 exploration control."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm import RoutingDecision
from tac_osm.exploration import EpsilonGreedySchedule, EpsilonGreedySelector


def test_schedule_is_frozen():
    schedule = EpsilonGreedySchedule()
    assert schedule.epsilon(0) == 0.30
    assert schedule.epsilon(250) == 0.15
    assert schedule.epsilon(500) == 0.0
    assert schedule.epsilon(1000) == 0.0


def test_epsilon_selector_preserves_scores():
    decision = RoutingDecision(
        selected=2,
        scores=(0.1, 0.2, 0.7, 0.3),
        provenance="test",
    )
    selected = EpsilonGreedySelector(seed=3).select(decision, step=500)
    assert selected == 2


def test_epsilon_selector_is_deterministic_for_fixed_seed():
    decision = RoutingDecision(
        selected=0,
        scores=(0.1, 0.2, 0.3, 0.4),
        provenance="test",
    )
    a = EpsilonGreedySelector(seed=11)
    b = EpsilonGreedySelector(seed=11)
    assert [a.select(decision, i) for i in range(32)] == [
        b.select(decision, i) for i in range(32)
    ]
