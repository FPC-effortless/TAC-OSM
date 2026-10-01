"""Preregistered training-time exploration for TACOSM-REP-004.

This module changes only how the training action is sampled. The graph-program
representation and learning rule remain unchanged. The schedule is frozen to
the exploration values already registered in TACOSM-LEARN-001.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import RoutingDecision


EPS_0 = 0.30
SCHEDULE_LENGTH = 500


@dataclass(frozen=True)
class EpsilonGreedySchedule:
    eps_0: float = EPS_0
    n_steps: int = SCHEDULE_LENGTH

    def __post_init__(self) -> None:
        if not 0.0 <= self.eps_0 <= 1.0:
            raise ValueError("eps_0 must be in [0, 1]")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")

    def epsilon(self, step: int) -> float:
        return self.eps_0 * max(0.0, 1.0 - step / self.n_steps)


@dataclass
class EpsilonGreedySelector:
    schedule: EpsilonGreedySchedule = EpsilonGreedySchedule()
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def select(self, decision: RoutingDecision, step: int) -> int:
        n = len(decision.scores)
        if n < 1:
            raise ValueError("cannot select from empty routing decision")
        if self._rng.random() < self.schedule.epsilon(step):
            return self._rng.randrange(n)
        return decision.selected
