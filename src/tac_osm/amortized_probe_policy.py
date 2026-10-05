"""Exact teacher and compact belief sketches for amortized active probing.

The teacher may inspect the exact candidate-predicted evidence partition. The learned
policy must not: it consumes a fixed-size action sketch that can be maintained
incrementally as the belief set changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Hashable, Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TeacherActionScore:
    action: Hashable
    budget_utility: float
    utility_gain: float
    acquisition_cost: float
    utility_per_work: float


def budget_utility_from_counts(counts: Sequence[int], budget: int) -> float:
    if budget <= 0:
        raise ValueError("budget must be positive")
    m = sum(int(n) for n in counts)
    if m <= 0 or any(int(n) < 0 for n in counts):
        raise ValueError("counts must be non-negative and sum to a positive population")
    return sum(min(budget, int(n)) for n in counts) / m


def _partition_counts(signatures: Sequence[Hashable]) -> tuple[int, ...]:
    buckets: dict[Hashable, int] = {}
    for signature in signatures:
        buckets[signature] = buckets.get(signature, 0) + 1
    return tuple(buckets.values())


def budget_teacher_score(
    signatures: Sequence[Hashable],
    *,
    budget: int,
    acquisition_cost: float,
) -> tuple[float, float]:
    if acquisition_cost <= 0:
        raise ValueError("acquisition_cost must be positive")
    if not signatures:
        raise ValueError("signatures must not be empty")
    post = budget_utility_from_counts(_partition_counts(signatures), budget)
    baseline = min(budget, len(signatures)) / len(signatures)
    return post, (post - baseline) / acquisition_cost


def choose_greedy_budget_action(
    evidence_by_action: Mapping[Hashable, Sequence[Hashable]],
    *,
    budget: int,
    acquisition_cost_by_action: Mapping[Hashable, float],
) -> TeacherActionScore:
    if not evidence_by_action:
        raise ValueError("evidence_by_action must not be empty")
    best: tuple[tuple, TeacherActionScore] | None = None
    for action in evidence_by_action:
        if action not in acquisition_cost_by_action:
            raise KeyError(f"missing acquisition cost for action {action!r}")
        post, utility_per_work = budget_teacher_score(
            evidence_by_action[action],
            budget=budget,
            acquisition_cost=float(acquisition_cost_by_action[action]),
        )
        baseline = min(budget, len(evidence_by_action[action])) / len(evidence_by_action[action])
        score = TeacherActionScore(
            action=action,
            budget_utility=post,
            utility_gain=post - baseline,
            acquisition_cost=float(acquisition_cost_by_action[action]),
            utility_per_work=utility_per_work,
        )
        key = (-score.utility_per_work, -score.budget_utility, score.action)
        if best is None or key < best[0]:
            best = (key, score)
    assert best is not None
    return best[1]


PAIR_INDICES = tuple((i, j) for i in range(6) for j in range(i + 1, 6))


class BitBeliefStats:
    """Mutable sufficient statistics for a set of six-bit evidence signatures."""

    def __init__(self, width: int = 6) -> None:
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = int(width)
        self.count = 0
        self.bit_sum = [0] * width
        self.pair_sum = {(i, j): 0 for i in range(width) for j in range(i + 1, width)}

    def add(self, signature: Sequence[int]) -> None:
        bits = tuple(int(x) for x in signature)
        if len(bits) != self.width or any(x not in (0, 1) for x in bits):
            raise ValueError("signature must be a binary vector of the configured width")
        self.count += 1
        for i, bit in enumerate(bits):
            self.bit_sum[i] += bit
        for pair in self.pair_sum:
            self.pair_sum[pair] += bits[pair[0]] * bits[pair[1]]

    def remove(self, signature: Sequence[int]) -> None:
        bits = tuple(int(x) for x in signature)
        if self.count <= 0:
            raise ValueError("cannot remove from empty statistics")
        if len(bits) != self.width or any(x not in (0, 1) for x in bits):
            raise ValueError("signature must be a binary vector of the configured width")
        self.count -= 1
        for i, bit in enumerate(bits):
            self.bit_sum[i] -= bit
        for pair in self.pair_sum:
            self.pair_sum[pair] -= bits[pair[0]] * bits[pair[1]]
        if self.count == 0 and (
            any(self.bit_sum) or any(self.pair_sum[p] for p in self.pair_sum)
        ):
            raise RuntimeError("statistics underflow")

    def features(
        self,
        *,
        action_row: Sequence[int],
        acquisition_cost: float,
        budget: int,
    ) -> list[float]:
        if self.count <= 0:
            raise ValueError("empty belief statistics")
        row = tuple(float(x) for x in action_row)
        if not row:
            raise ValueError("action_row must not be empty")
        denom = float(self.count)
        out = [log2(self.count + 1.0), float(budget) / self.count]
        out.extend(row)
        out.extend(value / denom for value in self.bit_sum)
        out.extend(self.pair_sum[p] / denom for p in self.pair_sum)
        out.append(float(acquisition_cost))
        return out


def make_action_sketch_matrix(
    signatures_by_action: Mapping[Hashable, Sequence[Sequence[int]]],
    action_rows: Mapping[Hashable, Sequence[int]],
    acquisition_cost_by_action: Mapping[Hashable, float],
    *,
    budget: int,
) -> tuple[Tensor, tuple[Hashable, ...]]:
    if not signatures_by_action:
        raise ValueError("signatures_by_action must not be empty")
    actions = tuple(signatures_by_action.keys())
    vectors: list[list[float]] = []
    for action in actions:
        stats = BitBeliefStats()
        for signature in signatures_by_action[action]:
            stats.add(signature)
        vectors.append(
            stats.features(
                action_row=action_rows[action],
                acquisition_cost=float(acquisition_cost_by_action[action]),
                budget=budget,
            )
        )
    return torch.tensor(vectors, dtype=torch.float32), actions


class AmortizedProbePolicy(nn.Module):
    """Small DAD-style policy over fixed-size per-action belief sketches."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, action_features: Tensor) -> Tensor:
        if action_features.ndim != 2:
            raise ValueError("action_features must have shape [actions, features]")
        return self.scorer(action_features).squeeze(-1)

    def choose(self, action_features: Tensor) -> int:
        with torch.no_grad():
            return int(torch.argmax(self.forward(action_features)).item())


__all__ = [
    "AmortizedProbePolicy",
    "BitBeliefStats",
    "TeacherActionScore",
    "budget_teacher_score",
    "budget_utility_from_counts",
    "choose_greedy_budget_action",
    "make_action_sketch_matrix",
]
