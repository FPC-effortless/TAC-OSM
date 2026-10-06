"""Exact inverted evidence index for target-identity-blind probe selection.

The index stores public candidate-predicted evidence signatures for each legal
probe action. Querying an index uses bitmap intersections rather than scanning
the entire candidate population. It is an exact computational transformation:
it must reproduce the exhaustive selector's action and score.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Hashable, Mapping, Sequence


@dataclass(frozen=True)
class IndexedProbeScore:
    action: Hashable
    information_gain_bits: float
    expected_remaining_candidates: float
    signature_count: int
    expected_cost: float
    information_per_work: float


class ExactEvidenceIndex:
    """Exact action -> evidence-signature -> candidate-bitmap index."""

    def __init__(
        self,
        postings: Mapping[Hashable, Mapping[Hashable, int]],
        population_size: int,
        evidence_width: int,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if evidence_width <= 0:
            raise ValueError("evidence_width must be positive")
        self._postings = {
            action: dict(buckets) for action, buckets in postings.items()
        }
        self.population_size = int(population_size)
        self.evidence_width = int(evidence_width)

    @classmethod
    def build(
        cls,
        evidence_by_action: Mapping[Hashable, Sequence[Hashable]],
        *,
        evidence_width: int,
    ) -> "ExactEvidenceIndex":
        if not evidence_by_action:
            raise ValueError("evidence_by_action must not be empty")
        lengths = {len(values) for values in evidence_by_action.values()}
        if len(lengths) != 1:
            raise ValueError("all actions must have the same candidate population")
        population_size = lengths.pop()
        if population_size <= 0:
            raise ValueError("candidate population must be positive")

        postings: dict[Hashable, dict[Hashable, int]] = {}
        for action, values in evidence_by_action.items():
            buckets: dict[Hashable, int] = {}
            for idx, signature in enumerate(values):
                buckets[signature] = buckets.get(signature, 0) | (1 << idx)
            postings[action] = buckets
        return cls(postings, population_size, evidence_width)

    @property
    def actions(self) -> tuple[Hashable, ...]:
        return tuple(self._postings)

    @property
    def signature_counts(self) -> dict[Hashable, int]:
        return {action: len(buckets) for action, buckets in self._postings.items()}

    @property
    def build_prediction_units(self) -> int:
        return self.population_size * len(self._postings) * self.evidence_width

    @property
    def build_posting_writes(self) -> int:
        return self.population_size * len(self._postings)

    @staticmethod
    def full_bitmap(population_size: int) -> int:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        return (1 << population_size) - 1

    def partition_counts(
        self,
        action: Hashable,
        compatible_bitmap: int,
        *,
        population_size: int,
        word_bits: int = 64,
    ) -> tuple[dict[Hashable, int], int]:
        if population_size <= 0 or population_size > self.population_size:
            raise ValueError("population_size outside indexed domain")
        if word_bits <= 0:
            raise ValueError("word_bits must be positive")
        mask = self.full_bitmap(population_size)
        current = int(compatible_bitmap) & mask
        buckets = self._postings[action]
        counts = {
            signature: (posting & current).bit_count()
            for signature, posting in buckets.items()
        }
        words = ceil(population_size / word_bits)
        query_units = len(buckets) * words
        return counts, query_units

    def score_action(
        self,
        action: Hashable,
        compatible_bitmap: int,
        *,
        population_size: int,
        expected_cost: float,
    ) -> tuple[IndexedProbeScore, int]:
        if expected_cost <= 0.0:
            raise ValueError("expected_cost must be positive")
        counts, work = self.partition_counts(
            action, compatible_bitmap, population_size=population_size
        )
        n = sum(counts.values())
        if n <= 0:
            raise ValueError("compatible bitmap must contain an indexed candidate")
        information = -sum(
            (count / n) * log2(count / n)
            for count in counts.values()
            if count
        )
        remaining = sum(count * count for count in counts.values()) / n
        return (
            IndexedProbeScore(
                action=action,
                information_gain_bits=float(information),
                expected_remaining_candidates=float(remaining),
                signature_count=sum(bool(c) for c in counts.values()),
                expected_cost=float(expected_cost),
                information_per_work=float(information / expected_cost),
            ),
            work,
        )

    def choose_best(
        self,
        compatible_bitmap: int,
        *,
        population_size: int,
        expected_cost_by_action: Mapping[Hashable, float],
    ) -> tuple[IndexedProbeScore, int]:
        best: tuple[tuple, IndexedProbeScore] | None = None
        total_work = 0
        for action in self.actions:
            if action not in expected_cost_by_action:
                raise KeyError(f"missing expected cost for action {action!r}")
            score, work = self.score_action(
                action,
                compatible_bitmap,
                population_size=population_size,
                expected_cost=float(expected_cost_by_action[action]),
            )
            total_work += work
            key = (
                -score.information_per_work,
                -score.information_gain_bits,
                score.expected_remaining_candidates,
                score.action if isinstance(score.action, (int, float, str)) else repr(score.action),
            )
            if best is None or key < best[0]:
                best = (key, score)
        if best is None:
            raise ValueError("index has no actions")
        return best[1], total_work


@dataclass(frozen=True)
class PrefixHistogramProbeScore:
    action: Hashable
    information_gain_bits: float
    expected_remaining_candidates: float
    signature_count: int
    expected_cost: float
    information_per_work: float


class PrefixEvidenceHistogramIndex:
    """Exact prefix-count index for one-step selection.

    The index is built once from public candidate-predicted evidence and
    cumulative action costs. A query for any prefix population M reads only
    the registered evidence alphabet counts and one cumulative cost total per
    action; it never scans candidate records. The O(M) construction cost is
    explicit and separately accountable.
    """

    def __init__(self, prefix_counts, prefix_cost_sums, population_size, evidence_width):
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if evidence_width <= 0:
            raise ValueError("evidence_width must be positive")
        self.population_size = int(population_size)
        self.evidence_width = int(evidence_width)
        self._prefix_counts = {action: dict(values) for action, values in prefix_counts.items()}
        self._prefix_cost_sums = {
            action: tuple(values) for action, values in prefix_cost_sums.items()
        }

    @classmethod
    def build(cls, evidence_by_action, cost_by_action, *, evidence_width):
        if not evidence_by_action:
            raise ValueError("evidence_by_action must not be empty")
        if set(evidence_by_action) != set(cost_by_action):
            raise ValueError("evidence and cost actions must match exactly")
        lengths = {len(v) for v in evidence_by_action.values()}
        cost_lengths = {len(v) for v in cost_by_action.values()}
        if len(lengths) != 1 or lengths != cost_lengths:
            raise ValueError("all actions must have the same candidate population")
        population_size = lengths.pop()
        if population_size <= 0:
            raise ValueError("candidate population must be positive")

        prefix_counts = {}
        prefix_cost_sums = {}
        for action in evidence_by_action:
            values = evidence_by_action[action]
            costs = cost_by_action[action]
            signatures = tuple(dict.fromkeys(values))
            counts = {signature: [0] * (population_size + 1) for signature in signatures}
            running = 0.0
            cost_prefix = [0.0]
            for i, (signature, cost) in enumerate(zip(values, costs), start=1):
                for key, series in counts.items():
                    series[i] = series[i - 1]
                counts[signature][i] += 1
                running += float(cost)
                cost_prefix.append(running)
            prefix_counts[action] = {k: tuple(v) for k, v in counts.items()}
            prefix_cost_sums[action] = tuple(cost_prefix)

        return cls(prefix_counts, prefix_cost_sums, population_size, evidence_width)

    @property
    def actions(self):
        return tuple(self._prefix_counts)

    @property
    def signature_counts(self):
        return {action: len(values) for action, values in self._prefix_counts.items()}

    @property
    def build_candidate_records(self):
        return self.population_size * len(self.actions)

    @property
    def max_query_bin_reads(self):
        return len(self.actions) * (2 ** self.evidence_width)

    def score_action(self, action, *, population_size):
        m = int(population_size)
        if m <= 0 or m > self.population_size:
            raise ValueError("population_size outside indexed domain")
        counts = {
            signature: series[m]
            for signature, series in self._prefix_counts[action].items()
            if series[m]
        }
        n = sum(counts.values())
        if n != m:
            raise RuntimeError("prefix histogram does not cover requested population")
        cost_sum = self._prefix_cost_sums[action][m]
        expected_cost = cost_sum / m
        information = -sum(
            (count / n) * log2(count / n)
            for count in counts.values()
            if count
        )
        remaining = sum(count * count for count in counts.values()) / n
        score = PrefixHistogramProbeScore(
            action=action,
            information_gain_bits=float(information),
            expected_remaining_candidates=float(remaining),
            signature_count=len(counts),
            expected_cost=float(expected_cost),
            information_per_work=float(information / expected_cost),
        )
        return score, len(counts)

    def choose_best(self, *, population_size):
        best = None
        total_bin_reads = 0
        for action in self.actions:
            score, reads = self.score_action(action, population_size=population_size)
            total_bin_reads += reads
            key = (
                -score.information_per_work,
                -score.information_gain_bits,
                score.expected_remaining_candidates,
                score.action if isinstance(score.action, (int, float, str)) else repr(score.action),
            )
            if best is None or key < best[0]:
                best = (key, score)
        if best is None:
            raise ValueError("index has no actions")
        return best[1], total_bin_reads


__all__ = [
    "IndexedProbeScore",
    "ExactEvidenceIndex",
    "PrefixHistogramProbeScore",
    "PrefixEvidenceHistogramIndex",
]
