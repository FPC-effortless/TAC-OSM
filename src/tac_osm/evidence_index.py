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
    ) -> tuple[IndexedProbeScore, int]:
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
            ),
            work,
        )

    def choose_best(
        self,
        compatible_bitmap: int,
        *,
        population_size: int,
    ) -> tuple[IndexedProbeScore, int]:
        best: tuple[tuple, IndexedProbeScore, int] | None = None
        total_work = 0
        for action in self.actions:
            score, work = self.score_action(
                action, compatible_bitmap, population_size=population_size
            )
            total_work += work
            key = (
                -score.information_gain_bits,
                score.expected_remaining_candidates,
                str(score.action),
            )
            if best is None or key < best[0]:
                best = (key, score, total_work)
        if best is None:
            raise ValueError("index has no actions")
        return best[1], total_work


__all__ = ["IndexedProbeScore", "ExactEvidenceIndex"]
