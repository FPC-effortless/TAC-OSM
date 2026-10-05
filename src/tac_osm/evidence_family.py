"""Exact evidence-family primitives for finite-domain active probing."""
from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Hashable, Sequence


@dataclass(frozen=True)
class EvidenceAction:
    kind: str
    row: tuple[int, ...]


def partition_counts(evidence_by_candidate: Sequence[Hashable]) -> dict[Hashable, int]:
    if not evidence_by_candidate:
        raise ValueError("evidence_by_candidate must not be empty")
    counts: dict[Hashable, int] = {}
    for signature in evidence_by_candidate:
        counts[signature] = counts.get(signature, 0) + 1
    return counts


def information_bits(evidence_by_candidate: Sequence[Hashable]) -> float:
    counts = partition_counts(evidence_by_candidate)
    n = len(evidence_by_candidate)
    return float(-sum((c / n) * log2(c / n) for c in counts.values()))


def expected_remaining(evidence_by_candidate: Sequence[Hashable]) -> float:
    counts = partition_counts(evidence_by_candidate)
    n = len(evidence_by_candidate)
    return float(sum(c * c for c in counts.values()) / n)


def effective_information_bits(evidence_by_candidate: Sequence[Hashable]) -> float:
    n = len(evidence_by_candidate)
    remaining = expected_remaining(evidence_by_candidate)
    return float(log2(n / remaining)) if remaining > 0 else 0.0


def information_per_work(
    evidence_by_candidate: Sequence[Hashable], environment_work: float
) -> float:
    if environment_work <= 0:
        raise ValueError("environment_work must be positive")
    return information_bits(evidence_by_candidate) / float(environment_work)


def budget_success(
    candidate_evidence: Sequence[Hashable],
    target_index: int,
    target_evidence: Hashable,
    budget: int,
) -> float:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not (0 <= target_index < len(candidate_evidence)):
        raise IndexError("target_index outside candidate set")
    bucket = [i for i, signature in enumerate(candidate_evidence) if signature == target_evidence]
    return float(target_index in set(bucket[:budget]))
