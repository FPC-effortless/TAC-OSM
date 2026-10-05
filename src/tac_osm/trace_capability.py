"""Helpers for the G-CASM-016C trace-to-capability bridge."""
from __future__ import annotations

from collections import Counter
from typing import Hashable, Sequence


def compatible_bucket(
    candidate_evidence: Sequence[Hashable], target_evidence: Hashable
) -> tuple[int, ...]:
    return tuple(
        i for i, evidence in enumerate(candidate_evidence)
        if evidence == target_evidence
    )


def shortlist(bucket: Sequence[int], budget: int) -> tuple[int, ...]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    return tuple(bucket[:budget])


def budget_capped_utility(
    candidate_evidence: Sequence[Hashable], budget: int
) -> float:
    """Expected exact-verifier success under a uniform target prior.

    Given a deterministic evidence partition and a terminal budget B, a fixed
    public ordering can retain at most B candidates from each evidence bucket.
    Averaging over a uniform target prior gives U_B = sum_e min(B, |H_e|) / M.
    """
    if not candidate_evidence:
        raise ValueError("candidate_evidence must not be empty")
    if budget <= 0:
        raise ValueError("budget must be positive")
    counts = Counter(candidate_evidence)
    return sum(min(budget, n) for n in counts.values()) / len(candidate_evidence)
