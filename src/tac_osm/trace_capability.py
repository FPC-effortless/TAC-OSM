"""Small helpers for the G-CASM-016C trace-to-capability bridge."""
from __future__ import annotations

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
