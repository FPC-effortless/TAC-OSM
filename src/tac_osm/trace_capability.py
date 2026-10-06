"""Helpers for the G-CASM-016C trace-to-capability bridge.

``assert_evidence_invariant`` lives here rather than in the 016C runner so that
the two conditions 016C-R1 violated are testable without importing the CASM
generator or torch.
"""
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


def assert_evidence_invariant(
    candidate_evidence: Sequence[Hashable],
    target_evidence: Hashable,
    channel: str = "",
    row: int = 0,
) -> None:
    """Two conditions that 016C-R1 violated, enforced on every trial.

    1. Type equality. ``compatible_bucket`` compares candidate evidence to
       target evidence with ``==``. A type mismatch makes every comparison
       False and the bucket deterministically empty -- which is exactly the R1
       failure: int candidate evidence against a 1-tuple target.
    2. Non-emptiness. The target is itself a candidate, so its own evidence is
       present in ``candidate_evidence`` and must match. An empty bucket
       therefore cannot be an empirical result here; it can only mean the
       candidate and target extraction paths disagree.

    R1's regression asserted type identity on a hand-written literal, which
    passed while the runner was broken. This runs on generated evidence.
    """
    if not candidate_evidence:
        raise ValueError("candidate_evidence must not be empty")
    first = candidate_evidence[0]
    if any(type(v) is not type(first) for v in candidate_evidence):
        raise RuntimeError(
            f"candidate evidence is not uniformly typed ({channel}, row {row})"
        )
    if type(target_evidence) is not type(first):
        raise RuntimeError(
            f"evidence type mismatch ({channel}, row {row}): target is "
            f"{type(target_evidence).__name__}, candidate is "
            f"{type(first).__name__}"
        )
    if target_evidence not in candidate_evidence:
        raise RuntimeError(
            f"compatible bucket is empty ({channel}, row {row}); the target "
            f"is a candidate so its evidence must be present -- this indicates "
            f"an extraction-path disagreement, not an empirical result"
        )


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
