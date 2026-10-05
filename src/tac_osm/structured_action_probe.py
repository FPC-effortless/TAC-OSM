"""Generic structured evidence and information-gain primitives for TAC-OSM.

The module is deliberately environment-agnostic. It does not know a target
identity. A selector can score candidate-predicted evidence signatures and
choose an action before the environment reveals the target's realized
observation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Hashable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ProbeAction:
    """An identity-blind action descriptor."""

    kind: str
    parameters: tuple[Hashable, ...] = ()


@dataclass(frozen=True)
class ProbeScore:
    """Exact deterministic score under a uniform hypothesis prior."""

    action: ProbeAction
    information_gain_bits: float
    expected_remaining_candidates: float
    signature_count: int
    expected_cost: float
    information_per_work: float


def signature_partition(
    evidence_by_candidate: Sequence[Hashable],
) -> Mapping[Hashable, int]:
    """Return evidence-bucket counts for the candidate population."""
    counts: dict[Hashable, int] = {}
    for signature in evidence_by_candidate:
        counts[signature] = counts.get(signature, 0) + 1
    return counts


def deterministic_information_gain_bits(
    evidence_by_candidate: Sequence[Hashable],
) -> float:
    """I(H;E) in bits for deterministic evidence and a uniform prior."""
    n = len(evidence_by_candidate)
    if n <= 1:
        return 0.0
    counts = signature_partition(evidence_by_candidate)
    return float(
        -sum((c / n) * log2(c / n) for c in counts.values())
    )


def expected_remaining_candidates(
    evidence_by_candidate: Sequence[Hashable],
) -> float:
    """Expected compatible-set size after observing the deterministic evidence."""
    n = len(evidence_by_candidate)
    if n == 0:
        raise ValueError("evidence_by_candidate must not be empty")
    counts = signature_partition(evidence_by_candidate)
    return float(sum(c * c for c in counts.values()) / n)


def candidate_count_lower_bound(
    population_size: int,
    signature_alphabet_size: int,
) -> float:
    """Collision lower bound M/q for target-weighted mean bucket size."""
    if population_size < 0:
        raise ValueError("population_size must be non-negative")
    if signature_alphabet_size <= 0:
        raise ValueError("signature_alphabet_size must be positive")
    if population_size == 0:
        return 0.0
    return population_size / signature_alphabet_size


def score_action(
    action: ProbeAction,
    evidence_by_candidate: Sequence[Hashable],
    expected_cost: float,
) -> ProbeScore:
    """Score one deterministic action without target identity."""
    if expected_cost <= 0:
        raise ValueError("expected_cost must be positive")
    if not evidence_by_candidate:
        raise ValueError("evidence_by_candidate must not be empty")
    information = deterministic_information_gain_bits(evidence_by_candidate)
    remaining = expected_remaining_candidates(evidence_by_candidate)
    return ProbeScore(
        action=action,
        information_gain_bits=information,
        expected_remaining_candidates=remaining,
        signature_count=len(signature_partition(evidence_by_candidate)),
        expected_cost=float(expected_cost),
        information_per_work=information / float(expected_cost),
    )


def choose_best_action(scores: Iterable[ProbeScore]) -> ProbeScore:
    """Select the deterministic information-per-work maximum."""
    scores = tuple(scores)
    if not scores:
        raise ValueError("scores must not be empty")
    return min(
        scores,
        key=lambda s: (
            -s.information_per_work,
            -s.information_gain_bits,
            s.expected_remaining_candidates,
            s.action.kind,
            s.action.parameters,
        ),
    )
