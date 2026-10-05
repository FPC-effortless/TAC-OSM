"""Exact finite-domain information and terminal-budget bounds.

These are mathematical infrastructure, not measurements. They make explicit the
two ceilings used by the G-CASM active-evidence experiments:

* at most q observable signatures induce a partition of M candidates;
* any deterministic B-item shortlist inside each signature bucket can solve at
  most min(B, bucket_size) targets in that bucket.

The M/q expression is the Cauchy-Schwarz collision lower bound. When q > M,
the exact balanced partition floor is 1 rather than M/q.
"""
from __future__ import annotations

from typing import Iterable


def collision_cauchy_lower_bound(m: int, q: int) -> float:
    """Lower bound E[|H(E)|] >= M/q for q observable signatures."""
    if m <= 0 or q <= 0:
        raise ValueError("m and q must be positive")
    return m / q


def balanced_collision_floor(m: int, q: int) -> float:
    """Exact minimum expected bucket size over integer q-way partitions of M."""
    if m <= 0 or q <= 0:
        raise ValueError("m and q must be positive")
    a, r = divmod(m, q)
    total = r * (a + 1) ** 2 + (q - r) * a**2
    return total / m


def terminal_success_upper_bound(m: int, q: int, budget: int) -> float:
    """Universal upper bound min(1, q*B/M) for B-per-bucket selection."""
    if m <= 0 or q <= 0 or budget <= 0:
        raise ValueError("m, q, and budget must be positive")
    return min(1.0, (q * budget) / m)


def partition_success_upper_bound(bucket_sizes: Iterable[int], budget: int) -> float:
    """Exact best success probability for a fixed partition with B per bucket."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    sizes = [int(x) for x in bucket_sizes]
    if not sizes or any(x < 0 for x in sizes) or sum(sizes) <= 0:
        raise ValueError("bucket_sizes must be non-empty and sum to a positive value")
    m = sum(sizes)
    return sum(min(budget, size) for size in sizes) / m


def signature_alphabet_ceiling(binary_observations: int) -> int:
    """Maximum deterministic signature count for K binary observations."""
    if binary_observations < 0:
        raise ValueError("binary_observations must be non-negative")
    return 2**binary_observations


__all__ = [
    "balanced_collision_floor",
    "collision_cauchy_lower_bound",
    "partition_success_upper_bound",
    "signature_alphabet_ceiling",
    "terminal_success_upper_bound",
]