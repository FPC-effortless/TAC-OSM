"""Runtime addressing separate from full-history candidate scoring.

The old retrieval path scans every candidate at every query. That can reduce
the expensive scorer's work, but it cannot establish a sublinear addressing
cost because the index itself is O(H) per query.

This module supplies a content-addressed benchmark control. The index is built
once when the candidate population is materialised; query-time lookup hashes
the query mask/value signature and returns a bounded bucket. Query-time work
therefore depends on the number of marked positions plus the returned bucket,
not on H.

This is intentionally an exact synthetic-relation index. It is infrastructure,
not evidence that semantic retrieval is solved. A future semantic/learned
addresser can implement the same interface and be compared against this
control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from . import Candidate, Query
from .environment import parse_query


def relation_signature(
    query: Query,
    reference: Sequence[int] | None,
) -> tuple[tuple[int, int], ...]:
    """Canonical (position, bit) signature for the active relation."""
    q_bits, _ = parse_query(query)
    marks = tuple(query.context)
    source = tuple(reference) if reference is not None else q_bits
    return tuple(
        (j, int(source[j]))
        for j in range(min(len(marks), len(source)))
        if marks[j]
    )


@dataclass(frozen=True)
class AddressHit:
    """The result of one query-time address resolution."""

    candidate_indices: tuple[int, ...]
    inspected_positions: int
    bucket_size: int


@dataclass
class ContentAddressIndex:
    """Exact signature index with O(1) bucket selection after build."""

    _buckets: dict[tuple[tuple[int, int], ...], tuple[int, ...]] = field(default_factory=dict)
    built_candidates: int = 0

    @classmethod
    def build(
        cls,
        candidates: Sequence[Candidate],
        *,
        context: Sequence[int],
    ) -> "ContentAddressIndex":
        buckets: dict[tuple[tuple[int, int], ...], list[int]] = {}
        for i, candidate in enumerate(candidates):
            signature = tuple(
                (j, int(candidate.descriptor[j]))
                for j in range(min(len(context), len(candidate.descriptor)))
                if context[j]
            )
            buckets.setdefault(signature, []).append(i)
        return cls(
            _buckets={key: tuple(value) for key, value in buckets.items()},
            built_candidates=len(candidates),
        )

    def lookup(
        self,
        query: Query,
        *,
        reference: Sequence[int] | None,
        k: int | None = None,
    ) -> AddressHit:
        signature = relation_signature(query, reference)
        bucket = self._buckets.get(signature, ())
        selected = bucket if k is None else bucket[: max(0, int(k))]
        return AddressHit(
            candidate_indices=tuple(selected),
            inspected_positions=len(signature),
            bucket_size=len(bucket),
        )


@dataclass(frozen=True)
class AddressCost:
    """Measured address-time cost, excluding one-time index construction."""

    build_candidates: int
    query_positions: int
    candidates_returned: int

    @property
    def query_growth(self) -> int:
        return self.query_positions + self.candidates_returned
