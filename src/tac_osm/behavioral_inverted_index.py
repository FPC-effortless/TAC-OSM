"""Exact public-behavior inverted index for G-CASM-013.

This is a retrieval ceiling, not a learned semantic router. Candidate program
truth tables are precomputed as public library metadata. A query only touches
posting lists for its observed support rows and intersects them.

The implementation uses Python integers as fixed-width bitmaps. The logical
operation count is reported separately from neural MACs or executor work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


BehaviorKey = tuple[tuple[int, ...], int]


@dataclass(frozen=True)
class IndexQueryCost:
    support_rows: int
    posting_lookups: int
    bitmap_words: int
    bitmap_and_word_ops: int
    result_candidates: int


class BehavioralInvertedIndex:
    """Map observed (input_bits, output) facts to candidate bitmaps."""

    def __init__(self, candidates: Sequence):
        self.size = len(candidates)
        self._postings: dict[BehaviorKey, int] = {}
        self._build_memberships = 0
        for i, ep in enumerate(candidates):
            bit = 1 << i
            for bits, output in ep.truth_table.items():
                key: BehaviorKey = (tuple(bits), int(output))
                self._postings[key] = self._postings.get(key, 0) | bit
                self._build_memberships += 1

    def retrieve(
        self,
        support: Iterable[tuple[Sequence[int], int]],
        *,
        population_size: int | None = None,
    ) -> tuple[list[int], IndexQueryCost]:
        """Return exactly the candidates compatible with every support row.

        population_size masks a larger prebuilt library so one index can serve
        nested M levels without rebuilding for each query.
        """
        rows = tuple((tuple(bits), int(y)) for bits, y in support)
        m = self.size if population_size is None else int(population_size)
        if not (0 < m <= self.size):
            raise ValueError("population_size must be in [1, index.size]")

        allowed = (1 << m) - 1
        result_mask = allowed
        for key in rows:
            result_mask &= self._postings.get(key, 0)
            if result_mask == 0:
                break

        indices: list[int] = []
        mask = result_mask
        while mask:
            low = mask & -mask
            indices.append(low.bit_length() - 1)
            mask ^= low

        words = (m + 63) // 64
        cost = IndexQueryCost(
            support_rows=len(rows),
            posting_lookups=len(rows),
            bitmap_words=words,
            bitmap_and_word_ops=max(0, len(rows) - 1) * words,
            result_candidates=len(indices),
        )
        return indices, cost

    @property
    def posting_count(self) -> int:
        return len(self._postings)

    @property
    def build_membership_operations(self) -> int:
        return self._build_memberships
