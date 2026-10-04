"""Target-identity-blind adaptive selection of behavioral probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ProbeDecision:
    row_index: int
    remaining_candidates: int
    partition_zero: int
    partition_one: int
    worst_case_remaining: int


def _row_key(row_index: int, row_keys: Sequence | None):
    return row_index if row_keys is None else row_keys[row_index]


class AdaptiveBehavioralProbeSelector:
    """Greedy minimax information probe over deterministic binary hypotheses."""

    def choose(
        self,
        candidates: Sequence,
        compatible_indices: Sequence[int],
        available_rows: Iterable[int],
        *,
        row_keys: Sequence | None = None,
    ) -> ProbeDecision:
        compatible = tuple(compatible_indices)
        available = tuple(sorted(set(int(r) for r in available_rows)))
        if not compatible:
            raise ValueError("compatible_indices must not be empty")
        if not available:
            raise ValueError("available_rows must not be empty")
        best = None
        for row_index in available:
            key = _row_key(row_index, row_keys)
            zero = sum(
                int(candidates[idx].truth_table[key]) == 0
                for idx in compatible
            )
            one = len(compatible) - zero
            decision = ProbeDecision(
                row_index=row_index,
                remaining_candidates=len(compatible),
                partition_zero=zero,
                partition_one=one,
                worst_case_remaining=max(zero, one),
            )
            key_for_order = (
                decision.worst_case_remaining,
                -min(decision.partition_zero, decision.partition_one),
                decision.row_index,
            )
            if best is None or key_for_order < (
                best.worst_case_remaining,
                -min(best.partition_zero, best.partition_one),
                best.row_index,
            ):
                best = decision
        return best

    @staticmethod
    def filter_compatible(
        candidates: Sequence,
        compatible_indices: Sequence[int],
        row_index: int,
        observed_output: int,
        *,
        row_keys: Sequence | None = None,
    ) -> tuple[int, ...]:
        y = int(observed_output)
        if y not in (0, 1):
            raise ValueError("observed_output must be binary")
        key = _row_key(row_index, row_keys)
        return tuple(
            idx
            for idx in compatible_indices
            if int(candidates[idx].truth_table[key]) == y
        )
