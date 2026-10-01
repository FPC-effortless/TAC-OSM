"""Deterministic exact semantic-state index ceiling.

This is an upper-bound control for query-time addressing cost. It does not
learn and it does not claim general semantic retrieval. It measures the cost
of building an exact inverted index over persistent semantic values, then the
cost of repeated query-time lookups against a stable state pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import Query
from .semantic_state_addressor import SemanticStateAddressor
from .temporal import TemporalPersistentState


@dataclass(frozen=True)
class StateIndexBuildDiagnostics:
    state_items: int
    build_reads: int
    build_entries: int


@dataclass(frozen=True)
class StateIndexLookup:
    found: bool
    address: str | None
    bucket_size: int
    query_operations: int


class ExactSemanticStateIndex:
    """Exact signature -> address-set inverted index."""

    def __init__(self) -> None:
        self._index: dict[tuple[int, ...], tuple[str, ...]] = {}
        self._built = False
        self._diagnostics = StateIndexBuildDiagnostics(0, 0, 0)

    @property
    def diagnostics(self) -> StateIndexBuildDiagnostics:
        return self._diagnostics

    @staticmethod
    def _query_signature(query: Query) -> tuple[int, ...]:
        raw = query.text.partition("\t")[0].strip()
        if not raw:
            raise ValueError("exact semantic index requires a non-empty query signature")
        return tuple(int(x) for x in raw.split())

    def build(
        self,
        state: TemporalPersistentState,
    ) -> StateIndexBuildDiagnostics:
        index: dict[tuple[int, ...], list[str]] = {}
        addresses = state.addresses()
        for address in addresses:
            read = state.read(
                Query(
                    text="\t" + address,
                    step=state.current_step,
                    provenance="exact_semantic_index_build",
                )
            )
            if not read.keys:
                continue
            value = tuple(read.values[0])
            index.setdefault(value, []).append(address)
        self._index = {k: tuple(v) for k, v in index.items()}
        self._built = True
        self._diagnostics = StateIndexBuildDiagnostics(
            state_items=len(addresses),
            build_reads=len(addresses),
            build_entries=len(self._index),
        )
        return self._diagnostics

    def lookup(self, query: Query) -> StateIndexLookup:
        if not self._built:
            raise RuntimeError("exact semantic index has not been built")
        signature = self._query_signature(query)
        bucket = self._index.get(signature, ())
        return StateIndexLookup(
            found=bool(bucket),
            address=bucket[0] if bucket else None,
            bucket_size=len(bucket),
            query_operations=1,
        )


def build_from_state(
    state: TemporalPersistentState,
) -> tuple[ExactSemanticStateIndex, StateIndexBuildDiagnostics]:
    index = ExactSemanticStateIndex()
    return index, index.build(state)
