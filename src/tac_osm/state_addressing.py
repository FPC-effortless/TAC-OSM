"""Successor state-addressing boundary for TAC-OSM.

This module makes state lookup a first-class operation:

    AddressState(Q_t, S_t) -> M_t

Only the addressed memory item is returned to the scorer.  The module does
not claim sublinear lookup by itself: the v0 PersistentStore interface
materialises a StateRead, so the measured query cost must be reported
separately from the semantic boundary.  A future indexed store can implement
the same AddressedMemory contract without changing the router.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import PersistentState, Query, StateRead


def query_address(query: Query) -> str:
    """Return the public state address carried by the query, if any."""
    return query.text.partition("\t")[2]


@dataclass(frozen=True)
class AddressedMemory:
    """The result of one state-addressing operation."""

    address: str
    found: bool
    key: str
    value: tuple[int, ...]
    inspected_slots: int
    pool_size: int

    @property
    def present(self) -> float:
        return 1.0 if self.found else 0.0


class StateAddressor:
    """Address persistent state without exposing unrelated state to routing."""

    def address(self, query: Query, state: PersistentState) -> AddressedMemory:
        addr = query_address(query)
        read: StateRead = state.read(query)
        if not addr:
            return AddressedMemory(
                address="",
                found=False,
                key="",
                value=(),
                inspected_slots=0,
                pool_size=len(read.keys),
            )

        for i, (key, value) in enumerate(zip(read.keys, read.values), start=1):
            if key == addr:
                return AddressedMemory(
                    address=addr,
                    found=True,
                    key=key,
                    value=tuple(int(x) for x in value),
                    inspected_slots=i,
                    pool_size=len(read.keys),
                )

        return AddressedMemory(
            address=addr,
            found=False,
            key="",
            value=(),
            inspected_slots=len(read.keys),
            pool_size=len(read.keys),
        )


class IndexedStateAddressor(StateAddressor):
    """Marker implementation for stores exposing an indexed lookup.

    It intentionally accepts the same protocol as StateAddressor.  The current
    PersistentStore does not expose a direct indexed-read API, so this class
    does not pretend to change asymptotic cost.  It exists to freeze the
    boundary that a genuinely indexed implementation can later satisfy.
    """

    pass


__all__ = ["AddressedMemory", "StateAddressor", "IndexedStateAddressor", "query_address"]
