"""Bitemporal immutable input reconstruction; Woyengi remains authoritative.

This is an engineering replay/checkpoint input adapter, not a trained time-travel
neural model, database ledger, legal compliance certification or source-of-truth.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from .contracts import ObservationStamp, utc

@dataclass(frozen=True)
class TemporalRevision:
    stamp: ObservationStamp
    fact_id: str
    revision_id: str
    payload: Any
    def __post_init__(self):
        if not self.fact_id or not self.revision_id:
            raise ValueError("stable fact and revision identity required")

@dataclass(frozen=True)
class ReconstructedFacts:
    entity_id: str
    valid_at: datetime
    known_at: datetime
    revisions: tuple[TemporalRevision, ...]
    def __post_init__(self):
        object.__setattr__(self, "valid_at", utc(self.valid_at))
        object.__setattr__(self, "known_at", utc(self.known_at))

class TemporalReconstructor:
    def __init__(self):
        self._records: list[TemporalRevision] = []
        self._identities: set[tuple[str, str, str]] = set()

    def append(self, revision: TemporalRevision) -> None:
        if not isinstance(revision, TemporalRevision):
            raise ValueError("revision required")
        key = (revision.stamp.entity_id, revision.fact_id, revision.revision_id)
        if key in self._identities:
            raise ValueError("revision ID cannot be silently overwritten")
        self._identities.add(key)
        self._records.append(revision)

    def as_of(self, entity_id: str, *, valid_at: datetime, known_at: datetime) -> ReconstructedFacts:
        valid, known = utc(valid_at), utc(known_at)
        if not entity_id:
            raise ValueError("entity required")
        latest: dict[str, TemporalRevision] = {}
        for row in self._records:
            if row.stamp.entity_id != entity_id or row.stamp.recorded_at > known:
                continue
            if row.stamp.valid_at > valid:
                continue
            prior = latest.get(row.fact_id)
            if prior is None or (row.stamp.recorded_at, row.revision_id) > (
                    prior.stamp.recorded_at, prior.revision_id):
                latest[row.fact_id] = row
        ordered = tuple(sorted(latest.values(),
                               key=lambda r: (r.stamp.valid_at, r.stamp.recorded_at, r.fact_id, r.revision_id)))
        return ReconstructedFacts(entity_id, valid, known, ordered)

    def replay(self, entity_id: str, *, valid_at: datetime, known_at: datetime,
               initial: Any, reduce: Callable[[Any, TemporalRevision], Any]) -> Any:
        """Reconstruct from admitted known revisions; no rollback authority implied."""
        if not callable(reduce):
            raise ValueError("pure reducer required")
        state = initial
        for row in self.as_of(entity_id, valid_at=valid_at, known_at=known_at).revisions:
            state = reduce(state, row)
        return state
