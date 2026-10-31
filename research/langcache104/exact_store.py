"""Explicit bounded exact-record storage, independent of neural memory.

Keys and writes must be supplied explicitly from legally observed source data.
This module does not implement learned language-to-key routing, learned admission,
or neural exact recall. Its storage and lookup costs belong in a separate resource
and ablation account, rather than being credited to the compressed carried state.

Capacity counts retained *versions*. Eviction is deterministic FIFO by insertion
sequence; reads never change eviction order. A historical lookup returns the latest
retained version at or before its cutoff, not an archival guarantee: evicted versions
are unavailable. Equal timestamps are resolved by insertion sequence. Times are
nonnegative integer observation ticks, not model-generated semantic timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping


def _integer(name: str, value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _scope_name(name: str, value: Any) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    # Validate that JSON/UTF-8 accounting is defined, including for unusual input.
    value.encode("utf-8")
    return value


def _keys(name: str, obj: Any, expected: set[str]) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping) or set(obj) != expected:
        raise ValueError(f"{name} has invalid fields")
    return obj


@dataclass(frozen=True)
class _Record:
    sequence: int
    observed_at: int
    key: bytes
    value: bytes


class BoundedExactStore:
    """An owner/entity-scoped, byte-exact, versioned FIFO store.

    Full keys are compared directly, without approximate similarity or a hash-only
    identity check. Lookup is O(capacity * max_key_bytes); this reference store
    makes no GPU throughput or constant-time lookup claim. Every operation requires
    the caller's scope and rejects a mismatch before inspecting the records.
    """

    FORMAT = "langcache104-exact-fifo-v1"

    def __init__(
        self,
        owner_id: str,
        entity_id: str,
        capacity: int,
        max_key_bytes: int = 256,
        max_value_bytes: int = 4096,
    ) -> None:
        self.owner_id = _scope_name("owner_id", owner_id)
        self.entity_id = _scope_name("entity_id", entity_id)
        self.capacity = _integer("capacity", capacity, 1)
        self.max_key_bytes = _integer("max_key_bytes", max_key_bytes, 1)
        self.max_value_bytes = _integer("max_value_bytes", max_value_bytes, 1)
        self._records: list[_Record] = []
        self._clock: int | None = None
        self._next_sequence = 0

    def _check_scope(self, owner_id: str, entity_id: str) -> None:
        if (type(owner_id) is not str or type(entity_id) is not str
                or owner_id != self.owner_id or entity_id != self.entity_id):
            raise PermissionError("exact-store owner/entity mismatch")

    @staticmethod
    def _bytes(name: str, value: Any, limit: int) -> bytes:
        if type(value) is not bytes:
            raise TypeError(f"{name} must be bytes")
        if len(value) > limit:
            raise ValueError(f"{name} exceeds its configured byte limit")
        return value

    def put(
        self,
        key: bytes,
        value: bytes,
        *,
        observed_at: int,
        owner_id: str,
        entity_id: str,
    ) -> None:
        """Retain a new version, evicting the oldest insertion if necessary."""
        self._check_scope(owner_id, entity_id)
        key = self._bytes("key", key, self.max_key_bytes)
        value = self._bytes("value", value, self.max_value_bytes)
        tick = _integer("observed_at", observed_at)
        if self._clock is not None and tick < self._clock:
            raise ValueError("observed_at cannot move backwards")
        self._records.append(_Record(self._next_sequence, tick, key, value))
        self._next_sequence += 1
        self._clock = tick
        if len(self._records) > self.capacity:
            del self._records[0]

    def get(
        self,
        key: bytes,
        *,
        as_of: int,
        owner_id: str,
        entity_id: str,
    ) -> bytes | None:
        """Return the latest retained matching value legally observed by cutoff.

        None means unavailable in the retained store, whether never admitted,
        not yet observed at the cutoff, or evicted. It does not establish that a
        fact never existed. Empty bytes are a valid value and differ from None.
        """
        self._check_scope(owner_id, entity_id)
        key = self._bytes("key", key, self.max_key_bytes)
        cutoff = _integer("as_of", as_of)
        for record in reversed(self._records):
            if record.observed_at <= cutoff and record.key == key:
                return record.value
        return None

    def to_dict(self, *, owner_id: str, entity_id: str) -> dict[str, Any]:
        """Return a JSON-safe snapshot; caller must be authorized for its scope."""
        self._check_scope(owner_id, entity_id)
        return {
            "format": self.FORMAT,
            "owner_id": self.owner_id,
            "entity_id": self.entity_id,
            "capacity": self.capacity,
            "max_key_bytes": self.max_key_bytes,
            "max_value_bytes": self.max_value_bytes,
            "clock": self._clock,
            "next_sequence": self._next_sequence,
            "records": [
                {
                    "sequence": r.sequence,
                    "observed_at": r.observed_at,
                    "key_hex": r.key.hex(),
                    "value_hex": r.value.hex(),
                }
                for r in self._records
            ],
        }

    @classmethod
    def from_dict(
        cls,
        snapshot: Mapping[str, Any],
        *,
        owner_id: str,
        entity_id: str,
    ) -> "BoundedExactStore":
        """Validate a snapshot against an independently supplied expected scope.

        Validation detects structural corruption; this is not an authenticated
        serialization or a sandbox against callers who can mutate Python objects.
        """
        data = _keys("snapshot", snapshot, {
            "format", "owner_id", "entity_id", "capacity", "max_key_bytes",
            "max_value_bytes", "clock", "next_sequence", "records",
        })
        if data["format"] != cls.FORMAT:
            raise ValueError("unsupported exact-store format")
        store = cls(data["owner_id"], data["entity_id"], data["capacity"],
                    data["max_key_bytes"], data["max_value_bytes"])
        store._check_scope(owner_id, entity_id)
        next_sequence = _integer("next_sequence", data["next_sequence"])
        raw_records = data["records"]
        if type(raw_records) is not list or len(raw_records) > store.capacity:
            raise ValueError("invalid retained-record count")
        if len(raw_records) != min(next_sequence, store.capacity):
            raise ValueError("retained-record count is inconsistent with FIFO history")
        prior_tick: int | None = None
        first_sequence = next_sequence - len(raw_records)
        for index, raw in enumerate(raw_records):
            record = _keys("record", raw, {
                "sequence", "observed_at", "key_hex", "value_hex",
            })
            sequence = _integer("sequence", record["sequence"])
            if sequence != first_sequence + index:
                raise ValueError("duplicate, reordered, or missing sequence ID")
            tick = _integer("observed_at", record["observed_at"])
            if prior_tick is not None and tick < prior_tick:
                raise ValueError("record timestamps move backwards")
            values: list[bytes] = []
            for name, limit in (("key_hex", store.max_key_bytes),
                                ("value_hex", store.max_value_bytes)):
                encoded = record[name]
                if type(encoded) is not str or len(encoded) > 2 * limit:
                    raise ValueError(f"invalid {name} length")
                try:
                    decoded = bytes.fromhex(encoded)
                except ValueError as exc:
                    raise ValueError(f"invalid {name} encoding") from exc
                if decoded.hex() != encoded:
                    raise ValueError(f"noncanonical {name} encoding")
                values.append(decoded)
            store._records.append(_Record(sequence, tick, *values))
            prior_tick = tick
        if raw_records:
            clock = _integer("clock", data["clock"])
            if clock != prior_tick:
                raise ValueError("clock disagrees with latest retained timestamp")
        elif data["clock"] is not None:
            raise ValueError("an empty, never-written store must have clock None")
        store._clock = prior_tick
        store._next_sequence = next_sequence
        return store

    def accounting(self, *, owner_id: str, entity_id: str) -> dict[str, Any]:
        """Exact payload/serialized sizes, explicitly excluding resident overhead.

        Payload counts each retained key/value version and the UTF-8 namespace
        once. Python containers, object headers, integer metadata, allocators,
        temporary lookup buffers and indexes are excluded from payload bytes.
        Serialized JSON bytes include their JSON metadata and hex expansion;
        neither figure is a measured process/device resident-memory footprint.
        """
        snapshot = self.to_dict(owner_id=owner_id, entity_id=entity_id)
        key_bytes = sum(len(r.key) for r in self._records)
        value_bytes = sum(len(r.value) for r in self._records)
        scope_bytes = len(self.owner_id.encode("utf-8")) + len(self.entity_id.encode("utf-8"))
        return {
            "retained_versions": len(self._records),
            "capacity_versions": self.capacity,
            "key_payload_bytes": key_bytes,
            "value_payload_bytes": value_bytes,
            "scope_payload_bytes": scope_bytes,
            "total_payload_bytes": key_bytes + value_bytes + scope_bytes,
            "maximum_record_payload_bytes": self.capacity * (self.max_key_bytes + self.max_value_bytes),
            "serialized_utf8_bytes": len(json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")),
            "resident_overhead_included": False,
            "resident_memory_measured": False,
            "eviction_policy": "fifo-version-insertion",
        }
