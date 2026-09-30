"""Approximate persistent-state index for one-bit-noisy semantic codes.

The index is intentionally deterministic and non-learned. It maps stored 10-bit
binary codes to opaque addresses, then probes the Hamming-radius-2 neighborhood
of a noisy query. A fixed shortlist cap K is applied before the learned state
addressor sees the retained items.

This is a selective-addressing mechanism experiment, not a claim of general
semantic memory or sublinear end-to-end computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from . import AddressedMemory, Query
from .temporal import TemporalPersistentState


@dataclass(frozen=True)
class HammingIndexLookup:
    addresses: tuple[str, ...]
    query_operations: int
    shortlist_size: int
    target_retrieved: bool = False


@dataclass(frozen=True)
class HammingIndexBuildDiagnostics:
    state_items: int
    build_reads: int
    unique_codes: int


class HammingStateIndex:
    """Fixed-width Hamming-radius index with a deterministic shortlist cap."""

    def __init__(self, *, bits: int = 10, radius: int = 2, shortlist_k: int = 4) -> None:
        if bits < 1 or bits > 20:
            raise ValueError("bits must be in [1, 20]")
        if radius < 0 or radius > bits:
            raise ValueError("radius must be in [0, bits]")
        if shortlist_k < 1:
            raise ValueError("shortlist_k must be >= 1")
        self.bits = bits
        self.radius = radius
        self.shortlist_k = shortlist_k
        self._index: dict[int, tuple[str, ...]] = {}
        self._built = False
        self._build = HammingIndexBuildDiagnostics(0, 0, 0)

    @property
    def build_diagnostics(self) -> HammingIndexBuildDiagnostics:
        return self._build

    @property
    def probe_count(self) -> int:
        return sum(__import__("math").comb(self.bits, d) for d in range(self.radius + 1))

    @staticmethod
    def bits_to_int(bits: Sequence[int]) -> int:
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return value

    @staticmethod
    def int_to_bits(value: int, bits: int) -> tuple[int, ...]:
        return tuple((value >> (bits - 1 - i)) & 1 for i in range(bits))

    def _parse_query(self, query: Query) -> tuple[int, ...]:
        raw = query.text.partition("\t")[0].strip()
        values = tuple(int(x) for x in raw.split()) if raw else ()
        if len(values) != self.bits:
            raise ValueError(f"query width {len(values)} != {self.bits}")
        if any(x not in (0, 1) for x in values):
            raise ValueError("query code must be binary")
        return values

    def build(self, state: TemporalPersistentState) -> HammingIndexBuildDiagnostics:
        index: dict[int, list[str]] = {}
        addresses = state.addresses()
        for address in addresses:
            read = state.read(Query(text="\t" + address, step=state.current_step))
            if not read.keys:
                continue
            value = tuple(int(x) for x in read.values[0])
            if len(value) != self.bits:
                raise ValueError(f"state code width {len(value)} != {self.bits}")
            code = self.bits_to_int(value)
            index.setdefault(code, []).append(address)
        self._index = {k: tuple(sorted(v)) for k, v in index.items()}
        self._build = HammingIndexBuildDiagnostics(
            state_items=len(addresses),
            build_reads=len(addresses),
            unique_codes=len(self._index),
        )
        self._built = True
        return self._build

    def _neighbors(self, code: int) -> tuple[tuple[int, int], ...]:
        """Return (distance, code) for every code within the registered radius."""
        out: list[tuple[int, int]] = [(0, code)]
        positions = range(self.bits)
        for distance in range(1, self.radius + 1):
            for flips in combinations(positions, distance):
                candidate = code
                for pos in flips:
                    candidate ^= 1 << (self.bits - 1 - pos)
                out.append((distance, candidate))
        return tuple(out)

    def lookup(self, query: Query) -> HammingIndexLookup:
        if not self._built:
            raise RuntimeError("Hamming state index has not been built")
        qbits = self._parse_query(query)
        qcode = self.bits_to_int(qbits)
        candidates: list[tuple[int, str]] = []
        for distance, code in self._neighbors(qcode):
            for address in self._index.get(code, ()):
                candidates.append((distance, address))
        candidates.sort(key=lambda x: (x[0], x[1]))
        addresses = tuple(address for _, address in candidates[: self.shortlist_k])
        return HammingIndexLookup(
            addresses=addresses,
            query_operations=self.probe_count,
            shortlist_size=len(addresses),
        )


def pool_for_addresses(
    state: TemporalPersistentState,
    addresses: Sequence[str],
) -> tuple[AddressedMemory, ...]:
    """Read only the indexed shortlist from persistent state."""
    out: list[AddressedMemory] = []
    for address in addresses:
        read = state.read(Query(text="\t" + address, step=state.current_step))
        if not read.keys:
            continue
        out.append(
            AddressedMemory(
                address=address,
                found=True,
                key=address,
                value=tuple(read.values[0]),
                inspected_slots=1,
                pool_size=len(addresses),
            )
        )
    return tuple(out)