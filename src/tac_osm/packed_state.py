"""Logical packed layout for sparse persistent-state retrieval.

This module records the memory-layout boundary separately from arithmetic
routing. It deliberately does not claim a hardware speedup: it gives the
retriever a stable row-major representation so candidate gathers can be
measured as contiguous rows rather than opaque address lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PackedStateLayout:
    """Immutable row-major state layout."""

    addresses: tuple[str, ...]
    embeddings: tuple[tuple[float, ...], ...]
    address_to_row: tuple[tuple[str, int], ...]

    @classmethod
    def from_items(
        cls,
        items: Sequence[tuple[str, Sequence[float]]],
    ) -> "PackedStateLayout":
        ordered = sorted(
            ((str(address), tuple(float(x) for x in embedding)) for address, embedding in items),
            key=lambda item: item[0],
        )
        if not ordered:
            raise ValueError("packed state requires at least one item")
        dimension = len(ordered[0][1])
        if dimension < 1:
            raise ValueError("packed embeddings must be non-empty")
        if any(len(embedding) != dimension for _, embedding in ordered):
            raise ValueError("packed embeddings must have uniform dimension")
        addresses = tuple(address for address, _ in ordered)
        embeddings = tuple(embedding for _, embedding in ordered)
        address_to_row = tuple((address, row) for row, address in enumerate(addresses))
        return cls(
            addresses=addresses,
            embeddings=embeddings,
            address_to_row=address_to_row,
        )

    @property
    def state_items(self) -> int:
        return len(self.addresses)

    @classmethod
    def from_blocks(
        cls,
        items: Sequence[tuple[str, Sequence[float]]],
        blocks: Sequence[Sequence[str]],
    ) -> "PackedStateLayout":
        """Pack state rows in block order, preserving each block contiguously."""
        item_map = {
            str(address): tuple(float(x) for x in embedding)
            for address, embedding in items
        }
        if not item_map:
            raise ValueError("packed state requires at least one item")
        ordered_addresses: list[str] = []
        seen: set[str] = set()
        for block in blocks:
            for raw_address in block:
                address = str(raw_address)
                if address not in item_map:
                    raise ValueError("block contains unknown state address")
                if address in seen:
                    raise ValueError("state address appears in multiple blocks")
                seen.add(address)
                ordered_addresses.append(address)
        if seen != set(item_map):
            raise ValueError("blocks must cover every state address exactly once")
        ordered = [(address, item_map[address]) for address in ordered_addresses]
        dimension = len(ordered[0][1])
        if dimension < 1 or any(len(embedding) != dimension for _, embedding in ordered):
            raise ValueError("packed embeddings must have uniform dimension")
        addresses = tuple(address for address, _ in ordered)
        embeddings = tuple(embedding for _, embedding in ordered)
        address_to_row = tuple((address, row) for row, address in enumerate(addresses))
        return cls(addresses=addresses, embeddings=embeddings, address_to_row=address_to_row)

    @property
    def embedding_dim(self) -> int:
        return len(self.embeddings[0])

    def row(self, address: str) -> int:
        mapping = dict(self.address_to_row)
        try:
            return mapping[address]
        except KeyError as exc:
            raise KeyError(f"unknown packed state address: {address}") from exc

    def rows_for_addresses(self, addresses: Sequence[str]) -> tuple[int, ...]:
        rows = tuple(self.row(address) for address in addresses)
        return tuple(sorted(rows))

    def embeddings_for_addresses(
        self,
        addresses: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        rows = self.rows_for_addresses(addresses)
        return tuple(self.embeddings[row] for row in rows)

    @staticmethod
    def contiguous_runs(rows: Sequence[int]) -> int:
        ordered = sorted(set(int(row) for row in rows))
        if not ordered:
            return 0
        runs = 1
        for previous, current in zip(ordered, ordered[1:]):
            if current != previous + 1:
                runs += 1
        return runs

    def layout_diagnostics(self, addresses: Sequence[str]) -> dict[str, int | float]:
        rows = self.rows_for_addresses(addresses)
        runs = self.contiguous_runs(rows)
        span = (rows[-1] - rows[0] + 1) if rows else 0
        density = (len(rows) / span) if span else 0.0
        return {
            "candidate_rows": len(rows),
            "contiguous_runs": runs,
            "row_span": span,
            "packing_density": density,
        }
