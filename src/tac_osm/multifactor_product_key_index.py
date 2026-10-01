"""N-factor product-key addressing for persistent-state retrieval."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MultiFactorProductKeyConfig:
    factor_count: int = 3
    factor_size: int = 8
    factor_beam: int = 3
    iterations: int = 8
    max_shortlist: int = 32
    max_rerank_candidates: int = 32

    def __post_init__(self) -> None:
        if self.factor_count < 2:
            raise ValueError("factor_count must be at least 2")
        if self.factor_size < 1:
            raise ValueError("factor_size must be positive")
        if self.factor_beam < 1 or self.factor_beam > self.factor_size:
            raise ValueError("factor_beam must be within factor_size")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.max_shortlist < 1:
            raise ValueError("max_shortlist must be positive")
        if self.max_rerank_candidates < 1:
            raise ValueError("max_rerank_candidates must be positive")


@dataclass(frozen=True)
class MultiFactorBuildDiagnostics:
    state_items: int
    codebook_training_items: int
    embedding_dim: int
    factor_count: int
    factor_size: int
    factor_dims: tuple[int, ...]
    total_build_macs: int
    nonempty_cells: int
    max_cell_size: int


@dataclass(frozen=True)
class MultiFactorLookup:
    candidate_addresses: tuple[str, ...]
    selected_address: str | None
    selected_cells: tuple[tuple[int, ...], ...]
    factor_score_macs: int
    pair_generation_ops: int
    state_candidates_scored: int
    state_rerank_macs: int
    factor_beam: int


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(x * x for x in values))
    if norm <= 1e-8:
        raise ValueError("cannot normalize a zero vector")
    return tuple(x / norm for x in values)


def _factor_normalize(values: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(x * x for x in values))
    if norm <= 1e-8:
        return tuple(0.0 for _ in values)
    return tuple(x / norm for x in values)


def _farthest_init(points: Sequence[tuple[float, ...]], k: int) -> list[tuple[float, ...]]:
    if len(points) < k:
        raise ValueError("not enough points for factor codebook")
    selected = [0]
    while len(selected) < k:
        idx = max(
            (i for i in range(len(points)) if i not in selected),
            key=lambda i: (-max(_dot(points[i], points[j]) for j in selected), -i),
        )
        selected.append(idx)
    return [points[i] for i in selected]


def _kmeans_cosine(
    points: Sequence[tuple[float, ...]],
    k: int,
    iterations: int,
) -> tuple[tuple[float, ...], ...]:
    centers = [list(_factor_normalize(point)) for point in _farthest_init(points, k)]
    for _ in range(iterations):
        groups: list[list[tuple[float, ...]]] = [[] for _ in range(k)]
        for point in points:
            best = max(range(k), key=lambda i: (_dot(point, centers[i]), -i))
            groups[best].append(point)
        next_centers: list[list[float]] = []
        for i, group in enumerate(groups):
            if not group:
                next_centers.append(centers[i])
                continue
            mean = [
                sum(point[d] for point in group) / len(group)
                for d in range(len(group[0]))
            ]
            next_centers.append(list(_factor_normalize(mean)))
        centers = next_centers
    return tuple(tuple(center) for center in centers)


def _factor_slices(dim: int, count: int) -> tuple[tuple[int, int], ...]:
    base, remainder = divmod(dim, count)
    if base < 1:
        raise ValueError("embedding dimension is too small for factor count")
    result = []
    start = 0
    for i in range(count):
        width = base + int(i < remainder)
        result.append((start, start + width))
        start += width
    return tuple(result)


class MultiFactorProductKeyStateIndex:
    """N-factor product-key index over normalized embeddings."""

    def __init__(self, config: MultiFactorProductKeyConfig | None = None) -> None:
        self.config = config or MultiFactorProductKeyConfig()
        self._factors: tuple[tuple[tuple[float, ...], ...], ...] = ()
        self._cells: dict[tuple[int, ...], tuple[str, ...]] = {}
        self._embeddings: dict[str, tuple[float, ...]] = {}
        self._address_cells: dict[str, tuple[int, ...]] = {}
        self._built = False
        self._slices: tuple[tuple[int, int], ...] = ()
        self._build = MultiFactorBuildDiagnostics(0, 0, 0, 0, 0, (), 0, 0, 0)

    def build(
        self,
        items: Sequence[tuple[str, Sequence[float]]],
        *,
        codebook_items: Sequence[tuple[str, Sequence[float]]] | None = None,
    ) -> MultiFactorBuildDiagnostics:
        if not items:
            raise ValueError("product-key state index requires items")
        embeddings = [(str(address), _normalize(embedding)) for address, embedding in items]
        training_raw = items if codebook_items is None else codebook_items
        if not training_raw:
            raise ValueError("codebook_items must not be empty")

        training_addresses = {str(address) for address, _ in training_raw}
        item_addresses = {address for address, _ in embeddings}
        if not training_addresses.issubset(item_addresses):
            raise ValueError("codebook_items must be a subset of items")

        training_map = {str(address): embedding for address, embedding in training_raw}
        training_embeddings = [
            (address, _normalize(training_map[address]))
            for address in sorted(training_addresses)
        ]

        dim = len(embeddings[0][1])
        if any(len(embedding) != dim for _, embedding in embeddings):
            raise ValueError("all product-key embeddings must have equal dimension")
        self._slices = _factor_slices(dim, self.config.factor_count)

        factors = []
        for start, end in self._slices:
            points = tuple(embedding[start:end] for _, embedding in training_embeddings)
            centers = _kmeans_cosine(
                points,
                min(self.config.factor_size, len(points)),
                self.config.iterations,
            )
            if len(centers) != self.config.factor_size:
                raise ValueError("training population cannot support requested codebook size")
            factors.append(centers)

        cells: dict[tuple[int, ...], list[str]] = {}
        for cell in itertools.product(
            range(self.config.factor_size),
            repeat=self.config.factor_count,
        ):
            cells[cell] = []

        for address, embedding in embeddings:
            indices = []
            for (start, end), centers in zip(self._slices, factors):
                part = embedding[start:end]
                indices.append(
                    max(range(self.config.factor_size), key=lambda c: (_dot(part, centers[c]), -c))
                )
            cells[tuple(indices)].append(address)

        self._factors = tuple(factors)
        self._cells = {
            cell: tuple(sorted(addresses))
            for cell, addresses in cells.items()
            if addresses
        }
        self._embeddings = dict(embeddings)
        self._address_cells = {
            address: cell
            for cell, addresses in self._cells.items()
            for address in addresses
        }
        self._built = True

        nonempty = len(self._cells)
        max_cell = max((len(v) for v in self._cells.values()), default=0)
        factor_dims = tuple(end - start for start, end in self._slices)
        codebook_macs = (
            len(training_embeddings)
            * self.config.factor_size
            * dim
            * self.config.iterations
        )
        assignment_macs = len(embeddings) * self.config.factor_size * dim
        self._build = MultiFactorBuildDiagnostics(
            state_items=len(embeddings),
            codebook_training_items=len(training_embeddings),
            embedding_dim=dim,
            factor_count=self.config.factor_count,
            factor_size=self.config.factor_size,
            factor_dims=factor_dims,
            total_build_macs=codebook_macs + assignment_macs,
            nonempty_cells=nonempty,
            max_cell_size=max_cell,
        )
        return self._build

    @property
    def build_diagnostics(self) -> MultiFactorBuildDiagnostics:
        return self._build

    def lookup(
        self,
        query_embedding: Sequence[float],
        *,
        beam: int | None = None,
        max_shortlist: int | None = None,
    ) -> MultiFactorLookup:
        if not self._built:
            raise RuntimeError("product-key state index has not been built")
        q = _normalize(query_embedding)
        if len(q) != self._build.embedding_dim:
            raise ValueError("query embedding width mismatch")

        width = self.config.factor_beam if beam is None else int(beam)
        limit = self.config.max_shortlist if max_shortlist is None else int(max_shortlist)
        if width < 1 or width > self.config.factor_size:
            raise ValueError("beam must be within factor_size")
        if limit < 1:
            raise ValueError("max_shortlist must be positive")

        top_factors = []
        factor_score_macs = 0
        for (start, end), centers in zip(self._slices, self._factors):
            scores = [_dot(q[start:end], center) for center in centers]
            top = sorted(range(self.config.factor_size), key=lambda i: (-scores[i], i))[:width]
            top_factors.append(top)
            factor_score_macs += self.config.factor_size * (end - start)

        cells = tuple(itertools.product(*top_factors))
        addresses = sorted({
            address
            for cell in cells
            for address in self._cells.get(cell, ())
        })

        # Optional cheap middle filter: rank candidate addresses by their
        # factor-center score and exact-rerank only a fixed number. This changes
        # only the rerank boundary; the product-key admission cells are fixed.
        rerank_cap = self.config.max_rerank_candidates
        coarse_scores = []
        factor_lookup = [
            {idx: scores[idx] for idx in range(self.config.factor_size)}
            for scores in [
                [_dot(q[start:end], center) for center in centers]
                for (start, end), centers in zip(self._slices, self._factors)
            ]
        ]
        for address in addresses:
            cell = self._address_cells[address]
            coarse_scores.append((
                sum(factor_lookup[f][cell[f]] for f in range(self.config.factor_count)),
                address,
            ))
        coarse_scores.sort(key=lambda item: (-item[0], item[1]))
        rerank_addresses = [address for _, address in coarse_scores[:min(rerank_cap, len(coarse_scores))]]

        scored = [
            (_dot(q, self._embeddings[address]), address)
            for address in rerank_addresses
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        retained = scored[:limit]
        selected = retained[0][1] if retained else None

        return MultiFactorLookup(
            candidate_addresses=tuple(address for _, address in retained),
            selected_address=selected,
            selected_cells=tuple(cells),
            factor_score_macs=factor_score_macs,
            pair_generation_ops=len(cells),
            state_candidates_scored=len(rerank_addresses),
            state_rerank_macs=len(rerank_addresses) * self._build.embedding_dim,
            factor_beam=width,
        )
