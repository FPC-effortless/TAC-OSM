"""Factorized product-key addressing for persistent-state retrieval.

The score is decomposed over two key factors. Each state embedding is split
into two subvectors; each factor is quantized to a small learned codebook.
Query-time routing searches the factor codebooks and forms a Cartesian beam of
candidate product cells. A final continuous reranker remains responsible for
state selection.

This is an experimental addressing layer. It does not claim universal
sublinear retrieval or hardware acceleration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ProductKeyConfig:
    factor_size: int = 8
    factor_beam: int = 2
    iterations: int = 8
    max_shortlist: int = 8

    def __post_init__(self) -> None:
        if self.factor_size < 1:
            raise ValueError("factor_size must be positive")
        if self.factor_beam < 1 or self.factor_beam > self.factor_size:
            raise ValueError("factor_beam must be within factor_size")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.max_shortlist < 1:
            raise ValueError("max_shortlist must be positive")


@dataclass(frozen=True)
class ProductKeyBuildDiagnostics:
    state_items: int
    codebook_training_items: int
    embedding_dim: int
    factor_size: int
    factor_dim: int
    build_similarity_macs: int
    nonempty_cells: int
    max_cell_size: int


@dataclass(frozen=True)
class ProductKeyLookup:
    candidate_addresses: tuple[str, ...]
    selected_address: str | None
    selected_cells: tuple[tuple[int, int], ...]
    factor_score_macs: int
    pair_generation_ops: int
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
        next_index = max(
            (i for i in range(len(points)) if i not in selected),
            key=lambda i: (
                -max(_dot(points[i], points[j]) for j in selected),
                -i,
            ),
        )
        selected.append(next_index)
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


class ProductKeyStateIndex:
    """Two-factor product-key index over packed state embeddings."""

    def __init__(
        self,
        config: ProductKeyConfig | None = None,
    ) -> None:
        self.config = config or ProductKeyConfig()
        self._factor1: tuple[tuple[float, ...], ...] = ()
        self._factor2: tuple[tuple[float, ...], ...] = ()
        self._cells: dict[tuple[int, int], tuple[str, ...]] = {}
        self._embeddings: dict[str, tuple[float, ...]] = {}
        self._built = False
        self._build = ProductKeyBuildDiagnostics(0, 0, 0, 0, 0, 0, 0, 0)

    def build(
        self,
        items: Sequence[tuple[str, Sequence[float]]],
        *,
        codebook_items: Sequence[tuple[str, Sequence[float]]] | None = None,
    ) -> ProductKeyBuildDiagnostics:
        """Build all runtime cells while optionally fitting codebooks on training-only items."""
        if not items:
            raise ValueError("product-key state index requires items")
        embeddings = [
            (str(address), _normalize(embedding))
            for address, embedding in items
        ]
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
        if dim < 2 or dim % 2:
            raise ValueError("product-key embeddings require an even dimension >= 2")
        if any(len(embedding) != dim for _, embedding in embeddings):
            raise ValueError("all product-key embeddings must have equal dimension")
        half = dim // 2
        first_points = tuple(embedding[:half] for _, embedding in training_embeddings)
        second_points = tuple(embedding[half:] for _, embedding in training_embeddings)
        factor1 = _kmeans_cosine(
            first_points,
            min(self.config.factor_size, len(first_points)),
            self.config.iterations,
        )
        factor2 = _kmeans_cosine(
            second_points,
            min(self.config.factor_size, len(second_points)),
            self.config.iterations,
        )
        if len(factor1) != self.config.factor_size or len(factor2) != self.config.factor_size:
            raise ValueError("state population must cover the requested factor codebooks")

        cells: dict[tuple[int, int], list[str]] = {
            (i, j): [] for i in range(self.config.factor_size) for j in range(self.config.factor_size)
        }
        for address, embedding in embeddings:
            part1 = embedding[:half]
            part2 = embedding[half:]
            i = max(range(self.config.factor_size), key=lambda c: (_dot(part1, factor1[c]), -c))
            j = max(range(self.config.factor_size), key=lambda c: (_dot(part2, factor2[c]), -c))
            cells[(i, j)].append(address)
        self._factor1 = factor1
        self._factor2 = factor2
        self._cells = {
            cell: tuple(sorted(addresses))
            for cell, addresses in cells.items()
            if addresses
        }
        self._embeddings = dict(embeddings)
        self._built = True
        nonempty = len(self._cells)
        max_cell = max((len(addresses) for addresses in self._cells.values()), default=0)
        factor_dim = half
        build_macs = (
            len(training_embeddings) * self.config.factor_size * dim
        )
        self._build = ProductKeyBuildDiagnostics(
            state_items=len(embeddings),
            codebook_training_items=len(training_embeddings),
            embedding_dim=dim,
            factor_size=self.config.factor_size,
            factor_dim=factor_dim,
            build_similarity_macs=build_macs,
            nonempty_cells=nonempty,
            max_cell_size=max_cell,
        )
        return self._build

    @property
    def build_diagnostics(self) -> ProductKeyBuildDiagnostics:
        return self._build

    def lookup(
        self,
        query_embedding: Sequence[float],
        *,
        beam: int | None = None,
    ) -> ProductKeyLookup:
        if not self._built:
            raise RuntimeError("product-key index has not been built")
        q = _normalize(query_embedding)
        dim = self._build.embedding_dim
        if len(q) != dim:
            raise ValueError("query embedding width mismatch")
        half = dim // 2
        q1, q2 = q[:half], q[half:]
        width = self.config.factor_beam if beam is None else int(beam)
        if width < 1 or width > self.config.factor_size:
            raise ValueError("beam must be within factor_size")

        factor1_scores = [_dot(q1, center) for center in self._factor1]
        factor2_scores = [_dot(q2, center) for center in self._factor2]
        top1 = sorted(
            range(self.config.factor_size),
            key=lambda i: (-factor1_scores[i], i),
        )[:width]
        top2 = sorted(
            range(self.config.factor_size),
            key=lambda i: (-factor2_scores[i], i),
        )[:width]

        cells = tuple((i, j) for i in top1 for j in top2)
        addresses = sorted({
            address
            for cell in cells
            for address in self._cells.get(cell, ())
        })
        # Over-fetch is deliberately bounded before the fine continuous reranker.
        if len(addresses) > self.config.max_shortlist:
            scored = [
                (_dot(q, self._embeddings[address]), address)
                for address in addresses
            ]
            scored.sort(key=lambda item: (-item[0], item[1]))
            addresses = [address for _, address in scored[: self.config.max_shortlist]]

        reranked = [
            (_dot(q, self._embeddings[address]), address)
            for address in addresses
        ]
        reranked.sort(key=lambda item: (-item[0], item[1]))
        selected = reranked[0][1] if reranked else None
        return ProductKeyLookup(
            candidate_addresses=tuple(addresses),
            selected_address=selected,
            selected_cells=cells,
            factor_score_macs=self.config.factor_size * dim,
            pair_generation_ops=len(cells),
            state_rerank_macs=len(addresses) * dim,
            factor_beam=width,
        )
