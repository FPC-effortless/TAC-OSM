"""Learned continuous prototype routing for bounded state retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from . import Query
from .learned_state_index import LearnedSemanticStateIndex
from .temporal import TemporalPersistentState

@dataclass(frozen=True)
class PrototypeStateIndexConfig:
    prototype_count: int = 16
    bucket_capacity: int = 4
    kmeans_iterations: int = 8

    def __post_init__(self) -> None:
        if self.prototype_count < 1:
            raise ValueError("prototype_count must be positive")
        if self.bucket_capacity < 1:
            raise ValueError("bucket_capacity must be positive")
        if self.kmeans_iterations < 1:
            raise ValueError("kmeans_iterations must be positive")

@dataclass(frozen=True)
class PrototypeBuildDiagnostics:
    prototype_count: int
    state_items: int
    bucket_capacity: int
    train_embedding_macs: int
    kmeans_macs: int
    state_embedding_macs: int
    state_assignment_macs: int
    total_build_macs: int
    max_bucket_size: int
    min_bucket_size: int

@dataclass(frozen=True)
class PrototypeLookup:
    address: str | None
    prototype_index: int
    candidate_addresses: tuple[str, ...]
    prototype_score_macs: int
    state_rerank_macs: int
    normalization_ops: int

class LearnedPrototypeStateIndex:
    """Learned continuous coarse routing with hard bucket capacity."""

    def __init__(
        self,
        learned_index: LearnedSemanticStateIndex,
        config: PrototypeStateIndexConfig | None = None,
    ) -> None:
        self.index = learned_index
        self.config = config or PrototypeStateIndexConfig()
        if self.index.config.latent_dim < 1:
            raise ValueError("latent dimension must be positive")
        self._prototypes: tuple[tuple[float, ...], ...] = ()
        self._buckets: dict[int, tuple[str, ...]] = {}
        self._state_embeddings: dict[str, tuple[float, ...]] = {}
        self._build = PrototypeBuildDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self._built = False

    @staticmethod
    def _normalize(values: Sequence[float]) -> tuple[float, ...]:
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 1e-8:
            raise ValueError("cannot normalize a zero embedding")
        return tuple(value / norm for value in values)

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _train_prototypes(
        self,
        train_codes: Sequence[Sequence[int]],
    ) -> tuple[tuple[tuple[float, ...], ...], int, int]:
        embeddings = [self._normalize(self.index.encode_state(code)) for code in train_codes]
        if len(embeddings) < self.config.prototype_count:
            raise ValueError("training-code count must cover all prototypes")
        p = self.config.prototype_count
        selected = [0]
        while len(selected) < p:
            best_idx = max(
                (i for i in range(len(embeddings)) if i not in selected),
                key=lambda i: (
                    -max(self._cosine(embeddings[i], embeddings[j]) for j in selected),
                    -i,
                ),
            )
            selected.append(best_idx)
        centers = [list(embeddings[i]) for i in selected]
        kmeans_macs = 0
        d = self.index.config.latent_dim
        for _ in range(self.config.kmeans_iterations):
            groups = [[] for _ in range(p)]
            for emb in embeddings:
                best = max(range(p), key=lambda j: (self._cosine(emb, centers[j]), -j))
                groups[best].append(emb)
            new_centers = []
            for j in range(p):
                if groups[j]:
                    mean = [
                        sum(emb[r] for emb in groups[j]) / len(groups[j])
                        for r in range(d)
                    ]
                    new_centers.append(list(self._normalize(mean)))
                else:
                    new_centers.append(centers[j])
            centers = new_centers
            kmeans_macs += len(embeddings) * p * d
        train_embedding_macs = len(train_codes) * self.index.state_embedding_macs
        return tuple(tuple(center) for center in centers), train_embedding_macs, kmeans_macs

    def _assign_states(
        self,
        state_embeddings: list[tuple[str, tuple[float, ...]]],
        centers: tuple[tuple[float, ...], ...],
    ) -> tuple[dict[int, tuple[str, ...]], int]:
        p = len(centers)
        capacity = self.config.bucket_capacity
        d = self.index.config.latent_dim
        pair_scores = []
        for state_idx, (address, emb) in enumerate(state_embeddings):
            for prototype_idx, center in enumerate(centers):
                pair_scores.append(
                    (
                        self._cosine(emb, center),
                        state_idx,
                        prototype_idx,
                        address,
                    )
                )
        pair_scores.sort(key=lambda item: (-item[0], item[2], item[3]))

        assigned: dict[int, int] = {}
        counts = [0] * p

        for _, state_idx, prototype_idx, _ in pair_scores:
            if state_idx in assigned or counts[prototype_idx] >= capacity:
                continue
            assigned[state_idx] = prototype_idx
            counts[prototype_idx] += 1

        if len(assigned) < len(state_embeddings):
            for state_idx, (_, emb) in enumerate(state_embeddings):
                if state_idx in assigned:
                    continue
                available = [j for j in range(p) if counts[j] < capacity]
                if not available:
                    raise AssertionError("capacity assignment exhausted unexpectedly")
                prototype_idx = max(
                    available,
                    key=lambda j: (self._cosine(emb, centers[j]), -j),
                )
                assigned[state_idx] = prototype_idx
                counts[prototype_idx] += 1

        buckets = {j: [] for j in range(p)}
        for state_idx, (address, _) in enumerate(state_embeddings):
            prototype_idx = assigned[state_idx]
            buckets[prototype_idx].append(address)

        buckets = {
            j: tuple(sorted(addresses))
            for j, addresses in buckets.items()
        }
        if sum(len(addresses) for addresses in buckets.values()) != len(state_embeddings):
            raise AssertionError("state bucket assignment lost items")
        if any(len(addresses) > capacity for addresses in buckets.values()):
            raise AssertionError("state bucket capacity exceeded")
        return buckets, len(state_embeddings) * p * d

    def build(
        self,
        state: TemporalPersistentState,
        train_codes: Sequence[Sequence[int]],
    ) -> PrototypeBuildDiagnostics:
        centers, train_macs, kmeans_macs = self._train_prototypes(train_codes)
        state_embeddings: list[tuple[str, tuple[float, ...]]] = []
        for address in state.addresses():
            read = state.read(Query(
                text="\t" + address,
                step=state.current_step,
                provenance="learned_prototype_state_build",
            ))
            if read.keys:
                state_embeddings.append((address, self._normalize(self.index.encode_state(read.values[0]))))
        if len(state_embeddings) != len(state.addresses()):
            raise AssertionError("all state items must be readable")
        buckets, assignment_macs = self._assign_states(state_embeddings, centers)
        self._prototypes = centers
        self._state_embeddings = dict(state_embeddings)
        self._buckets = buckets
        capacity_values = [len(self._buckets[j]) for j in range(len(centers))]
        state_macs = len(state_embeddings) * self.index.state_embedding_macs
        self._build = PrototypeBuildDiagnostics(
            prototype_count=len(centers),
            state_items=len(state_embeddings),
            bucket_capacity=self.config.bucket_capacity,
            train_embedding_macs=train_macs,
            kmeans_macs=kmeans_macs,
            state_embedding_macs=state_macs,
            state_assignment_macs=assignment_macs,
            total_build_macs=train_macs + kmeans_macs + state_macs + assignment_macs,
            max_bucket_size=max(capacity_values),
            min_bucket_size=min(capacity_values),
        )
        self._built = True
        return self._build

    @property
    def build_diagnostics(self) -> PrototypeBuildDiagnostics:
        return self._build

    def lookup(self, query: Query) -> PrototypeLookup:
        if not self._built:
            raise RuntimeError("prototype state index has not been built")
        q = self._normalize(self.index.encode_query(query))
        prototype_scores = [self._cosine(q, center) for center in self._prototypes]
        selected = max(range(len(prototype_scores)), key=lambda j: (prototype_scores[j], -j))
        candidates = self._buckets[selected]
        scored = [
            (self._cosine(q, self._state_embeddings[address]), address)
            for address in candidates
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        address = scored[0][1] if scored else None
        return PrototypeLookup(
            address=address,
            prototype_index=selected,
            candidate_addresses=tuple(candidates),
            prototype_score_macs=len(self._prototypes) * self.index.config.latent_dim,
            state_rerank_macs=len(candidates) * self.index.config.latent_dim,
            normalization_ops=2,
        )

    @property
    def query_projection_macs(self) -> int:
        return self.index.query_embedding_macs

    @property
    def max_shortlist(self) -> int:
        return self.config.bucket_capacity