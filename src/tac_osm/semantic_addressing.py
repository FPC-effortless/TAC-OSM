"""Representation-based semantic addressing with bounded LSH lookup.

This is an R1 runtime control, not a learned-capability result. The address
encoder is injected so it can later be supplied by CDL/PLM training without
changing the runtime boundary.

Build:
    candidate -> representation -> LSH bucket

Query:
    query -> representation -> bounded bucket probes -> retained candidates

The implementation deliberately reports query bucket probes separately from
candidate scoring. It must not receive hidden acceptable actions or world
truth.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from . import Candidate, Query
from .addressing import AddressHit

QueryEncoder = Callable[[Query, Any], Sequence[float]]
CandidateEncoder = Callable[[Candidate, Any], Sequence[float]]


@dataclass(frozen=True)
class SemanticAddressConfig:
    dimensions: int
    n_planes: int = 8
    seed: int = 0
    hamming_radius: int = 0

    def __post_init__(self) -> None:
        if self.dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        if self.n_planes < 1:
            raise ValueError("n_planes must be >= 1")
        if self.hamming_radius < 0 or self.hamming_radius > 1:
            raise ValueError("hamming_radius must be 0 or 1")


@dataclass(frozen=True)
class SemanticAddressHit(AddressHit):
    bucket_probes: int = 0
    representation_dimension: int = 0


@dataclass
class RepresentationAddressIndex:
    """LSH address index over injected query/candidate representations."""

    query_encoder: QueryEncoder
    candidate_encoder: CandidateEncoder
    config: SemanticAddressConfig
    _planes: tuple[tuple[float, ...], ...] = field(default_factory=tuple)
    _buckets: dict[int, tuple[int, ...]] = field(default_factory=dict)
    built_candidates: int = 0

    def __post_init__(self) -> None:
        if self._planes:
            return
        rng = random.Random(self.config.seed)
        planes = []
        for _ in range(self.config.n_planes):
            planes.append(
                tuple(
                    rng.gauss(0.0, 1.0)
                    for _ in range(self.config.dimensions)
                )
            )
        object.__setattr__(self, "_planes", tuple(planes))

    @classmethod
    def create(
        cls,
        query_encoder: QueryEncoder,
        candidate_encoder: CandidateEncoder,
        *,
        config: SemanticAddressConfig,
    ) -> "RepresentationAddressIndex":
        return cls(
            query_encoder=query_encoder,
            candidate_encoder=candidate_encoder,
            config=config,
        )

    def rebuild(
        self,
        candidates: Sequence[Candidate],
        *,
        context: Sequence[int],
        state: object | None = None,
    ) -> "RepresentationAddressIndex":
        # Context is intentionally not used as hidden truth. Encoders receive
        # only the public query/state or candidate/state interfaces.
        buckets: dict[int, list[int]] = {}
        del context
        for i, candidate in enumerate(candidates):
            vector = tuple(float(x) for x in self.candidate_encoder(candidate, state))
            self._validate_dimension(vector)
            signature = self._signature(vector)
            buckets.setdefault(signature, []).append(i)
        return RepresentationAddressIndex(
            query_encoder=self.query_encoder,
            candidate_encoder=self.candidate_encoder,
            config=self.config,
            _planes=self._planes,
            _buckets={k: tuple(v) for k, v in buckets.items()},
            built_candidates=len(candidates),
        )

    def _validate_dimension(self, vector: Sequence[float]) -> None:
        if len(vector) != self.config.dimensions:
            raise ValueError(
                f"representation dimension {len(vector)} != "
                f"configured dimension {self.config.dimensions}"
            )

    def _signature(self, vector: Sequence[float]) -> int:
        bits = 0
        for i, plane in enumerate(self._planes):
            dot = sum(float(a) * float(b) for a, b in zip(vector, plane))
            if dot >= 0.0:
                bits |= 1 << i
        return bits

    def _probe_signatures(self, signature: int) -> tuple[int, ...]:
        probes = [signature]
        if self.config.hamming_radius == 1:
            probes.extend(signature ^ (1 << i) for i in range(self.config.n_planes))
        return tuple(probes)

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = math.sqrt(sum(float(x) * float(x) for x in a))
        nb = math.sqrt(sum(float(x) * float(x) for x in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    accepts_state = True

    def lookup(
        self,
        query: Query,
        *,
        state: object | None = None,
        k: int | None = None,
        relation: str | None = None,
    ) -> SemanticAddressHit:
        del relation  # Relation is not an addressing oracle.
        state_obj = state
        q = tuple(float(x) for x in self.query_encoder(query, state_obj))
        self._validate_dimension(q)
        signature = self._signature(q)
        probes = self._probe_signatures(signature)
        candidates = []
        seen = set()
        for bucket_id in probes:
            for index in self._buckets.get(bucket_id, ()):
                if index not in seen:
                    seen.add(index)
                    candidates.append(index)
        if k is not None:
            candidates = candidates[: max(0, int(k))]
        return SemanticAddressHit(
            candidate_indices=tuple(candidates),
            inspected_positions=len(probes),
            bucket_size=len(candidates),
            bucket_probes=len(probes),
            representation_dimension=self.config.dimensions,
        )
