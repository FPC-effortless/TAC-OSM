"""Learned semantic coarse index for persistent-state retrieval.

This module separates the learned semantic representation from the index
mechanism. A dual linear encoder is trained only on a disjoint set of semantic
codes. At runtime, stored state items are quantized to a compact binary code
and placed in hash buckets; a noisy query computes its binary code and probes
a bounded Hamming neighborhood.

The index is an experiment boundary, not a claim of general semantic memory.
Index construction and query embedding costs are reported separately from
dictionary probes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from . import Query
from .temporal import TemporalPersistentState


@dataclass(frozen=True)
class LearnedStateIndexConfig:
    input_dim: int = 10
    latent_dim: int = 8
    learning_rate: float = 0.02
    margin: float = 0.25
    epochs: int = 32
    bucket_bits: int = 8
    probe_radius: int = 1
    shortlist_k: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.latent_dim < 1:
            raise ValueError("dimensions must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if not 1 <= self.bucket_bits <= self.latent_dim:
            raise ValueError("bucket_bits must be in [1, latent_dim]")
        if not 0 <= self.probe_radius <= self.bucket_bits:
            raise ValueError("probe_radius must be in [0, bucket_bits]")
        if self.shortlist_k < 1:
            raise ValueError("shortlist_k must be positive")


@dataclass(frozen=True)
class LearnedIndexBuildDiagnostics:
    state_items: int
    build_encodes: int
    unique_buckets: int


@dataclass(frozen=True)
class LearnedIndexLookup:
    addresses: tuple[str, ...]
    query_probes: int
    bucket_candidates: int
    shortlist_size: int


class LearnedSemanticStateIndex:
    """A trained semantic encoder followed by a bounded binary bucket index."""

    def __init__(self, config: LearnedStateIndexConfig | None = None) -> None:
        self.config = config or LearnedStateIndexConfig()
        rng = random.Random(self.config.seed)
        scale = 0.08
        self.wq = [
            [rng.uniform(-scale, scale) for _ in range(self.config.input_dim)]
            for _ in range(self.config.latent_dim)
        ]
        self.ws = [
            [rng.uniform(-scale, scale) for _ in range(self.config.input_dim)]
            for _ in range(self.config.latent_dim)
        ]
        self.bq = [0.0] * self.config.latent_dim
        self.bs = [0.0] * self.config.latent_dim
        self._trained_pairs = 0
        self._buckets: dict[int, tuple[str, ...]] = {}
        self._build = LearnedIndexBuildDiagnostics(0, 0, 0)
        self._built = False

    @property
    def trained_pairs(self) -> int:
        return self._trained_pairs

    @property
    def build_diagnostics(self) -> LearnedIndexBuildDiagnostics:
        return self._build

    @property
    def query_embedding_macs(self) -> int:
        return self.config.latent_dim * self.config.input_dim

    @property
    def state_embedding_macs(self) -> int:
        return self.config.latent_dim * self.config.input_dim

    def _encode(
        self,
        weights: Sequence[Sequence[float]],
        bias: Sequence[float],
        value: Sequence[float],
    ) -> list[float]:
        return [
            sum(w * x for w, x in zip(row, value)) + bias[i]
            for i, row in enumerate(weights)
        ]

    @staticmethod
    def _signed(bits: Sequence[int]) -> list[float]:
        return [1.0 if int(x) else -1.0 for x in bits]

    @staticmethod
    def _query_bits(query: Query, width: int) -> tuple[int, ...]:
        raw = query.text.partition("\t")[0].strip()
        bits = tuple(int(x) for x in raw.split()) if raw else ()
        if len(bits) != width or any(x not in (0, 1) for x in bits):
            raise ValueError(f"query code must be binary width {width}")
        return bits

    def encode_query(self, query: Query) -> list[float]:
        return self._encode(
            self.wq,
            self.bq,
            self._signed(self._query_bits(query, self.config.input_dim)),
        )

    def encode_state(self, value: Sequence[int]) -> list[float]:
        bits = tuple(int(x) for x in value)
        if len(bits) != self.config.input_dim or any(x not in (0, 1) for x in bits):
            raise ValueError(f"state code must be binary width {self.config.input_dim}")
        return self._encode(self.ws, self.bs, self._signed(bits))

    def bucket_code(self, embedding: Sequence[float]) -> int:
        code = 0
        for value in embedding[: self.config.bucket_bits]:
            code = (code << 1) | int(value >= 0.0)
        return code

    def train(
        self,
        train_codes: Sequence[Sequence[int]],
        *,
        noise_cycle: int = 10,
    ) -> int:
        """Train on semantic codes disjoint from evaluation targets.

        Each update uses a noisy query for one positive code and a deterministic
        negative code chosen from the training set. No evaluation state, address,
        or evaluation target is consulted.
        """
        codes = [tuple(int(x) for x in code) for code in train_codes]
        if len(codes) < 2:
            raise ValueError("at least two training codes are required")
        for code in codes:
            if len(code) != self.config.input_dim:
                raise ValueError("training code width mismatch")
        rng = random.Random(self.config.seed * 1009 + 17)
        self._trained_pairs = 0
        for epoch in range(self.config.epochs):
            order = list(range(len(codes)))
            rng.shuffle(order)
            for local, idx in enumerate(order):
                positive = codes[idx]
                neg_idx = (idx + 1 + epoch + local) % len(codes)
                if neg_idx == idx:
                    neg_idx = (neg_idx + 1) % len(codes)
                negative = codes[neg_idx]

                bits = list(positive)
                bits[(idx + epoch + noise_cycle) % self.config.input_dim] ^= 1
                query_x = self._signed(bits)
                pos_x = self._signed(positive)
                neg_x = self._signed(negative)

                zq = self._encode(self.wq, self.bq, query_x)
                zp = self._encode(self.ws, self.bs, pos_x)
                zn = self._encode(self.ws, self.bs, neg_x)
                diff = sum(a * b for a, b in zip(zq, zp)) - sum(
                    a * b for a, b in zip(zq, zn)
                )
                clipped = max(-60.0, min(60.0, self.config.margin - diff))
                gate = 1.0 / (1.0 + math.exp(-clipped))
                for r in range(self.config.latent_dim):
                    delta = gate * (zp[r] - zn[r])
                    for j in range(self.config.input_dim):
                        self.wq[r][j] += self.config.learning_rate * delta * query_x[j]
                    self.bq[r] += self.config.learning_rate * delta

                    d = self.config.learning_rate * gate * zq[r]
                    for j in range(self.config.input_dim):
                        self.ws[r][j] += d * (pos_x[j] - neg_x[j])
                self._trained_pairs += 1
        return self._trained_pairs

    def _neighbors(self, code: int) -> tuple[tuple[int, int], ...]:
        out = [(0, code)]
        for distance in range(1, self.config.probe_radius + 1):
            for flips in combinations(range(self.config.bucket_bits), distance):
                candidate = code
                for pos in flips:
                    candidate ^= 1 << pos
                out.append((distance, candidate))
        return tuple(out)

    def build(self, state: TemporalPersistentState) -> LearnedIndexBuildDiagnostics:
        buckets: dict[int, list[str]] = {}
        addresses = state.addresses()
        for address in addresses:
            read = state.read(
                Query(
                    text="\t" + address,
                    step=state.current_step,
                    provenance="learned_state_index_build",
                )
            )
            if not read.keys:
                continue
            code = self.bucket_code(self.encode_state(read.values[0]))
            buckets.setdefault(code, []).append(address)
        self._buckets = {
            key: tuple(sorted(value)) for key, value in buckets.items()
        }
        self._build = LearnedIndexBuildDiagnostics(
            state_items=len(addresses),
            build_encodes=len(addresses),
            unique_buckets=len(self._buckets),
        )
        self._built = True
        return self._build

    def lookup(self, query: Query) -> LearnedIndexLookup:
        if not self._built:
            raise RuntimeError("learned state index has not been built")
        query_code = self.bucket_code(self.encode_query(query))
        candidates: list[tuple[int, str]] = []
        for distance, bucket in self._neighbors(query_code):
            for address in self._buckets.get(bucket, ()):
                candidates.append((distance, address))
        candidates.sort(key=lambda x: (x[0], x[1]))
        addresses = tuple(
            address for _, address in candidates[: self.config.shortlist_k]
        )
        return LearnedIndexLookup(
            addresses=addresses,
            query_probes=len(self._neighbors(query_code)),
            bucket_candidates=len(candidates),
            shortlist_size=len(addresses),
        )
