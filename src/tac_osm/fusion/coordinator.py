"""Sparse local-module recruitment and long-range coordination."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from .. import Query, StateRead
from .interfaces import CoordinatorDecision, MemoryContext, RegimeContext


@dataclass(frozen=True)
class CoordinatorConfig:
    n_modules: int = 8
    top_k: int = 2
    mode: str = "sparse"
    dim: int = 8
    lr: float = 0.03
    temperature: float = 0.5
    n_buckets: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_modules < 1:
            raise ValueError("n_modules must be >= 1")
        if not 1 <= self.top_k <= self.n_modules:
            raise ValueError("top_k must be in [1,n_modules]")
        if self.mode not in {"sparse", "dense", "single", "all", "disabled"}:
            raise ValueError(f"unknown coordination mode {self.mode!r}")
        if self.temperature <= 0.0 or self.lr <= 0.0:
            raise ValueError("temperature and lr must be positive")
        if self.n_buckets < 1:
            raise ValueError("n_buckets must be >= 1")


class SparseCoordinator:
    """Learnable module selector.

    Sparse mode first chooses a deterministic bucket from the public query and
    evaluates only modules in that bucket. The bucket contains at least one
    module for every module id by construction. This keeps the routing work
    explicitly measurable and prevents the sparse claim from being inferred
    from a top-k count alone.
    """

    def __init__(self, config: CoordinatorConfig | None = None) -> None:
        self.config = config if config is not None else CoordinatorConfig()
        rng = random.Random(self.config.seed)
        self._embeddings = [
            [rng.uniform(-0.1, 0.1) for _ in range(self.config.dim)]
            for _ in range(self.config.n_modules)
        ]
        self._bias = [0.0 for _ in range(self.config.n_modules)]
        self._updates = 0
        self._selected = [0 for _ in range(self.config.n_modules)]

    def route(
        self,
        *,
        query: Query,
        state_read: StateRead,
        memory: MemoryContext,
        regime: RegimeContext,
    ) -> CoordinatorDecision:
        base = _coord_features(query, state_read, memory, regime, self.config.dim)
        candidates = self._candidate_modules(query)
        if self.config.mode == "disabled":
            selected = tuple(range(min(self.config.top_k, self.config.n_modules)))
            candidates = selected
        elif self.config.mode == "single":
            candidates = (0,)
            selected = (0,)
        elif self.config.mode == "all":
            candidates = tuple(range(self.config.n_modules))
            selected = candidates
        else:
            scores = [self._score(m, base) for m in candidates]
            if self.config.mode == "dense":
                ranked = sorted(
                    zip(candidates, scores), key=lambda x: (x[1], -x[0]), reverse=True
                )
            else:
                ranked = sorted(
                    zip(candidates, scores), key=lambda x: (x[1], -x[0]), reverse=True
                )
            k = min(self.config.top_k, len(ranked))
            selected = tuple(m for m, _ in ranked[:k])

        for module_id in selected:
            self._selected[module_id] += 1
        score_vec = tuple(
            self._score(m, base) if m in candidates else float("-inf")
            for m in range(self.config.n_modules)
        )
        return CoordinatorDecision(
            selected_modules=selected,
            candidate_module_ids=tuple(candidates),
            scores=score_vec,
            routing_work=len(candidates),
            provenance=f"coordinator:{self.config.mode}",
        )

    def update(self, *, decision: CoordinatorDecision, reward: float) -> None:
        if self.config.mode in {"disabled", "all", "single"}:
            self._updates += 1
            return
        centred = float(reward) - (1.0 / max(1, self.config.n_modules))
        for module_id in decision.selected_modules:
            if 0 <= module_id < self.config.n_modules:
                self._bias[module_id] += self.config.lr * centred
        self._updates += 1

    def inspect(self) -> dict[str, object]:
        total = sum(self._selected)
        return {
            "mode": self.config.mode,
            "n_modules": self.config.n_modules,
            "top_k": self.config.top_k,
            "updates": self._updates,
            "routing_work_mean": (
                sum(self.config.n_modules for _ in range(1)) if total == 0
                else None
            ),
            "module_selection_counts": dict(enumerate(self._selected)),
        }

    def _score(self, module_id: int, features: Sequence[float]) -> float:
        return self._bias[module_id] + sum(
            a * b for a, b in zip(self._embeddings[module_id], features)
        )

    def _candidate_modules(self, query: Query) -> tuple[int, ...]:
        if self.config.mode == "dense":
            return tuple(range(self.config.n_modules))
        if self.config.mode in {"single", "all", "disabled"}:
            return tuple(range(self.config.n_modules))
        bits = tuple(int(b) for b in query.text.partition("\t")[0].split() if b in {"0", "1"})
        signature = 0
        for b in bits:
            signature = ((signature << 1) ^ b) & 0xFFFFFFFF
        bucket = signature % self.config.n_buckets
        selected = tuple(
            m for m in range(self.config.n_modules)
            if m % self.config.n_buckets == bucket
        )
        return selected or (bucket % self.config.n_modules,)


def _coord_features(
    query: Query,
    state_read: StateRead,
    memory: MemoryContext,
    regime: RegimeContext,
    dim: int,
) -> tuple[float, ...]:
    bits = tuple(
        int(b) for b in query.text.partition("\t")[0].split()
        if b in {"0", "1"}
    )
    x = [0.0 for _ in range(dim)]
    for i in range(min(dim, len(bits))):
        x[i] = 1.0 if bits[i] else -1.0
    if query.context:
        for i in range(min(dim, len(query.context))):
            x[i] *= 1.0 if query.context[i] else 0.5
    if memory.priming:
        for i in range(min(dim, len(memory.priming))):
            x[i] += 0.25 * memory.priming[i]
    if regime.embedding:
        for i in range(min(dim, len(regime.embedding))):
            x[i] += 0.5 * regime.embedding[i]
    if state_read.values:
        x[0] += min(1.0, len(state_read.values) / 16.0)
    return tuple(math.tanh(v) for v in x)
