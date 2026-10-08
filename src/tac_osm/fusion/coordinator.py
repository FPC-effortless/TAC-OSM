"""Sparse local-module recruitment and long-range coordination.

The coordinator is an independently swappable approximation of long-range
ascending/descending control: it recruits a small set of local operators,
accounts for routing work, and supports controlled lesions.
"""

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
    disabled_modules: tuple[int, ...] = ()
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
        if any(m < 0 or m >= self.n_modules for m in self.disabled_modules):
            raise ValueError("disabled module id out of range")

    @property
    def active_modules(self) -> tuple[int, ...]:
        disabled = set(self.disabled_modules)
        return tuple(m for m in range(self.n_modules) if m not in disabled)


class SparseCoordinator:
    """Learnable module selector with explicit routing-work accounting."""

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
        self._work = 0

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
        if not candidates:
            return CoordinatorDecision(
                selected_modules=(),
                candidate_module_ids=(),
                scores=tuple(float("-inf") for _ in range(self.config.n_modules)),
                routing_work=0,
                provenance=f"coordinator:{self.config.mode}:empty_after_lesion",
            )

        if self.config.mode == "disabled":
            selected = tuple(candidates[: min(self.config.top_k, len(candidates))])
        elif self.config.mode == "single":
            selected = tuple(candidates[:1])
        elif self.config.mode == "all":
            selected = tuple(candidates)
        else:
            ranked = sorted(
                ((m, self._score(m, base)) for m in candidates),
                key=lambda x: (x[1], -x[0]),
                reverse=True,
            )
            selected = tuple(
                m for m, _ in ranked[: min(self.config.top_k, len(ranked))]
            )

        self._work += len(candidates)
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
            self._bias[module_id] += self.config.lr * centred
        self._updates += 1

    def inspect(self) -> dict[str, object]:
        return {
            "mode": self.config.mode,
            "n_modules": self.config.n_modules,
            "active_modules": self.config.active_modules,
            "top_k": self.config.top_k,
            "updates": self._updates,
            "routing_work_total": self._work,
            "routing_work_mean_per_step": self._work / max(1, self._updates),
            "module_selection_counts": dict(enumerate(self._selected)),
            "disabled_modules": self.config.disabled_modules,
        }

    def _score(self, module_id: int, features: Sequence[float]) -> float:
        return self._bias[module_id] + sum(
            a * b for a, b in zip(self._embeddings[module_id], features)
        )

    def _candidate_modules(self, query: Query) -> tuple[int, ...]:
        active = self.config.active_modules
        if not active:
            return ()
        if self.config.mode in {"dense", "all", "disabled"}:
            return active
        if self.config.mode == "single":
            return active[:1]

        bits = tuple(
            int(b) for b in query.text.partition("	")[0].split()
            if b in {"0", "1"}
        )
        signature = 0
        for b in bits:
            signature = ((signature << 1) ^ b) & 0xFFFFFFFF
        bucket = signature % self.config.n_buckets
        selected = tuple(
            m for m in active if m % self.config.n_buckets == bucket
        )
        return selected or (active[0],)


def _coord_features(
    query: Query,
    state_read: StateRead,
    memory: MemoryContext,
    regime: RegimeContext,
    dim: int,
) -> tuple[float, ...]:
    bits = tuple(
        int(b) for b in query.text.partition("	")[0].split()
        if b in {"0", "1"}
    )
    x = [0.0] * dim
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
