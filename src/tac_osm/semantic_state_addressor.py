"""Semantic addressing over opaque persistent state items.

Inference boundary:
    query semantic requirement -> select one state item -> read its value

The address is deliberately opaque and absent from the query. This module
does not claim sublinear lookup: it enumerates the current state addresses and
reads each item through the temporal state API. The cost is measured as an
explicit state-addressing term.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from . import Query
from .state_addressing import AddressedMemory
from .temporal import TemporalPersistentState


@dataclass(frozen=True)
class SemanticAddressingConfig:
    input_dim: int = 5
    latent_dim: int = 8
    learning_rate: float = 0.01
    margin: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.latent_dim < 1:
            raise ValueError("dimensions must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")


@dataclass(frozen=True)
class StateAddressingDecision:
    selected_index: int
    selected_address: str
    scores: tuple[float, ...]
    target_rank: int
    inspected_items: int
    pool_size: int
    total_macs: int


class SemanticStateAddressor:
    """Learned semantic matcher over persistent state values."""

    def __init__(self, config: SemanticAddressingConfig | None = None) -> None:
        self.config = (
            config if config is not None else SemanticAddressingConfig()
        )
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
        self._updates = 0

    @property
    def updates(self) -> int:
        return self._updates

    @staticmethod
    def _bits(query: Query) -> list[float]:
        raw = query.text.partition("\t")[0].strip()
        bits = [int(x) for x in raw.split()] if raw else []
        return [1.0 if x else -1.0 for x in bits]

    @staticmethod
    def _signed_value(value: Sequence[int]) -> list[float]:
        return [1.0 if int(x) else -1.0 for x in value]

    def _linear(
        self,
        weights: Sequence[Sequence[float]],
        bias: Sequence[float],
        x: Sequence[float],
    ) -> list[float]:
        return [
            sum(w * xv for w, xv in zip(row, x)) + bias[i]
            for i, row in enumerate(weights)
        ]

    def _encode_query(self, query: Query) -> list[float]:
        x = self._bits(query)
        if len(x) != self.config.input_dim:
            raise ValueError("semantic query width mismatch")
        return self._linear(self.wq, self.bq, x)

    def _encode_state(self, value: Sequence[int]) -> list[float]:
        x = self._signed_value(value)
        if len(x) != self.config.input_dim:
            raise ValueError("semantic state value width mismatch")
        return self._linear(self.ws, self.bs, x)

    @staticmethod
    def _score(q: Sequence[float], s: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(q, s))

    def state_pool(
        self, query: Query, state: TemporalPersistentState
    ) -> tuple[AddressedMemory, ...]:
        del query
        items: list[AddressedMemory] = []
        for address in state.addresses():
            read_query = Query(text="\t" + address, step=state.current_step)
            read = state.read(read_query)
            if not read.keys:
                continue
            items.append(
                AddressedMemory(
                    address=address,
                    found=True,
                    key=address,
                    value=tuple(read.values[0]),
                    inspected_slots=1,
                    pool_size=len(state.addresses()),
                )
            )
        return tuple(items)

    def score_pool(
        self,
        query: Query,
        pool: Sequence[AddressedMemory],
    ) -> tuple[list[float], int]:
        zq = self._encode_query(query)
        scores = [self._score(zq, self._encode_state(item.value)) for item in pool]
        macs = self.config.latent_dim * self.config.input_dim
        macs += len(pool) * self.config.latent_dim * self.config.input_dim
        macs += len(pool) * self.config.latent_dim
        return scores, macs

    def select(
        self,
        query: Query,
        state: TemporalPersistentState,
        *,
        target_address: str | None = None,
    ) -> StateAddressingDecision:
        pool = self.state_pool(query, state)
        if not pool:
            raise ValueError("semantic state pool is empty")
        scores, macs = self.score_pool(query, pool)
        selected = max(range(len(scores)), key=lambda i: (scores[i], -i))
        rank = 1
        if target_address is not None:
            target_indices = [i for i, item in enumerate(pool) if item.address == target_address]
            if len(target_indices) != 1:
                raise ValueError("target address absent or duplicated in state pool")
            target_score = scores[target_indices[0]]
            rank = 1 + sum(score > target_score for score in scores)
        return StateAddressingDecision(
            selected_index=selected,
            selected_address=pool[selected].address,
            scores=tuple(scores),
            target_rank=rank,
            inspected_items=len(pool),
            pool_size=len(pool),
            total_macs=macs,
        )

    def set_identity(self) -> None:
        for r in range(self.config.latent_dim):
            for j in range(self.config.input_dim):
                value = 1.0 if r == j else 0.0
                self.wq[r][j] = value
                self.ws[r][j] = value
            self.bq[r] = 0.0
            self.bs[r] = 0.0

    def learn_from_success(
        self,
        query: Query,
        pool: Sequence[AddressedMemory],
        selected_index: int,
        *,
        success: bool,
        scores: Sequence[float],
    ) -> float:
        if not success:
            return 0.0
        if not (0 <= selected_index < len(pool)):
            raise IndexError("selected index outside state pool")

        qx = self._bits(query)
        zq = self._linear(self.wq, self.bq, qx)
        zp = self._encode_state(pool[selected_index].value)
        negatives = [i for i in range(len(pool)) if i != selected_index]
        if not negatives:
            return 0.0
        negative = max(negatives, key=lambda i: (scores[i], -i))
        zn = self._encode_state(pool[negative].value)
        diff = self._score(zq, zp) - self._score(zq, zn)
        gate = 1.0 / (
            1.0 + math.exp(max(-60.0, min(60.0, diff - self.config.margin)))
        )
        delta = [gate * (p - n) for p, n in zip(zp, zn)]
        lr = self.config.learning_rate

        for r in range(self.config.latent_dim):
            for j in range(self.config.input_dim):
                self.wq[r][j] += lr * delta[r] * qx[j]
            self.bq[r] += lr * delta[r]

        for r in range(self.config.latent_dim):
            dpos = gate * zq[r]
            dneg = gate * zq[r]
            for j in range(self.config.input_dim):
                self.ws[r][j] += lr * dpos * self._signed_value(pool[selected_index].value)[j]
                self.ws[r][j] -= lr * dneg * self._signed_value(pool[negative].value)[j]
            self.bs[r] += lr * (dpos - dneg)

        self._updates += 1
        return math.log1p(
            math.exp(max(-60.0, min(60.0, self.config.margin - diff)))
        )
