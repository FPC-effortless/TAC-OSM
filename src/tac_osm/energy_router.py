"""Successor representation/energy router for TAC-OSM.

The legacy router is a hand-designed linear bandit.  This module introduces
the successor boundary:

    M_t = AddressState(Q_t, S_t)
    z_q = f_theta(Q_t, M_t)
    z_i = g_phi(C_i)
    E(q, c_i) = - <z_q, z_i>
    R_t = TopK_B(-E)

The representation maps are learned from outcome-derived positives.  For a
successful execution, the selected candidate is a positive and the strongest
current competitor is a hard negative.  This is intentionally a small
pure-Python implementation so the integration repository stays dependency
free.  It is a research implementation, not a claim that this optimizer is
the final PLM router.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from . import Candidate, PersistentState, Query, RoutingDecision
from .state_addressing import AddressedMemory, StateAddressor

__all__ = [
    "EnergyRouterConfig",
    "RoutingDiagnostics",
    "RepresentationEnergyRouter",
]


@dataclass(frozen=True)
class EnergyRouterConfig:
    input_dim: int = 8
    latent_dim: int = 16
    learning_rate: float = 0.01
    margin: float = 0.1
    top_k: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.margin < 0.0:
            raise ValueError("margin must be non-negative")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")


@dataclass(frozen=True)
class RoutingDiagnostics:
    candidate_count: int
    candidates_scored: int
    candidate_coverage: float
    selected_rank: int
    selected_energy: float
    hard_negative_margin: float | None
    state_found: bool
    state_inspected_slots: int
    query_encode_macs: int
    candidate_encode_macs: int
    similarity_macs: int
    total_macs: int

    @classmethod
    def from_scores(
        cls,
        scores: Sequence[float],
        selected: int,
        memory: AddressedMemory,
        *,
        latent_dim: int,
        raw_query_dim: int,
        input_dim: int,
    ) -> "RoutingDiagnostics":
        if not scores:
            raise ValueError("scores cannot be empty")
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        rank = order.index(selected) + 1
        selected_energy = -float(scores[selected])
        if len(order) > 1:
            hard_negative = scores[order[0] if order[0] != selected else order[1]]
            margin = float(scores[selected] - hard_negative)
        else:
            margin = None
        query_macs = latent_dim * raw_query_dim
        candidate_macs = len(scores) * latent_dim * input_dim
        similarity_macs = len(scores) * latent_dim
        return cls(
            candidate_count=len(scores),
            candidates_scored=len(scores),
            candidate_coverage=1.0,
            selected_rank=rank,
            selected_energy=selected_energy,
            hard_negative_margin=margin,
            state_found=memory.found,
            state_inspected_slots=memory.inspected_slots,
            query_encode_macs=query_macs,
            candidate_encode_macs=candidate_macs,
            similarity_macs=similarity_macs,
            total_macs=query_macs + candidate_macs + similarity_macs,
        )


class RepresentationEnergyRouter:
    """Learned query/candidate representations with pairwise hard negatives."""

    def __init__(
        self,
        config: EnergyRouterConfig | None = None,
        *,
        addressor: StateAddressor | None = None,
    ) -> None:
        self.config = config if config is not None else EnergyRouterConfig()
        self.addressor = addressor if addressor is not None else StateAddressor()
        rng = random.Random(self.config.seed)
        scale = 0.05
        self.wq = [
            [rng.uniform(-scale, scale) for _ in range(self._raw_query_dim)]
            for _ in range(self.config.latent_dim)
        ]
        self.wc = [
            [rng.uniform(-scale, scale) for _ in range(self.config.input_dim)]
            for _ in range(self.config.latent_dim)
        ]
        self.bq = [0.0] * self.config.latent_dim
        self.bc = [0.0] * self.config.latent_dim
        self._updates = 0
        self._last_diagnostics: RoutingDiagnostics | None = None
        self._last_memory: AddressedMemory | None = None

    @property
    def _raw_query_dim(self) -> int:
        # query bits + context + addressed-memory bits + presence flag
        return 3 * self.config.input_dim + 1

    @property
    def updates(self) -> int:
        return self._updates

    @property
    def last_diagnostics(self) -> RoutingDiagnostics | None:
        return self._last_diagnostics

    @property
    def last_memory(self) -> AddressedMemory | None:
        return self._last_memory

    def _bits(self, query: Query) -> tuple[int, ...]:
        raw = query.text.partition("\t")[0]
        return tuple(int(x) for x in raw.split()) if raw.strip() else ()

    def _pad(self, values: Sequence[float]) -> list[float]:
        out = [0.0] * self.config.input_dim
        for i, value in enumerate(values[: self.config.input_dim]):
            out[i] = float(value)
        return out

    def _query_input(self, query: Query, memory: AddressedMemory) -> list[float]:
        bits = self._pad(self._bits(query))
        context = self._pad(query.context)
        mem = self._pad(memory.value)
        return bits + context + mem + [memory.present]

    def _candidate_input(self, candidate: Candidate) -> list[float]:
        # Candidate action/index is deliberately excluded.
        return self._pad(candidate.descriptor)

    @staticmethod
    def _linear(weights: Sequence[Sequence[float]], bias: Sequence[float],
                x: Sequence[float]) -> list[float]:
        return [
            sum(w * xv for w, xv in zip(row, x)) + bias[i]
            for i, row in enumerate(weights)
        ]

    def encode_query(self, query: Query, memory: AddressedMemory) -> tuple[float, ...]:
        return tuple(self._linear(self.wq, self.bq, self._query_input(query, memory)))

    def encode_candidate(self, candidate: Candidate) -> tuple[float, ...]:
        return tuple(self._linear(self.wc, self.bc, self._candidate_input(candidate)))

    @staticmethod
    def _score(zq: Sequence[float], zc: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(zq, zc))

    def score(
        self,
        query: Query,
        memory: AddressedMemory,
        candidates: Sequence[Candidate],
    ) -> list[float]:
        zq = self.encode_query(query, memory)
        return [
            self._score(zq, self.encode_candidate(candidate))
            for candidate in candidates
        ]

    def route(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
    ) -> RoutingDecision:
        if not candidates:
            raise ValueError("cannot route an empty candidate set")
        memory = self.addressor.address(query, state)
        scores = self.score(query, memory, candidates)
        selected = max(range(len(scores)), key=lambda i: (scores[i], -i))
        self._last_memory = memory
        self._last_diagnostics = RoutingDiagnostics.from_scores(
            scores,
            selected,
            memory,
            latent_dim=self.config.latent_dim,
            raw_query_dim=self._raw_query_dim,
            input_dim=self.config.input_dim,
        )
        return RoutingDecision(
            selected=selected,
            scores=tuple(scores),
            provenance="representation_energy_v1",
        )

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0.0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def learn_from_outcome(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
        selected: int,
        *,
        success: bool,
        scores: Sequence[float] | None = None,
    ) -> float:
        """One pairwise update using the strongest current hard negative.

        A failed action supplies no invented positive label.  The update is
        therefore deliberately conservative: only verified success creates a
        positive.  This is one reason the next capability experiment must
        report update count and success yield, not only final accuracy.
        """
        if not success:
            return 0.0
        if not (0 <= selected < len(candidates)):
            raise IndexError("selected candidate outside candidate set")

        memory = self._last_memory
        if memory is None:
            memory = self.addressor.address(query, state)
        qx = self._query_input(query, memory)
        cx_pos = self._candidate_input(candidates[selected])
        score_values = (
            [float(x) for x in scores]
            if scores is not None
            else self.score(query, memory, candidates)
        )
        negatives = [i for i in range(len(candidates)) if i != selected]
        if not negatives:
            return 0.0
        negative = max(negatives, key=lambda i: (score_values[i], -i))
        cx_neg = self._candidate_input(candidates[negative])

        zq = self._linear(self.wq, self.bq, qx)
        zp = self._linear(self.wc, self.bc, cx_pos)
        zn = self._linear(self.wc, self.bc, cx_neg)

        diff = self._score(zq, zp) - self._score(zq, zn)
        gate = self._sigmoid(self.config.margin - diff)

        delta_c = [zp_i - zn_i for zp_i, zn_i in zip(zp, zn)]
        lr = self.config.learning_rate

        for r in range(self.config.latent_dim):
            dq = gate * delta_c[r]
            for j in range(self._raw_query_dim):
                self.wq[r][j] += lr * dq * qx[j]
            self.bq[r] += lr * dq

        for r in range(self.config.latent_dim):
            dpos = gate * zq[r]
            dneg = gate * zq[r]
            for j in range(self.config.input_dim):
                self.wc[r][j] += lr * dpos * cx_pos[j]
                self.wc[r][j] -= lr * dneg * cx_neg[j]
            self.bc[r] += lr * (dpos - dneg)

        self._updates += 1
        return math.log1p(math.exp(self.config.margin - diff))

    def diagnostics_with_target(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
        target_index: int,
    ) -> RoutingDiagnostics:
        """Post-hoc diagnostics; target never enters routing or training inputs."""
        if not (0 <= target_index < len(candidates)):
            raise IndexError("target_index outside candidate set")
        memory = self.addressor.address(query, state)
        scores = self.score(query, memory, candidates)
        ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        selected = ranking[0]
        selected_rank = ranking.index(target_index) + 1
        hard_negative = next((i for i in ranking if i != target_index), None)
        margin = (
            scores[target_index] - scores[hard_negative]
            if hard_negative is not None
            else None
        )
        query_macs = self.config.latent_dim * self._raw_query_dim
        candidate_macs = len(candidates) * self.config.latent_dim * self.config.input_dim
        similarity_macs = len(candidates) * self.config.latent_dim
        return RoutingDiagnostics(
            candidate_count=len(candidates),
            candidates_scored=len(candidates),
            candidate_coverage=1.0,
            selected_rank=selected_rank,
            selected_energy=-scores[target_index],
            hard_negative_margin=margin,
            state_found=memory.found,
            state_inspected_slots=memory.inspected_slots,
            query_encode_macs=query_macs,
            candidate_encode_macs=candidate_macs,
            similarity_macs=similarity_macs,
            total_macs=query_macs + candidate_macs + similarity_macs,
        )
