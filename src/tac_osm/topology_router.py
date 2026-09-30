"""Explicit-program energy router for TACOSM-REP-002.

Unlike the REP-001 router, candidate encoding includes the candidate's explicit
executable edge set. The public query contains the desired edge-set mask.

No target index, action index, outcome, or environment-side gold field enters
the router.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from . import Candidate, PersistentState, Query, RoutingDecision
from .topology_tasks import TOPOLOGY_EDGE_UNIVERSE, edge_mask

__all__ = [
    "ExplicitProgramRouterConfig",
    "ExplicitProgramRoutingDiagnostics",
    "ExplicitProgramEnergyRouter",
]


@dataclass(frozen=True)
class ExplicitProgramRouterConfig:
    input_dim: int = len(TOPOLOGY_EDGE_UNIVERSE)
    latent_dim: int = len(TOPOLOGY_EDGE_UNIVERSE)
    learning_rate: float = 0.01
    margin: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.input_dim != len(TOPOLOGY_EDGE_UNIVERSE):
            raise ValueError(
                f"input_dim must equal topology mask width {len(TOPOLOGY_EDGE_UNIVERSE)}"
            )
        if self.latent_dim < self.input_dim:
            raise ValueError("latent_dim must be >= input_dim")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.margin < 0.0:
            raise ValueError("margin must be non-negative")


@dataclass(frozen=True)
class ExplicitProgramRoutingDiagnostics:
    candidate_count: int
    selected_rank: int
    hard_negative_margin: float | None
    query_encode_macs: int
    candidate_encode_macs: int
    similarity_macs: int
    total_macs: int


class ExplicitProgramEnergyRouter:
    """Learned query/program representations with pairwise hard negatives."""

    def __init__(self, config: ExplicitProgramRouterConfig | None = None) -> None:
        self.config = config if config is not None else ExplicitProgramRouterConfig()
        rng = random.Random(self.config.seed)
        scale = 0.05
        self.wq = [
            [rng.uniform(-scale, scale) for _ in range(self.config.input_dim)]
            for _ in range(self.config.latent_dim)
        ]
        self.wc = [
            [rng.uniform(-scale, scale) for _ in range(self.config.input_dim)]
            for _ in range(self.config.latent_dim)
        ]
        self.bq = [0.0] * self.config.latent_dim
        self.bc = [0.0] * self.config.latent_dim
        self._updates = 0
        self._last: ExplicitProgramRoutingDiagnostics | None = None

    @property
    def updates(self) -> int:
        return self._updates

    @property
    def last_diagnostics(self) -> ExplicitProgramRoutingDiagnostics | None:
        return self._last

    def _signed_query(self, query: Query) -> list[float]:
        raw = query.text.partition("\t")[0]
        bits = [int(x) for x in raw.split()] if raw.strip() else []
        if len(bits) != self.config.input_dim:
            raise ValueError(
                f"query topology mask has width {len(bits)}, expected {self.config.input_dim}"
            )
        return [1.0 if bit else -1.0 for bit in bits]

    def _signed_candidate(self, candidate: Candidate) -> list[float]:
        if not candidate.executable_edges:
            raise ValueError("candidate has no executable edge set")
        mask = edge_mask(candidate.executable_edges)
        return [1.0 if bit else -1.0 for bit in mask]

    @staticmethod
    def _linear(
        weights: Sequence[Sequence[float]],
        bias: Sequence[float],
        x: Sequence[float],
    ) -> list[float]:
        return [
            sum(w * xv for w, xv in zip(row, x)) + bias[i]
            for i, row in enumerate(weights)
        ]

    def encode_query(self, query: Query) -> tuple[float, ...]:
        return tuple(self._linear(self.wq, self.bq, self._signed_query(query)))

    def encode_candidate(self, candidate: Candidate) -> tuple[float, ...]:
        return tuple(
            self._linear(self.wc, self.bc, self._signed_candidate(candidate))
        )

    @staticmethod
    def _score(zq: Sequence[float], zc: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(zq, zc))

    def score(self, query: Query, candidates: Sequence[Candidate]) -> list[float]:
        zq = self.encode_query(query)
        return [self._score(zq, self.encode_candidate(c)) for c in candidates]

    def route(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
    ) -> RoutingDecision:
        del state
        if not candidates:
            raise ValueError("cannot route an empty candidate set")
        scores = self.score(query, candidates)
        selected = max(range(len(scores)), key=lambda i: (scores[i], -i))
        self._last = self._diagnostics(scores, selected)
        return RoutingDecision(
            selected=selected,
            scores=tuple(scores),
            provenance="explicit_program_energy",
        )

    def _diagnostics(
        self, scores: Sequence[float], selected: int
    ) -> ExplicitProgramRoutingDiagnostics:
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        rank = order.index(selected) + 1
        hard_negative = next((i for i in order if i != selected), None)
        margin = None if hard_negative is None else scores[selected] - scores[hard_negative]
        d = self.config.latent_dim
        q_macs = d * self.config.input_dim
        c_macs = len(scores) * d * self.config.input_dim
        s_macs = len(scores) * d
        return ExplicitProgramRoutingDiagnostics(
            candidate_count=len(scores),
            selected_rank=rank,
            hard_negative_margin=margin,
            query_encode_macs=q_macs,
            candidate_encode_macs=c_macs,
            similarity_macs=s_macs,
            total_macs=q_macs + c_macs + s_macs,
        )

    def set_analytic_relation(self) -> None:
        """Install identity embeddings; equal masks receive the maximal score."""
        for r in range(self.config.latent_dim):
            for j in range(self.config.input_dim):
                self.wq[r][j] = 1.0 if r == j else 0.0
                self.wc[r][j] = 1.0 if r == j else 0.0
            self.bq[r] = 0.0
            self.bc[r] = 0.0

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
        del state
        if not success:
            return 0.0
        if not (0 <= selected < len(candidates)):
            raise IndexError("selected candidate outside candidate set")

        qx = self._signed_query(query)
        cx_pos = self._signed_candidate(candidates[selected])
        score_values = (
            [float(x) for x in scores]
            if scores is not None
            else self.score(query, candidates)
        )
        negatives = [i for i in range(len(candidates)) if i != selected]
        if not negatives:
            return 0.0
        negative = max(negatives, key=lambda i: (score_values[i], -i))
        cx_neg = self._signed_candidate(candidates[negative])

        zq = self._linear(self.wq, self.bq, qx)
        zp = self._linear(self.wc, self.bc, cx_pos)
        zn = self._linear(self.wc, self.bc, cx_neg)
        diff = self._score(zq, zp) - self._score(zq, zn)

        # Only a verified success creates the positive pair. The negative is
        # the strongest current in-pool competitor.
        clipped = max(-60.0, min(60.0, diff - self.config.margin))
        gate = 1.0 / (1.0 + math.exp(clipped))
        dc = [zp_i - zn_i for zp_i, zn_i in zip(zp, zn)]
        lr = self.config.learning_rate

        for r in range(self.config.latent_dim):
            dq = gate * dc[r]
            for j in range(self.config.input_dim):
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
        loss_arg = max(-60.0, min(60.0, self.config.margin - diff))
        return math.log1p(math.exp(loss_arg))

    def diagnostics_with_target(
        self,
        query: Query,
        candidates: Sequence[Candidate],
        target_index: int,
    ) -> ExplicitProgramRoutingDiagnostics:
        if not (0 <= target_index < len(candidates)):
            raise IndexError("target_index outside candidate set")
        scores = self.score(query, candidates)
        ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        selected = ranking[0]
        target_rank = ranking.index(target_index) + 1
        hard_negative = next((i for i in ranking if i != target_index), None)
        margin = (
            scores[target_index] - scores[hard_negative]
            if hard_negative is not None
            else None
        )
        d = self.config.latent_dim
        q_macs = d * self.config.input_dim
        c_macs = len(candidates) * d * self.config.input_dim
        s_macs = len(candidates) * d
        return ExplicitProgramRoutingDiagnostics(
            candidate_count=len(candidates),
            selected_rank=target_rank,
            hard_negative_margin=margin,
            query_encode_macs=q_macs,
            candidate_encode_macs=c_macs,
            similarity_macs=s_macs,
            total_macs=q_macs + c_macs + s_macs,
        )

    def candidate_observation(self, candidate: Candidate) -> tuple[int, ...]:
        """Return the executable-topology observation; action is excluded."""
        return edge_mask(candidate.executable_edges)
