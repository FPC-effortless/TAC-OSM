"""Learned graph-program router for TACOSM-REP-003.

The candidate encoder receives only the candidate's explicit edge mask.
A two-layer tanh network is used so path-compositional features are not baked
into the representation. The query encoder receives only the five-bit semantic
dependency signature.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from . import Candidate, PersistentState, Query, RoutingDecision
from .topology_tasks import TOPOLOGY_EDGE_UNIVERSE
from .semantic_topology import SEMANTIC_SIGNATURE_WIDTH, semantic_signature


@dataclass(frozen=True)
class GraphProgramRouterConfig:
    edge_dim: int = len(TOPOLOGY_EDGE_UNIVERSE)
    query_dim: int = SEMANTIC_SIGNATURE_WIDTH
    hidden_dim: int = 16
    latent_dim: int = 8
    learning_rate: float = 0.01
    margin: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.edge_dim < 1 or self.query_dim < 1:
            raise ValueError("input dimensions must be positive")
        if self.hidden_dim < 1 or self.latent_dim < 1:
            raise ValueError("hidden/latent dimensions must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.margin < 0.0:
            raise ValueError("margin must be non-negative")


@dataclass(frozen=True)
class GraphProgramRoutingDiagnostics:
    candidate_count: int
    selected_rank: int
    hard_negative_margin: float | None
    query_encode_macs: int
    candidate_encode_macs: int
    similarity_macs: int
    total_macs: int


class GraphProgramRouter:
    def __init__(self, config: GraphProgramRouterConfig | None = None) -> None:
        self.config = config if config is not None else GraphProgramRouterConfig()
        rng = random.Random(self.config.seed)
        scale = 0.08
        self.wq = [
            [rng.uniform(-scale, scale) for _ in range(self.config.query_dim)]
            for _ in range(self.config.latent_dim)
        ]
        self.bq = [0.0] * self.config.latent_dim
        self.w1 = [
            [rng.uniform(-scale, scale) for _ in range(self.config.edge_dim)]
            for _ in range(self.config.hidden_dim)
        ]
        self.b1 = [0.0] * self.config.hidden_dim
        self.w2 = [
            [rng.uniform(-scale, scale) for _ in range(self.config.hidden_dim)]
            for _ in range(self.config.latent_dim)
        ]
        self.b2 = [0.0] * self.config.latent_dim
        self._updates = 0
        self._last: GraphProgramRoutingDiagnostics | None = None

    @property
    def updates(self) -> int:
        return self._updates

    @property
    def last_diagnostics(self) -> GraphProgramRoutingDiagnostics | None:
        return self._last

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

    def _query_x(self, query: Query) -> list[float]:
        bits = [int(x) for x in query.text.partition("\t")[0].split()]
        if len(bits) != self.config.query_dim:
            raise ValueError("semantic query width mismatch")
        return [1.0 if x else -1.0 for x in bits]

    def _candidate_x(self, candidate: Candidate) -> list[float]:
        if not candidate.executable_edges:
            raise ValueError("candidate has no explicit executable topology")
        mask = [
            1 if edge in candidate.executable_edges else 0
            for edge in TOPOLOGY_EDGE_UNIVERSE
        ]
        return [1.0 if x else -1.0 for x in mask]

    def _encode_candidate(
        self, x: Sequence[float]
    ) -> tuple[list[float], list[float]]:
        pre = self._linear(self.w1, self.b1, x)
        hidden = [math.tanh(v) for v in pre]
        latent = self._linear(self.w2, self.b2, hidden)
        return hidden, latent

    def encode_query(self, query: Query) -> tuple[float, ...]:
        x = self._query_x(query)
        return tuple(self._linear(self.wq, self.bq, x))

    def encode_candidate(self, candidate: Candidate) -> tuple[float, ...]:
        _, z = self._encode_candidate(self._candidate_x(candidate))
        return tuple(z)

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
            provenance="graph_program",
        )

    def _diagnostics(self, scores: Sequence[float], selected: int):
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        rank = order.index(selected) + 1
        negative = next((i for i in order if i != selected), None)
        margin = None if negative is None else scores[selected] - scores[negative]
        q_macs = self.config.latent_dim * self.config.query_dim
        # Two dense candidate layers: edge->hidden and hidden->latent.
        c_macs = len(scores) * (
            self.config.hidden_dim * self.config.edge_dim
            + self.config.latent_dim * self.config.hidden_dim
        )
        s_macs = len(scores) * self.config.latent_dim
        return GraphProgramRoutingDiagnostics(
            candidate_count=len(scores),
            selected_rank=rank,
            hard_negative_margin=margin,
            query_encode_macs=q_macs,
            candidate_encode_macs=c_macs,
            similarity_macs=s_macs,
            total_macs=q_macs + c_macs + s_macs,
        )

    def diagnostics_with_target(
        self,
        query: Query,
        candidates: Sequence[Candidate],
        target_index: int,
    ) -> GraphProgramRoutingDiagnostics:
        if not (0 <= target_index < len(candidates)):
            raise IndexError("target_index outside candidate set")
        scores = self.score(query, candidates)
        ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        negative = next((i for i in ranking if i != target_index), None)
        margin = None if negative is None else scores[target_index] - scores[negative]
        return GraphProgramRoutingDiagnostics(
            candidate_count=len(scores),
            selected_rank=ranking.index(target_index) + 1,
            hard_negative_margin=margin,
            query_encode_macs=self.config.latent_dim * self.config.query_dim,
            candidate_encode_macs=len(scores) * (
                self.config.hidden_dim * self.config.edge_dim
                + self.config.latent_dim * self.config.hidden_dim
            ),
            similarity_macs=len(scores) * self.config.latent_dim,
            total_macs=(
                self.config.latent_dim * self.config.query_dim
                + len(scores) * (
                    self.config.hidden_dim * self.config.edge_dim
                    + self.config.latent_dim * self.config.hidden_dim
                    + self.config.latent_dim
                )
            ),
        )

    def semantic_observation(self, candidate: Candidate) -> tuple[int, ...]:
        return semantic_signature(candidate.executable_edges)

    def topology_observation(self, candidate: Candidate) -> tuple[int, ...]:
        return tuple(int(x) for x in self._candidate_x(candidate))

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

        qx = self._query_x(query)
        zq = self._linear(self.wq, self.bq, qx)
        positive_x = self._candidate_x(candidates[selected])
        positive_hidden, zp = self._encode_candidate(positive_x)

        values = [float(x) for x in scores] if scores is not None else self.score(query, candidates)
        negative_candidates = [i for i in range(len(candidates)) if i != selected]
        if not negative_candidates:
            return 0.0
        negative = max(negative_candidates, key=lambda i: (values[i], -i))
        negative_x = self._candidate_x(candidates[negative])
        negative_hidden, zn = self._encode_candidate(negative_x)

        diff = self._score(zq, zp) - self._score(zq, zn)
        clipped = max(-60.0, min(60.0, self.config.margin - diff))
        gate = 1.0 / (1.0 + math.exp(clipped))

        # Query-gradient for +gate * d(score_pos - score_neg)/d zq.
        gzq = [gate * (p - n) for p, n in zip(zp, zn)]
        lr = self.config.learning_rate
        for r in range(self.config.latent_dim):
            for j in range(self.config.query_dim):
                self.wq[r][j] += lr * gzq[r] * qx[j]
            self.bq[r] += lr * gzq[r]

        def backprop(
            x: Sequence[float],
            hidden: Sequence[float],
            grad_latent: Sequence[float],
            direction: float,
        ) -> None:
            # grad_latent is the derivative of score_pos - score_neg with
            # respect to this candidate latent, including the sign for the
            # positive/negative branch.
            old_w2 = [row[:] for row in self.w2]
            grad_hidden = [0.0] * self.config.hidden_dim
            for r in range(self.config.latent_dim):
                for h in range(self.config.hidden_dim):
                    grad_hidden[h] += (
                        old_w2[r][h] * grad_latent[r] * direction
                    )
            for r in range(self.config.latent_dim):
                for h in range(self.config.hidden_dim):
                    self.w2[r][h] += lr * direction * grad_latent[r] * hidden[h]
                self.b2[r] += lr * direction * grad_latent[r]
            for h in range(self.config.hidden_dim):
                grad_pre = grad_hidden[h] * (1.0 - hidden[h] * hidden[h])
                for j in range(self.config.edge_dim):
                    self.w1[h][j] += lr * grad_pre * x[j]
                self.b1[h] += lr * grad_pre

        backprop(positive_x, positive_hidden, zq, +gate)
        backprop(negative_x, negative_hidden, zq, -gate)

        self._updates += 1
        loss_arg = max(-60.0, min(60.0, self.config.margin - diff))
        return math.log1p(math.exp(loss_arg))


def semantic_match_rank(
    query: Query,
    candidates: Sequence[Candidate],
) -> tuple[int, float | None]:
    """Exact analytic structural matcher, used only as representability witness."""
    q = tuple(int(x) for x in query.text.partition("\t")[0].split())
    scores = []
    for candidate in candidates:
        signature = semantic_signature(candidate.executable_edges)
        signed = tuple(1.0 if x else -1.0 for x in signature)
        q_signed = tuple(1.0 if x else -1.0 for x in q)
        scores.append(sum(a * b for a, b in zip(q_signed, signed)))
    selected = max(range(len(scores)), key=lambda i: (scores[i], -i))
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    negative = next((i for i in order if i != selected), None)
    margin = None if negative is None else scores[selected] - scores[negative]
    return order.index(selected) + 1, margin
