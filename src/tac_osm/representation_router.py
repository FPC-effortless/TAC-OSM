"""R1 relevance-router boundary: representations are separate from policy.

The v0 router couples feature construction, scoring, and REINFORCE updates into
one hand-designed linear model. This module introduces the next architectural
boundary without claiming a new capability result:

    public query + persistent state -> query representation
    candidate + persistent state   -> candidate representation
    similarity(z_query, z_candidate) -> routing score

An implementation can later be backed by CDL distillation, a PLM encoder, or a
small learned model. The runtime call graph does not need to change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from . import Candidate, PersistentState, Query, RoutingDecision


QueryEncoder = Callable[[Query, PersistentState], Sequence[float]]
CandidateEncoder = Callable[[Candidate, PersistentState], Sequence[float]]


@dataclass(frozen=True)
class RepresentationRouterConfig:
    """Runtime choices for the representation-based router."""

    normalize: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.normalize, bool):
            raise TypeError("normalize must be bool")


class RepresentationSimilarityRouter:
    """R1 router using externally supplied representations and dot/cosine score."""

    def __init__(
        self,
        query_encoder: QueryEncoder,
        candidate_encoder: CandidateEncoder,
        *,
        config: RepresentationRouterConfig | None = None,
    ) -> None:
        self.query_encoder = query_encoder
        self.candidate_encoder = candidate_encoder
        self.config = config if config is not None else RepresentationRouterConfig()

    @staticmethod
    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b):
            raise ValueError(
                f"representation dimensions differ: {len(a)} != {len(b)}"
            )
        return sum(float(x) * float(y) for x, y in zip(a, b))

    @staticmethod
    def _norm(a: Sequence[float]) -> float:
        return math.sqrt(sum(float(x) * float(x) for x in a))

    def _score(self, q: Sequence[float], c: Sequence[float]) -> float:
        score = self._dot(q, c)
        if not self.config.normalize:
            return score
        nq, nc = self._norm(q), self._norm(c)
        if nq == 0.0 or nc == 0.0:
            return 0.0
        return score / (nq * nc)

    def route(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
    ) -> RoutingDecision:
        if not candidates:
            raise ValueError("cannot route an empty candidate set")
        zq = tuple(float(x) for x in self.query_encoder(query, state))
        scores = tuple(
            self._score(
                zq,
                tuple(float(x) for x in self.candidate_encoder(candidate, state)),
            )
            for candidate in candidates
        )
        selected = max(range(len(candidates)), key=lambda i: (scores[i], -i))
        return RoutingDecision(
            selected=selected,
            scores=scores,
            provenance="representation_similarity_r1",
        )
