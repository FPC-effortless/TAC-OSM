"""Verifier-driven representation learning with the real product-key state index.

The representation learner receives post-execution verifier feedback. The
factorized index is inference-time only: it is rebuilt from the current
candidate embeddings at controlled synchronization points. This separation
lets the experiment report both learned capability and addressing/build cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import Candidate, Outcome, PersistentState, Query, RoutingDecision, VerificationResult
from .execution_feedback import VerifierDrivenEnergyRouter
from .multifactor_product_key_index import (
    MultiFactorProductKeyConfig,
    MultiFactorProductKeyStateIndex,
)


@dataclass(frozen=True)
class ProductKeyRouteDiagnostics:
    candidate_count: int
    admitted_count: int
    selected_index: int
    state_candidates_scored: int
    factor_score_macs: int
    pair_generation_ops: int
    state_rerank_macs: int
    build_total_macs: int
    refresh_count: int


class ExecutionFeedbackProductKeyRouter:
    """Dense verifier-trained representation + sparse product-key inference."""

    def __init__(
        self,
        *,
        seed: int,
        input_dim: int,
        latent_dim: int = 16,
        factor_count: int = 3,
        factor_size: int = 16,
        factor_beam: int = 7,
        max_shortlist: int = 32,
        refresh_interval_updates: int = 32,
        learning_rate: float = 0.01,
        margin: float = 0.1,
    ) -> None:
        if refresh_interval_updates < 1:
            raise ValueError("refresh_interval_updates must be positive")
        self.representation = VerifierDrivenEnergyRouter(
            config=__import__("tac_osm.energy_router", fromlist=["EnergyRouterConfig"]).EnergyRouterConfig(
                input_dim=input_dim,
                latent_dim=latent_dim,
                learning_rate=learning_rate,
                margin=margin,
                top_k=1,
                seed=seed,
            )
        )
        self.index_config = MultiFactorProductKeyConfig(
            factor_count=factor_count,
            factor_size=factor_size,
            factor_beam=factor_beam,
            iterations=8,
            max_shortlist=max_shortlist,
        )
        self.refresh_interval_updates = refresh_interval_updates
        self._index = MultiFactorProductKeyStateIndex(self.index_config)
        self._candidates: tuple[Candidate, ...] = ()
        self._candidate_by_key: dict[str, int] = {}
        self._candidate_embeddings: dict[str, tuple[float, ...]] = {}
        self._training_keys: tuple[str, ...] = ()
        self._refresh_count = 0
        self._last_update_count = 0
        self._build_macs = 0

    @property
    def updates(self) -> int:
        return self.representation.updates

    @property
    def refresh_count(self) -> int:
        return self._refresh_count

    @property
    def build_total_macs(self) -> int:
        return self._build_macs

    def sync(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
        *,
        training_keys: Sequence[str],
    ) -> None:
        items = []
        training_set = {str(x) for x in training_keys}
        if not training_set:
            raise ValueError("training_keys must not be empty")
        for candidate in candidates:
            items.append(
                (candidate.key, self.representation.encode_candidate(candidate))
            )
        training_items = [item for item in items if item[0] in training_set]
        if len(training_items) < self.index_config.factor_size:
            raise ValueError(
                "training population must contain at least factor_size unique items"
            )
        self._index = MultiFactorProductKeyStateIndex(self.index_config)
        diag = self._index.build(items, codebook_items=training_items)
        self._candidates = tuple(candidates)
        self._candidate_by_key = {
            candidate.key: i for i, candidate in enumerate(candidates)
        }
        self._candidate_embeddings = dict(items)
        self._training_keys = tuple(sorted(training_set))
        self._build_macs = diag.total_build_macs
        self._refresh_count += 1
        self._last_update_count = self.updates

    def _ensure_index(
        self, query: Query, state: PersistentState, candidates: Sequence[Candidate],
        training_keys: Sequence[str],
    ) -> None:
        changed = tuple(candidate.key for candidate in candidates) != tuple(
            candidate.key for candidate in self._candidates
        )
        stale = self.updates - self._last_update_count >= self.refresh_interval_updates
        if not self._candidates or changed or stale:
            self.sync(query, state, candidates, training_keys=training_keys)

    def route(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
        *,
        training_keys: Sequence[str],
    ) -> tuple[RoutingDecision, ProductKeyRouteDiagnostics]:
        if not candidates:
            raise ValueError("cannot route empty candidates")
        self._ensure_index(query, state, candidates, training_keys=training_keys)
        memory = self.representation.addressor.address(query, state)
        zq = self.representation.encode_query(query, memory)
        hit = self._index.lookup(
            zq,
            beam=self.index_config.factor_beam,
            max_shortlist=self.index_config.max_shortlist,
        )
        if hit.selected_address is None:
            raise RuntimeError("product-key index returned no candidate")
        selected = self._candidate_by_key[hit.selected_address]

        # Only shortlisted candidates receive a score in the routing decision.
        # This avoids an O(M) inference-time dense score just for diagnostics.
        scores = [-float("inf")] * len(candidates)
        for address in hit.candidate_addresses:
            idx = self._candidate_by_key[address]
            scores[idx] = sum(
                a * b
                for a, b in zip(zq, self._candidate_embeddings[address])
            )
        decision = RoutingDecision(
            selected=selected,
            scores=tuple(scores),
            provenance="execution_feedback_product_key_v1",
        )
        diagnostics = ProductKeyRouteDiagnostics(
            candidate_count=len(candidates),
            admitted_count=len(hit.candidate_addresses),
            selected_index=selected,
            state_candidates_scored=hit.state_candidates_scored,
            factor_score_macs=hit.factor_score_macs,
            pair_generation_ops=hit.pair_generation_ops,
            state_rerank_macs=hit.state_rerank_macs,
            build_total_macs=self._build_macs if self.refresh_count else 0,
            refresh_count=self._refresh_count,
        )
        return decision, diagnostics

    def learn_from_verifier(
        self,
        *,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
        selected: int,
        outcome: Outcome,
        verification: VerificationResult,
        scores: Sequence[float],
    ) -> None:
        self.representation.learn_from_verifier(
            query=query,
            state=state,
            candidates=candidates,
            selected=selected,
            outcome=outcome,
            verification=verification,
            scores=scores,
        )
