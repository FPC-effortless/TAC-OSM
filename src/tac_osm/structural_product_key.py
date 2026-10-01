"""Deterministic structural product-key routing for TAC-OSM.

This is the next-phase intervention after TACOSM-FUSED-FRONTIER-001.
The learned candidate encoder is replaced by a compositional operator descriptor
for the address layer, while StructMeans and PST remain independently measured.

Pipeline:

    goal/state -> structural family gate -> product-key descriptor index
              -> structural exact rerank -> optional PST rerank

The router never receives the target candidate index. The target is used only
post-hoc for evaluation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from . import Candidate, PersistentState, Query, RoutingDecision
from .multifactor_product_key_index import (
    MultiFactorProductKeyConfig,
    MultiFactorProductKeyStateIndex,
)
from .operator_learning import StructMeans, TransitionRecord, transition_features
from .fused_architecture import KINDS, OperatorDescriptor, _goal_mask, candidate_operator


@dataclass(frozen=True)
class StructuralPKMDiagnostics:
    candidate_count: int
    admitted_count: int
    selected_index: int
    factor_score_macs: int
    pair_generation_ops: int
    internal_exact_addresses_scored: int
    internal_exact_rerank_ops: int
    structural_gate_ops: int
    pst_prediction_ops: int
    final_budget: int


@dataclass(frozen=True)
class StructuralPKMRoute:
    decision: RoutingDecision
    diagnostics: StructuralPKMDiagnostics
    admitted_indices: tuple[int, ...]
    final_indices: tuple[int, ...]
    allowed_kinds: tuple[str, ...]


def _descriptor_vector(kind: str, mask: Sequence[int]) -> tuple[float, ...]:
    return tuple(float(x) for x in OperatorDescriptor(kind, tuple(int(x) for x in mask)).encode())


def _kind_for_cluster(
    structmeans: StructMeans,
    training_records: Sequence[TransitionRecord],
) -> tuple[str, ...]:
    labels: list[str] = []
    for cluster in range(structmeans.k):
        counts: Counter[str] = Counter(
            r.operator.kind
            for r in training_records
            if r.verified and structmeans.assign(r) == cluster
        )
        labels.append(max(counts, key=lambda k: (counts[k], k)) if counts else KINDS[cluster % len(KINDS)])
    return tuple(labels)


class StructuralProductKeyRouter:
    """Factorized index over a deterministic operator descriptor."""

    def __init__(
        self,
        *,
        factor_count: int = 3,
        factor_size: int = 16,
        factor_beam: int = 7,
        max_shortlist: int = 32,
        iterations: int = 8,
    ) -> None:
        self.config = MultiFactorProductKeyConfig(
            factor_count=factor_count,
            factor_size=factor_size,
            factor_beam=factor_beam,
            iterations=iterations,
            max_shortlist=max_shortlist,
        )
        self.index = MultiFactorProductKeyStateIndex(self.config)
        self._addresses: tuple[str, ...] = ()
        self._candidate_by_key: dict[str, int] = {}
        self._descriptors: dict[str, tuple[int, ...]] = {}
        self._built_signature: tuple[str, ...] = ()
        self._family_labels: tuple[str, ...] = ()
        self._structmeans: StructMeans | None = None

    def build(
        self,
        candidates: Sequence[Candidate],
        *,
        training_records: Sequence[TransitionRecord],
        structmeans: StructMeans | None = None,
    ) -> None:
        if not candidates:
            raise ValueError("cannot build an index without candidates")
        items = [
            (candidate.key, _descriptor_vector(
                OperatorDescriptor.decode(candidate.descriptor).kind,
                OperatorDescriptor.decode(candidate.descriptor).mask,
            ))
            for candidate in candidates
        ]
        training_keys = {
            f"op" for _ in ()
        }
        # Codebook ownership is explicit: only records marked verified by the
        # supplied training history contribute codebook items.
        training_signatures = {
            r.operator.signature
            for r in training_records
            if r.verified
        }
        codebook = [
            item
            for item, candidate in zip(items, candidates)
            if candidate_operator(candidate).signature in training_signatures
        ]
        if len(codebook) < self.config.factor_size:
            raise ValueError("insufficient verified training signatures for codebook")
        self.index.build(items, codebook_items=codebook)
        self._addresses = tuple(c.key for c in candidates)
        self._candidate_by_key = {c.key: i for i, c in enumerate(candidates)}
        self._descriptors = {c.key: tuple(int(x) for x in c.descriptor) for c in candidates}
        self._built_signature = tuple(c.key for c in candidates)
        self._structmeans = structmeans
        self._family_labels = _kind_for_cluster(structmeans, training_records) if structmeans else ()

    @staticmethod
    def _query_kinds(
        *,
        structmeans: StructMeans | None,
        training_records: Sequence[TransitionRecord],
        state: Sequence[int],
        goal: Sequence[int],
        cluster_budget: int,
    ) -> tuple[str, ...]:
        if structmeans is None:
            return KINDS
        if cluster_budget < 1:
            raise ValueError("cluster_budget must be positive")
        labels = _kind_for_cluster(structmeans, training_records)
        scored: list[tuple[float, int]] = []
        for cluster, centroid in enumerate(structmeans.centroids):
            best = float("inf")
            for kind in KINDS:
                mask = _goal_mask(kind, state, goal)
                vector = transition_features(
                    TransitionRecord(
                        before=tuple(int(x) for x in state),
                        operator=__import__("tac_osm.operator_learning", fromlist=["PrimitiveOperator"]).PrimitiveOperator(kind, tuple(mask)),
                        after=tuple(int(x) for x in goal),
                        verified=True,
                        episode=-1,
                        step=-1,
                    ),
                    KINDS,
                )
                best = min(best, StructMeans._dist(vector, centroid))
            scored.append((best, cluster))
        chosen = sorted(scored, key=lambda x: (x[0], x[1]))[:cluster_budget]
        # Deduplicate while preserving cluster-distance order.
        kinds: list[str] = []
        for _, cluster in chosen:
            kind = labels[cluster]
            if kind not in kinds:
                kinds.append(kind)
        return tuple(kinds) if kinds else KINDS

    @staticmethod
    def _exact_score(
        candidate: Candidate,
        query_vectors: Sequence[tuple[float, ...]],
    ) -> float:
        desc = tuple(float(x) for x in candidate.descriptor)
        return max(
            sum(a * b for a, b in zip(desc, q))
            for q in query_vectors
        )

    def route(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
        *,
        training_records: Sequence[TransitionRecord],
        budget: int,
        cluster_budget: int = 2,
        pst=None,
    ) -> StructuralPKMRoute:
        if not candidates:
            raise ValueError("cannot route empty candidates")
        if budget < 1:
            raise ValueError("budget must be positive")
        signature = tuple(c.key for c in candidates)
        if signature != self._built_signature:
            self.build(candidates, training_records=training_records, structmeans=self._structmeans)

        raw_state = query.text.partition("\\t")[0]
        state_bits = tuple(int(x) for x in raw_state.split())
        goal = tuple(int(x) for x in query.context)
        if len(state_bits) != len(goal):
            raise ValueError("state and goal dimensions must match")

        allowed_kinds = self._query_kinds(
            structmeans=self._structmeans,
            training_records=training_records,
            state=state_bits,
            goal=goal,
            cluster_budget=cluster_budget,
        )
        query_vectors = tuple(
            _descriptor_vector(kind, _goal_mask(kind, state_bits, goal))
            for kind in allowed_kinds
        )

        admitted: dict[int, float] = {}
        factor_ops = 0
        pair_ops = 0
        exact_scored = 0
        for query_vector in query_vectors:
            hit = self.index.lookup(
                query_vector,
                beam=self.config.factor_beam,
                max_shortlist=self.config.max_shortlist,
            )
            factor_ops += hit.factor_score_macs
            pair_ops += hit.pair_generation_ops
            exact_scored += hit.state_candidates_scored
            for address in hit.candidate_addresses:
                idx = self._candidate_by_key[address]
                admitted[idx] = max(
                    admitted.get(idx, float("-inf")),
                    self._exact_score(candidates[idx], (query_vector,)),
                )

        ordered = sorted(admitted, key=lambda i: (-admitted[i], i))
        final_pool = ordered[:budget]
        pst_ops = 0

        if pst is not None and final_pool:
            ranked: list[tuple[int, float, int]] = []
            for order, idx in enumerate(final_pool):
                predicted = pst.predict(state_bits, candidate_operator(candidates[idx]))
                distance = sum(int(a != b) for a, b in zip(predicted, goal))
                ranked.append((distance, -admitted[idx], idx))
            ranked.sort()
            final_pool = [idx for _, _, idx in ranked[:budget]]
            pst_ops = len(final_pool) * (len(state_bits) * 2)

        selected = final_pool[0] if final_pool else -1
        scores = [-float("inf")] * len(candidates)
        for idx in final_pool:
            scores[idx] = admitted[idx]

        gate_ops = (
            (self._structmeans.k * 27) if self._structmeans is not None else 0
        ) + len(query_vectors) * 11

        decision = RoutingDecision(
            selected=selected,
            scores=tuple(scores),
            provenance="deterministic_structural_product_key_v1",
        )
        diagnostics = StructuralPKMDiagnostics(
            candidate_count=len(candidates),
            admitted_count=len(admitted),
            selected_index=selected,
            factor_score_macs=factor_ops,
            pair_generation_ops=pair_ops,
            internal_exact_addresses_scored=exact_scored,
            internal_exact_rerank_ops=exact_scored * len(candidates[0].descriptor),
            structural_gate_ops=gate_ops,
            pst_prediction_ops=pst_ops,
            final_budget=budget,
        )
        return StructuralPKMRoute(
            decision=decision,
            diagnostics=diagnostics,
            admitted_indices=tuple(ordered),
            final_indices=tuple(final_pool),
            allowed_kinds=allowed_kinds,
        )
