"""Fused TAC-OSM research architecture: sparse addressing + structural admission +
predictive transition routing + verified operator consolidation.

Research path:

    product-key -> StructMeans -> PST -> AXON/REGM -> SECA -> verifier

The product-key stage is an address generator, not proof of bounded routing.
Its internal candidate-rerank work is reported separately from the final fixed
absolute budget B used by StructMeans/PST/execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import Candidate, Query, PersistentState, RoutingDecision, Outcome, VerificationResult
from .execution_product_key import ExecutionFeedbackProductKeyRouter, ProductKeyRouteDiagnostics
from .operator_learning import (
    ExperienceStore,
    MacroOperator,
    PrimitiveOperator,
    PSTLearner,
    StructMeans,
    TransitionRecord,
    apply_operator,
)

KINDS = ("toggle", "set1", "set0")


@dataclass(frozen=True)
class OperatorDescriptor:
    """Lossless encoding used as a Candidate descriptor."""

    kind: str
    mask: tuple[int, ...]

    def encode(self) -> tuple[int, ...]:
        if self.kind not in KINDS:
            raise ValueError(f"unknown operator kind: {self.kind}")
        if any(int(x) not in (0, 1) for x in self.mask):
            raise ValueError("operator mask must be binary")
        return tuple(int(self.kind == k) for k in KINDS) + tuple(int(x) for x in self.mask)

    @classmethod
    def decode(cls, descriptor: Sequence[int]) -> "OperatorDescriptor":
        values = tuple(int(x) for x in descriptor)
        if len(values) != 11:
            raise ValueError(f"expected 11 descriptor values, got {len(values)}")
        kind_idx = max(range(3), key=lambda i: (values[i], -i))
        kind = KINDS[kind_idx]
        mask = tuple(values[3:])
        if any(x not in (0, 1) for x in mask):
            raise ValueError("decoded mask is not binary")
        return cls(kind, mask)


def candidate_operator(candidate: Candidate) -> PrimitiveOperator:
    desc = OperatorDescriptor.decode(candidate.descriptor)
    return PrimitiveOperator(desc.kind, desc.mask)


def candidate_record(candidate: Candidate, *, episode: int = -1) -> TransitionRecord:
    """Canonical structural witness for one bound operator."""
    op = candidate_operator(candidate)
    before = tuple(0 for _ in op.mask)
    after = apply_operator(before, op)
    return TransitionRecord(
        before=before,
        operator=op,
        after=after,
        verified=True,
        episode=episode,
        step=0,
    )


@dataclass(frozen=True)
class StructuralAdmission:
    candidates: tuple[Candidate, ...]
    cluster_scores: tuple[float, ...]
    selected_clusters: tuple[int, ...]
    structural_ops: int


def _goal_mask(kind: str, state: Sequence[int], goal: Sequence[int]) -> tuple[int, ...]:
    if kind == "toggle":
        return tuple(int(a != b) for a, b in zip(state, goal))
    if kind == "set1":
        return tuple(int(a != 1 and b == 1) for a, b in zip(state, goal))
    if kind == "set0":
        return tuple(int(a != 0 and b == 0) for a, b in zip(state, goal))
    raise ValueError(kind)


def structural_query_scores(
    structmeans: StructMeans,
    state: Sequence[int],
    goal: Sequence[int],
) -> tuple[float, ...]:
    """Distance from the desired transition to each StructMeans centroid."""
    if not structmeans.centroids:
        raise RuntimeError("StructMeans is not fitted")
    scores: list[float] = []
    for centroid in structmeans.centroids:
        best = float("inf")
        for kind in KINDS:
            mask = _goal_mask(kind, state, goal)
            vector = tuple(
                [8.0 if kind == k else 0.0 for k in KINDS]
                + [float(x) for x in state]
                + [float(x) for x in mask]
                + [float(x) for x in goal]
            )
            best = min(best, StructMeans._dist(vector, centroid))
        scores.append(best)
    return tuple(scores)


def structural_admit(
    candidates: Sequence[Candidate],
    pkm_indices: Sequence[int],
    *,
    structmeans: StructMeans,
    cluster_by_index: Sequence[int],
    state: Sequence[int],
    goal: Sequence[int],
    budget: int,
    cluster_budget: int = 2,
) -> StructuralAdmission:
    if budget < 1:
        raise ValueError("budget must be positive")
    if cluster_budget < 1:
        raise ValueError("cluster_budget must be positive")
    if not pkm_indices:
        return StructuralAdmission((), (), (), len(structmeans.centroids) * 27)

    cluster_scores = structural_query_scores(structmeans, state, goal)
    ordered_clusters = sorted(range(len(cluster_scores)), key=lambda j: (cluster_scores[j], j))
    selected_clusters = tuple(ordered_clusters[:min(cluster_budget, len(ordered_clusters))])
    selected = set(selected_clusters)

    eligible = [i for i in pkm_indices if cluster_by_index[i] in selected]
    chosen = tuple(eligible[:budget])
    structural_ops = len(cluster_scores) * 27 + len(pkm_indices)
    return StructuralAdmission(
        candidates=tuple(candidates[i] for i in chosen),
        cluster_scores=cluster_scores,
        selected_clusters=selected_clusters,
        structural_ops=structural_ops,
    )


def pst_rerank(
    *,
    candidates: Sequence[Candidate],
    state: Sequence[int],
    goal: Sequence[int],
    pst: PSTLearner,
    budget: int,
) -> tuple[tuple[Candidate, ...], int]:
    """Rank a bounded set by predicted next-state distance."""
    if budget < 1:
        raise ValueError("budget must be positive")
    rows: list[tuple[int, int, str]] = []
    for order, candidate in enumerate(candidates):
        op = candidate_operator(candidate)
        predicted = pst.predict(state, op)
        distance = sum(int(a != b) for a, b in zip(predicted, goal))
        rows.append((distance, order, candidate.key))
    rows.sort()
    selected = tuple(candidates[order] for _, order, _ in rows[:budget])
    prediction_ops = min(budget, len(candidates)) * (len(state) * 2)
    return selected, prediction_ops


@dataclass
class VerifiedOperatorMemory:
    """REGM-like store for arbitrary executable macros."""

    records: list[tuple[MacroOperator, tuple[int, ...], tuple[int, ...], str]]

    def __init__(self) -> None:
        self.records = []

    def append(
        self,
        macro: MacroOperator,
        state: Sequence[int],
        goal: Sequence[int],
        evidence: str,
    ) -> None:
        self.records.append((macro, tuple(state), tuple(goal), evidence))

    @property
    def operators(self) -> tuple[MacroOperator, ...]:
        seen: dict[str, MacroOperator] = {}
        for macro, _, _, _ in self.records:
            seen[macro.name] = macro
        return tuple(seen.values())

    @property
    def verified_count(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class FusedRoute:
    pkm: RoutingDecision
    pkm_diag: ProductKeyRouteDiagnostics
    pkm_indices: tuple[int, ...]
    structural: StructuralAdmission | None
    final_candidates: tuple[Candidate, ...]
    pkm_selected_candidate: Candidate | None
    final_selected_candidate: Candidate | None
    pst_prediction_ops: int
    exact_selected_candidate: Candidate | None
    mode: str


class FusedOperatorRouter:
    """Hierarchical inference router.

    Modes:
      pkm    = product-key admission then fixed absolute budget
      struct = product-key admission -> StructMeans -> exact choice
      pst    = product-key admission -> StructMeans -> PST rerank

    The route budget is absolute and independent of population size M.
    """

    def __init__(
        self,
        pkm: ExecutionFeedbackProductKeyRouter,
        *,
        structmeans: StructMeans,
        cluster_by_index: Sequence[int],
        pst: PSTLearner,
        budget: int,
        cluster_budget: int = 2,
        mode: str = "pst",
    ) -> None:
        if mode not in {"pkm", "struct", "pst"}:
            raise ValueError("mode must be pkm, struct, or pst")
        if budget < 1:
            raise ValueError("budget must be positive")
        self.pkm = pkm
        self.structmeans = structmeans
        self.cluster_by_index = tuple(cluster_by_index)
        self.pst = pst
        self.budget = budget
        self.cluster_budget = cluster_budget
        self.mode = mode

    def route(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
        *,
        training_keys: Sequence[str],
    ) -> FusedRoute:
        pkm_decision, pkm_diag = self.pkm.route(
            query, state, candidates, training_keys=training_keys
        )
        pkm_indices = tuple(
            i
            for i, _ in sorted(
                ((i, score) for i, score in enumerate(pkm_decision.scores) if score != float("-inf")),
                key=lambda pair: (-pair[1], pair[0]),
            )
        )
        pkm_selected = candidates[pkm_decision.selected] if pkm_decision.selected >= 0 else None

        state_bits = _query_bits(query)
        goal = tuple(int(x) for x in query.context)
        if len(state_bits) < len(goal):
            state_bits = tuple(state_bits) + (0,) * (len(goal) - len(state_bits))
        state_bits = tuple(state_bits[: len(goal)])

        structural = None
        if self.mode in {"struct", "pst"}:
            structural = structural_admit(
                candidates,
                pkm_indices,
                structmeans=self.structmeans,
                cluster_by_index=self.cluster_by_index,
                state=state_bits,
                goal=goal,
                budget=self.budget,
                cluster_budget=self.cluster_budget,
            )
            pool = structural.candidates
        else:
            pool = tuple(candidates[i] for i in pkm_indices[: self.budget])

        pst_ops = 0
        if self.mode == "pst":
            final, pst_ops = pst_rerank(
                candidates=pool,
                state=state_bits,
                goal=goal,
                pst=self.pst,
                budget=self.budget,
            )
        else:
            final = pool

        selected = final[0] if final else None
        return FusedRoute(
            pkm=pkm_decision,
            pkm_diag=pkm_diag,
            pkm_indices=pkm_indices,
            structural=structural,
            final_candidates=tuple(final),
            pkm_selected_candidate=pkm_selected,
            final_selected_candidate=selected,
            pst_prediction_ops=pst_ops,
            exact_selected_candidate=selected,
            mode=self.mode,
        )


def _query_bits(query: Query) -> tuple[int, ...]:
    raw = query.text.partition("\\t")[0]
    if not raw.strip():
        return ()
    return tuple(int(x) for x in raw.split())


@dataclass(frozen=True)
class VerifiedExecution:
    candidate: Candidate | None
    operator: MacroOperator | PrimitiveOperator | None
    outcome: Outcome
    verification: VerificationResult


def execute_and_verify(
    candidate: Candidate,
    *,
    state: Sequence[int],
    goal: Sequence[int],
) -> VerifiedExecution:
    primitive = candidate_operator(candidate)
    output = apply_operator(state, primitive)
    ok = tuple(output) == tuple(goal)
    outcome = Outcome(
        success=ok,
        value=float(ok),
        feedback="success" if ok else "wrong_transition",
        detail={"target": tuple(goal)},
    )
    verification = VerificationResult(
        passed=ok,
        feedback="accept" if ok else "reject",
    )
    return VerifiedExecution(candidate, primitive, outcome, verification)


def macro_from_candidate(candidate: Candidate) -> MacroOperator:
    primitive = candidate_operator(candidate)
    return MacroOperator(
        name=f"axon-bound:{candidate.key}",
        steps=(primitive,),
        support=1,
        source_kind=primitive.kind,
        parameterized=False,
    )


def make_composite(
    first: MacroOperator,
    second: MacroOperator,
    *,
    name_prefix: str = "seca",
) -> MacroOperator:
    if first.parameterized or second.parameterized:
        raise ValueError("SECA composition requires bound executable operators")
    return MacroOperator(
        name=f"{name_prefix}:{first.name}+{second.name}",
        steps=first.steps + second.steps,
        support=first.support + second.support,
        source_kind=f"{first.source_kind}+{second.source_kind}",
        parameterized=False,
    )


def verify_macro(
    macro: MacroOperator,
    *,
    state: Sequence[int],
    goal: Sequence[int],
) -> VerificationResult:
    predicted = macro.execute(state)
    ok = tuple(predicted) == tuple(goal)
    return VerificationResult(passed=ok, feedback="accept" if ok else "reject")


def record_verified_primitive(
    store: ExperienceStore,
    candidate: Candidate,
    state: Sequence[int],
    goal: Sequence[int],
    *,
    episode: int,
    step: int,
) -> bool:
    primitive = candidate_operator(candidate)
    return store.append(
        TransitionRecord(
            before=tuple(state),
            operator=primitive,
            after=tuple(goal),
            verified=True,
            episode=episode,
            step=step,
        )
    )


def consolidate_axon(
    store: ExperienceStore,
    *,
    min_support: int = 2,
) -> tuple[MacroOperator, ...]:
    from .operator_learning import FixedAXONConsolidator
    return FixedAXONConsolidator(min_support=min_support).consolidate(store.records)


def build_structmeans_and_pst(
    records: Sequence[TransitionRecord],
    *,
    seed: int = 0,
    clusters: int = 3,
) -> tuple[StructMeans, PSTLearner]:
    if not records:
        raise ValueError("records cannot be empty")
    sm = StructMeans(clusters, KINDS, seed=seed)
    sm.fit(records)
    pst = PSTLearner(KINDS)
    pst.fit(records)
    return sm, pst
