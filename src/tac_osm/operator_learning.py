"""Structured transition learning and operator consolidation.

PST learns reusable state transition laws from verified transitions.
StructMeans compresses transition instances into structural prototypes.
AXON consolidates repeated verified laws into parameterized operators.
REGM stores reconstructible verified experience.
SECA creates novel verified compositions.
SSA performs bounded operator retrieval before exact execution.

All mechanisms are dependency-free and intentionally small enough to audit.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "PrimitiveOperator",
    "TransitionRecord",
    "PSTLearner",
    "StructMeans",
    "MacroOperator",
    "AXONConsolidator",
    "FixedAXONConsolidator",
    "ExperienceStore",
    "SparseOperatorRouter",
    "SECAEngine",
    "LoopUpdate",
    "VerifiedOperatorLoop",
    "apply_operator",
]


@dataclass(frozen=True)
class PrimitiveOperator:
    kind: str
    mask: tuple[int, ...]

    @property
    def signature(self) -> tuple[str, tuple[int, ...]]:
        return self.kind, self.mask


@dataclass(frozen=True)
class TransitionRecord:
    before: tuple[int, ...]
    operator: PrimitiveOperator
    after: tuple[int, ...]
    verified: bool
    episode: int
    step: int


def apply_operator(state: Sequence[int], operator: PrimitiveOperator) -> tuple[int, ...]:
    """Ground-truth structured transition law used by the environment."""
    if len(state) != len(operator.mask):
        raise ValueError("state and operator dimensions differ")
    out = list(state)
    if operator.kind == "toggle":
        for i, active in enumerate(operator.mask):
            if active:
                out[i] = 1 - out[i]
    elif operator.kind == "set1":
        for i, active in enumerate(operator.mask):
            if active:
                out[i] = 1
    elif operator.kind == "set0":
        for i, active in enumerate(operator.mask):
            if active:
                out[i] = 0
    else:
        raise ValueError(f"unknown operator kind: {operator.kind}")
    return tuple(out)


class PSTLearner:
    """Learn per-kind/per-bit transition laws from verified examples."""

    def __init__(self, kinds: Sequence[str]) -> None:
        self.kinds = tuple(kinds)
        self._counts: dict[tuple[str, int, int], Counter[int]] = defaultdict(Counter)
        self._support: Counter[str] = Counter()

    def fit(self, records: Iterable[TransitionRecord]) -> None:
        for rec in records:
            if not rec.verified:
                continue
            if len(rec.before) != len(rec.after) or len(rec.before) != len(rec.operator.mask):
                raise ValueError("transition dimensions do not match")
            if rec.operator.kind not in self.kinds:
                raise ValueError(f"unknown operator kind: {rec.operator.kind}")
            for before, active, after in zip(
                rec.before, rec.operator.mask, rec.after
            ):
                self._counts[(rec.operator.kind, int(active), int(before))][int(after)] += 1
            self._support[rec.operator.kind] += 1

    def predict(
        self, before: Sequence[int], operator: PrimitiveOperator
    ) -> tuple[int, ...]:
        if operator.kind not in self.kinds:
            raise ValueError(f"unseen operator kind: {operator.kind}")
        if len(before) != len(operator.mask):
            raise ValueError("state and operator dimensions differ")
        out: list[int] = []
        for bit, active in zip(before, operator.mask):
            counts = self._counts.get((operator.kind, int(active), int(bit)))
            out.append(int(counts.most_common(1)[0][0]) if counts else int(bit))
        return tuple(out)

    def transition_accuracy(self, records: Iterable[TransitionRecord]) -> float:
        rows = list(records)
        if not rows:
            return 0.0
        return sum(
            self.predict(r.before, r.operator) == r.after for r in rows
        ) / len(rows)

    @property
    def support(self) -> dict[str, int]:
        return dict(self._support)


def transition_features(rec: TransitionRecord, kinds: Sequence[str]) -> tuple[float, ...]:
    # Give the typed operator identity enough weight to compete with the
    # nuisance variation in state/mask values. StructMeans is intended to
    # discover recurring structural roles, not cluster arbitrary bit patterns.
    onehot = [8.0 if rec.operator.kind == k else 0.0 for k in kinds]
    before = [float(x) for x in rec.before]
    mask = [float(x) for x in rec.operator.mask]
    after = [float(x) for x in rec.after]
    return tuple(onehot + before + mask + after)


class StructMeans:
    """K-means structural abstraction over verified transition instances."""

    def __init__(self, k: int, kinds: Sequence[str], seed: int = 0) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k
        self.kinds = tuple(kinds)
        self.seed = seed
        self.centroids: list[tuple[float, ...]] = []

    @staticmethod
    def _dist(a: Sequence[float], b: Sequence[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def fit(self, records: Sequence[TransitionRecord], iterations: int = 20) -> None:
        rows = [r for r in records if r.verified]
        if len(rows) < self.k:
            raise ValueError("insufficient verified records for k")
        vectors = [transition_features(r, self.kinds) for r in rows]
        rng = random.Random(self.seed)
        self.centroids = [tuple(v) for v in rng.sample(vectors, self.k)]
        for _ in range(iterations):
            groups: list[list[tuple[float, ...]]] = [[] for _ in range(self.k)]
            for vector in vectors:
                idx = min(range(self.k), key=lambda j: self._dist(vector, self.centroids[j]))
                groups[idx].append(vector)
            updated: list[tuple[float, ...]] = []
            for idx, group in enumerate(groups):
                if not group:
                    updated.append(self.centroids[idx])
                    continue
                updated.append(
                    tuple(
                        sum(v[i] for v in group) / len(group)
                        for i in range(len(group[0]))
                    )
                )
            self.centroids = updated

    def assign(self, record: TransitionRecord) -> int:
        if not self.centroids:
            raise RuntimeError("StructMeans is not fitted")
        vector = transition_features(record, self.kinds)
        return min(
            range(self.k),
            key=lambda j: self._dist(vector, self.centroids[j]),
        )

    def purity(self, records: Sequence[TransitionRecord]) -> float:
        rows = [r for r in records if r.verified]
        if not rows:
            return 0.0
        grouped: dict[int, Counter[str]] = defaultdict(Counter)
        for rec in rows:
            grouped[self.assign(rec)][rec.operator.kind] += 1
        correct = sum(max(counter.values()) for counter in grouped.values())
        return correct / len(rows)

    def compression_ratio(self, records: Sequence[TransitionRecord]) -> float:
        return len([r for r in records if r.verified]) / max(1, len(self.centroids))


    def signature_purity(self, records: Sequence[TransitionRecord]) -> float:
        """Cluster purity using exact operator signatures as structural labels."""
        rows = [r for r in records if r.verified]
        if not rows:
            return 0.0
        grouped: dict[int, Counter[tuple[str, tuple[int, ...]]]] = defaultdict(Counter)
        for rec in rows:
            grouped[self.assign(rec)][rec.operator.signature] += 1
        correct = sum(max(counter.values()) for counter in grouped.values())
        return correct / len(rows)


def _mask_for_goal(
    kind: str,
    state: Sequence[int],
    goal: Sequence[int],
) -> tuple[int, ...]:
    if len(state) != len(goal):
        raise ValueError("state and goal dimensions differ")
    if kind == "toggle":
        return tuple(int(a != b) for a, b in zip(state, goal))
    if kind == "set1":
        return tuple(int(a != 1 and b == 1) for a, b in zip(state, goal))
    if kind == "set0":
        return tuple(int(a != 0 and b == 0) for a, b in zip(state, goal))
    raise ValueError(f"unknown parameterized kind: {kind}")


@dataclass(frozen=True)
class MacroOperator:
    name: str
    steps: tuple[PrimitiveOperator, ...]
    support: int
    source_kind: str
    parameterized: bool = False

    def bind_to_goal(
        self, state: Sequence[int], goal: Sequence[int]
    ) -> "MacroOperator":
        if not self.parameterized:
            return self
        if len(self.steps) != 1:
            raise ValueError("only one-step parameterized macros are supported")
        step = self.steps[0]
        mask = _mask_for_goal(step.kind, state, goal)
        return MacroOperator(
            name=self.name,
            steps=(PrimitiveOperator(step.kind, mask),),
            support=self.support,
            source_kind=self.source_kind,
            parameterized=False,
        )

    def predict(
        self,
        state: Sequence[int],
        pst: PSTLearner,
        goal: Sequence[int] | None = None,
    ) -> tuple[int, ...]:
        bound = self.bind_to_goal(state, goal) if self.parameterized and goal is not None else self
        current = tuple(state)
        for op in bound.steps:
            current = pst.predict(current, op)
        return current

    def execute(
        self, state: Sequence[int], goal: Sequence[int] | None = None
    ) -> tuple[int, ...]:
        bound = self.bind_to_goal(state, goal) if self.parameterized and goal is not None else self
        current = tuple(state)
        for op in bound.steps:
            current = apply_operator(current, op)
        return current


class AXONConsolidator:
    """Turn repeated verified transition laws into reusable macros."""

    def __init__(self, min_support: int = 4) -> None:
        self.min_support = min_support

    def consolidate(self, records: Sequence[TransitionRecord]) -> tuple[MacroOperator, ...]:
        support = Counter(r.operator.kind for r in records if r.verified)
        macros: list[MacroOperator] = []
        for kind in sorted(support):
            if support[kind] < self.min_support:
                continue
            macros.append(
                MacroOperator(
                    name=f"axon:{kind}",
                    steps=(PrimitiveOperator(kind, ()),),
                    support=support[kind],
                    source_kind=kind,
                    parameterized=True,
                )
            )
        return tuple(macros)


class FixedAXONConsolidator:
    """Consolidate repeated *typed-and-bound* operators without widening them."""

    def __init__(self, min_support: int = 4) -> None:
        self.min_support = min_support

    def consolidate(self, records: Sequence[TransitionRecord]) -> tuple[MacroOperator, ...]:
        counts: Counter[tuple[str, tuple[int, ...]]] = Counter(
            (r.operator.kind, r.operator.mask)
            for r in records
            if r.verified
        )
        macros: list[MacroOperator] = []
        for (kind, mask), support in sorted(counts.items()):
            if support < self.min_support:
                continue
            macros.append(
                MacroOperator(
                    name=f"axon:{kind}:{''.join(map(str, mask))}",
                    steps=(PrimitiveOperator(kind, mask),),
                    support=support,
                    source_kind=kind,
                    parameterized=False,
                )
            )
        return tuple(macros)

class ExperienceStore:
    """REGM-like reconstructible experience.

    Only verified transitions are committed. Raw rejected trajectories can be
    retained transiently by a caller but are not part of durable competence.
    """

    def __init__(self) -> None:
        self.records: list[TransitionRecord] = []

    def append(self, record: TransitionRecord) -> bool:
        if not record.verified:
            return False
        self.records.append(record)
        return True

    def reconstruct_pst(self, kinds: Sequence[str]) -> PSTLearner:
        pst = PSTLearner(kinds)
        pst.fit(self.records)
        return pst

    @property
    def stored_records(self) -> int:
        return len(self.records)


class SparseOperatorRouter:
    """Retrieve a bounded operator set using PST-predicted next states."""

    def __init__(self, pst: PSTLearner) -> None:
        self.pst = pst

    @staticmethod
    def _distance(a: Sequence[int], b: Sequence[int]) -> int:
        return sum(int(x != y) for x, y in zip(a, b))

    def route(
        self,
        state: Sequence[int],
        goal: Sequence[int],
        operators: Sequence[MacroOperator],
        budget: int,
    ) -> tuple[tuple[MacroOperator, ...], list[tuple[MacroOperator, int]]]:
        if budget < 1:
            raise ValueError("budget must be positive")
        scored: list[tuple[MacroOperator, int]] = []
        for op in operators:
            predicted = op.predict(state, self.pst, goal)
            scored.append((op, self._distance(predicted, goal)))
        scored.sort(key=lambda pair: (pair[1], pair[0].name))
        return tuple(op for op, _ in scored[:budget]), scored


@dataclass(frozen=True)
class LoopUpdate:
    ingested: bool
    pst_accuracy: float
    structure_purity: float
    operator_count: int
    novel_operator_count: int


class VerifiedOperatorLoop:
    """One auditable experience -> operator -> verification learning loop.

    The loop deliberately separates post-execution observation from routing.
    A caller first supplies a verified TransitionRecord. That record is the
    only durable learning input. Rebuild then updates PST, StructMeans and
    AXON; SECA may propose new bound compositions and retains only those that
    agree with an independently supplied verifier/reference.
    """

    def __init__(
        self,
        kinds: Sequence[str],
        *,
        structmeans_k: int,
        structmeans_seed: int = 0,
        axon_min_support: int = 4,
    ) -> None:
        if not kinds:
            raise ValueError("kinds must not be empty")
        self.kinds = tuple(kinds)
        self.store = ExperienceStore()
        self.pst = PSTLearner(self.kinds)
        self.structmeans = StructMeans(
            structmeans_k, self.kinds, seed=structmeans_seed
        )
        self.axon = FixedAXONConsolidator(min_support=axon_min_support)
        self.seca = SECAEngine()
        self.macros: tuple[MacroOperator, ...] = ()
        self.novel_macros: tuple[MacroOperator, ...] = ()

    @property
    def library(self) -> tuple[MacroOperator, ...]:
        return self.macros + self.novel_macros

    def ingest(self, record: TransitionRecord) -> bool:
        committed = self.store.append(record)
        if committed:
            self.rebuild()
        return committed

    def ingest_many(self, records: Iterable[TransitionRecord]) -> int:
        committed = sum(self.store.append(record) for record in records)
        if committed:
            self.rebuild()
        return committed

    def rebuild(self) -> LoopUpdate:
        if not self.store.records:
            self.macros = ()
            self.novel_macros = ()
            return LoopUpdate(False, 0.0, 0.0, 0, 0)

        self.pst = self.store.reconstruct_pst(self.kinds)
        self.structmeans.fit(self.store.records)
        self.macros = self.axon.consolidate(self.store.records)
        return LoopUpdate(
            ingested=True,
            pst_accuracy=self.pst.transition_accuracy(self.store.records),
            structure_purity=self.structmeans.signature_purity(self.store.records),
            operator_count=len(self.macros),
            novel_operator_count=len(self.novel_macros),
        )

    def route(
        self,
        state: Sequence[int],
        goal: Sequence[int],
        *,
        budget: int = 1,
    ) -> tuple[tuple[MacroOperator, ...], list[tuple[MacroOperator, int]]]:
        router = SparseOperatorRouter(self.pst)
        return router.route(state, goal, self.library, budget=budget)

    def discover(
        self,
        states: Sequence[tuple[int, ...]],
        reference,
        *,
        max_pairs: int = 32,
    ) -> tuple[MacroOperator, ...]:
        proposals = self.seca.propose(self.macros, max_pairs=max_pairs)
        accepted = self.seca.verify(proposals, states, reference)
        existing = {m.name for m in self.novel_macros}
        fresh = tuple(m for m in accepted if m.name not in existing)
        self.novel_macros = self.novel_macros + fresh
        return fresh

    def step(
        self,
        record: TransitionRecord,
        *,
        discovery_states: Sequence[tuple[int, ...]] = (),
        reference=None,
    ) -> LoopUpdate:
        self.ingest(record)
        novel = ()
        if discovery_states and reference is not None:
            novel = self.discover(discovery_states, reference)
        return LoopUpdate(
            ingested=True,
            pst_accuracy=self.pst.transition_accuracy(self.store.records),
            structure_purity=self.structmeans.signature_purity(self.store.records),
            operator_count=len(self.macros),
            novel_operator_count=len(self.novel_macros),
        )

class SECAEngine:
    """Generate novel operator compositions and retain verified ones."""

    def propose(
        self,
        macros: Sequence[MacroOperator],
        *,
        max_pairs: int = 32,
    ) -> tuple[MacroOperator, ...]:
        out: list[MacroOperator] = []
        for first in macros:
            for second in macros:
                if first.name == second.name:
                    continue
                if first.parameterized or second.parameterized:
                    # SECA only composes already-bound executable structures.
                    continue
                if len(out) >= max_pairs:
                    return tuple(out)
                out.append(
                    MacroOperator(
                        name=f"seca:{first.name}+{second.name}",
                        steps=first.steps + second.steps,
                        support=first.support + second.support,
                        source_kind=f"{first.source_kind}+{second.source_kind}",
                        parameterized=False,
                    )
                )
        return tuple(out)

    def verify(
        self,
        candidates: Sequence[MacroOperator],
        states: Sequence[tuple[int, ...]],
        reference,
    ) -> tuple[MacroOperator, ...]:
        """Keep only compositions that agree with an independent reference."""
        if not callable(reference):
            raise TypeError("reference must be callable")
        accepted: list[MacroOperator] = []
        for macro in candidates:
            if all(macro.execute(state) == reference(state, macro) for state in states):
                accepted.append(macro)
        return tuple(accepted)

