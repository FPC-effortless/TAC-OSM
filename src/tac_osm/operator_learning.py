"""Structured transition learning and operator consolidation.

This module is the next research layer above the TAC-OSM execution loop.
It intentionally uses small, dependency-free learners so that failures are
attributable to the mechanism under test rather than a large optimizer.

PST learns reusable state transition laws from verified transitions.
StructMeans compresses transition instances into structural prototypes.
AXON consolidates repeated verified laws into reusable operators.
REGM stores only reconstructible verified experience.
SECA composes existing operators and keeps only execution-verified novel
operators.
SSA provides a bounded operator shortlist before exact execution.
"""
from __future__ import annotations

import math
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
    "ExperienceStore",
    "SparseOperatorRouter",
    "SECAEngine",
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
    """Ground-truth structured transition law used only by the environment."""
    out = list(state)
    kind = operator.kind
    for i, active in enumerate(operator.mask):
        if not active:
            continue
        if kind == "toggle":
            out[i] = 1 - out[i]
        elif kind == "set1":
            out[i] = 1
        elif kind == "set0":
            out[i] = 0
        elif kind == "swap":
            # swap is interpreted on consecutive active pairs.
            pass
        else:
            raise ValueError(f"unknown operator kind: {kind}")
    if kind == "swap":
        active = [i for i, bit in enumerate(operator.mask) if bit]
        for a, b in zip(active[::2], active[1::2]):
            out[a], out[b] = out[b], out[a]
    return tuple(out)


class PSTLearner:
    """Learn per-kind, per-bit transition laws from verified examples."""

    def __init__(self, kinds: Sequence[str]) -> None:
        self.kinds = tuple(kinds)
        self._counts: dict[tuple[str, int, int], Counter[int]] = defaultdict(Counter)
        self._support = Counter()

    def fit(self, records: Iterable[TransitionRecord]) -> None:
        for rec in records:
            if not rec.verified:
                continue
            if len(rec.before) != len(rec.after) or len(rec.before) != len(rec.operator.mask):
                raise ValueError("transition dimensions do not match")
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
        out: list[int] = []
        for i, bit in enumerate(before):
            key = (operator.kind, int(operator.mask[i]), int(bit))
            counts = self._counts.get(key)
            if not counts:
                out.append(int(bit))
            else:
                out.append(int(counts.most_common(1)[0][0]))
        return tuple(out)

    def transition_accuracy(
        self, records: Iterable[TransitionRecord]
    ) -> float:
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
    """Numeric structural signature used by StructMeans."""
    onehot = [1.0 if rec.operator.kind == k else 0.0 for k in kinds]
    before = [float(x) for x in rec.before]
    mask = [float(x) for x in rec.operator.mask]
    after = [float(x) for x in rec.after]
    return tuple(onehot + before + mask + after)


class StructMeans:
    """Small k-means compressor over transition structure."""

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
        if not records:
            raise ValueError("StructMeans needs records")
        vectors = [transition_features(r, self.kinds) for r in records]
        rng = random.Random(self.seed)
        if len(vectors) < self.k:
            raise ValueError("k exceeds number of transition examples")
        seeds = rng.sample(vectors, self.k)
        self.centroids = [tuple(v) for v in seeds]
        for _ in range(iterations):
            groups: list[list[tuple[float, ...]]] = [[] for _ in range(self.k)]
            for v in vectors:
                idx = min(range(self.k), key=lambda j: self._dist(v, self.centroids[j]))
                groups[idx].append(v)
            updated: list[tuple[float, ...]] = []
            for j, group in enumerate(groups):
                if not group:
                    updated.append(self.centroids[j])
                    continue
                updated.append(
                    tuple(sum(v[i] for v in group) / len(group) for i in range(len(group[0])))
                )
            self.centroids = updated

    def assign(self, record: TransitionRecord) -> int:
        if not self.centroids:
            raise RuntimeError("StructMeans is not fitted")
        v = transition_features(record, self.kinds)
        return min(range(self.k), key=lambda j: self._dist(v, self.centroids[j]))

    def purity(self, records: Sequence[TransitionRecord]) -> float:
        if not records:
            return 0.0
        grouped: dict[int, Counter[str]] = defaultdict(Counter)
        for r in records:
            grouped[self.assign(r)][r.operator.kind] += 1
        correct = sum(max(c.values()) for c in grouped.values())
        return correct / len(records)

    def compression_ratio(self, records: Sequence[TransitionRecord]) -> float:
        return len(records) / max(1, len(self.centroids))


@dataclass(frozen=True)
class MacroOperator:
    name: str
    steps: tuple[PrimitiveOperator, ...]
    support: int
    source_kind: str

    def predict(self, state: Sequence[int], pst: PSTLearner) -> tuple[int, ...]:
        current = tuple(state)
        for op in self.steps:
            current = pst.predict(current, op)
        return current

    def execute(self, state: Sequence[int]) -> tuple[int, ...]:
        current = tuple(state)
        for op in self.steps:
            current = apply_operator(current, op)
        return current


class AXONConsolidator:
    """Consolidate repeated verified primitive laws into reusable macros."""

    def __init__(self, min_support: int = 4) -> None:
        self.min_support = min_support

    def consolidate(
        self, records: Sequence[TransitionRecord], pst: PSTLearner
    ) -> tuple[MacroOperator, ...]:
        by_kind: dict[str, list[PrimitiveOperator]] = defaultdict(list)
        for r in records:
            if r.verified:
                by_kind[r.operator.kind].append(r.operator)
        macros: list[MacroOperator] = []
        for kind, ops in sorted(by_kind.items()):
            if len(ops) < self.min_support:
                continue
            representatives: dict[tuple[int, ...], int] = Counter(
                op.mask for op in ops
            )
            mask = max(representatives, key=representatives.get)
            macros.append(
                MacroOperator(
                    name=f"axon:{kind}",
                    steps=(PrimitiveOperator(kind, mask),),
                    support=len(ops),
                    source_kind=kind,
                )
            )
        return tuple(macros)


@dataclass
class ExperienceStore:
    """REGM-like reconstructible memory; unverified experience is not committed."""

    records: list[TransitionRecord]

    def __init__(self) -> None:
        self.records = []

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
    def verified_fraction(self) -> float:
        return 1.0 if self.records else 0.0


class SparseOperatorRouter:
    """Retrieve a bounded subset of operators by predicted state distance."""

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
            pred = op.predict(state, self.pst)
            scored.append((op, self._distance(pred, goal)))
        scored.sort(key=lambda x: (x[1], x[0].name))
        return tuple(op for op, _ in scored[:budget]), scored


class SECAEngine:
    """Create and retain novel verified compositions of existing operators."""

    def propose(
        self,
        macros: Sequence[MacroOperator],
        *,
        max_pairs: int = 16,
    ) -> tuple[MacroOperator, ...]:
        out: list[MacroOperator] = []
        for a in macros:
            for b in macros:
                if a.name == b.name:
                    continue
                out.append(
                    MacroOperator(
                        name=f"seca:{a.name}+{b.name}",
                        steps=a.steps + b.steps,
                        support=a.support + b.support,
                        source_kind=f"{a.source_kind}+{b.source_kind}",
                    )
                )
                if len(out) >= max_pairs:
                    return tuple(out)
        return tuple(out)

    def verify(
        self,
        candidates: Sequence[MacroOperator],
        states: Sequence[tuple[int, ...]],
    ) -> tuple[MacroOperator, ...]:
        accepted: list[MacroOperator] = []
        for macro in candidates:
            if not states:
                continue
            ok = all(
                macro.execute(state)
                == self._compose_ground_truth(state, macro.steps)
                for state in states
            )
            if ok:
                accepted.append(macro)
        return tuple(accepted)

    @staticmethod
    def _compose_ground_truth(
        state: Sequence[int], steps: Sequence[PrimitiveOperator]
    ) -> tuple[int, ...]:
        current = tuple(state)
        for op in steps:
            current = apply_operator(current, op)
        return current
