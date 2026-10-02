"""Verified representation synthesis primitives.

This module defines the representation boundary used by the VRS research line.
It deliberately contains no LLM client and no benchmark-specific scoring logic.

The scientific object is a *versioned representation contract*:

    proposal -> calibration -> static validation -> dynamic validation
             -> downstream computation -> counterexample/repair signal

A representation is not admitted as persistent computational state merely
because its coordinates are parseable, compact, or semantically plausible.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Callable, Iterable, Mapping, Sequence


Scalar = float
Vector = tuple[Scalar, ...]


@dataclass(frozen=True)
class RepresentationAnchor:
    """One calibration anchor for one representation dimension."""

    name: str
    value: Scalar
    state_id: str


@dataclass(frozen=True)
class RepresentationFeature:
    """A named semantic coordinate produced by a proposal."""

    name: str
    description: str
    anchors: tuple[RepresentationAnchor, ...] = ()
    unit_interval: bool = True


@dataclass(frozen=True)
class RepresentationProposal:
    """Frozen proposal metadata and semantic feature definitions."""

    proposal_id: str
    proposer: str
    prompt_hash: str
    features: tuple[RepresentationFeature, ...]
    schema_version: str = "v1"
    decoding: Mapping[str, Any] | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposer": self.proposer,
            "prompt_hash": self.prompt_hash,
            "schema_version": self.schema_version,
            "decoding": self.decoding or {},
            "features": [
                {
                    "name": f.name,
                    "description": f.description,
                    "unit_interval": f.unit_interval,
                    "anchors": [
                        {
                            "name": a.name,
                            "value": a.value,
                            "state_id": a.state_id,
                        }
                        for a in f.anchors
                    ],
                }
                for f in self.features
            ],
        }

    def digest(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(payload).hexdigest()


@dataclass(frozen=True)
class RepresentationEvaluation:
    state_id: str
    vector: Vector


@dataclass(frozen=True)
class FixedTransitionCase:
    """Representation-independent dynamic validation case.

    Pair membership and action schedules must be created before a representation
    is scored. The representation may only classify a fixed case as 'near'.
    """

    pair_id: str
    state_a: str
    state_b: str
    actions: tuple[Any, ...]
    horizon: int
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DynamicCounterexample:
    pair_id: str
    state_a: str
    state_b: str
    action: Any
    horizon: int
    distance_before: float
    distance_after: float
    outcome_a: Any
    outcome_b: Any
    reason: str


@dataclass(frozen=True)
class DynamicValidationResult:
    considered_pairs: int
    near_pairs: int
    checked_transitions: int
    violations: tuple[DynamicCounterexample, ...]

    @property
    def violation_rate(self) -> float:
        if self.checked_transitions == 0:
            return 0.0
        return len(self.violations) / self.checked_transitions

    @property
    def passed(self) -> bool:
        return not self.violations


class RepresentationError(ValueError):
    """Invalid representation contract or validation configuration."""


class RepresentationValidator:
    """Static and dynamic checks for a frozen representation."""

    def __init__(
        self,
        encode: Callable[[str], Sequence[float]],
        *,
        distance: Callable[[Sequence[float], Sequence[float]], float] | None = None,
    ) -> None:
        self.encode = encode
        self.distance = distance or euclidean_distance

    def evaluate(self, state_ids: Iterable[str]) -> tuple[RepresentationEvaluation, ...]:
        rows = tuple(
            RepresentationEvaluation(state_id=sid, vector=tuple(map(float, self.encode(sid))))
            for sid in state_ids
        )
        for row in rows:
            if not row.vector:
                raise RepresentationError(f"empty representation for {row.state_id}")
            if any(not _finite(x) for x in row.vector):
                raise RepresentationError(f"non-finite representation for {row.state_id}")
        dims = {len(r.vector) for r in rows}
        if len(dims) > 1:
            raise RepresentationError(f"inconsistent representation dimensions: {sorted(dims)}")
        return rows

    def check_determinism(self, state_ids: Iterable[str]) -> None:
        for sid in state_ids:
            first = tuple(map(float, self.encode(sid)))
            second = tuple(map(float, self.encode(sid)))
            if first != second:
                raise RepresentationError(f"non-deterministic representation for {sid}")

    def check_anchors(
        self,
        proposal: RepresentationProposal,
        *,
        tolerance: float = 1e-6,
    ) -> None:
        if tolerance < 0:
            raise RepresentationError("anchor tolerance must be non-negative")
        for feature_index, feature in enumerate(proposal.features):
            for anchor in feature.anchors:
                vector = tuple(map(float, self.encode(anchor.state_id)))
                if feature_index >= len(vector):
                    raise RepresentationError(
                        f"anchor feature index {feature_index} missing for {feature.name}"
                    )
                if abs(vector[feature_index] - anchor.value) > tolerance:
                    raise RepresentationError(
                        f"anchor mismatch for {feature.name}/{anchor.name}: "
                        f"{vector[feature_index]} != {anchor.value}"
                    )

    def validate_dynamic(
        self,
        cases: Sequence[FixedTransitionCase],
        *,
        transition: Callable[[str, Any], str],
        outcome: Callable[[str, Any], Any],
        near_threshold: float,
        successor_tolerance: float,
        action_extractor: Callable[[FixedTransitionCase], Iterable[Any]] | None = None,
    ) -> DynamicValidationResult:
        if near_threshold < 0 or successor_tolerance < 0:
            raise RepresentationError("dynamic thresholds must be non-negative")
        violations: list[DynamicCounterexample] = []
        considered = len(cases)
        near_count = 0
        checked = 0

        for case in cases:
            va = tuple(map(float, self.encode(case.state_a)))
            vb = tuple(map(float, self.encode(case.state_b)))
            before = self.distance(va, vb)
            if before > near_threshold:
                continue
            near_count += 1
            actions = tuple(
                action_extractor(case)
                if action_extractor is not None
                else case.actions
            )
            if case.horizon != 0 and not actions:
                raise RepresentationError(f"no actions for nonzero horizon: {case.pair_id}")

            current_a, current_b = case.state_a, case.state_b
            for action in actions:
                checked += 1
                next_a = transition(current_a, action)
                next_b = transition(current_b, action)
                after = self.distance(
                    tuple(map(float, self.encode(next_a))),
                    tuple(map(float, self.encode(next_b))),
                )
                oa = outcome(current_a, action)
                ob = outcome(current_b, action)
                if after > successor_tolerance or oa != ob:
                    violations.append(
                        DynamicCounterexample(
                            pair_id=case.pair_id,
                            state_a=current_a,
                            state_b=current_b,
                            action=action,
                            horizon=case.horizon,
                            distance_before=before,
                            distance_after=after,
                            outcome_a=oa,
                            outcome_b=ob,
                            reason="future-behavior divergence",
                        )
                    )
                current_a, current_b = next_a, next_b

        return DynamicValidationResult(
            considered_pairs=considered,
            near_pairs=near_count,
            checked_transitions=checked,
            violations=tuple(violations),
        )


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise RepresentationError("distance vectors must have equal dimension")
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5


def _finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


__all__ = [
    "DynamicCounterexample",
    "DynamicValidationResult",
    "FixedTransitionCase",
    "RepresentationAnchor",
    "RepresentationError",
    "RepresentationEvaluation",
    "RepresentationFeature",
    "RepresentationProposal",
    "RepresentationValidator",
    "euclidean_distance",
]
