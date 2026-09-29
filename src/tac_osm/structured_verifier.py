"""Structured verification and bounded, executable repair.

The legacy verifier mostly reduced verification to a value-range check. This
module makes epistemic evidence explicit and gives repair a real execution
boundary: failed verification returns a failed constraint and repair retries a
different candidate or computation under a fixed attempt budget.

Gold is never used by the verifier or repair controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from . import Computation, Outcome


@dataclass(frozen=True)
class VerificationEvidence:
    valid: bool
    failed_constraint: str | None = None
    counterexample: str | None = None
    repair_target: str | None = None
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.valid and self.failed_constraint is not None:
            raise ValueError("valid verification cannot name a failed constraint")


@dataclass(frozen=True)
class StructuredRepair:
    attempts: int
    passed: bool
    selected_index: int | None
    patch: str | None
    verification: VerificationEvidence


class SemanticVerifier:
    """Verifier with explicit constraints and counterexample reporting."""

    def verify(
        self,
        computation: Computation,
        outcome: Outcome,
        *,
        expected_range: tuple[float, float] = (0.0, 1.0),
    ) -> VerificationEvidence:
        lo, hi = expected_range
        observed = float(outcome.value)
        if not lo <= observed <= hi:
            return VerificationEvidence(
                valid=False,
                failed_constraint="outcome_range",
                counterexample=(
                    f"observed value {observed:.6f} outside "
                    f"[{lo:.6f},{hi:.6f}]"
                ),
                repair_target="computation.output",
                confidence=1.0,
                evidence=("observed_outcome",),
            )
        trace = tuple(float(v) for v in computation.trace)
        bad = [i for i, value in enumerate(trace) if not lo <= value <= hi]
        if bad:
            i = bad[0]
            return VerificationEvidence(
                valid=False,
                failed_constraint="trace_range",
                counterexample=(
                    f"node {i}={trace[i]:.6f} outside "
                    f"[{lo:.6f},{hi:.6f}]"
                ),
                repair_target=f"trace.node[{i}]",
                confidence=1.0,
                evidence=("executed_trace",),
            )
        return VerificationEvidence(
            valid=True,
            confidence=1.0,
            evidence=("observed_outcome", "executed_trace"),
        )


class BoundedExecutableRepair:
    """Retry execution under verification evidence, without gold access."""

    def __init__(self, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.max_attempts = max_attempts

    def repair(
        self,
        candidates: Sequence[object],
        *,
        compute: Callable[[object], Computation],
        verify: Callable[[Computation], VerificationEvidence],
        selected_index: int,
    ) -> StructuredRepair:
        attempted: set[int] = set()
        last_verification = VerificationEvidence(
            valid=False,
            failed_constraint="no_attempt",
            repair_target="candidate",
        )
        current = selected_index
        for _ in range(self.max_attempts):
            if current in attempted or not 0 <= current < len(candidates):
                break
            attempted.add(current)
            computation = compute(candidates[current])
            last_verification = verify(computation)
            if last_verification.valid:
                patch = None if current == selected_index else (
                    f"replace_candidate:{selected_index}->{current}"
                )
                return StructuredRepair(
                    attempts=len(attempted),
                    passed=True,
                    selected_index=current,
                    patch=patch,
                    verification=last_verification,
                )
            current = next(
                (
                    i for i in range(current + 1, len(candidates))
                    if i not in attempted
                ),
                -1,
            )
        return StructuredRepair(
            attempts=len(attempted),
            passed=False,
            selected_index=None,
            patch=None,
            verification=last_verification,
        )
