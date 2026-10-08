"""Post-action verification, conservative writes, and bounded repair signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .. import Candidate, Outcome, Query
from .interfaces import ModuleExecution


@dataclass(frozen=True)
class FusionVerifierConfig:
    mode: str = "path"
    require_positive_outcome_for_commit: bool = True
    path_tolerance: float = 1.0
    repair_attempts: int = 2

    def __post_init__(self) -> None:
        if self.mode not in {"none", "final", "path"}:
            raise ValueError(f"unknown verifier mode {self.mode!r}")
        if self.path_tolerance < 0.0:
            raise ValueError("path_tolerance must be >= 0")
        if self.repair_attempts < 0:
            raise ValueError("repair_attempts must be >= 0")


class FusionVerifier:
    """Verifier that cannot turn a failed action into authoritative memory."""

    def __init__(self, config: FusionVerifierConfig | None = None) -> None:
        self.config = config if config is not None else FusionVerifierConfig()

    def verify(
        self,
        *,
        execution: Sequence[ModuleExecution],
        selected: int,
        candidates: Sequence[Candidate],
        outcome: Outcome,
    ) -> tuple[bool, str]:
        if self.config.mode == "none":
            return True, "no_verifier"

        if not outcome.success and self.config.require_positive_outcome_for_commit:
            return False, "negative_outcome_not_authoritative"

        if not execution:
            return False, "no_execution_trace"

        # Module consensus is evidence, not a gold label. Require the selected
        # candidate to be the selected maximum of every participating module.
        for e in execution:
            if selected >= len(e.candidate_scores):
                return False, "selected_index_out_of_range"
            top = max(range(len(e.candidate_scores)), key=lambda i: (e.candidate_scores[i], -i))
            if top != selected:
                return False, f"module_{e.module_id}_disagrees"

        if self.config.mode == "final":
            return True, "final_consensus_ok"

        for e in execution:
            if e.trace:
                output = float(e.trace[0])
                if not 0.0 <= (output + 1.0) / 2.0 <= 1.0:
                    return False, f"module_{e.module_id}_output_range"
            if any(abs(float(v)) > self.config.path_tolerance for v in e.trace):
                return False, f"module_{e.module_id}_path_range"
        return True, "path_consensus_ok"

    def commit_allowed(self, *, passed: bool, outcome: Outcome) -> bool:
        if self.config.mode == "none":
            return True
        if not passed:
            return False
        if self.config.require_positive_outcome_for_commit:
            return bool(outcome.success)
        return True


@dataclass
class RepairPolicy:
    """Bounded post-verification repair signal.

    Repair never gets a gold index. It records a control action that changes
    future exploration/priming instead of retroactively rewriting the outcome.
    """

    enabled: bool = True
    max_attempts: int = 2

    def propose(self, *, passed: bool, feedback: str) -> str | None:
        if not self.enabled or passed or self.max_attempts < 1:
            return None
        return f"repair_next_step:increase_exploration;reason={feedback}"
