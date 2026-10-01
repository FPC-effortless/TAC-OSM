"""Verifier-driven learning for the successor routing boundary.

The hard Top-K decision is non-differentiable. The training path therefore
uses the verifier only after execution to produce a pairwise supervised
signal. Inference can remain discrete while parameters are updated
continuously.

No verifier label is passed into route().
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from . import Candidate, Outcome, PersistentState, Query, VerificationResult
from .energy_router import RepresentationEnergyRouter

__all__ = [
    "ExecutionFeedback",
    "execution_feedback_from_verdict",
    "VerifierDrivenEnergyRouter",
]


@dataclass(frozen=True)
class ExecutionFeedback:
    verdict: str
    positive_indices: tuple[int, ...]
    negative_indices: tuple[int, ...]
    target_index: int | None
    error_type: str
    repair_key: str | None = None


def _target_index(detail: Any) -> int | None:
    value = getattr(detail, "gold_index", None)
    if isinstance(value, int):
        return value
    value = getattr(detail, "target_action", None)
    if isinstance(value, int):
        return value
    return None


def _repair_key(task_detail: Any) -> str | None:
    candidates = getattr(task_detail, "candidates", None)
    target = getattr(task_detail, "target_action", None)
    if candidates is not None and isinstance(target, int) and 0 <= target < len(candidates):
        return getattr(candidates[target], "key", None)
    return None


def execution_feedback_from_verdict(
    candidates: Sequence[Candidate],
    selected: int,
    outcome: Outcome,
    verification: VerificationResult,
) -> ExecutionFeedback:
    """Convert post-hoc execution/verifier information into labels.

    The target is read only from outcome.detail after the action has executed.
    It is never part of the router-visible Query or route() call.
    """
    target = _target_index(outcome.detail)
    accepted = bool(verification.passed and outcome.success)
    if accepted:
        negatives = tuple(i for i in range(len(candidates)) if i != selected)
        return ExecutionFeedback(
            verdict="accept",
            positive_indices=(selected,),
            negative_indices=negatives,
            target_index=target,
            error_type="",
            repair_key=_repair_key(outcome.detail),
        )
    if target is not None and 0 <= target < len(candidates):
        return ExecutionFeedback(
            verdict="repair",
            positive_indices=(target,),
            negative_indices=(selected,),
            target_index=target,
            error_type="execution_reject",
            repair_key=_repair_key(outcome.detail),
        )
    return ExecutionFeedback(
        verdict="reject",
        positive_indices=(),
        negative_indices=(selected,) if 0 <= selected < len(candidates) else (),
        target_index=None,
        error_type="unlocalized_reject",
        repair_key=None,
    )


class VerifierDrivenEnergyRouter(RepresentationEnergyRouter):
    """EnergyRouter trained from verified execution rather than teacher labels."""

    def learn_from_verifier(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
        selected: int,
        outcome: Outcome,
        verification: VerificationResult,
        *,
        scores: Sequence[float] | None = None,
    ) -> tuple[float, ExecutionFeedback]:
        feedback = execution_feedback_from_verdict(
            candidates, selected, outcome, verification
        )
        if not feedback.positive_indices or not feedback.negative_indices:
            return 0.0, feedback

        memory = self.last_memory
        if memory is None:
            memory = self.addressor.address(query, state)
        qx = self._query_input(query, memory)
        zq = self._linear(self.wq, self.bq, qx)

        score_values = (
            [float(x) for x in scores]
            if scores is not None
            else self.score(query, memory, candidates)
        )

        pos_idx = feedback.positive_indices[0]
        ordered = sorted(
            feedback.negative_indices, key=lambda i: (-score_values[i], i)
        )
        negatives = ordered[:2]
        total_loss = 0.0

        for neg_idx in negatives:
            total_loss += self._pair_update(
                qx,
                zq,
                candidates[pos_idx],
                candidates[neg_idx],
            )
        return total_loss / max(1, len(negatives)), feedback

    def _pair_update(
        self,
        qx: Sequence[float],
        zq: Sequence[float],
        positive: Candidate,
        negative: Candidate,
    ) -> float:
        cx_pos = self._candidate_input(positive)
        cx_neg = self._candidate_input(negative)
        zp = self._linear(self.wc, self.bc, cx_pos)
        zn = self._linear(self.wc, self.bc, cx_neg)

        diff = self._score(zq, zp) - self._score(zq, zn)
        gap = self.config.margin - diff
        gate = self._sigmoid(gap)
        lr = self.config.learning_rate

        delta_c = [zp[i] - zn[i] for i in range(self.config.latent_dim)]
        for r in range(self.config.latent_dim):
            dq = gate * delta_c[r]
            for j in range(self._raw_query_dim):
                self.wq[r][j] += lr * dq * qx[j]
            self.bq[r] += lr * dq

        for r in range(self.config.latent_dim):
            for j in range(self.config.input_dim):
                self.wc[r][j] += lr * gate * zq[r] * (cx_pos[j] - cx_neg[j])

        self._updates += 1
        return math.log1p(math.exp(min(60.0, gap)))
