"""GP-UNC-003: constructed-only chronological residual forecast replay contract.

This is not a real-data training/evaluation runner, not a source authorizer,
and not an evaluator signature. It accepts source-claimed prior forecasts
and outcomes, performs prediction BEFORE feedback, and keeps separate
inducing-GP states by entity. The caller must prove publisher provenance,
frozen inducing basis, model checkpoint and exact source-time information.
Online assimilation of heldout labels changes the evaluation protocol:
it requires a SPECIFIC preregistered prequential adaptive regime.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from collections.abc import Mapping, Sequence

from .gp_rankone_nofit import RankOneProjectedResidualGP
from .gp_uncertainty_nofit import _vector


@dataclass(frozen=True)
class RecordedForecast:
    row_id: str
    entity_id: str
    features: tuple[float, ...]
    base_prediction: float
    truth: float
    feature_at: float
    issued_at: float
    outcome_observed_at: float
    outcome_available_at: float


@dataclass(frozen=True)
class ReplayForecast:
    row_id: str
    entity_id: str
    issued_at: float
    base_prediction: float
    corrected_prediction: float
    gp_latent_variance: float
    gp_observed_variance: float
    eligible_past_updates: int


def _validate(rows: Sequence[RecordedForecast], inducing_by_entity: Mapping) -> None:
    seen = set()
    last_issued = float("-inf")
    for row in rows:
        if not isinstance(row, RecordedForecast):
            raise ValueError("RecordedForecast rows required")
        if (not isinstance(row.row_id, str) or not row.row_id or
            row.row_id in seen or
            not isinstance(row.entity_id, str) or not row.entity_id or
            row.entity_id not in inducing_by_entity):
            raise ValueError("duplicate row or unknown entity; independently frozen inducing basis required")
        seen.add(row.row_id)
        _vector(row.features)
        times = (row.feature_at, row.issued_at, row.outcome_observed_at,
                 row.outcome_available_at, row.base_prediction, row.truth)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in times):
            raise ValueError("invalid timing, original prediction or outcome")
        if not (last_issued <= row.issued_at and
                row.feature_at <= row.issued_at < row.outcome_observed_at <=
                row.outcome_available_at):
            raise ValueError("forecast rows must be chronological and issue before outcome")
        last_issued = row.issued_at


def replay_constructed_only(rows: Sequence[RecordedForecast], inducing_by_entity: Mapping,
                            *, max_pending: int = 256, noise_variance: float = .1):
    """Compute frozen-base + post-outcome GP correction in prequential order.

    No model fit is authorized for original observational records by this
    reference; consumer must add external source/signature checks and
    registered comparison budget before any genuine-data use.
    """
    if type(max_pending) is not int or not 1 <= max_pending <= 100_000:
        raise ValueError("explicit positive bounded pending-label budget required")
    if not isinstance(inducing_by_entity, Mapping) or not inducing_by_entity:
        raise ValueError("separately frozen inducing set per entity required")
    _validate(rows, inducing_by_entity)
    models = {entity: RankOneProjectedResidualGP(inducing, noise_variance=noise_variance)
              for entity, inducing in inducing_by_entity.items()}
    pending = []  # buffering delayed observed labels is separate memory from GP state
    outputs = []
    for serial, row in enumerate(rows):
        # Release only prior outcomes legally available by the CURRENT issue time.
        while pending and pending[0][0] <= row.issued_at:
            _, _, old = heapq.heappop(pending)
            models[old.entity_id].observe(
                old.features, old.truth - old.base_prediction,
                feature_at=old.feature_at, issued_at=old.issued_at,
                label_at=old.outcome_available_at, cutoff=row.issued_at)
        model = models[row.entity_id]
        prediction = model.predict(row.features)
        outputs.append(ReplayForecast(
            row.row_id, row.entity_id, row.issued_at, row.base_prediction,
            row.base_prediction + prediction.mean,
            prediction.latent_variance, prediction.observed_variance,
            model.observations))
        heapq.heappush(pending, (row.outcome_available_at, serial, row))
        if len(pending) > max_pending:
            raise ValueError("pending labels exceed explicitly registered memory bound")
    return tuple(outputs)
