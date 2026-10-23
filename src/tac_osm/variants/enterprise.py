"""Read-only enterprise event-forecast contract, attach an actual trained model separately.

The historical ME073 TrajectoryModel/generate retains its frozen module path
and checkpoints. This adapter is NOT a new enterprise trained checkpoint.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping
from .contracts import ForecastEvidence, ObservationStamp, PredictionEnvelope, utc

HORIZONS = (1, 2, 4, 8)

@dataclass(frozen=True)
class EnterpriseEvent:
    stamp: ObservationStamp
    activity: str
    resource: str = ""
    def __post_init__(self):
        if not self.activity:
            raise ValueError("activity required")

class EnterpriseAdvisory:
    """Caller injects a checkpoint-loaded forecasting function; no weights trained."""
    def __init__(self, predictor: Callable[[tuple[EnterpriseEvent, ...]], Mapping[int, Any]],
                 evidence: ForecastEvidence):
        if not callable(predictor):
            raise ValueError("checkpoint-loaded predictor required")
        self.predictor, self.evidence = predictor, evidence

    def forecast(self, observed: list[EnterpriseEvent] | tuple[EnterpriseEvent, ...],
                 *, known_as_of: datetime, valid_at: datetime) -> PredictionEnvelope:
        record = tuple(observed)
        if not record or not all(isinstance(x, EnterpriseEvent) for x in record):
            raise ValueError("nonempty registered event sequence required")
        entity = record[0].stamp.entity_id
        recorded_cut, valid_cut = utc(known_as_of), utc(valid_at)
        if any(x.stamp.entity_id != entity or x.stamp.recorded_at > recorded_cut or
               x.stamp.valid_at > valid_cut for x in record):
            raise ValueError("cross-entity or lookahead enterprise record")
        raw = self.predictor(record)
        if not isinstance(raw, Mapping) or set(raw) != set(HORIZONS):
            raise ValueError("predictor must report exactly 1/2/4/8 horizons")
        return PredictionEnvelope(
            "enterprise", entity, recorded_cut,
            {"futures": dict(raw), "checkpoint": self.evidence.original_checkpoint_sha256,
             "evidence_tier": self.evidence.evidence_tier,
             "end_of_record_is_verified_business_completion": False})
