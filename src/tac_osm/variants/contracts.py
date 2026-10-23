"""Minimal strict boundary common to six TAC research regimes; no trainable code."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
import math

def utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("offset-aware datetime required")
    return value.astimezone(timezone.utc)

def finite_vector(values: tuple[float, ...] | list[float], *, width: int | None = None) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("non-empty numerical vector required")
    if width is not None and len(values) != width:
        raise ValueError("vector dimensions do not match the registered adapter")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
        raise ValueError("finite numeric inputs required")
    return tuple(float(v) for v in values)

@dataclass(frozen=True)
class ObservationStamp:
    entity_id: str
    valid_at: datetime
    recorded_at: datetime
    source_id: str
    def __post_init__(self):
        if not self.entity_id or not self.source_id:
            raise ValueError("entity_id and source_id required")
        object.__setattr__(self, "valid_at", utc(self.valid_at))
        object.__setattr__(self, "recorded_at", utc(self.recorded_at))

@dataclass(frozen=True)
class ForecastEvidence:
    """Technical evidence, not authority or validated operational utility."""
    original_checkpoint_sha256: str
    source_identity: str
    evidence_tier: str
    authentic_outcome_qualification: bool = False
    def __post_init__(self):
        if not self.source_identity:
            raise ValueError("source identity is required")
        if len(self.original_checkpoint_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.original_checkpoint_sha256
        ):
            raise ValueError("original checkpoint SHA256 required")
        if self.evidence_tier not in {
            "TRAINED_REPLAY_VERIFIED", "DEV_CAPABILITY_OBSERVED", "DEV_GAIN",
            "MIXED_TRADEOFF", "INDEPENDENT_GAIN", "TRANSFER_REPLICATED",
            "OPERATIONAL_VALUE_VERIFIED",
        }:
            raise ValueError("unknown capability evidence tier")

@dataclass(frozen=True)
class PredictionEnvelope:
    lane: str
    entity_id: str
    forecast_as_of: datetime
    payload: Mapping[str, Any] = field(default_factory=dict)
    scope: str = "advisory_only"
    def __post_init__(self):
        if self.scope != "advisory_only":
            raise ValueError("TAC research interface does not authorize execution")
        if not self.entity_id or not self.lane:
            raise ValueError("lane and entity required")
        object.__setattr__(self, "forecast_as_of", utc(self.forecast_as_of))
