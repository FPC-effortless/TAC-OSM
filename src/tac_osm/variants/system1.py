"""Immediate prediction adapter, NOT a trusted safety/execution controller.

Latency numbers, immediate-action value, and allowed deployment require
independent real hardware and policy evidence; ME074 only diagnosed readouts.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from time import perf_counter_ns
import math
from .contracts import ObservationStamp, finite_vector, PredictionEnvelope

@dataclass(frozen=True)
class ReactiveResult:
    proposed_risk: float
    forecast: PredictionEnvelope
    measured_call_ns: int
    def __post_init__(self):
        if not 0 <= self.proposed_risk <= 1 or not math.isfinite(self.proposed_risk):
            raise ValueError("risk must be a finite probability")
        if self.measured_call_ns < 0:
            raise ValueError("invalid measured duration")

class ReactiveAdvisory:
    """Inferred one-step risk, never ALLOW/ABORT authority or a latency guarantee."""
    def __init__(self, scorer: Callable[[tuple[float, ...]], float], feature_width: int):
        if not callable(scorer) or feature_width < 1:
            raise ValueError("a scorer and positive dimension are required")
        self.scorer, self.width = scorer, feature_width

    def forecast(self, stamp: ObservationStamp, features: list[float] | tuple[float, ...],
                 *, known_as_of) -> ReactiveResult:
        from .contracts import utc
        if utc(known_as_of) < stamp.recorded_at:
            raise ValueError("cannot forecast using an observation not yet known")
        values = finite_vector(features, width=self.width)
        t = perf_counter_ns()
        risk = float(self.scorer(values))
        elapsed = perf_counter_ns() - t
        if not math.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("scorer returned invalid risk")
        return ReactiveResult(risk, PredictionEnvelope(
            "system1", stamp.entity_id, known_as_of,
            {"risk_estimate": risk, "verified_policy_permission": False,
             "hardware_sla_qualified": False}), elapsed)
