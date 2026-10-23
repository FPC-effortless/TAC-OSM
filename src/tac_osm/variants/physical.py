"""Physical state and causal-action adapter with explicit persistence/servo fallback.

These adapters do not retrain ME06x weights and do not prove neural corrections
better than the physical baseline on authentic external transfer.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from .contracts import finite_vector

@dataclass(frozen=True)
class PhysicalForecast:
    states: tuple[tuple[float, ...], ...]
    method: str
    capability_qualified: bool = False
    def __post_init__(self):
        if self.method not in {"persistence", "servo", "qualified_neural_residual"}:
            raise ValueError("method must identify actual predictor")

class PhysicalAdvisory:
    def __init__(self, *, dimension: int,
                 servo_step: Callable[[tuple[float, ...], tuple[float, ...]], tuple[float, ...]] | None = None,
                 residual_step: Callable[[tuple[float, ...], tuple[float, ...]], tuple[float, ...]] | None = None,
                 residual_real_transfer_qualified: bool = False,
                 baseline: str = "persistence"):
        if dimension < 1 or baseline not in ("persistence", "servo"):
            raise ValueError("invalid physical dimension or FIT-registered baseline")
        if baseline == "servo" and not callable(servo_step):
            raise ValueError("servo baseline requires a recorded-law implementation")
        if residual_real_transfer_qualified and not callable(residual_step):
            raise ValueError("cannot qualify absent trained residual")
        self.dimension, self.servo_step, self.residual_step = dimension, servo_step, residual_step
        self.qualified, self.baseline = residual_real_transfer_qualified, baseline

    def forecast(self, current, planned_actions) -> PhysicalForecast:
        state = finite_vector(current, width=self.dimension)
        if not isinstance(planned_actions, (list, tuple)) or not planned_actions:
            raise ValueError("nonempty known-at-time proposed action sequence required")
        out = []
        for action in planned_actions:
            a = finite_vector(action, width=self.dimension)
            if self.baseline == "persistence":
                candidate = state
            else:
                candidate = finite_vector(self.servo_step(state, a), width=self.dimension)
            if self.qualified:
                candidate = finite_vector(self.residual_step(state, a), width=self.dimension)
            state = candidate
            out.append(state)
        return PhysicalForecast(tuple(out),
                                "qualified_neural_residual" if self.qualified else self.baseline,
                                self.qualified)
