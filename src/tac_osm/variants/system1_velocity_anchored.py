"""System1 velocity-anchored point predictor, UNTRAINED research candidate.

The observed last two states give a deterministic, causal constant-velocity
reference. The pre-existing System1ParallelWorld can only add a bounded
innovation to that reference. Logged actions are prior observed evidence, not
interventions. This DOES NOT prove improvement or high-motion risk calibration.

No historical S1MOTION001 trained weights are modified, upgraded or transferred.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional
import math

import torch
from torch import nn

from .system1_parallel import System1ParallelWorld, System1Forecast
from .streaming_continuous_world import WorldStreamState


@dataclass(frozen=True)
class MotionStreamState:
    """Full TAC (88+64) state plus one 14D previous measured observation."""
    core_state: WorldStreamState
    previous_observation: torch.Tensor
    has_previous: bool


class VelocityAnchoredSystem1(nn.Module):
    """Same shared trainable core; fixed physically anchored correction.

    x_next = 2*x - previous_x + gain*(core_next - x).
    The core correction is scale*tanh(head), so for each dimension:
    |x_next - (2*x-previous_x)| <= gain*TRAIN_delta_scale,
    when input/weights are finite (real-number bound, not exact IEEE proof).
    The extra last-actual-observation buffer is explicit, 14 FP32 floats
    (56 B), raising floating stream state from 608 to 664 bytes for D=14.
    """

    def __init__(self, state_dim: int = 14, action_dim: int = 14,
                 width: int = 88, decoder: int = 132, head_width: int = 12,
                 correction_gain: float = 0.1):
        super().__init__()
        if (not isinstance(correction_gain, (float, int))
            or isinstance(correction_gain, bool)
            or not math.isfinite(correction_gain)
            or not (0 <= correction_gain <= 0.2)):
            raise ValueError("finite fixed correction gain must be in [0,0.2]")
        self.core = System1ParallelWorld(
            state_dim, action_dim, width, decoder, head_width)
        self.correction_gain = float(correction_gain)

    def set_train_delta_scale(self, scale: torch.Tensor) -> None:
        self.core.set_train_delta_scale(scale)

    @property
    def carried_fp32_bytes(self) -> int:
        return self.core.carried_fp32_bytes + 4 * self.core.core.state_dim

    def _prior(self, observed: torch.Tensor,
               initial_previous_observation: Optional[torch.Tensor]) -> torch.Tensor:
        first = observed[:, :1]
        if initial_previous_observation is not None:
            if (initial_previous_observation.shape != observed[:, 0].shape or
                initial_previous_observation.dtype != observed.dtype or
                initial_previous_observation.device != observed.device or
                not torch.isfinite(initial_previous_observation).all().item()):
                raise ValueError("initial previous observation must be valid known past")
            first = initial_previous_observation.unsqueeze(1)
        return torch.cat((first, observed[:, :-1]), dim=1)

    def forward(self, observed: torch.Tensor, prior_logged_action: torch.Tensor,
                *, initial_previous_observation: Optional[torch.Tensor] = None
                ) -> System1Forecast:
        """Causal all-time batch scan; no next-state or future input access."""
        result = self.core(observed, prior_logged_action)
        previous = self._prior(observed, initial_previous_observation)
        velocity_base = 2 * observed - previous
        correction = self.correction_gain * (result.next_observation - observed)
        return replace(result, next_observation=velocity_base + correction)

    def initial_state(self, batch_size: int,
                      *, previous_observation: Optional[torch.Tensor] = None
                      ) -> MotionStreamState:
        state = self.core.initial_state(batch_size)
        prev = state.fast_sum.new_zeros(batch_size, self.core.core.state_dim)
        seen = previous_observation is not None
        if seen:
            if (previous_observation.shape != prev.shape or
                previous_observation.dtype != prev.dtype or
                previous_observation.device != prev.device or
                not torch.isfinite(previous_observation).all().item()):
                raise ValueError("invalid observed historical priming state")
            prev = previous_observation.detach().clone()
        return MotionStreamState(state, prev, seen)

    def step(self, state: MotionStreamState, observed: torch.Tensor,
             prior_logged_action: torch.Tensor
             ) -> tuple[System1Forecast, MotionStreamState]:
        if not isinstance(state, MotionStreamState) or type(state.has_previous) is not bool:
            raise ValueError("invalid carried motion state")
        result, next_core = self.core.step(
            state.core_state, observed, prior_logged_action)
        prev = state.previous_observation if state.has_previous else observed
        if (prev.shape != observed.shape or prev.dtype != observed.dtype or
            prev.device != observed.device or
            not torch.isfinite(prev).all().item()):
            raise ValueError("invalid source-verified previous observation")
        anchored = 2 * observed - prev
        corrected = anchored + self.correction_gain * (
            result.next_observation - observed)
        return (replace(result, next_observation=corrected),
                MotionStreamState(next_core, observed, True))
