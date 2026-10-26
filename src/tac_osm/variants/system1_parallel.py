"""Untrained System 1 *observational* head on the shared parallel TAC world core.

This candidate does not infer causal action effects, license action execution,
provide calibrated event probabilities, or establish real-world superiority.
All inputs at t are observed state and the action already logged at t; future
candidate actions MUST NOT be substituted for logged, executed actions.

The trainable backbone is the one already used by the six-family parallel
TAC experiment. Both causal batch and single-step prediction share weights.
A single compact head produces a binary adverse-event logit and strictly
positive per-dimension observational next-state residual scales. Real-data
training, independent outcome validation, label definitions, calibration,
comparator matching, timing and safety qualification are all still required.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .parallel_continuous_world import ParallelTACWorld
from .streaming_continuous_world import (
    WorldStreamState, init_world_stream, world_stream_step,
)


@dataclass(frozen=True)
class System1Forecast:
    """Uncalibrated, non-executable observational candidate output."""
    next_observation: torch.Tensor
    adverse_event_logit: torch.Tensor
    adverse_event_score: torch.Tensor
    residual_scale: torch.Tensor
    scope: str = "advisory_only"
    is_calibrated: bool = False
    authorized_to_act: bool = False

    def __post_init__(self):
        if self.scope != "advisory_only" or self.is_calibrated or self.authorized_to_act:
            raise ValueError("System 1 candidate cannot assert qualification")


class System1ParallelWorld(nn.Module):
    """One shared state engine and one risk/uncertainty observational head.

    Default state_dim=14, action_dim=14, width=88, decoder=132,
    head_width=12 yields 95,907 trainable parameters. The streaming state
    holds 88+64=152 FP32 numbers per entity (608 B) plus event count.
    No entity identity, event timestamp, provenance, or missingness is inferred;
    callers must enforce the existing ObservationStamp/as-of contract.
    """

    def __init__(self, state_dim: int = 14, action_dim: int = 14,
                 width: int = 88, decoder: int = 132, head_width: int = 12):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in
               (state_dim, action_dim, width, decoder, head_width)):
            raise ValueError("positive integer model dimensions required")
        self.core = ParallelTACWorld(state_dim, action_dim, width, decoder)
        feature_dim = 2 * width + 8 + state_dim
        self.reactive_head = nn.Sequential(
            nn.Linear(feature_dim, head_width),
            nn.SiLU(),
            nn.Linear(head_width, state_dim + 1),
        )

    def set_train_delta_scale(self, scale: torch.Tensor) -> None:
        """Accept a delta scale estimated ONLY from the fitting partition."""
        self.core.set_train_delta_scale(scale)

    def _validate_input(self, observed: torch.Tensor,
                        logged_action: torch.Tensor, *, sequence: bool) -> None:
        expected = 3 if sequence else 2
        if (observed.ndim != expected or logged_action.ndim != expected or
            observed.shape[:-1] != logged_action.shape[:-1] or
            observed.shape[-1] != self.core.state_dim or
            logged_action.shape[-1] != self.core.action_dim or
            any(d < 1 for d in observed.shape[:-1])):
            raise ValueError("nonempty matching observed/logged-action tensor shapes required")
        param = next(self.parameters())
        if (observed.dtype != param.dtype or logged_action.dtype != param.dtype or
            observed.device != param.device or logged_action.device != param.device or
            not torch.isfinite(observed).all().item() or
            not torch.isfinite(logged_action).all().item()):
            raise ValueError("inputs must be finite and match model dtype/device")

    def _project(self, predicted: torch.Tensor, fast: torch.Tensor,
                 slow: torch.Tensor, old_read: torch.Tensor) -> System1Forecast:
        evidence = torch.cat((fast, slow, old_read, predicted), dim=-1)
        raw = self.reactive_head(evidence)
        logit = raw[..., 0]
        # Positivity is structural, but scale calibration is NOT established.
        residual_scale = F.softplus(raw[..., 1:]) + 1.0e-4
        return System1Forecast(predicted, logit, logit.sigmoid(), residual_scale)

    def forward(self, observed: torch.Tensor,
                logged_action: torch.Tensor) -> System1Forecast:
        """Process [B,T,D] causal recorded prefixes with shared scan."""
        self._validate_input(observed, logged_action, sequence=True)
        predicted, trace = self.core(observed, logged_action, return_trace=True)
        return self._project(predicted, trace["fast"],
                             trace["slow"], trace["old_read"])

    def initial_state(self, batch_size: int) -> WorldStreamState:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("positive integer batch size required")
        return init_world_stream(self.core, batch_size)

    def step(self, state: WorldStreamState, observed: torch.Tensor,
             logged_action: torch.Tensor) -> tuple[System1Forecast, WorldStreamState]:
        """Advance a single observed time step; NO proposed-action rollout."""
        self._validate_input(observed, logged_action, sequence=False)
        if (not isinstance(state, WorldStreamState) or
            state.fast_sum.shape != (observed.shape[0], self.core.width) or
            state.memory.shape != (observed.shape[0], 8, 8) or
            type(state.count) is not int or state.count < 0 or
            any(x.dtype != observed.dtype or x.device != observed.device or
                not torch.isfinite(x).all().item() for x in
                (state.fast_sum, state.memory))):
            raise ValueError("invalid finite carried state")
        predicted, next_state, trace = world_stream_step(
            self.core, state, observed, logged_action, return_trace=True)
        return self._project(predicted, trace["fast"], trace["slow"],
                             trace["old_read"]), next_state

    @property
    def carried_fp32_bytes(self) -> int:
        """Streaming floating state only; excludes metadata and weights."""
        return 4 * (self.core.width + 64)
