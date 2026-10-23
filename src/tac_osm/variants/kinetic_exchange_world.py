"""ESWAM004: source-causal kinetic evidence and bounded motion activation.

Full native fast/slow + antisymmetric/symmetric exchange memory inherited from
ESWAM003. No synthetic performance claim. No ability to choose causal actions.
External observed episode identity remains required. No learned policy authority.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from tac_osm.variants.selective_exchange_world import (
    SelectiveExchangeWorldModel, SelectiveWorldState,
)


class KineticState(NamedTuple):
    prior: SelectiveWorldState
    kinetic_ema: torch.Tensor
    last_command: torch.Tensor
    last_command_known: torch.Tensor


class KineticForecast(NamedTuple):
    mean_next: torch.Tensor
    scale_next: torch.Tensor
    proposed_action: torch.Tensor
    state: KineticState
    movement_logit: torch.Tensor
    movement_probability: torch.Tensor
    previous_prediction_present: torch.Tensor
    observation_written: torch.Tensor
    prediction_written: torch.Tensor
    kinetic_gate: torch.Tensor


class KineticExchangeWorldModel(SelectiveExchangeWorldModel):
    """Learned motion event with physically grounded motion-activation path.

    Input actions are historically *already issued* commands. No future movement
    labels, future actions or evaluation-time persistence oracle are used.
    """
    architecture = "eswam004_kinetic_exchange_native"

    def __init__(self, dimension: int = 14, action_dim: int = 14,
                 width: int = 90, memory_width: int = 16, slots: int = 8,
                 state_scale=None):
        super().__init__(dimension, action_dim, width, memory_width, slots, state_scale)
        # Per-joint causal |velocity|, |EMA velocity|, |command tracking error|
        # and |command change|. No source future used for routing.
        self.kinetic_event = nn.Sequential(
            nn.Linear(4 * dimension, 8), nn.SiLU(), nn.Linear(8, 1))
        self.log_kinetic_gain = nn.Parameter(torch.tensor(2.0))
        self.prior_velocity_logits = nn.Parameter(torch.full((dimension,), -0.5))
        nn.init.zeros_(self.kinetic_event[-1].weight)
        nn.init.zeros_(self.kinetic_event[-1].bias)

    def initial_kinetic(self, batch: int, *, device=None, dtype=None) -> KineticState:
        prior = self.initial_selective(batch, device=device, dtype=dtype)
        return KineticState(
            prior=prior,
            kinetic_ema=torch.zeros(batch, self.dimension, dtype=prior.world.fast.dtype, device=device),
            last_command=torch.zeros(batch, self.action_dim, dtype=prior.world.fast.dtype, device=device),
            last_command_known=torch.zeros(batch, dtype=torch.bool, device=device),
        )

    def forward(self, observation: torch.Tensor, action_at_t: torch.Tensor,
                elapsed_seconds: torch.Tensor, entity_ids: torch.Tensor,
                observed_mask: torch.Tensor, goal: torch.Tensor | None = None,
                state: KineticState | None = None) -> KineticForecast:
        if observation.ndim != 2 or observation.shape[-1] != self.dimension:
            raise ValueError("physical observation required")
        batch = len(observation)
        if state is None:
            state = self.initial_kinetic(batch, device=observation.device, dtype=observation.dtype)
        if (state.kinetic_ema.shape != observation.shape
                or state.last_command.shape != action_at_t.shape
                or state.last_command_known.shape != (batch,)
                or state.last_command_known.dtype != torch.bool
                or not torch.isfinite(state.kinetic_ema).all()
                or not torch.isfinite(state.last_command).all()):
            raise ValueError("source history malformed")
        old = state.prior
        # Mean next-state from ESWAM003, including its original full
        # exchange-structured recurrent pathway; new branch receives its errors.
        forecast = super().forward(
            observation, action_at_t, elapsed_seconds, entity_ids,
            observed_mask, goal=goal, state=old)
        current = torch.where(
            observed_mask, observation,
            torch.where(old.last_observed_mask,old.last_observation,torch.zeros_like(observation)))
        valid_vel = observed_mask & old.last_observed_mask
        vel = torch.where(valid_vel,current-old.last_observation,torch.zeros_like(current))
        vel = vel.clamp(-1,1)
        velocity_abs = vel.abs()
        next_ema = 0.8 * state.kinetic_ema + 0.2 * velocity_abs
        command_change = torch.where(
            state.last_command_known[:,None],
            (action_at_t-state.last_command).abs(),
            torch.zeros_like(action_at_t))
        error = (action_at_t-current).abs()
        risk_input = torch.cat((velocity_abs,next_ema,error,command_change),dim=-1)
        movement_logit = forecast.movement_logit + self.kinetic_event(risk_input).squeeze(-1)
        p = movement_logit.sigmoid()
        # Noisy-OR between supervised prospective movement estimate (squared
        # so low-confidence predictions don't move static joints) and causal
        # actually observed kinetic evidence. Both zero -> exact persistence.
        evidence = (velocity_abs + next_ema + 0.25 * command_change).mean(-1)
        gain = F.softplus(self.log_kinetic_gain)
        kinetic_open = -torch.expm1(-gain * evidence)
        gate = 1.0 - (1.0-p.square()) * (1.0-kinetic_open)
        gate = gate.clamp(0,1)
        residual = forecast.mean_next - current
        # Source-measured continuation correction is learnable per joint, not
        # oracle future motion and not a physical action causal model.
        continuation = torch.sigmoid(self.prior_velocity_logits) * vel
        mean_next = current + gate[:,None] * (residual + continuation)
        return KineticForecast(
            mean_next, forecast.scale_next, forecast.proposed_action,
            KineticState(forecast.state,next_ema,action_at_t,state.last_command_known.new_ones((batch,))),
            movement_logit,p,forecast.previous_prediction_present,
            forecast.observation_written,forecast.prediction_written,gate)
