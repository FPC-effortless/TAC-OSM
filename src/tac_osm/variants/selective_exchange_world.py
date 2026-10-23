"""ESWAM003: one upgraded exchange-memory world model, source-causal selective physics.

This is conventional classical neural mathematics, not quantum dynamics.
The existing trained ESWAM002 checkpoint remains unchanged. No policy/control
authorization or causal do(action) is established by observational joint logs.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from tac_osm.variants.exchange_world_action import (
    ExchangeWorldActionModel, WorldState, WorldActionForecast,
)


class SelectiveWorldState(NamedTuple):
    world: WorldState
    last_observation: torch.Tensor
    last_observed_mask: torch.Tensor


class SelectiveForecast(NamedTuple):
    mean_next: torch.Tensor
    scale_next: torch.Tensor
    proposed_action: torch.Tensor
    state: SelectiveWorldState
    movement_logit: torch.Tensor
    movement_probability: torch.Tensor
    previous_prediction_present: torch.Tensor
    observation_written: torch.Tensor
    prediction_written: torch.Tensor


class SelectiveExchangeWorldModel(ExchangeWorldActionModel):
    """ES-WAM with causal velocity, movement-aware shrinkage, and anchored action.

    The actual future movement flag is never an input. The model predicts a
    probability of movement on physical observations; it is only calibrated if
    a real outcome-specific calibration study subsequently establishes that.

    Forward action_at_t is the *last already-issued logged qpos*, not a proposed
    intervention. proposed_action predicts the next recorded command, not an
    executable autonomous policy.
    """
    architecture = "eswam003_selective_exchange_native"

    def __init__(self, dimension: int = 14, action_dim: int = 14,
                 width: int = 90, memory_width: int = 16, slots: int = 8,
                 state_scale=None):
        super().__init__(dimension, action_dim, width, memory_width, slots, state_scale)
        self.motion_head = nn.Sequential(
            nn.Linear(2 * width + 2 * dimension, 4),
            nn.SiLU(),
            nn.Linear(4, 1),
        )
        self.shrink_log_scale = nn.Parameter(torch.full((dimension,), -1.8))
        self.velocity_log_gain = nn.Parameter(torch.zeros(dimension))
        self.action_step_logit = nn.Parameter(torch.full((action_dim,), -2.5))
        with torch.no_grad():
            self.motion_head[-1].bias.fill_(-1.5)

    def initial_selective(self, batch: int, *, device=None, dtype=None) -> SelectiveWorldState:
        world = self.initial(batch, device=device, dtype=dtype)
        return SelectiveWorldState(
            world,
            torch.zeros(batch, self.dimension, device=device, dtype=world.fast.dtype),
            torch.zeros(batch, self.dimension, device=device, dtype=torch.bool),
        )

    def forward(self, observation: torch.Tensor, action_at_t: torch.Tensor,
                elapsed_seconds: torch.Tensor, entity_ids: torch.Tensor,
                observed_mask: torch.Tensor, goal: torch.Tensor | None = None,
                state: SelectiveWorldState | None = None) -> SelectiveForecast:
        if observation.ndim != 2 or observed_mask.shape != observation.shape:
            raise ValueError("observed source vector and factual mask required")
        batch, dim = observation.shape
        if state is None:
            state = self.initial_selective(batch, device=observation.device, dtype=observation.dtype)
        if (state.last_observation.shape != observation.shape
                or state.last_observed_mask.shape != observed_mask.shape
                or state.last_observed_mask.dtype != torch.bool
                or not torch.isfinite(state.last_observation).all()):
            raise ValueError("invalid source-causal previous observed state")
        # Existing fast/slow recurrent and exchange-structured role-memory remain
        # trainable and are NOT replaced by a stationary-only shortcut.
        base: WorldActionForecast = super().forward(
            observation, action_at_t, elapsed_seconds, entity_ids, observed_mask,
            goal=goal, state=state.world)
        known_current = torch.where(
            observed_mask, observation,
            torch.where(state.last_observed_mask, state.last_observation, torch.zeros_like(observation))
        )
        joint_has_velocity = observed_mask & state.last_observed_mask
        velocity = torch.where(
            joint_has_velocity, known_current - state.last_observation,
            torch.zeros_like(known_current))
        velocity = velocity.clamp(-1.0, 1.0)
        action_error = action_at_t - known_current
        features = torch.cat((base.state.fast, base.state.slow, velocity, action_error), -1)
        movement_logit = self.motion_head(features).squeeze(-1)
        p_move = movement_logit.sigmoid()
        gain = torch.sigmoid(self.velocity_log_gain + 2.0)
        raw = gain * velocity + (base.mean_next - torch.where(observed_mask, observation, torch.zeros_like(observation)))
        # Rational adaptive shrinkage: never increases |raw| for fixed base,
        # but learned whole-model accuracy has to be evaluated on real outcomes.
        tau = F.softplus(self.shrink_log_scale) * ((1.0 - p_move) / (p_move + 0.05))[:, None]
        raw_squared = raw.square()
        bounded = raw * raw_squared / (raw_squared + tau.square() + 1e-12)
        mean_next = known_current + bounded
        # Retain a residual scale head; do not claim movement risk and per-joint
        # scale constitute a calibrated joint distribution.
        scale_next = base.scale_next
        # The source next action almost always repeats the last logged qpos.
        # Learn only a bounded change instead of an absolute tanh action.
        next_logged_command = action_at_t + (
            0.10 * self.action_step_logit.sigmoid()
            * torch.tanh(base.proposed_action))
        next_observed = torch.where(observed_mask, observation,
                                    state.last_observation)
        next_mask = observed_mask | state.last_observed_mask
        return SelectiveForecast(
            mean_next, scale_next, next_logged_command,
            SelectiveWorldState(base.state, next_observed, next_mask),
            movement_logit, p_move, base.previous_prediction_present,
            base.observation_written, base.prediction_written,
        )
