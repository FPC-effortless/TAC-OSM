"""Experimental post-scan FFN for the six-family parallel TAC research core.

This extends, but DOES NOT replace, ParallelTACWorld. All memory gates remain
input-conditioned before the associative scan; the FFN executes strictly after
the read, so it cannot introduce a state-dependent scan schedule.

Input actions are original *observed* records, not interventions or proposed
future controls. No training, independent real-world skill, or calibration is
claimed by this construction. Six-family domain heads remain unqualified.
"""
from __future__ import annotations

import torch
from torch import nn

from .parallel_continuous_world import ParallelTACWorld
from .streaming_continuous_world import (
    WorldStreamState, init_world_stream, world_stream_step,
)


class ParallelResidualWorld(nn.Module):
    """One model with batch-scan and O(1)-history streaming prediction.

    The residual head consumes present observation embeddings and the
    read-only *previous* memory read, together with postscan fast/slow
    proposals. The only carried per-entity state remains 88+8*8=152 FP32
    values for the default geometry, plus one integer count. Learned model
    weights and external entity metadata are NOT part of that state count.

    The default 14/14/88/132, FFN width 12 configuration has 96,303
    genuinely trainable parameters. This does not attest to matchable
    training FLOPs or good real-world predictions.
    """

    def __init__(
        self, state_dim: int = 14, action_dim: int = 14,
        width: int = 88, decoder: int = 132, ffn_hidden: int = 12,
    ):
        super().__init__()
        dimensions = (state_dim, action_dim, width, decoder, ffn_hidden)
        if any(type(d) is not int or d <= 0 for d in dimensions):
            raise ValueError("all model dimensions must be positive integers")
        self.core = ParallelTACWorld(
            state_dim=state_dim, action_dim=action_dim,
            width=width, decoder=decoder,
        )
        features = 2 * width + 8 + 48
        self.residual_ffn = nn.Sequential(
            nn.Linear(features, ffn_hidden),
            nn.SiLU(),
            nn.Linear(ffn_hidden, state_dim),
        )
        # Starts small but differentiable: full FFN and the gate receive
        # gradients on the first backward step. It is not identity initialized.
        self.residual_logit = nn.Parameter(torch.tensor(-4.0))

    @property
    def state_dim(self) -> int:
        return self.core.state_dim

    @property
    def action_dim(self) -> int:
        return self.core.action_dim

    def set_train_delta_scale(self, scale: torch.Tensor) -> None:
        """Only TRAIN-defined scales; same validation as the parent core."""
        self.core.set_train_delta_scale(scale)

    def _correction(self, fast, slow, old_read, observation, action):
        # e_t only uses the available observation and recorded action at t.
        embedding = self.core.encoder(
            torch.cat((observation, action), dim=-1)
        ).tanh()
        features = torch.cat((fast, slow, old_read, embedding), dim=-1)
        gated = self.residual_logit.sigmoid() * self.residual_ffn(features).tanh()
        return gated

    def forward(self, observation, observed_action, *, return_trace=False):
        base, trace = self.core(
            observation, observed_action, return_trace=True,
        )
        correction = self._correction(
            trace["fast"], trace["slow"], trace["old_read"],
            observation, observed_action,
        )
        prediction = base + self.core.delta_scale * correction
        if return_trace:
            return prediction, {
                **trace,
                "post_scan_correction": correction,
                "post_scan_gate": self.residual_logit.sigmoid(),
            }
        return prediction

    def init_stream(self, batch: int) -> WorldStreamState:
        return init_world_stream(self.core, batch)

    def stream_step(
        self, state: WorldStreamState, observation, observed_action, *,
        return_trace=False,
    ):
        base, next_state, trace = world_stream_step(
            self.core, state, observation, observed_action, return_trace=True,
        )
        correction = self._correction(
            trace["fast"], trace["slow"], trace["old_read"],
            observation, observed_action,
        )
        prediction = base + self.core.delta_scale * correction
        if return_trace:
            return prediction, next_state, {
                **trace,
                "post_scan_correction": correction,
                "post_scan_gate": self.residual_logit.sigmoid(),
            }
        return prediction, next_state
