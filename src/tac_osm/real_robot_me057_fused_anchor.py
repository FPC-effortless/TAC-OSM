"""ME-057 fused predictive-anchor TAC.

The persistent cell uses one belief-space anchor projection for both boundary
consolidation and dense transition supervision.  This replaces two small
serial projections with one larger projection while preserving the native
fast/slow/relevance state and giving every coefficient a predictive target.
"""
from __future__ import annotations

import torch
from torch import nn

from .real_robot_me055_rollout import CompactBelief, DOF
from .real_robot_me056_boundary import BoundaryServoRolloutTAC


class FusedAnchorServoRolloutTAC(BoundaryServoRolloutTAC):
    """Boundary TAC with one jointly trained predictive anchor map."""

    def __init__(self, delta_scale, state_mean, state_scale, action_mean,
                 action_scale, servo_gain, servo_step_limit):
        super().__init__(delta_scale, state_mean, state_scale, action_mean,
                         action_scale, servo_gain, servo_step_limit)
        boundary = self.boundary_projection
        state = self.state_projection
        # Common input: encoded observation16 + fast32 + slow anchor13 +
        # recalled relevance trace8.  The 90 outputs jointly parameterize the
        # 17D write anchor and 73D slow/relevance update.
        self.anchor_projection = nn.Linear(69, 90)
        with torch.no_grad():
            self.anchor_projection.weight.zero_()
            self.anchor_projection.bias.zero_()
            # Preserve the complete ME-056 boundary map.  The new slow-anchor
            # coefficients start as a small learned residual.
            self.anchor_projection.weight[:17, :48].copy_(
                boundary.weight[:, :48])
            self.anchor_projection.weight[:17, 61:].copy_(
                boundary.weight[:, 48:])
            self.anchor_projection.bias[:17].copy_(boundary.bias)
            nn.init.xavier_uniform_(self.anchor_projection.weight[:17, 48:61])
            # Preserve fast, the first thirteen slow coordinates, and the
            # complete recalled trace from the ME-056 state map.  Observation
            # coefficients replace the remaining slow-only coefficients.
            self.anchor_projection.weight[17:, 16:48].copy_(
                state.weight[:, :32])
            self.anchor_projection.weight[17:, 48:61].copy_(
                state.weight[:, 32:45])
            self.anchor_projection.weight[17:, 61:].copy_(
                state.weight[:, 64:])
            self.anchor_projection.bias[17:].copy_(state.bias)
            nn.init.xavier_uniform_(self.anchor_projection.weight[17:, :16])
        del self.boundary_projection, self.state_projection

    def dense_transition_auxiliary(self, fast: torch.Tensor,
            slow: torch.Tensor, memory: torch.Tensor,
            current_state: torch.Tensor, next_state: torch.Tensor,
            action: torch.Tensor, authority: torch.Tensor,
            next_authority: torch.Tensor):
        if (fast.ndim != 2 or fast.shape[1] != 32 or slow.shape != fast.shape
                or memory.shape != (len(fast), 8)
                or current_state.shape != (len(fast), DOF)
                or next_state.shape != current_state.shape
                or action.shape != current_state.shape):
            raise ValueError("flattened authentic transition batch required")
        return self._dense_transition_auxiliary_training(
            fast, slow, memory, current_state, next_state, action,
            authority, next_authority)

    @torch.jit.export
    def _dense_transition_auxiliary_training(self, fast: torch.Tensor,
            slow: torch.Tensor, memory: torch.Tensor,
            current_state: torch.Tensor, next_state: torch.Tensor,
            action: torch.Tensor, authority: torch.Tensor,
            next_authority: torch.Tensor) -> torch.Tensor:
        delta = torch.tanh(
            (next_state-current_state)/self.factual_delta_scale)
        action_error = torch.tanh(action-current_state)
        onset = (~authority & next_authority).to(delta.dtype)
        delta_mean = delta.mean(-1, keepdim=True)
        magnitude = delta.abs().mean(-1, keepdim=True)
        transition = torch.cat((delta, action_error, onset,
            authority.to(delta.dtype), delta_mean, magnitude), -1)
        boundary_target = torch.cat(
            (delta, onset, delta_mean, magnitude), -1)
        state_target = torch.cat(
            (delta[:, :8], magnitude, transition, transition.abs()), -1)
        # The dense target trains the anchor map.  The observation encoder is
        # already trained end to end by every rollout step; detaching here
        # avoids a second, redundant encoder-gradient path over old history.
        u = torch.tanh(self.native.observation(current_state)).detach()
        anchor = self.anchor_projection(torch.cat(
            (u, fast, slow[:, :13], memory), -1))
        boundary, state = anchor.split((17, 73), -1)
        return ((boundary-boundary_target).square().mean()
                + (state-state_target).square().mean())

    @torch.jit.export
    def _consolidate_training(self, belief: CompactBelief,
                              context: torch.Tensor) -> CompactBelief:
        native = self.native
        u = torch.tanh(native.observation(context))
        query = native.address(belief.fast[:, :8]+belief.slow[:, :8])
        recalled = belief.memory*query
        anchor = self.anchor_projection(torch.cat(
            (u, belief.fast, belief.slow[:, :13], recalled), -1))
        boundary, projected = anchor.split((17, 73), -1)
        key, residual, strength = boundary.split((8, 8, 1), -1)
        key = native.address(
            key+torch.sigmoid(strength)*torch.tanh(residual))
        value, write = projected[:, :8], projected[:, 8:9]
        innovation, gate = projected[:, 9:41], projected[:, 41:]
        retained = (.95+.0499*torch.sigmoid(native.retention))*belief.memory
        memory = retained+torch.sigmoid(write)*(
            torch.tanh(value)-retained)*key
        slow = belief.slow+torch.sigmoid(gate)*(
            torch.tanh(belief.slow+torch.tanh(innovation))-belief.slow)
        return CompactBelief(belief.fast, slow, torch.tanh(memory))
