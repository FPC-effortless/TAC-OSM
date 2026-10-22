"""ME-056 boundary-consolidated TAC with prediction-time latent refinement.

The fast state updates on each observation. Slow/associative state is
consolidated at explicit segment boundaries, while the same learned projections
perform non-mutating latent refinement for every action-conditioned prediction.
This keeps the native parameters active without rewriting persistent world
state during internal computation.
"""
from __future__ import annotations

import torch
from torch import nn

from .real_robot_me055_rollout import (
    CompactBelief, DOF, RobotForecast, ServoRolloutTAC)


class BoundaryServoRolloutTAC(ServoRolloutTAC):
    """Fast evidence tracking plus anchored slow-memory consolidation."""

    def __init__(self, delta_scale, state_mean, state_scale, action_mean,
                 action_scale, servo_gain, servo_step_limit):
        super().__init__(delta_scale,state_mean,state_scale,action_mean,
                         action_scale,servo_gain,servo_step_limit)
        old=self.fast_projection
        self.fast_projection=nn.Linear(56,64)
        self.boundary_projection=nn.Linear(56,17)
        with torch.no_grad():
            self.fast_projection.weight.copy_(old.weight[:64])
            self.fast_projection.bias.copy_(old.bias[:64])
            self.boundary_projection.weight.copy_(old.weight[64:])
            self.boundary_projection.bias.copy_(old.bias[64:])

    @torch.jit.export
    def _observe_training(self, belief: CompactBelief,
                          context: torch.Tensor) -> CompactBelief:
        u=torch.tanh(self.native.observation(context))
        projected=self.fast_projection(torch.cat(
            (u,belief.fast,belief.memory),-1))
        proposal,gate=projected.chunk(2,-1)
        fast=belief.fast+torch.sigmoid(gate)*(
            torch.tanh(proposal)-belief.fast)
        return CompactBelief(fast,belief.slow,belief.memory)

    @torch.jit.export
    def _predict_training(self,belief: CompactBelief,
                          measured_state: torch.Tensor,
                          candidate_actions: torch.Tensor,
                          current_human: torch.Tensor) -> RobotForecast:
        return ServoRolloutTAC._predict_training(
            self,belief,measured_state,candidate_actions,current_human)

    def dense_transition_auxiliary(self,fast: torch.Tensor,slow: torch.Tensor,
            memory: torch.Tensor,current_state: torch.Tensor,
            next_state: torch.Tensor,action: torch.Tensor,
            authority: torch.Tensor,next_authority: torch.Tensor):
        """Dense FIT transition targets evaluated as two large batched maps."""
        if (fast.ndim!=2 or fast.shape[1]!=32 or slow.shape!=fast.shape or
                memory.shape!=(len(fast),8) or current_state.shape!=(len(fast),DOF)
                or next_state.shape!=current_state.shape or action.shape!=current_state.shape):
            raise ValueError("flattened authentic transition batch required")
        return self._dense_transition_auxiliary_training(
            fast,slow,memory,current_state,next_state,action,
            authority,next_authority)

    @torch.jit.export
    def _dense_transition_auxiliary_training(self,fast: torch.Tensor,
            slow: torch.Tensor,memory: torch.Tensor,
            current_state: torch.Tensor,next_state: torch.Tensor,
            action: torch.Tensor,authority: torch.Tensor,
            next_authority: torch.Tensor) -> torch.Tensor:
        delta=torch.tanh((next_state-current_state)/self.factual_delta_scale)
        action_error=torch.tanh(action-current_state)
        onset=(~authority&next_authority).to(delta.dtype)
        delta_mean=delta.mean(-1,keepdim=True)
        magnitude=delta.abs().mean(-1,keepdim=True)
        transition=torch.cat((delta,action_error,onset,
                              authority.to(delta.dtype),delta_mean,magnitude),-1)
        boundary_target=torch.cat((delta,onset,delta_mean,magnitude),-1)
        state_target=torch.cat((delta[:,:8],magnitude,transition,
                                transition.abs()),-1)
        u=torch.tanh(self.native.observation(current_state))
        boundary=self.boundary_projection(torch.cat((u,fast,memory),-1))
        state=self.state_projection(torch.cat((fast,slow,memory),-1))
        return (boundary-boundary_target).square().mean()+(
            state-state_target).square().mean()

    @torch.jit.export
    def _consolidate_training(self,belief: CompactBelief,
                              context: torch.Tensor) -> CompactBelief:
        """Mutate slow/trace state once at an observed segment boundary."""
        native=self.native
        u=torch.tanh(native.observation(context))
        query=native.address(belief.fast[:,:8]+belief.slow[:,:8])
        recalled=belief.memory*query
        key,residual,strength=self.boundary_projection(torch.cat(
            (u,belief.fast,recalled),-1)).split((8,8,1),-1)
        key=native.address(key+torch.sigmoid(strength)*torch.tanh(residual))
        projected=self.state_projection(torch.cat(
            (belief.fast,belief.slow,recalled),-1))
        value,write=projected[:,:8],projected[:,8:9]
        innovation,gate=projected[:,9:41],projected[:,41:]
        retained=(.95+.0499*torch.sigmoid(native.retention))*belief.memory
        memory=retained+torch.sigmoid(write)*(
            torch.tanh(value)-retained)*key
        slow=belief.slow+torch.sigmoid(gate)*(
            torch.tanh(belief.slow+torch.tanh(innovation))-belief.slow)
        return CompactBelief(belief.fast,slow,torch.tanh(memory))

    @torch.jit.export
    def _burnin_boundary_training(self,belief: CompactBelief,
                         current_state: torch.Tensor,
                         executed_action: torch.Tensor,
                         current_human: torch.Tensor,
                         next_state: torch.Tensor,
                         next_human: torch.Tensor) -> CompactBelief:
        belief=self._burnin_training(
            belief,current_state,executed_action,current_human,
            next_state,next_human)
        return self._consolidate_training(belief,next_state)

