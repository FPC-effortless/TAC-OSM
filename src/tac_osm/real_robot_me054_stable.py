"""ME054 stable, identity-anchored factual real physical world dynamics.

Mathematical correction motivated by actual ME052/ME053 measured robot failures:
TAC additive postprediction correction reached recurrent norm >1050 and
physical MAE >55; both TAC and GRU were far worse than last-state persistence.
These known stability/system-identification techniques are engineering, not
a claim of novel SSM/JEPA/transformer design or already learned intelligence.

Both arms inherit their original learned param count: native 37985,
GRU 38295. Both predict *physical DELTAS* about exactly observed x_t
using the same FIT-only per-channel physical-delta scale; each initialized
output head is exactly zero (identity physical persistence prior).
TAC verification is a convex combination of bounded states whose mixing
coefficient is zero for zero physically measured innovation.
"""
from __future__ import annotations

import torch
from torch import nn
from .real_robot_predictive_state import RealRobotPredictiveState, RobotForecast
from .real_robot_gru_world_control import GRURobotWorldModel
from .predictive_state_agent import Belief


def checked_scale(delta_scale):
    v=torch.as_tensor(delta_scale,dtype=torch.float32)
    if v.shape!=(14,) or not torch.isfinite(v).all() or not torch.all(v>0):
        raise ValueError("positive FIT-only original 14D physical-delta scales required")
    return v


class StableRealTAC(RealRobotPredictiveState):
    def __init__(self,delta_scale):
        super().__init__()
        self.register_buffer("factual_delta_scale",checked_scale(delta_scale))
        with torch.no_grad():
            self.native.next_observation.weight.zero_()
            self.native.next_observation.bias.zero_()

    def observe(self,belief,measured_state):
        z=super().observe(belief,measured_state)
        # Bounded associative representation; never change external fast/slow
        # geometry or source observation channels.
        return Belief(z.fast,z.slow,torch.tanh(z.memory))

    def predict(self,belief,measured_state,candidate_actions,current_human):
        o=super().predict(belief,measured_state,candidate_actions,current_human)
        predicted=measured_state.unsqueeze(1)+self.factual_delta_scale*o.next_joint_state
        return RobotForecast(o.takeover_logits,predicted)

    def verify_real_feedback(self,belief,executed_action,current_human,
            actual_next_state,predicted_next_state,physical_available,
            observed_takeover,predicted_takeover_logit,takeover_available):
        b=self.native._validate(belief)
        authority=self._authority(current_human,b)
        physical=self._avail(physical_available,b,"physical")
        onset=self._avail(takeover_available,b,"onset")
        if torch.any(authority&onset):
            raise ValueError("new takeover undefined while human already controlling")
        if (actual_next_state.shape!=(b,14) or predicted_next_state.shape!=(b,14)
                or observed_takeover.shape!=(b,1)
                or predicted_takeover_logit.shape!=(b,1)):
            raise ValueError("original real observed feedback size mismatch")
        self._finite_when_available(actual_next_state,physical)
        self._finite_when_available(predicted_next_state,physical)
        self._finite_when_available(observed_takeover,onset)
        self._finite_when_available(predicted_takeover_logit,onset)
        active=observed_takeover[onset]
        if active.numel() and not torch.isin(
                active,torch.tensor([0.,1.],dtype=active.dtype,device=active.device)).all():
            raise ValueError("real next human takeover observation must be binary")

        x=self.action_features(executed_action,authority)
        a=torch.tanh(self.native.action(x))
        joint=torch.cat((belief.fast,belief.slow),-1)
        memory=belief.memory

        # Bounded physical innovation: exact observed==forecast implies
        # zero mixing, and hence exact identity belief update for this lane.
        p=torch.tanh((
            torch.where(physical,actual_next_state,torch.zeros_like(actual_next_state))-
            torch.where(physical,predicted_next_state.detach(),torch.zeros_like(predicted_next_state))
            )/self.factual_delta_scale)
        intensity=p.abs().mean(-1,keepdim=True)
        proposal,gate=self.physical_map(torch.cat((joint,a,p),-1)).chunk(2,-1)
        mix=.10*intensity*torch.sigmoid(gate)
        joint=joint+mix*(torch.tanh(proposal)-joint)
        key=self.native.address(self.physical_key(a))
        value=torch.tanh(self.physical_value(p))
        memory_target=value.unsqueeze(-1)*key.unsqueeze(-2)
        memory=memory+.10*intensity.unsqueeze(-1)*(memory_target-memory)

        # Real takeover-onset correction is also innovation-controlled and
        # bounded by a convex mixing coefficient. It is never fitted for
        # already-human-controlled frames.
        binary_residual=torch.tanh(
            torch.where(onset,observed_takeover,torch.zeros_like(observed_takeover))-
            torch.where(onset,torch.sigmoid(predicted_takeover_logit.detach()),
                        torch.zeros_like(predicted_takeover_logit)))
        scalar_prop,scalar_gate=self.native.verify_map(
            torch.cat((joint,a,binary_residual),-1)).chunk(2,-1)
        scalar_mix=.10*binary_residual.abs()*torch.sigmoid(scalar_gate)
        joint=joint+scalar_mix*(torch.tanh(scalar_prop)-joint)
        scalar_key=self.native.address(self.native.verify_key(a))
        scalar_value=torch.tanh(self.native.verify_value(binary_residual))
        scalar_memory_target=scalar_value.unsqueeze(-1)*scalar_key.unsqueeze(-2)
        memory=memory+.10*binary_residual.abs().unsqueeze(-1)*(scalar_memory_target-memory)
        return Belief(joint[:,:32],joint[:,32:],memory)


class StableRealGRU(GRURobotWorldModel):
    def __init__(self,delta_scale):
        super().__init__(hidden_size=40)
        self.register_buffer("factual_delta_scale",checked_scale(delta_scale))
        with torch.no_grad():
            self.physical_head[-1].weight.zero_()
            self.physical_head[-1].bias.zero_()

    def predict(self,state,measured_state,candidate_actions,current_human):
        o=super().predict(state,measured_state,candidate_actions,current_human)
        predicted=measured_state.unsqueeze(1)+self.factual_delta_scale*o.next_joint_state
        return RobotForecast(o.takeover_logits,predicted)


def actual_real_physical_and_takeover_loss(model,forecast,current_state,
                                           genuine_next_state,observed_onset,eligible_onset):
    """Exact same genuine source supervision for both actual trained models."""
    if (current_state.shape!=genuine_next_state.shape or
            current_state.ndim!=2 or current_state.shape[-1]!=14):
        raise ValueError("original 14D measured physical state required")
    losses=model.genuine_logged_losses(
        forecast,genuine_next_state,observed_onset,eligible_onset)
    # Standardize the original *actual* small one-frame measured dynamics,
    # rather than regress an arbitrary absolute pose.
    delta_error=(forecast.next_joint_state[:,0]-genuine_next_state)/model.factual_delta_scale
    physical=delta_error.square().mean()
    return physical+.2*losses["masked_real_takeover_bce"],{
        "physical_MSE_FIT_only_delta_scaled":physical,
        "real_policy_takeover_masked_BCE":losses["masked_real_takeover_bce"]}
