"""Established recurrent world-model control for authentic real robot data.

UNTRAINED comparator. GRUCell recurrent state, action-conditioned real 14D
next-joint-state and a separate masked human-takeover head, post-outcome
feedback via another learned GRUCell. This is a substantive recurrent world
model, not an intentionally weak last-state or static context control.

Same actual prediction-time information and feedback availability as native
RealRobotPredictiveState. No causal do(a), policy/value reward, fabricated
success, hidden privileged state, or simulated performance evidence.

The hidden size is set *before* original training, following numeric prefit
resource measurement. Parameter matching by itself does NOT establish FLOP or
CPU parity, and may fail. No dummy unused capacity or sleep-time padding.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

DOF=14

class GRURobotWorldModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        if type(hidden_size) is not int or hidden_size<16:
            raise ValueError("meaningful positive recurrent hidden size required")
        h=hidden_size
        self.hidden_size=h
        self.observe_encoder=nn.Linear(DOF,h)
        self.observation_cell=nn.GRUCell(h,h)
        self.action_encoder=nn.Linear(DOF+1,h)
        self.physical_head=nn.Sequential(nn.Linear(h*2,h),nn.SiLU(),nn.Linear(h,DOF))
        self.takeover_head=nn.Sequential(nn.Linear(h*2,h),nn.SiLU(),nn.Linear(h,1))
        self.physical_residual=nn.Linear(DOF,h)
        self.takeover_residual=nn.Linear(1,h)
        self.feedback_cell=nn.GRUCell(h*3,h)
        self.register_buffer("_zero",torch.tensor(0.0),persistent=False)

    def initial(self,batch_size):
        if type(batch_size) is not int or batch_size<1:
            raise ValueError("positive batch required")
        return self._zero.new_zeros((batch_size,self.hidden_size))

    def observe(self,state,measured_state):
        b=state.shape[0]
        if state.shape!=(b,self.hidden_size) or measured_state.shape!=(b,DOF):
            raise ValueError("invalid measured state/recurrent state size")
        if not torch.isfinite(measured_state).all():
            raise ValueError("nonfinite physical observation")
        encoded=torch.tanh(self.observe_encoder(measured_state))
        return self.observation_cell(encoded,state)

    def _action(self,actions,authority):
        b=actions.shape[0]
        if authority.shape!=(b,1) or authority.dtype!=torch.bool:
            raise ValueError("current operator authority must be Bx1 boolean")
        if actions.shape[-1]!=DOF or not torch.isfinite(actions).all():
            raise ValueError("14 measured commanded action channels required")
        if actions.ndim==3:
            auth=authority.to(actions.dtype).unsqueeze(1).expand(-1,actions.shape[1],-1)
        elif actions.ndim==2:
            auth=authority.to(actions.dtype)
        else:
            raise ValueError("real action dimensionality invalid")
        return torch.tanh(self.action_encoder(torch.cat((actions,auth),-1)))

    def predict(self,state,measured_state,candidate_actions,current_human):
        b=state.shape[0]
        if (state.shape!=(b,self.hidden_size) or measured_state.shape!=(b,DOF)
            or candidate_actions.ndim!=3 or candidate_actions.shape[0]!=b
            or candidate_actions.shape[2]!=DOF):
            raise ValueError("state14 and candidate logged action14 required")
        if not torch.isfinite(measured_state).all():
            raise ValueError("nonfinite current physical observation")
        encoded=self._action(candidate_actions,current_human)
        z=torch.cat((state.unsqueeze(1).expand(-1,encoded.shape[1],-1),encoded),-1)
        # Head order identical to native contract.
        from .real_robot_predictive_state import RobotForecast
        return RobotForecast(self.takeover_head(z),self.physical_head(z))

    def verify_real_feedback(self,state,executed_action,current_human,
            actual_next_state,predicted_next_state,physical_available,
            observed_takeover,predicted_takeover_logit,takeover_available):
        b=state.shape[0]
        if state.shape!=(b,self.hidden_size):
            raise ValueError("persistent recurrent state shape invalid")
        if (actual_next_state.shape!=(b,DOF) or predicted_next_state.shape!=(b,DOF)
            or observed_takeover.shape!=(b,1) or predicted_takeover_logit.shape!=(b,1)):
            raise ValueError("factual feedback shape invalid")
        for mask in (physical_available,takeover_available):
            if mask.shape!=(b,1) or mask.dtype!=torch.bool:
                raise ValueError("independent factual feedback availability mask required")
        if torch.any(current_human & takeover_available):
            raise ValueError("new operator onset undefined when operator already controlling")
        for data,mask in ((actual_next_state,physical_available),
                          (predicted_next_state,physical_available),
                          (observed_takeover,takeover_available),
                          (predicted_takeover_logit,takeover_available)):
            if not torch.isfinite(data[mask.expand_as(data)]).all():
                raise ValueError("available source feedback nonfinite")
        labels=observed_takeover[takeover_available]
        if labels.numel() and not torch.isin(labels,torch.tensor(
                [0.,1.],dtype=labels.dtype,device=labels.device)).all():
            raise ValueError("observed takeover not binary")
        if not (physical_available | takeover_available).any():
            return state # identity with absent labels, including NaN
        action=self._action(executed_action,current_human)
        physical=torch.tanh(
            torch.where(physical_available,actual_next_state,torch.zeros_like(actual_next_state))-
            torch.where(physical_available,predicted_next_state.detach(),torch.zeros_like(predicted_next_state)))
        residual=torch.tanh(
            torch.where(takeover_available,observed_takeover,torch.zeros_like(observed_takeover))-
            torch.where(takeover_available,torch.sigmoid(predicted_takeover_logit.detach()),
                        torch.zeros_like(predicted_takeover_logit)))
        v=torch.cat((action,self.physical_residual(physical),
                     self.takeover_residual(residual)),dim=-1)
        candidate=self.feedback_cell(v,state)
        return torch.where(physical_available|takeover_available,candidate,state)

    @staticmethod
    def genuine_logged_losses(forecast,measured_next_state,observed_takeover,takeover_available):
        # Exactly the same supervision rule and emitted loss names as the
        # native model: no advantage by labels, auxiliary oracle, or tuning.
        from .real_robot_predictive_state import RealRobotPredictiveState
        return RealRobotPredictiveState.genuine_logged_losses(
            forecast,measured_next_state,observed_takeover,takeover_available)
