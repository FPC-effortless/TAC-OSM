"""Native TAC integrated real-robot world prediction and factual correction.

This is an UNTRAINED physical-state/behavior-feedback adapter for the already
integrated PredictiveStateAgent. No Transformer/SSM/scripted planner, no
unexecuted-action labels, no fabricated task success/reward. Predict BEFORE the
real next frame; correct only AFTER the measured target becomes observable.

The physical next-qpos and optional takeover-onset are distinct real targets.
Do not confuse missing onset under existing human control with a negative.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .predictive_state_agent import Belief, PredictiveStateAgent

DOF=14


class RobotForecast(NamedTuple):
    takeover_logits: torch.Tensor  # B x A x 1; factual only for logged arm
    next_joint_state: torch.Tensor  # B x A x 14; measured at subsequent frame


class RealRobotPredictiveState(nn.Module):
    """Persistent state, action-dependent physical prediction, dual real feedback."""

    def __init__(self):
        super().__init__()
        # Last action feature is KNOWN current control authority; not future.
        # The native readout predicts next operator takeover; its separate
        # next_observation head predicts actual 14-dimensional robot state.
        self.native=PredictiveStateAgent(context_size=DOF,action_size=DOF+1,
                                         outcome_size=1)
        # Full vector of physical surprise is necessary: reducing 14 joint
        # errors to the binary takeover target would throw away real state.
        self.physical_map=nn.Linear(64+8+DOF,128)
        self.physical_key=nn.Linear(8,8)
        self.physical_value=nn.Linear(DOF,8)

    def initial(self,batch_size):
        return self.native.initial(batch_size)

    def observe(self,belief,measured_state):
        return self.native.observe(belief,measured_state)

    @staticmethod
    def _authority(current_human,b):
        if (current_human.shape!=(b,1) or current_human.dtype!=torch.bool):
            raise ValueError("known current physical-control authority must be Bx1 boolean")
        return current_human

    @staticmethod
    def _avail(mask,b,description):
        if mask.shape!=(b,1) or mask.dtype!=torch.bool:
            raise ValueError(description+" must be Bx1 boolean availability")
        return mask

    @staticmethod
    def _finite_when_available(value,mask):
        if not torch.isfinite(value[mask.expand_as(value)]).all():
            raise ValueError("real available physical/outcome values must be finite")

    def action_features(self,executed_action,current_human):
        b=len(executed_action)
        if executed_action.shape!=(b,DOF):
            raise ValueError("14-dimensional actually executed robot action required")
        self._authority(current_human,b)
        if not torch.isfinite(executed_action).all():
            raise ValueError("executed physical robot action nonfinite")
        return torch.cat((executed_action,current_human.to(executed_action.dtype)),dim=-1)

    def predict(self,belief,measured_state,candidate_actions,current_human):
        """Candidate query has NO state mutation; only logged choices supervised."""
        b=len(measured_state)
        self._authority(current_human,b)
        if (measured_state.shape!=(b,DOF) or candidate_actions.ndim!=3 or
                candidate_actions.shape[0]!=b or candidate_actions.shape[2]!=DOF):
            raise ValueError("B x 14 observation and B x A x 14 candidate actions required")
        if not (torch.isfinite(measured_state).all() and
                torch.isfinite(candidate_actions).all()):
            raise ValueError("candidate physical inputs must be finite")
        flag=current_human.to(candidate_actions.dtype).unsqueeze(1)
        action_with_known_authority=torch.cat(
            (candidate_actions,flag.expand(-1,candidate_actions.shape[1],-1)),dim=-1)
        onset,physical=self.native.predict(belief,measured_state,
                                             action_with_known_authority)
        return RobotForecast(onset,physical)

    def verify_real_feedback(self,belief,executed_action,current_human,
            actual_next_state,predicted_next_state,physical_available,
            observed_takeover,predicted_takeover_logit,takeover_available):
        """Update belief only after factual next frame or onset label arrives.

        Under current human control, new takeover onset is undefined and
        takeover_available MUST be False. Either label can independently be
        missing (even represented NaN); masks are applied before subtraction.
        Each continuous-state correction magnitude is bounded <=.25 per
        coordinate; takeover correction has its own native <=.25 bound.
        Combined corrections <=.5 per coordinate, not a general stability
        theorem and not a claim of learned capability.
        """
        b=self.native._validate(belief)
        authority=self._authority(current_human,b)
        phys=self._avail(physical_available,b,"physical")
        onset=self._avail(takeover_available,b,"onset")
        if torch.any(authority & onset):
            raise ValueError("cannot label a new human takeover while already human-controlled")
        if (actual_next_state.shape!=(b,DOF) or predicted_next_state.shape!=(b,DOF)
                or observed_takeover.shape!=(b,1)
                or predicted_takeover_logit.shape!=(b,1)):
            raise ValueError("real feedback physical14/onset1 shape mismatch")
        self._finite_when_available(actual_next_state,phys)
        self._finite_when_available(predicted_next_state,phys)
        self._finite_when_available(observed_takeover,onset)
        self._finite_when_available(predicted_takeover_logit,onset)
        labels=observed_takeover[onset]
        if labels.numel() and not torch.isin(labels,
                   torch.tensor([0.,1.],device=labels.device,dtype=labels.dtype)).all():
            raise ValueError("factual observed onset must be binary")
        act=self.action_features(executed_action,authority)
        # The native takeover-verification path is used only when an actual
        # human onset label exists, never by filling a missing label with zero.
        from_onset=self.native.verify(belief,act,observed_takeover,
                          torch.sigmoid(predicted_takeover_logit),onset)
        # A distinct learned physical correction uses the complete observed
        # 14-dimensional physical surprise, not a fake intervention reward.
        r=torch.tanh(
            torch.where(phys,actual_next_state,torch.zeros_like(actual_next_state))-
            torch.where(phys,predicted_next_state.detach(),
                        torch.zeros_like(predicted_next_state)))
        a=torch.tanh(self.native.action(act))
        joint=torch.cat((from_onset.fast,from_onset.slow),dim=-1)
        proposal,gate=self.physical_map(torch.cat((joint,a,r),dim=-1)).chunk(2,-1)
        correction=.25*torch.sigmoid(gate)*torch.tanh(proposal)
        updated=joint+torch.where(phys,correction,torch.zeros_like(correction))
        key=self.native.address(self.physical_key(a))
        value=torch.tanh(self.physical_value(r))
        delta=.25*(value-(from_onset.memory@key.unsqueeze(-1)).squeeze(-1)
                   ).unsqueeze(-1)*key.unsqueeze(-2)
        matrix=from_onset.memory+torch.where(phys.unsqueeze(-1),delta,
                                           torch.zeros_like(delta))
        return Belief(updated[:,:32],updated[:,32:],matrix)

    @staticmethod
    def genuine_logged_losses(forecast,measured_next_state,observed_takeover,
                              takeover_available):
        """Two factual REAL supervised objectives; no weighting is selected here.

        The eventual lambda and normalized physical units require source-only
        FIT statistics and a prefit matched-compute preregistration. Returning
        separate losses PREVENTS silently choosing task weights on TEST.
        """
        if (forecast.next_joint_state.ndim!=3 or
                forecast.next_joint_state.shape[1:]!=(1,DOF) or
                forecast.takeover_logits.shape!=(len(measured_next_state),1,1)):
            raise ValueError("training may supervise the actually logged action only")
        b=len(measured_next_state)
        if measured_next_state.shape!=(b,DOF):
            raise ValueError("measured factual next state must be 14D")
        if (observed_takeover.shape!=(b,1) or takeover_available.shape!=(b,1)
                or takeover_available.dtype!=torch.bool):
            raise ValueError("real takeover target must carry independent availability mask")
        if not torch.isfinite(measured_next_state).all():
            raise ValueError("factual next physical observation must be finite")
        if not torch.isfinite(observed_takeover[takeover_available]).all():
            raise ValueError("available actual human flags must be finite")
        active=observed_takeover[takeover_available]
        if active.numel() and not torch.isin(active,torch.tensor(
                [0.,1.],device=active.device,dtype=active.dtype)).all():
            raise ValueError("real human flag must be binary")
        physical=F.mse_loss(forecast.next_joint_state[:,0],measured_next_state)
        targets=torch.where(takeover_available,observed_takeover,
                            torch.zeros_like(observed_takeover))
        per_row=F.binary_cross_entropy_with_logits(
            forecast.takeover_logits[:,0],targets,reduction="none")
        onset=(per_row*takeover_available).sum()/takeover_available.sum().clamp_min(1)
        return {"observed_physical_mse":physical,
                "masked_real_takeover_bce":onset,
                "actually_supervised_takeover_count":int(takeover_available.sum())}
