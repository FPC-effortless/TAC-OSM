"""ME056: physical robot adapter for persistent concept-swarm recurrent CED.

Real observations, executed robot actions and separately masked measured
physical/onset feedback only. NO simulated or counterfactual targets.
The persistent native fast/slow/associative recurrence is shared with
ConceptSwarmCED: decoder passes update TEMPORARY inference only.
No training or comparative skill is established by instantiation.
"""
from __future__ import annotations

from typing import NamedTuple
import torch
from torch import nn
from .concept_swarm_ced import ConceptBelief, ConceptSwarmCED
from .predictive_state_agent import Belief
from .real_robot_predictive_state import RobotForecast, DOF
from .real_robot_me054_stable import StableRealTAC, checked_scale
from .real_robot_gru_world_control import GRURobotWorldModel


class ZeroUnsupervisedHead(nn.Module):
    """No fictitious policy or information target on this factual corpus."""
    def forward(self,x):
        return x[..., :1].detach()*0


class RealRobotConceptCED(StableRealTAC):
    """Physics-anchored concept inference over unchanged native belief."""

    def __init__(self, delta_scale, *, concept_count=6, concept_dim=16,
                 refinement_steps=4):
        super().__init__(delta_scale)
        self.swarm = ConceptSwarmCED(DOF, DOF + 1, 1,
                                    concept_count=concept_count,
                                    concept_dim=concept_dim,
                                    refinement_steps=refinement_steps)
        # The same ONE trained native persistent state is used by both
        # routes, not a second duplicated core with hidden additional weights.
        self.swarm.core = self.native
        # Robot corpus has NO realized reward or information-gain targets.
        # Eliminate unused trainable heads instead of padding model size.
        self.swarm.future_map = nn.Identity()
        self.swarm.policy_head = ZeroUnsupervisedHead()
        self.swarm.information_head = ZeroUnsupervisedHead()
        self.physical_to_concept = nn.Linear(DOF,concept_dim)
        self.takeover_to_concept = nn.Linear(1,concept_dim)
        self.concept_feedback = nn.Linear(2*concept_dim,concept_dim)
        self.concept_feedback_gate = nn.Linear(2*concept_dim,concept_dim)
        # ME054 physics reference: the untrained head starts at identity.
        with torch.no_grad():
            self.swarm.next_delta.weight.zero_()
            self.swarm.next_delta.bias.zero_()

    def initial(self,batch_size):
        return self.swarm.initial(batch_size)

    def observe(self, belief, measured_state):
        self.swarm._validate(belief)
        if measured_state.shape!=(belief.concepts.shape[0],DOF) or not torch.isfinite(measured_state).all():
            raise ValueError("observed physical robot 14D state required")
        new=self.swarm.observe(belief,measured_state)
        core=Belief(new.core.fast,new.core.slow,torch.tanh(new.core.memory))
        return ConceptBelief(core,new.concepts)

    def predict(self,belief,measured_state,candidate_actions,current_human):
        b=self.swarm._validate(belief)
        authority=self._authority(current_human,b)
        if (measured_state.shape!=(b,DOF) or candidate_actions.ndim!=3
              or candidate_actions.shape[0]!=b or candidate_actions.shape[2]!=DOF):
            raise ValueError("real robot prediction requires Bx14 and BxAx14")
        if not (torch.isfinite(measured_state).all() and torch.isfinite(candidate_actions).all()):
            raise ValueError("nonfinite available robot prediction inputs")
        flag=authority.to(candidate_actions.dtype).unsqueeze(1).expand(-1,candidate_actions.shape[1],-1)
        factual_action=torch.cat((candidate_actions,flag),dim=-1)
        p=self.swarm.predict(belief,measured_state,factual_action)
        physical=measured_state.unsqueeze(1) + self.factual_delta_scale * p.next_observation
        return RobotForecast(p.outcome_logits,physical)

    def predict_trace(self,belief,measured_state,candidate_actions,current_human):
        """Optional diagnostic; same read-only candidate query as predict."""
        b=self.swarm._validate(belief)
        self._authority(current_human,b)
        if (measured_state.shape!=(b,DOF) or candidate_actions.ndim!=3
                or candidate_actions.shape[0]!=b or candidate_actions.shape[2]!=DOF):
            raise ValueError("invalid real robot trace query")
        flag=current_human.to(candidate_actions.dtype).unsqueeze(1).expand(-1,candidate_actions.shape[1],-1)
        return self.swarm.predict(belief,measured_state,torch.cat((candidate_actions,flag),-1))

    def verify_real_feedback(self,belief,executed_action,current_human,
            actual_next_state,predicted_next_state,physical_available,
            observed_takeover,predicted_takeover_logit,takeover_available):
        b=self.swarm._validate(belief)
        # Existing ME054 core verification enforces chronological finite/label
        # masks and bounded zero-innovation physical/onset state corrections.
        core=super().verify_real_feedback(
            belief.core,executed_action,current_human,actual_next_state,
            predicted_next_state,physical_available,observed_takeover,
            predicted_takeover_logit,takeover_available)
        p=torch.tanh((
            torch.where(physical_available,actual_next_state,torch.zeros_like(actual_next_state))-
            torch.where(physical_available,predicted_next_state.detach(),torch.zeros_like(predicted_next_state))
            )/self.factual_delta_scale)
        onset=torch.tanh(
            torch.where(takeover_available,observed_takeover,torch.zeros_like(observed_takeover))-
            torch.where(takeover_available,torch.sigmoid(predicted_takeover_logit.detach()),torch.zeros_like(predicted_takeover_logit)))
        signal=torch.tanh(self.physical_to_concept(p)+self.takeover_to_concept(onset))
        pair=torch.cat((belief.concepts,signal[:,None,:].expand_as(belief.concepts)),-1)
        propose=torch.tanh(self.concept_feedback(pair))
        gate=torch.sigmoid(self.concept_feedback_gate(pair))
        intensity=torch.clamp(p.abs().mean(-1,keepdim=True)+onset.abs(),max=1.0)
        # No observed innovation -> exact no-op, including all persistent
        # concepts. The correction is a bounded convex state interpolation.
        concepts=belief.concepts + .10*intensity.unsqueeze(-1)*gate*(propose-belief.concepts)
        return ConceptBelief(core,concepts)


class MatchedStableRobotGRU(GRURobotWorldModel):
    """Substantive all-trained GRU, widened only to match active capacity."""

    def __init__(self,delta_scale,hidden_size=40):
        super().__init__(hidden_size=hidden_size)
        self.register_buffer("factual_delta_scale",checked_scale(delta_scale))
        with torch.no_grad():
            self.physical_head[-1].weight.zero_()
            self.physical_head[-1].bias.zero_()

    def predict(self,state,measured_state,candidate_actions,current_human):
        p=super().predict(state,measured_state,candidate_actions,current_human)
        return RobotForecast(p.takeover_logits,
             measured_state.unsqueeze(1)+self.factual_delta_scale*p.next_joint_state)


def count_active(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def nearest_real_gru_width(delta_scale,*,lower=40,upper=80):
    """Deterministic architecture-only search, NEVER selected by outcomes."""
    tac=count_active(RealRobotConceptCED(delta_scale))
    sizes={width:count_active(MatchedStableRobotGRU(delta_scale,width))
           for width in range(lower,upper+1)}
    width=min(sizes,key=lambda w:(abs(sizes[w]-tac),w))
    gap=abs(sizes[width]-tac)/max(sizes[width],tac)
    return {"native_tac":tac,"gru_world":sizes[width],"gru_width":width,
            "relative_gap":gap}
