"""ME055 candidate: bounded servo-residual TAC trajectory world model.

This is untrained architecture code. It combines a FIT-only identified,
shared robot servo prior with bounded learned residual dynamics and exposes
open-loop action-sequence rollout. Both TAC and GRU controls use the exact same
physical prior. Numerical tests are not evidence of real-world capability.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .real_robot_me054_stable import StableRealGRU, StableRealTAC
from .predictive_state_agent import Belief
from .real_robot_predictive_state import RobotForecast

DOF = 14


class TrajectoryForecast(NamedTuple):
    takeover_logits: torch.Tensor  # B x H x 1
    joint_state: torch.Tensor       # B x H x 14


class CompactBelief(NamedTuple):
    """Fast/slow state plus a diagonal relevance-conditioned memory trace."""
    fast: torch.Tensor
    slow: torch.Tensor
    memory: torch.Tensor


def _vector(value, name, *, strictly_positive=False, unit_interval=False):
    result = torch.as_tensor(value, dtype=torch.float32)
    if result.shape != (DOF,) or not torch.isfinite(result).all():
        raise ValueError(f"finite FIT-only {name} vector required")
    if strictly_positive and not torch.all(result > 0):
        raise ValueError(f"positive FIT-only {name} vector required")
    if unit_interval and not torch.all((result >= 0) & (result <= 1)):
        raise ValueError(f"FIT-only {name} must be in [0,1]")
    return result


class _ServoRollout(nn.Module):
    """Shared causal physical prior and open-loop rollout implementation."""

    def _init_servo(self, state_mean, state_scale, action_mean, action_scale,
                    servo_gain, servo_step_limit):
        self.register_buffer("servo_state_mean", _vector(state_mean, "state mean"))
        self.register_buffer("servo_state_scale", _vector(
            state_scale, "state scale", strictly_positive=True))
        self.register_buffer("servo_action_mean", _vector(action_mean, "action mean"))
        self.register_buffer("servo_action_scale", _vector(
            action_scale, "action scale", strictly_positive=True))
        self.register_buffer("servo_gain", _vector(
            servo_gain, "servo gain", unit_interval=True))
        self.register_buffer("servo_step_limit", _vector(
            servo_step_limit, "servo step limit", strictly_positive=True))

    def _action_target_in_state_coordinates(self, normalized_action: torch.Tensor) -> torch.Tensor:
        raw = normalized_action * self.servo_action_scale + self.servo_action_mean
        return (raw - self.servo_state_mean) / self.servo_state_scale

    def _servo_residual_prediction(self, inherited: RobotForecast,
                                   measured_state: torch.Tensor,
                                   candidate_actions: torch.Tensor) -> RobotForecast:
        target = self._action_target_in_state_coordinates(candidate_actions)
        displacement = self.servo_gain * (target - measured_state.unsqueeze(1))
        servo_step = torch.maximum(torch.minimum(displacement,
                                  self.servo_step_limit), -self.servo_step_limit)
        # The inherited head emits x + D*r. Rebound r with tanh so repeated
        # open-loop residuals cannot create the ME052-style unbounded jump.
        inherited_delta = inherited.next_joint_state - measured_state.unsqueeze(1)
        learned = self.factual_delta_scale * torch.tanh(
            inherited_delta / self.factual_delta_scale)
        state = measured_state.unsqueeze(1) + servo_step + learned
        return RobotForecast(inherited.takeover_logits, state)

    def rollout(self, belief, measured_state, planned_actions, current_human):
        """Predict H physical states under a supplied causal action plan.

        No future observation or feedback is read. The rollout is intended for
        autonomous starts; after a real takeover, future logged human actions
        are not a valid ex-ante plan and must be censored by the training/eval
        contract rather than supplied as privileged information.
        """
        if (planned_actions.ndim != 3 or planned_actions.shape[0] != len(measured_state)
                or planned_actions.shape[2] != DOF or planned_actions.shape[1] < 1):
            raise ValueError("B x H x 14 planned action sequence required")
        if current_human.shape != (len(measured_state), 1) or current_human.dtype != torch.bool:
            raise ValueError("known current control authority must be Bx1 boolean")
        if current_human.any():
            raise ValueError("open-loop candidate planning starts under robot control")
        if not torch.isfinite(planned_actions).all():
            raise ValueError("planned action sequence must be finite")
        state = measured_state
        memory = belief
        states, logits = [], []
        for step in range(planned_actions.shape[1]):
            memory = self._observe_training(memory, state)
            one = self._predict_training(
                memory, state, planned_actions[:, step:step+1], current_human)
            state = one.next_joint_state[:, 0]
            states.append(state)
            logits.append(one.takeover_logits[:, 0])
        return TrajectoryForecast(torch.stack(logits, 1), torch.stack(states, 1))


class ServoRolloutTAC(_ServoRollout, StableRealTAC):
    """Native fast/slow/associative state with bounded physical rollout."""

    def __init__(self, delta_scale, state_mean, state_scale, action_mean,
                 action_scale, servo_gain, servo_step_limit):
        StableRealTAC.__init__(self, delta_scale)
        self._init_servo(state_mean, state_scale, action_mean, action_scale,
                         servo_gain, servo_step_limit)
        # Fuse proposal, gate and address-key projections that consume the
        # same [observation16, fast32, recalled8] vector. The key gains a useful
        # recalled-memory input (64 coefficients), while all other coefficients
        # are copied exactly from the prior native cell.
        enc,transition,read_fast=self.native.enc,self.native.transition,self.native.read_fast
        gate,key=self.native.gate,self.native.key
        self.fast_projection=nn.Linear(56,81)
        with torch.no_grad():
            self.fast_projection.weight.zero_();self.fast_projection.bias.zero_()
            self.fast_projection.weight[:32,:16].copy_(enc.weight)
            self.fast_projection.weight[:32,16:48].copy_(transition.weight)
            self.fast_projection.weight[:32,48:].copy_(read_fast.weight)
            self.fast_projection.bias[:32].copy_(enc.bias)
            self.fast_projection.weight[32:64].copy_(gate.weight)
            self.fast_projection.bias[32:64].copy_(gate.bias)
            self.fast_projection.weight[64:72,:48].copy_(key.weight)
            self.fast_projection.bias[64:72].copy_(key.bias)
            nn.init.xavier_uniform_(self.fast_projection.weight[72:])
        del self.native.enc,self.native.transition,self.native.read_fast
        del self.native.gate,self.native.key
        # The associative read address is the bounded first eight coupled
        # fast/slow coordinates. Reinvest the former separate query map into
        # the fused projection as a learned write-address residual and strength.
        del self.native.query
        # Compute the associative write and next slow-state proposal from one
        # synchronous state snapshot. This removes the serial write-then-read
        # dependency inside a cell and turns the native update into one larger,
        # hardware-efficient projection. All added coefficients connect the
        # recalled memory to the slow gate or the coupled state to its proposal.
        old=(self.native.value,self.native.write,self.native.slow,
             self.native.slow_gate)
        self.state_projection=nn.Linear(72,73)
        with torch.no_grad():
            self.state_projection.weight.zero_()
            self.state_projection.bias.zero_()
            self.state_projection.weight[:9,:64].copy_(torch.cat(
                [layer.weight for layer in old[:2]],0))
            self.state_projection.bias[:9].copy_(torch.cat(
                [layer.bias for layer in old[:2]],0))
            self.state_projection.weight[9:41].copy_(old[2].weight)
            self.state_projection.bias[9:41].copy_(old[2].bias)
            self.state_projection.weight[41:,:64].copy_(old[3].weight)
            self.state_projection.bias[41:].copy_(old[3].bias)
            nn.init.xavier_uniform_(self.state_projection.weight[41:,64:])
        del self.native.value,self.native.write,self.native.slow
        del self.native.slow_gate
        # Physical and takeover verification use the same action address and
        # arrive together in this source. Fuse their keys, and let both 8D
        # values use the complete physical14+onset1 innovation. The added
        # cross-innovation coefficients are active learned capacity.
        pkey,skey=self.physical_key,self.native.verify_key
        pvalue,svalue=self.physical_value,self.native.verify_value
        self.feedback_key=nn.Linear(8,16)
        self.feedback_value=nn.Linear(15,16)
        with torch.no_grad():
            self.feedback_key.weight.copy_(torch.cat((pkey.weight,skey.weight),0))
            self.feedback_key.bias.copy_(torch.cat((pkey.bias,skey.bias),0))
            self.feedback_value.weight.zero_();self.feedback_value.bias.zero_()
            self.feedback_value.weight[:8,:14].copy_(pvalue.weight)
            self.feedback_value.bias[:8].copy_(pvalue.bias)
            self.feedback_value.weight[8:,14:].copy_(svalue.weight)
            self.feedback_value.bias[8:].copy_(svalue.bias)
        del self.physical_key,self.physical_value
        del self.native.verify_key,self.native.verify_value
        # One cross-modal innovation map replaces serial physical and takeover
        # correction maps. It can condition each bounded state correction on
        # both error sources at once. Reinvest the removed parameters in a
        # wider action/state prediction head, where they directly improve the
        # multi-step objective rather than adding recurrent kernel launches.
        physical_map,verify_map=self.physical_map,self.native.verify_map
        self.feedback_map=nn.Linear(87,128)
        with torch.no_grad():
            self.feedback_map.weight.zero_();self.feedback_map.bias.zero_()
            self.feedback_map.weight[:,:64].copy_(physical_map.weight[:,:64])
            self.feedback_map.weight[:,64:72].copy_(physical_map.weight[:,64:72])
            self.feedback_map.weight[:,72:86].copy_(physical_map.weight[:,72:])
            self.feedback_map.bias.copy_(physical_map.bias)
            nn.init.xavier_uniform_(self.feedback_map.weight[:,86:])
        del self.physical_map,self.native.verify_map
        old_readout=self.native.readout
        widened=nn.Sequential(nn.Linear(94,147),old_readout[1],nn.Linear(147,1))
        with torch.no_grad():
            widened[0].weight[:64].copy_(old_readout[0].weight)
            widened[0].bias[:64].copy_(old_readout[0].bias)
            nn.init.xavier_uniform_(widened[0].weight[64:])
            widened[0].bias[64:].zero_()
            widened[2].weight[:,:64].copy_(old_readout[2].weight)
            nn.init.xavier_uniform_(widened[2].weight[:,64:])
            widened[2].bias.copy_(old_readout[2].bias)
        self.native.readout=widened
        self.episode_initializer=nn.Linear(DOF,72)
        self.initializer_gate=nn.Parameter(torch.zeros(72))

    def initial(self, batch: int, device=None) -> CompactBelief:
        if batch < 1:
            raise ValueError("positive batch required")
        reference = next(self.parameters())
        options = {"device": reference.device if device is None else device,
                   "dtype": reference.dtype}
        return CompactBelief(torch.zeros(batch,32,**options),
                             torch.zeros(batch,32,**options),
                             torch.zeros(batch,8,**options))

    def initial_from_observation(self, context: torch.Tensor) -> CompactBelief:
        if context.ndim!=2 or context.shape[1]!=DOF or not torch.isfinite(context).all():
            raise ValueError("finite Bx14 initial observation required")
        encoded=torch.tanh(self.episode_initializer(context))*torch.sigmoid(
            self.initializer_gate)
        return CompactBelief(encoded[:,:32],encoded[:,32:64],encoded[:,64:])

    @staticmethod
    def _validate_compact(belief: CompactBelief) -> int:
        batch=belief.fast.shape[0]
        if (belief.fast.shape!=(batch,32) or belief.slow.shape!=(batch,32)
                or belief.memory.shape!=(batch,8)):
            raise ValueError("compact TAC belief must be Bx32, Bx32, Bx8")
        return batch

    def observe(self,belief,context):
        batch=self._validate_compact(belief)
        if context.shape!=(batch,DOF) or not torch.isfinite(context).all():
            raise ValueError("finite measured Bx14 state required")
        return self._observe_training(belief,context)

    def predict(self, belief, measured_state, candidate_actions, current_human):
        batch=self._validate_compact(belief)
        self._authority(current_human,batch)
        if (measured_state.shape!=(batch,DOF) or candidate_actions.ndim!=3
                or candidate_actions.shape[0]!=batch
                or candidate_actions.shape[2]!=DOF):
            raise ValueError("B x 14 state and B x A x 14 actions required")
        if not (torch.isfinite(measured_state).all()
                and torch.isfinite(candidate_actions).all()):
            raise ValueError("candidate physical inputs must be finite")
        return self._predict_training(
            belief,measured_state,candidate_actions,current_human)

    @torch.jit.export
    def _observe_training(self, belief: CompactBelief,
                          context: torch.Tensor) -> CompactBelief:
        """Shape-checked runner calls this equivalent fused training path."""
        native=self.native
        u=torch.tanh(native.observation(context))
        f,s,matrix=belief
        query=native.address(f[:,:8]+s[:,:8])
        recalled=matrix*query
        proposal,gate,key,key_residual,key_strength=self.fast_projection(
            torch.cat((u,f,recalled),-1)).split((32,32,8,8,1),-1)
        gate=torch.sigmoid(gate)
        f=f+gate*(torch.tanh(proposal)-f)
        joint=torch.cat((f,s),-1)
        value,write,innovation,slow_gate=self.state_projection(
            torch.cat((joint,recalled),-1)).split((8,1,32,32),-1)
        key=native.address(key+torch.sigmoid(key_strength)*torch.tanh(key_residual))
        value=torch.tanh(value)
        beta=torch.sigmoid(write)
        retained=(.95+.0499*torch.sigmoid(native.retention))*matrix
        matrix=retained+beta*(value-retained)*key
        s=s+torch.sigmoid(slow_gate)*(torch.tanh(s+torch.tanh(innovation))-s)
        return CompactBelief(f,s,torch.tanh(matrix))

    @torch.jit.export
    def _predict_training(self,belief: CompactBelief,measured_state: torch.Tensor,
                          candidate_actions: torch.Tensor,
                          current_human: torch.Tensor) -> RobotForecast:
        native=self.native
        flag=current_human.to(candidate_actions.dtype).unsqueeze(1)
        features=torch.cat((candidate_actions,flag.expand(-1,candidate_actions.shape[1],-1)),-1)
        action=torch.tanh(native.action(features))
        joint=torch.cat((belief.fast,belief.slow),-1)
        latent=joint.unsqueeze(1).expand(-1,action.shape[1],-1)
        recall=belief.memory.unsqueeze(1)*native.address(action)
        logits=native.readout(torch.cat((measured_state.unsqueeze(1).expand(
            -1,action.shape[1],-1),latent,action,recall),-1))
        raw=native.next_observation(torch.cat((latent,action),-1))
        inherited=RobotForecast(logits,measured_state.unsqueeze(1)+
                                 self.factual_delta_scale*raw)
        return self._servo_residual_prediction(inherited,measured_state,candidate_actions)

    @torch.jit.export
    def _verify_training(self,belief: CompactBelief,executed_action: torch.Tensor,
                         current_human: torch.Tensor,actual_next_state: torch.Tensor,
                         predicted_next_state: torch.Tensor,
                         observed_takeover: torch.Tensor,
                         predicted_takeover_logit: torch.Tensor) -> CompactBelief:
        native=self.native
        features=torch.cat((executed_action,current_human.to(executed_action.dtype)),-1)
        action=torch.tanh(native.action(features))
        joint=torch.cat((belief.fast,belief.slow),-1)
        physical=torch.tanh((actual_next_state-predicted_next_state.detach())/
                            self.factual_delta_scale)
        available=~current_human
        residual=torch.tanh(torch.where(available,observed_takeover,
                                        torch.zeros_like(observed_takeover))-
                            torch.where(available,torch.sigmoid(
                                predicted_takeover_logit.detach()),
                                torch.zeros_like(predicted_takeover_logit)))
        physical_value,scalar_value=self.feedback_value(
            torch.cat((physical,residual),-1)).split((8,8),-1)
        physical_key,scalar_key=self.feedback_key(action).split((8,8),-1)
        physical_intensity=physical.abs().mean(-1,keepdim=True)
        key=native.address(physical_key)
        value=torch.tanh(physical_value)
        target=value*key
        memory=belief.memory+.10*physical_intensity*(target-belief.memory)
        proposal,gate=self.feedback_map(
            torch.cat((joint,action,physical,residual),-1)).chunk(2,-1)
        intensity=1-(1-physical_intensity)*(1-residual.abs())
        joint=joint+.10*intensity*torch.sigmoid(gate)*(torch.tanh(proposal)-joint)
        key=native.address(scalar_key)
        value=torch.tanh(scalar_value)
        target=value*key
        memory=memory+.10*residual.abs()*(target-memory)
        return CompactBelief(joint[:,:32],joint[:,32:],memory)

    @torch.jit.export
    def _burnin_training(self, belief: CompactBelief,
                         current_state: torch.Tensor,
                         executed_action: torch.Tensor,
                         current_human: torch.Tensor,
                         next_state: torch.Tensor,
                         next_human: torch.Tensor) -> CompactBelief:
        """One fused observed transition for efficient recurrent burn-in."""
        belief = self._observe_training(belief, current_state)
        prediction = self._predict_training(
            belief, current_state, executed_action.unsqueeze(1), current_human)
        onset = (~current_human & next_human).to(current_state.dtype)
        return self._verify_training(
            belief, executed_action, current_human, next_state,
            prediction.next_joint_state[:, 0], onset,
            prediction.takeover_logits[:, 0])

    def verify_real_feedback(self,belief,executed_action,current_human,
            actual_next_state,predicted_next_state,physical_available,
            observed_takeover,predicted_takeover_logit,takeover_available):
        batch=self._validate_compact(belief)
        authority=self._authority(current_human,batch)
        physical=self._avail(physical_available,batch,"physical")
        onset=self._avail(takeover_available,batch,"onset")
        if not physical.all() or not torch.equal(onset,~authority):
            raise ValueError("ME055 training kernel requires measured physical feedback and causal onset availability")
        for value,mask in ((actual_next_state,physical),
                           (predicted_next_state,physical),
                           (observed_takeover,onset),
                           (predicted_takeover_logit,onset)):
            self._finite_when_available(value,mask)
        return self._verify_training(belief,executed_action,authority,
            actual_next_state,predicted_next_state,observed_takeover,
            predicted_takeover_logit)


class ServoRolloutGRU(_ServoRollout, StableRealGRU):
    """Established GRU control with identical bounded physical rollout prior."""

    def __init__(self, delta_scale, state_mean, state_scale, action_mean,
                 action_scale, servo_gain, servo_step_limit):
        StableRealGRU.__init__(self, delta_scale)
        self._init_servo(state_mean, state_scale, action_mean, action_scale,
                         servo_gain, servo_step_limit)

    def predict(self, state, measured_state, candidate_actions, current_human):
        inherited = StableRealGRU.predict(
            self, state, measured_state, candidate_actions, current_human)
        return self._servo_residual_prediction(
            inherited, measured_state, candidate_actions)

    @torch.jit.export
    def _observe_training(self,state: torch.Tensor,
                          measured_state: torch.Tensor) -> torch.Tensor:
        return self.observation_cell(torch.tanh(
            self.observe_encoder(measured_state)),state)

    @torch.jit.export
    def _predict_training(self,state: torch.Tensor,measured_state: torch.Tensor,
                          candidate_actions: torch.Tensor,
                          current_human: torch.Tensor) -> RobotForecast:
        authority=current_human.to(candidate_actions.dtype).unsqueeze(1)
        action=torch.tanh(self.action_encoder(torch.cat((candidate_actions,
            authority.expand(-1,candidate_actions.shape[1],-1)),-1)))
        latent=torch.cat((state.unsqueeze(1).expand(-1,action.shape[1],-1),action),-1)
        inherited=RobotForecast(self.takeover_head(latent),
            measured_state.unsqueeze(1)+self.factual_delta_scale*self.physical_head(latent))
        return self._servo_residual_prediction(inherited,measured_state,candidate_actions)

    @torch.jit.export
    def _verify_training(self,state: torch.Tensor,executed_action: torch.Tensor,
                         current_human: torch.Tensor,actual_next_state: torch.Tensor,
                         predicted_next_state: torch.Tensor,
                         observed_takeover: torch.Tensor,
                         predicted_takeover_logit: torch.Tensor) -> torch.Tensor:
        action=torch.tanh(self.action_encoder(torch.cat((executed_action,
            current_human.to(executed_action.dtype)),-1)))
        physical=torch.tanh(actual_next_state-predicted_next_state.detach())
        available=~current_human
        residual=torch.tanh(torch.where(available,observed_takeover,
                                        torch.zeros_like(observed_takeover))-
                            torch.where(available,torch.sigmoid(
                                predicted_takeover_logit.detach()),
                                torch.zeros_like(predicted_takeover_logit)))
        value=torch.cat((action,self.physical_residual(physical),
                         self.takeover_residual(residual)),-1)
        return self.feedback_cell(value,state)

    @torch.jit.export
    def _burnin_training(self, state: torch.Tensor,
                         current_state: torch.Tensor,
                         executed_action: torch.Tensor,
                         current_human: torch.Tensor,
                         next_state: torch.Tensor,
                         next_human: torch.Tensor) -> torch.Tensor:
        """One fused observed transition for efficient recurrent burn-in."""
        state = self._observe_training(state, current_state)
        prediction = self._predict_training(
            state, current_state, executed_action.unsqueeze(1), current_human)
        onset = (~current_human & next_human).to(current_state.dtype)
        return self._verify_training(
            state, executed_action, current_human, next_state,
            prediction.next_joint_state[:, 0], onset,
            prediction.takeover_logits[:, 0])


def factual_multistep_loss(model, forecast, measured_future,
                           physical_available, observed_onset, onset_available):
    """Score only genuine pre-intervention physical states and observed onsets."""
    if measured_future.shape != forecast.joint_state.shape:
        raise ValueError("measured future trajectory shape mismatch")
    b, horizon, width = measured_future.shape
    if width != DOF or forecast.takeover_logits.shape != (b, horizon, 1):
        raise ValueError("physical14/onset1 forecast contract required")
    if (physical_available.shape != (b, horizon, 1) or
            onset_available.shape != (b, horizon, 1) or
            physical_available.dtype != torch.bool or
            onset_available.dtype != torch.bool or
            observed_onset.shape != (b, horizon, 1)):
        raise ValueError("independent BxHx1 availability masks required")
    if not torch.isfinite(measured_future[physical_available.expand_as(measured_future)]).all():
        raise ValueError("available physical targets must be finite")
    labels = observed_onset[onset_available]
    if (not torch.isfinite(labels).all() or
            (labels.numel() and not torch.isin(labels, labels.new_tensor([0., 1.])).all())):
        raise ValueError("available onset targets must be finite binary values")
    normalized = ((forecast.joint_state - torch.where(
        physical_available, measured_future, torch.zeros_like(measured_future))) /
        model.factual_delta_scale).square().mean(-1, keepdim=True)
    physical = (normalized * physical_available).sum() / physical_available.sum().clamp_min(1)
    target = torch.where(onset_available, observed_onset, torch.zeros_like(observed_onset))
    bce = F.binary_cross_entropy_with_logits(
        forecast.takeover_logits, target, reduction="none")
    onset = (bce * onset_available).sum() / onset_available.sum().clamp_min(1)
    return physical + .2 * onset, {
        "open_loop_physical_delta_scaled_MSE": physical,
        "masked_takeover_onset_BCE": onset,
        "supervised_physical_steps": int(physical_available.sum()),
        "supervised_onset_steps": int(onset_available.sum()),
    }
