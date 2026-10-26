"""S1MOTION006: 14D zero-new-parameter causal measured-velocity ablation.

The native System1 state, parallel associative scan, bounded velocity-anchored
point prediction, uncalibrated motion head and stream state are preserved.
Only the already-existing last 14D auxiliary evidence vector changes:
 control  -> core_predicted_state
 treatment-> core_predicted_state + (x[t] - x[t-1])
No learned params or extra carried state. This is observational, not causal
command rollout, reward, collision or safety estimation.

Identical forward base output is an exact source regression, not a new train.
"""
from __future__ import annotations
from dataclasses import replace
from typing import Literal
import torch
from torch.nn import functional as F
from .system1_velocity_anchored import VelocityAnchoredSystem1, MotionStreamState
from .system1_parallel import System1Forecast
from .streaming_continuous_world import world_stream_step

FEATURES=("core_prediction","core_plus_measured_velocity")

class VelocityEventSystem1(VelocityAnchoredSystem1):
    def __init__(self,*,feature_mode:Literal["core_prediction","core_plus_measured_velocity"]="core_prediction"):
        if feature_mode not in FEATURES:
            raise ValueError("source-authorized evidence variant required")
        super().__init__()
        self.feature_mode=feature_mode

    def _forecast(self, core_prediction, fast, slow, old_read,
                  observed, prev):
        velocity=observed-prev
        aux_feature=(core_prediction if self.feature_mode=="core_prediction"
                     else core_prediction+velocity)
        evidence=torch.cat((fast,slow,old_read,aux_feature),dim=-1)
        raw=self.core.reactive_head(evidence)
        logit=raw[...,0]
        dispersion=F.softplus(raw[...,1:])+1e-4
        forecast=2*observed-prev+self.correction_gain*(core_prediction-observed)
        return System1Forecast(forecast,logit,logit.sigmoid(),dispersion)

    def forward(self, observed, prior_logged_action, *,
                initial_previous_observation=None):
        self.core._validate_input(observed,prior_logged_action,sequence=True)
        prev=self._prior(observed,initial_previous_observation)
        prediction,trace=self.core.core(observed,prior_logged_action,return_trace=True)
        return self._forecast(prediction,trace["fast"],trace["slow"],
            trace["old_read"],observed,prev)

    def step(self,state:MotionStreamState,observed,prior_logged_action):
        if not isinstance(state,MotionStreamState) or type(state.has_previous) is not bool:
            raise ValueError("invalid motion stream state")
        self.core._validate_input(observed,prior_logged_action,sequence=False)
        old=state.previous_observation if state.has_previous else observed
        if (old.shape!=observed.shape or old.device!=observed.device
                or old.dtype!=observed.dtype or
                not torch.isfinite(old).all().item()):
            raise ValueError("invalid source preceding measured state")
        core=state.core_state
        if (core.fast_sum.shape!=(observed.shape[0],self.core.core.width) or
                core.memory.shape!=(observed.shape[0],8,8) or
                type(core.count) is not int or core.count<0):
            raise ValueError("invalid actual carried world memory")
        prediction,next_core,trace=world_stream_step(
            self.core.core,core,observed,prior_logged_action,return_trace=True)
        f=self._forecast(prediction,trace["fast"],trace["slow"],
            trace["old_read"],observed,old)
        return f,MotionStreamState(next_core,observed.detach().clone(),True)
