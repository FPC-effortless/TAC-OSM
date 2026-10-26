"""Trusted actual slow16 adapter for PR #422 model-agnostic Observatory.

The adapter, not the generic engine, knows which 72/16 carried tensors have
actual persistent temporal meaning. Any no-state/unsupported check is
INCONCLUSIVE rather than a substituted fake memory or safety assertion.
"""
from __future__ import annotations
from dataclasses import replace
import torch
from .research_probe import ResearchProbeAdapter
from .research_system1_adapter import (
    system1_velocity_adapter,
    validate_system1_source,
    forecast,
    start,
    advance,
)
from .variants.s1temp006_persistent_slow import VelocityAnchoredSlowSystem1
from .variants.system1_velocity_anchored import MotionStreamState
from .variants.streaming_continuous_world import WorldStreamState

def validate(model,sample):
    if (not isinstance(model,VelocityAnchoredSlowSystem1)
        or model.memory_mode!="slow16"):
        raise ValueError("persistent slow memory adapter only for actual slow16 source")
    validate_system1_source(model,sample)

def views(state):
    if not isinstance(state,MotionStreamState):
        raise ValueError("invalid model stream")
    fast=state.core_state.fast_sum
    if fast.ndim!=2 or fast.shape[-1]!=88:
        raise ValueError("slow memory expected original width 88")
    return dict(additive_fast72=fast[:,:72],
        persistent_slow16=fast[:,72:],
        associative_memory=state.core_state.memory,
        preceding_measured_observation=state.previous_observation)

def patch(state,name,value):
    original=views(state)
    if (name not in original or value.shape!=original[name].shape
        or value.dtype!=original[name].dtype
        or value.device!=original[name].device
        or not torch.isfinite(value).all().item()):
        raise ValueError("unauthorized/out-of-shape slow-state replacement")
    parts=dict(original);parts[name]=value
    core=WorldStreamState(
        torch.cat((parts["additive_fast72"],parts["persistent_slow16"]),dim=-1),
        parts["associative_memory"],state.core_state.count)
    return MotionStreamState(core,parts["preceding_measured_observation"],
                             state.has_previous)

def graph(model,state,sample):
    tensors={k:v.detach().clone().requires_grad_(True)
             for k,v in views(state).items()}
    leaf=patch(state,"additive_fast72",tensors["additive_fast72"])
    leaf=patch(leaf,"persistent_slow16",tensors["persistent_slow16"])
    leaf=patch(leaf,"associative_memory",tensors["associative_memory"])
    leaf=patch(leaf,"preceding_measured_observation",
               tensors["preceding_measured_observation"])
    x=sample["observation_history"][:,-1].detach().clone().requires_grad_(True)
    u=sample["prior_logged_action_history"][:,-1].detach().clone().requires_grad_(True)
    candidate=dict(sample,
        observation_history=torch.cat((sample["observation_history"][:,:-1].detach(),
                                       x[:,None]),dim=1),
        prior_logged_action_history=torch.cat((
            sample["prior_logged_action_history"][:,:-1].detach(),u[:,None]),dim=1))
    return forecast(model,leaf,candidate),dict(
        current_observed_state=x,prior_logged_action=u,**tensors)

persistent_slow16_adapter=replace(system1_velocity_adapter,
    name="S1TEMP006-actual-persistent-slow16-observational-v1",
    validate=validate,
    state_views=views,
    patch_state=patch,
    gradient_graph=graph,
)
