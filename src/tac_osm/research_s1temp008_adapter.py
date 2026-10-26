"""Native S1TEMP008 exact source-time ID adapter to generic Observatory.

Explicit [B,T] source slots allowed, NO candidate interventions or oracle
labels. Trust of OfferID's original chronological provenance is an external
source-reader obligation and is NOT inferred from the integer.
"""
from __future__ import annotations
from dataclasses import replace
import torch
from .research_probe import ResearchProbeAdapter
from .research_system1_adapter import (
    validate_system1_source,start as source_start)
from .research_s1temp006_adapter import views,patch
from .variants.system1_velocity_anchored import MotionStreamState
from .variants.streaming_continuous_world import WorldStreamState
from .variants.s1temp008_entity_retention import validate_event_slots
from .variants.s1temp008_source_entity_world import SourceEntityVelocitySystem1
SOURCE=frozenset({
 "observation_history","prior_logged_action_history",
 "preceding_measured_observation","source_entity_slot_history"})
def validate(model,sample):
    if (not isinstance(model,SourceEntityVelocitySystem1) or
        type(sample) is not dict or set(sample)!=SOURCE):
        raise ValueError("actual source-indexed System1 and only allowed fields")
    subset={k:v for k,v in sample.items() if k!="source_entity_slot_history"}
    validate_system1_source(model,subset)
    ids=sample["source_entity_slot_history"]
    if not isinstance(ids,torch.Tensor):
        raise ValueError("actual source slots must be torch tensor")
    validate_event_slots(ids,tuple(sample["observation_history"].shape[:2]))
    if ids.device!=sample["observation_history"].device:
        raise ValueError("source ID device mismatch")
def start(model,sample):
    return source_start(model,sample)
def advance(model,state,sample,t):
    if not 0<=t<sample["observation_history"].shape[1]-1:
        raise ValueError("cannot advance beyond legal prefix")
    _,next_state=model.step_source(state,
        sample["observation_history"][:,t],
        sample["prior_logged_action_history"][:,t],
        sample["source_entity_slot_history"][:,t])
    return next_state
def forecast(model,state,sample):
    y,_=model.step_source(state,
        sample["observation_history"][:,-1],
        sample["prior_logged_action_history"][:,-1],
        sample["source_entity_slot_history"][:,-1])
    if y.is_calibrated or y.authorized_to_act:
        raise ValueError("cannot authorize unqualified source model")
    return {
       "next_observed_state":y.next_observation,
       "observed_motion_proxy_logit":y.adverse_event_logit,
       "observed_motion_proxy_score":y.adverse_event_score,
       "uncalibrated_dispersion":y.residual_scale,
    }
def graph(model,state,sample):
    named={k:v.detach().clone().requires_grad_(True)
           for k,v in views(state).items()}
    from .research_s1temp006_adapter import patch as replace_part
    leaf=state
    for name,replacement in named.items():
        leaf=replace_part(leaf,name,replacement)
    x=sample["observation_history"][:,-1].detach().clone().requires_grad_(True)
    u=sample["prior_logged_action_history"][:,-1].detach().clone().requires_grad_(True)
    query=dict(sample,
        observation_history=torch.cat((
            sample["observation_history"][:,:-1].detach(),x[:,None]),dim=1),
        prior_logged_action_history=torch.cat((
            sample["prior_logged_action_history"][:,:-1].detach(),u[:,None]),dim=1))
    return forecast(model,leaf,query),{
       "current_asof_observation":x,
       "prior_logged_observed_action":u,
       **named,
    }
source_entity_system1_adapter=ResearchProbeAdapter(
    name="S1TEMP008-original-source-entity-8slot-observational",
    validate=validate,
    steps=lambda s:s["observation_history"].shape[1]-1,
    start=start,advance=advance,forecast=forecast,
    state_views=views,patch_state=patch,
    batch_size=lambda s:s["observation_history"].shape[0],
    batch_item=lambda s,i:{k:v[i:i+1] for k,v in s.items()},
    gradient_graph=graph,
    permute_candidates=None,
)
