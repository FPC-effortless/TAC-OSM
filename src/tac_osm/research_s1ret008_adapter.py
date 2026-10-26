"""S1RET008 Research Observatory adapter: REAL source-typed drive ID bank.

Generic PR422 ResearchProbeAdapter remains architecture/version agnostic.
This trusted version-specific adapter explicitly maps source-known drive
identity (categorical), observed SMART vector, observed sensor-presence and
current source-priority to the actual native System1 graph.

Caveat: Backblaze has NO recorded repair actions. The 14D second encoder
channel is verified observed S.M.A.R.T. presence, never "logged action".
This adapter does not prove source provenance by reading tensor values;
the external original Backblaze member-SHA/date/serial reader is mandatory.
"""
from __future__ import annotations
from dataclasses import replace
import torch
from .research_probe import ResearchProbeAdapter
from .research_system1_adapter import (
    validate_system1_source,views,patch_state)
from .variants.s1ret008_entity_system1 import SourceEntitySystem1
from .variants.s1ret008_entity_affine import check_entity_inputs
from .variants.system1_velocity_anchored import MotionStreamState
from .variants.streaming_continuous_world import WorldStreamState

FIELDS=frozenset(("observation_history",
    "observed_smart_presence_history",
    "preceding_measured_observation",
    "source_entity_slot_history",
    "observed_priority_history"))

def source_inputs(s):
    if type(s) is not dict or set(s)!=FIELDS:
        raise ValueError("unauthorized drive input: no future outcome, repair, oracle")
    return dict(
        observation_history=s["observation_history"],
        prior_logged_action_history=s["observed_smart_presence_history"],
        preceding_measured_observation=s["preceding_measured_observation"])

def validate(model,s):
    if not isinstance(model,SourceEntitySystem1):
        raise ValueError("actual native source-entity System1 required")
    if model.forecast_anchor!="persistence":
        raise ValueError("cross-drive previous-event velocity would be invalid")
    core=source_inputs(s)
    validate_system1_source(model,core)
    check_entity_inputs(s["source_entity_slot_history"],
        s["observed_priority_history"],s["observation_history"],sequence=True)
    presence=s["observed_smart_presence_history"]
    if not bool(((presence==0)|(presence==1)).all().item()):
        raise ValueError("source SMART observed presence must be exact 0/1")
    if not bool((s["observation_history"].abs()<1e8).all().item()):
        raise ValueError("source SMART numerical value overflow")

def start(model,s):
    return model.initial_state(s["observation_history"].shape[0],
          previous_observation=s["preceding_measured_observation"])

def advance(model,state,s,t):
    validate(model,s)
    if not (0<=t<s["observation_history"].shape[1]-1):
        raise ValueError("no future or final source state may be advanced")
    _,next_state=model.step(state,s["observation_history"][:,t],
        s["observed_smart_presence_history"][:,t],
        source_slot=s["source_entity_slot_history"][:,t],
        observed_priority=s["observed_priority_history"][:,t])
    return next_state

def forecast(model,state,s):
    validate(model,s)
    y,_=model.step(state,s["observation_history"][:,-1],
        s["observed_smart_presence_history"][:,-1],
        source_slot=s["source_entity_slot_history"][:,-1],
        observed_priority=s["observed_priority_history"][:,-1])
    if y.is_calibrated or y.authorized_to_act:
        raise ValueError("failure-score model may not claim risk/safety qualification")
    return dict(
        source_next_day_observation=y.next_observation,
        source_future7day_failure_logit=y.adverse_event_logit,
        source_future7day_unqualified_score=y.adverse_event_score,
        uncalibrated_source_residual_scale=y.residual_scale)

def graph(model,state,s):
    validate(model,s)
    leaves={k:v.detach().clone().requires_grad_(True) for k,v in views(state).items()}
    leaf=MotionStreamState(WorldStreamState(
        leaves["fast_running_sum"],leaves["associative_memory"],
        state.core_state.count),leaves["previous_measured_observation"],
        state.has_previous)
    sensor=s["observation_history"][:,-1].detach().clone().requires_grad_(True)
    presence=s["observed_smart_presence_history"][:,-1].detach().clone().requires_grad_(True)
    priority=s["observed_priority_history"][:,-1].detach().clone().requires_grad_(True)
    sample=dict(s,
        observation_history=torch.cat((s["observation_history"][:,:-1].detach(),
                                      sensor[:,None]),dim=1),
        observed_smart_presence_history=torch.cat((
             s["observed_smart_presence_history"][:,:-1].detach(),
             presence[:,None]),dim=1),
        observed_priority_history=torch.cat((
             s["observed_priority_history"][:,:-1].detach(),
             priority[:,None]),dim=1))
    # Categorical entity slot cannot have a real derivative; it must be
    # evaluated via discrete source-slot permutation and isolation tests.
    return forecast(model,leaf,sample),dict(
        current_real_SMART_source=sensor,
        current_real_SMART_presence=presence,
        observed_relevance_score=priority,
        **leaves)

source_entity_system1_adapter=ResearchProbeAdapter(
    name="S1RET008-real-backblaze-source-entity-memory-v1",
    validate=validate,
    steps=lambda s:s["observation_history"].shape[1]-1,
    start=start,
    advance=advance,
    forecast=forecast,
    state_views=views,
    patch_state=patch_state,
    batch_size=lambda s:s["observation_history"].shape[0],
    batch_item=lambda s,i:{name:v[i:i+1] for name,v in s.items()},
    gradient_graph=graph,
    permute_candidates=None,
)
