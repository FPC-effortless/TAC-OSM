"""S1RET008: native System1 source-addressed observational entity memory.

All learned modules, initial weights, 95,907 params and the original 166
float / 664 byte streaming state are unchanged. The existing 8x8 matrix
becomes 8 individually stable, fixed-source-identity rows (8 floats each).
Read/write use same source_slot for both fit/control arms; observed
relevance differs only in write-admission (entity_all vs entity_priority).

Source slot is not internally invented or hashed. A provenance/audit layer
must map approved identifiers to exactly eight stable cohort slots before
each run, and verify current observation date, entity and future cutoff.
For interleaved cross-drive telemetry use forecast_anchor="persistence", because
previous stream event may belong to a DIFFERENT drive; do not invent its
velocity. All real drive arms, including global and no-memory, use same anchor.
The GLOBAL legacy comparator can ignore identity; it has unequal semantic
side information and cannot establish fully input-matched superiority.

Observed previous actions are recorded facts only, not interventions;
Backblaze SMART lacks executed maintenance commands. Never claim do(a).
"""
from __future__ import annotations
from dataclasses import replace
import torch
from torch.nn import functional as F
from .parallel_continuous_world import ParallelTACWorld
from .parallel_affine_memory import associative_affine_schedule,work_efficient_affine_prefix
from .streaming_continuous_world import WorldStreamState,world_stream_step
from .system1_parallel import System1ParallelWorld
from .system1_velocity_anchored import VelocityAnchoredSystem1,MotionStreamState
from .s1ret008_entity_affine import (
    MODES,SLOTS,THRESHOLD,check_entity_inputs,
    entity_affine_prefix,entity_row_step,source_row_read,
)

class SourceEntityTACWorld(ParallelTACWorld):
    def __init__(self,mode="entity_priority"):
        if mode not in MODES:
            raise ValueError("actual source entity mode required")
        super().__init__()
        self.entity_mode=mode

    def forward(self,obs,action,*,source_slot,observed_priority,return_trace=False):
        check_entity_inputs(source_slot,observed_priority,obs,sequence=True)
        if self.entity_mode=="global":
            return super().forward(obs,action,return_trace=return_trace)
        if (obs.ndim!=3 or action.shape!=obs.shape or
            obs.shape[-1]!=self.state_dim):
            raise ValueError("actual source continuous 14D observed state/action needed")
        B,T,_=obs.shape
        e=self.encoder(torch.cat((obs,action),dim=-1)).tanh()
        # Source slot fixes the row, while learned address shapes the row read.
        k=F.normalize(self.address(e),dim=-1,eps=1e-6)
        v=F.normalize(self.content(e),dim=-1,eps=1e-6)
        zf=e.new_zeros(B,T,self.width)
        zr=e.new_zeros(B,T,8)
        basis,_=self.fast_map(torch.cat((e,zf,zr),-1)).chunk(2,-1)
        basis=basis.tanh()
        context=F.pad(basis.cumsum(dim=1)[:,:-1],(0,0,1,0))
        context=context/torch.arange(T,device=e.device,dtype=e.dtype).clamp_min(1).view(1,T,1)
        p,g=self.fast_map(torch.cat((e,context,zr),-1)).chunk(2,-1)
        pf=torch.lerp(context,p.tanh(),g.sigmoid())
        d,g=self.slow_map(torch.cat((pf,context,zr),-1)).chunk(2,-1)
        ps=torch.lerp(context,(context+d.tanh()).tanh(),.25*g.sigmoid())
        alpha,beta=self.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1)
        admitted_gain=beta*(1-alpha/4)
        history,_,_=entity_affine_prefix(
            v,admitted_gain.squeeze(-1),source_slot,observed_priority,self.entity_mode)
        previous=torch.cat((history.new_zeros(B,1,SLOTS,8),
                            history[:,:-1]),dim=1)
        old_read=source_row_read(previous,source_slot)*k
        f,g=self.fast_map(torch.cat((e,pf,old_read),-1)).chunk(2,-1)
        fast=torch.lerp(pf,f.tanh(),g.sigmoid())
        d,g=self.slow_map(torch.cat((fast,ps,old_read),-1)).chunk(2,-1)
        slow=torch.lerp(ps,(ps+d.tanh()).tanh(),.25*g.sigmoid())
        correction=self.readout(torch.cat((fast,slow,old_read,e),-1)).tanh()
        prediction=obs+self.delta_scale*correction
        if return_trace:
            return prediction,dict(memory=history,old_read=old_read,
                fast=fast,slow=slow,sum_basis=basis.cumsum(dim=1))
        return prediction

def entity_source_step(model,state,observed,prior_action,slot,priority,*,return_trace=False):
    if model.entity_mode=="global":
        return world_stream_step(model,state,observed,prior_action,
                                 return_trace=return_trace)
    B=observed.shape[0]
    check_entity_inputs(slot,priority,observed,sequence=False)
    if (prior_action.shape!=observed.shape or observed.shape!=(B,14)
        or state.fast_sum.shape!=(B,88) or state.memory.shape!=(B,8,8)
        or type(state.count) is not int or state.count<0):
        raise ValueError("invalid source-timestamped state")
    x=observed[:,None];u=prior_action[:,None]
    e=model.encoder(torch.cat((x,u),-1)).tanh()
    k=F.normalize(model.address(e),dim=-1,eps=1e-6)
    v=F.normalize(model.content(e),dim=-1,eps=1e-6)
    zf=e.new_zeros(B,1,88);zr=e.new_zeros(B,1,8)
    basis,_=model.fast_map(torch.cat((e,zf,zr),-1)).chunk(2,-1)
    basis=basis.tanh()
    ctx=(state.fast_sum/max(1,state.count))[:,None]
    p,g=model.fast_map(torch.cat((e,ctx,zr),-1)).chunk(2,-1)
    pf=torch.lerp(ctx,p.tanh(),g.sigmoid())
    d,g=model.slow_map(torch.cat((pf,ctx,zr),-1)).chunk(2,-1)
    ps=torch.lerp(ctx,(ctx+d.tanh()).tanh(),.25*g.sigmoid())
    alpha,beta=model.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1)
    admitted_gain=beta*(1-alpha/4)
    old_read=(source_row_read(state.memory,slot)[:,None]*k)
    updated=entity_row_step(state.memory,v[:,0],admitted_gain[:,0,0],slot,
                            priority,model.entity_mode)
    fast,g=model.fast_map(torch.cat((e,pf,old_read),-1)).chunk(2,-1)
    fast=torch.lerp(pf,fast.tanh(),g.sigmoid())
    d,g=model.slow_map(torch.cat((fast,ps,old_read),-1)).chunk(2,-1)
    slow=torch.lerp(ps,(ps+d.tanh()).tanh(),.25*g.sigmoid())
    point=observed+model.delta_scale*model.readout(
        torch.cat((fast,slow,old_read,e),-1)).tanh()[:,0]
    new_state=WorldStreamState(state.fast_sum+basis[:,0],updated,state.count+1)
    if return_trace:
        return point,new_state,dict(memory=updated,
            fast=fast[:,0],slow=slow[:,0],old_read=old_read[:,0])
    return point,new_state

class SourceEntitySystem1Core(System1ParallelWorld):
    def __init__(self,mode):
        super().__init__()
        self.core=SourceEntityTACWorld(mode)
    def forward(self,observed,logged_action,*,source_slot,observed_priority):
        self._validate_input(observed,logged_action,sequence=True)
        p,t=self.core(observed,logged_action,source_slot=source_slot,
                      observed_priority=observed_priority,return_trace=True)
        return self._project(p,t["fast"],t["slow"],t["old_read"])
    def step(self,state,observed,logged_action,*,source_slot,observed_priority):
        self._validate_input(observed,logged_action,sequence=False)
        if (not isinstance(state,WorldStreamState) or
            state.fast_sum.shape!=(observed.shape[0],88) or
            state.memory.shape!=(observed.shape[0],8,8) or
            not torch.isfinite(state.fast_sum).all().item() or
            not torch.isfinite(state.memory).all().item()):
            raise ValueError("unauthorized/malformed source entity bank")
        p,next_state,t=entity_source_step(self.core,state,observed,logged_action,
            source_slot,observed_priority,return_trace=True)
        return self._project(p,t["fast"],t["slow"],t["old_read"]),next_state

class SourceEntitySystem1(VelocityAnchoredSystem1):
    def __init__(self,*,entity_mode="entity_priority",forecast_anchor="velocity"):
        if forecast_anchor not in ("velocity","persistence"):
            raise ValueError("explicit causal physical forecast prior required")
        super().__init__()
        self.core=SourceEntitySystem1Core(entity_mode)
        self.entity_mode=entity_mode
        self.forecast_anchor=forecast_anchor
    def forward(self,observed,prior_logged_action,*,source_slot,
                observed_priority,initial_previous_observation=None):
        check_entity_inputs(source_slot,observed_priority,observed,sequence=True)
        forecast=self.core(observed,prior_logged_action,
            source_slot=source_slot,observed_priority=observed_priority)
        previous=self._prior(observed,initial_previous_observation)
        anchor=(2*observed-previous if self.forecast_anchor=="velocity"
                else observed)
        physical=anchor+self.correction_gain*(forecast.next_observation-observed)
        return replace(forecast,next_observation=physical)
    def step(self,state,observed,prior_logged_action,*,source_slot,observed_priority):
        if not isinstance(state,MotionStreamState):
            raise ValueError("invalid source prior state")
        check_entity_inputs(source_slot,observed_priority,observed,sequence=False)
        f,next_state=self.core.step(state.core_state,observed,prior_logged_action,
                                    source_slot=source_slot,
                                    observed_priority=observed_priority)
        previous=state.previous_observation if state.has_previous else observed
        if previous.shape!=observed.shape or not torch.isfinite(previous).all().item():
            raise ValueError("untrusted previous source observation")
        anchor=(2*observed-previous if self.forecast_anchor=="velocity"
                else observed)
        p=anchor+self.correction_gain*(f.next_observation-observed)
        return replace(f,next_observation=p),MotionStreamState(
            next_state,observed.detach().clone(),True)
