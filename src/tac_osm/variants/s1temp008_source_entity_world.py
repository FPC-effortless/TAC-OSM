"""S1TEMP008: native System1 source-indexed 8x8 memory, unchanged trained weights.

Uses the already available 72D fast + 16D multi-timescale slow carry and
8x8 associative matrix. Trustworthy external source-slot [B,T] or [B]
is compulsory in BOTH modes. 'source' k is source-authorized onehot;
'oblivious' k is learned from the same source observation without identity.
Both modes use EXACTLY THE SAME trainable matrices (95,907 parameters),
same 664-byte floating carried state and same local gamma gate/readout.

The address weight participates in both via value=(content+.1*address),
so the feature does not silently disable trainable parameters in the source
arm. This is an experimental paired controls graph, not a trained model,
loan approval advice, causal do(action), or calibrated event risk.

XES parser must assign source offer slots strictly from prior chronology.
"""
from __future__ import annotations
from dataclasses import replace
import torch
from torch.nn import functional as F
from .s1temp006_persistent_slow import (
    PersistentSlowTACWorld,PersistentSlowSystem1Core,
    VelocityAnchoredSlowSystem1,temporal_previous_context,
    FAST_DIM,MULTISCALE_RATES,ALPHA)
from .s1temp008_entity_retention import (
    indexed_affine_schedule,validate_event_slots)
from .parallel_affine_memory import work_efficient_affine_prefix
from .streaming_continuous_world import WorldStreamState
from .system1_velocity_anchored import MotionStreamState

MODES=("source_indexed","oblivious")
def key_schedule(e,slots,mode,address):
    """Return learned-gate-compatible keys, no future source inputs."""
    if mode not in MODES:raise ValueError("invalid original source identity arm")
    if mode=="source_indexed":
        return F.one_hot(slots,8).to(dtype=e.dtype)
    return F.normalize(address(e),dim=-1,eps=1e-6)

def _affine_schedule(k,v,alpha,beta):
    """Identical per-address selective update for both control and treatment."""
    gamma=beta*(1-alpha)
    eye=torch.eye(8,dtype=k.dtype,device=k.device).view(1,1,8,8)
    P=eye-gamma.unsqueeze(-1)*(k.unsqueeze(-1)*k.unsqueeze(-2))
    Q=gamma.unsqueeze(-1)*v.unsqueeze(-1)*k.unsqueeze(-2)
    return P,Q

class SourceEntityTACWorld(PersistentSlowTACWorld):
    def __init__(self,*,address_mode:str="source_indexed"):
        if address_mode not in MODES:
            raise ValueError("source or oblivious mode only")
        super().__init__("multiscale16")
        self.address_mode=address_mode

    def forward(self,obs,action,source_slot,*,return_trace=False):
        if (obs.ndim!=3 or action.shape!=obs.shape
            or obs.shape[-1]!=14 or obs.shape[1]<1
            or obs.dtype!=next(self.parameters()).dtype):
            raise ValueError("observed source real state/actions need [B,T,14]")
        B,T,_=obs.shape
        validate_event_slots(source_slot,(B,T))
        if source_slot.device!=obs.device:
            raise ValueError("entity slots must be on source's own device")
        e=self.encoder(torch.cat((obs,action),-1)).tanh()
        address=self.address(e)
        k=key_schedule(e,source_slot,self.address_mode,self.address)
        v=F.normalize(self.content(e)+.1*address,dim=-1,eps=1e-6)
        zeros_f=e.new_zeros(B,T,88);zeros_r=e.new_zeros(B,T,8)
        basis,_=self.fast_map(torch.cat((e,zeros_f,zeros_r),-1)).chunk(2,-1)
        basis=basis.tanh()
        ctx,history=temporal_previous_context(basis,"multiscale16")
        p,fg=self.fast_map(torch.cat((e,ctx,zeros_r),-1)).chunk(2,-1)
        pf=torch.lerp(ctx,p.tanh(),fg.sigmoid())
        d,g=self.slow_map(torch.cat((pf,ctx,zeros_r),-1)).chunk(2,-1)
        ps=torch.lerp(ctx,(ctx+d.tanh()).tanh(),.25*g.sigmoid())
        alpha,beta=self.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1)
        alpha=alpha*(1-beta/4.)
        P,Q=_affine_schedule(k,v,alpha,beta)
        memory=work_efficient_affine_prefix(P,Q)
        old=torch.cat((memory.new_zeros(B,1,8,8),memory[:,:-1]),1)
        read=(old@k.unsqueeze(-1)).squeeze(-1)
        f,g=self.fast_map(torch.cat((e,pf,read),-1)).chunk(2,-1)
        fast=torch.lerp(pf,f.tanh(),g.sigmoid())
        d,g=self.slow_map(torch.cat((fast,ps,read),-1)).chunk(2,-1)
        slow=torch.lerp(ps,(ps+d.tanh()).tanh(),.25*g.sigmoid())
        correction=self.readout(torch.cat((fast,slow,read,e),-1)).tanh()
        result=obs+self.delta_scale*correction
        if return_trace:
            return result,dict(fast=fast,slow=slow,old_read=read,
                memory=memory,sum_basis=history,
                persistent_slow=history[:,:,FAST_DIM:],
                addressed_source_slot=source_slot)
        return result

    def step_source(self,state:WorldStreamState,obs,action,source_slot,*,return_trace=False):
        B=obs.shape[0]
        if (obs.shape!=(B,14) or action.shape!=(B,14)
            or state.fast_sum.shape!=(B,88) or state.memory.shape!=(B,8,8)
            or type(state.count) is not int or state.count<0):
            raise ValueError("invalid actual source history/current observation")
        validate_event_slots(source_slot,(B,))
        if source_slot.device!=obs.device:
            raise ValueError("mismatched source entity slot device")
        e=self.encoder(torch.cat((obs,action),-1)[:,None]).tanh()
        address=self.address(e)
        k=key_schedule(e,source_slot[:,None],self.address_mode,self.address)
        v=F.normalize(self.content(e)+.1*address,dim=-1,eps=1e-6)
        zf=torch.zeros_like(state.fast_sum)[:,None]
        zr=e.new_zeros(B,1,8)
        basis,_=self.fast_map(torch.cat((e,zf,zr),-1)).chunk(2,-1)
        basis=basis.tanh()
        ctx=torch.cat((
            state.fast_sum[:,:FAST_DIM]/max(state.count,1),
            state.fast_sum[:,FAST_DIM:]),dim=-1)[:,None]
        p,g=self.fast_map(torch.cat((e,ctx,zr),-1)).chunk(2,-1)
        pf=torch.lerp(ctx,p.tanh(),g.sigmoid())
        d,g=self.slow_map(torch.cat((pf,ctx,zr),-1)).chunk(2,-1)
        ps=torch.lerp(ctx,(ctx+d.tanh()).tanh(),.25*g.sigmoid())
        alpha,beta=self.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1)
        alpha=alpha*(1-beta/4.)
        P,Q=_affine_schedule(k,v,alpha,beta)
        old=state.memory[:,None]
        read=(old@k.unsqueeze(-1)).squeeze(-1)
        updated=(old@P+Q)[:,0]
        fast,g=self.fast_map(torch.cat((e,pf,read),-1)).chunk(2,-1)
        fast=torch.lerp(pf,fast.tanh(),g.sigmoid())
        d,g=self.slow_map(torch.cat((fast,ps,read),-1)).chunk(2,-1)
        slow=torch.lerp(ps,(ps+d.tanh()).tanh(),.25*g.sigmoid())
        correction=self.readout(torch.cat((fast,slow,read,e),-1)).tanh()[:,0]
        predicted=obs+self.delta_scale*correction
        rates=obs.new_tensor(MULTISCALE_RATES).repeat_interleave(4)
        next_fast=torch.cat((
            state.fast_sum[:,:FAST_DIM]+basis[:,0,:FAST_DIM],
            (1-rates)*state.fast_sum[:,FAST_DIM:]+
            rates*basis[:,0,FAST_DIM:]),dim=-1)
        next_state=WorldStreamState(next_fast,updated,state.count+1)
        if return_trace:
            return predicted,next_state,dict(
                fast=fast[:,0],slow=slow[:,0],old_read=read[:,0],
                memory=updated,
                persistent_slow=next_fast[:,FAST_DIM:])
        return predicted,next_state

class SourceEntitySystem1Core(PersistentSlowSystem1Core):
    def __init__(self,*,address_mode:str):
        super().__init__("multiscale16")
        self.core=SourceEntityTACWorld(address_mode=address_mode)

    def forward(self,observed,logged_action,source_slot):
        self._validate_input(observed,logged_action,sequence=True)
        predicted,tr=self.core(observed,logged_action,source_slot,return_trace=True)
        return self._project(predicted,tr["fast"],tr["slow"],tr["old_read"])

    def step_source(self,state,observed,logged_action,source_slot):
        self._validate_input(observed,logged_action,sequence=False)
        predicted,next_state,tr=self.core.step_source(
            state,observed,logged_action,source_slot,return_trace=True)
        return self._project(predicted,tr["fast"],tr["slow"],tr["old_read"]),next_state

class SourceEntityVelocitySystem1(VelocityAnchoredSlowSystem1):
    """Observational actual as-of offer source entity and shared 664B graph."""
    def __init__(self,*,address_mode:str="source_indexed"):
        super().__init__(memory_mode="multiscale16")
        self.core=SourceEntitySystem1Core(address_mode=address_mode)
        self.address_mode=address_mode

    def forward(self,observed,prior_logged_action,source_slot,*,
                initial_previous_observation=None):
        result=self.core(observed,prior_logged_action,source_slot)
        prev=self._prior(observed,initial_previous_observation)
        return replace(result,next_observation=2*observed-prev+
            self.correction_gain*(result.next_observation-observed))

    def step_source(self,state:MotionStreamState,observed,prior_logged_action,
                    source_slot):
        if not isinstance(state,MotionStreamState):
            raise ValueError("invalid real source stream state")
        result,next_state=self.core.step_source(
            state.core_state,observed,prior_logged_action,source_slot)
        prev=state.previous_observation if state.has_previous else observed
        if (prev.shape!=observed.shape or prev.dtype!=observed.dtype
            or prev.device!=observed.device or not torch.isfinite(prev).all().item()):
            raise ValueError("source previous state invalid")
        output=replace(result,next_observation=2*observed-prev+
            self.correction_gain*(result.next_observation-observed))
        return output,MotionStreamState(next_state,observed.detach().clone(),True)

    def step(self,*args,**kwargs):
        raise ValueError("explicit source_entity_slot required: call step_source")
