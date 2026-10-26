"""S1TEMP006: actual persistent slow memory inside SAME 88D state capacity.

Two implementations of one native 95,907-param streaming/batch graph:
  additive: exactly original 88D commutative accumulated fast summary.
  slow16:   first 72D cumulative fast summary, last 16D causal EMA slow.
Both retain associative 8x8, observed previous 14D, same output heads,
same total 166 floats/entity, and same learned weight sizes. Alpha=.0625 is
fixed a priori, not fitted from any TEST/DEV labels. No policy/causal claims.

Slow EMA batched computation uses bounded 32-frame parallel block scans and
one boundary carried state per 32-frame chunk: no unbounded q**(-T) values.
Streaming uses exactly the same recurrence, one O(16) additional state update.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .parallel_continuous_world import ParallelTACWorld
from .system1_parallel import System1ParallelWorld
from .system1_velocity_anchored import VelocityAnchoredSystem1
from .streaming_continuous_world import (
    WorldStreamState,world_stream_step
)

FAST_DIM=72
SLOW_DIM=16
ALPHA=0.0625
CHUNK=32
MODES=("additive","slow16")

def slow_scan(basis:torch.Tensor,alpha:float=ALPHA):
    """Exact recurrence up to float rounding; [B,T,16] -> [B,T,16]."""
    if (basis.ndim!=3 or basis.shape[-1]!=SLOW_DIM or basis.shape[1]<1
        or not torch.isfinite(basis).all().item()
        or not 0<alpha<1 or not math.isfinite(alpha)):
        raise ValueError("invalid original source slow-memory scan")
    q=1.-alpha
    prev=torch.zeros_like(basis[:,0])
    stages=[]
    for block in basis.split(CHUNK,dim=1):
        t=block.shape[1]
        power=q**torch.arange(1,t+1,device=block.device,dtype=block.dtype)
        # Stable: no inverse powers across more than one bounded 32 frame block
        prefix=torch.cumsum(alpha*block/power[None,:,None],dim=1)
        states=power[None,:,None]*(prev[:,None,:]+prefix)
        stages.append(states)
        prev=states[:,-1]
    return torch.cat(stages,dim=1)

def temporal_previous_context(basis:torch.Tensor,mode:str):
    if mode not in MODES or basis.ndim!=3 or basis.shape[-1]!=88:
        raise ValueError("invalid 88D matched temporal memory")
    B,T,W=basis.shape
    divider=torch.arange(T,device=basis.device,dtype=basis.dtype).clamp_min(1)[None,:,None]
    if mode=="additive":
        prior_sum=F.pad(basis.cumsum(dim=1)[:,:-1],(0,0,1,0))
        return prior_sum/divider,basis.cumsum(dim=1)
    fast_sum=basis[:,:,:FAST_DIM].cumsum(dim=1)
    prior_fast=F.pad(fast_sum[:,:-1],(0,0,1,0))/divider
    slow=slow_scan(basis[:,:,FAST_DIM:])
    prior_slow=F.pad(slow[:,:-1],(0,0,1,0))
    return torch.cat((prior_fast,prior_slow),dim=-1),torch.cat((fast_sum,slow),dim=-1)

class PersistentSlowTACWorld(ParallelTACWorld):
    """No new learned matrix or added state: mode changes only state update."""
    def __init__(self,mode:str="additive"):
        if mode not in MODES:
            raise ValueError("valid matched fast/slow memory mode required")
        super().__init__()
        self.temporal_mode=mode

    def forward(self,obs,action,*,return_trace=False):
        if self.temporal_mode=="additive":
            return super().forward(obs,action,return_trace=return_trace)
        if (obs.ndim!=3 or action.ndim!=3 or obs.shape[:2]!=action.shape[:2]
             or obs.shape[-1]!=self.state_dim or action.shape[-1]!=self.action_dim
             or obs.shape[1]<1):
            raise ValueError("invalid source state/action batch")
        B,T,_=obs.shape
        e=self.encoder(torch.cat((obs,action),-1)).tanh()
        k=F.normalize(self.address(e),dim=-1,eps=1e-6)
        v=F.normalize(self.content(e),dim=-1,eps=1e-6)
        zf=e.new_zeros(B,T,self.width)
        zr=e.new_zeros(B,T,8)
        basis,_=self.fast_map(torch.cat((e,zf,zr),-1)).chunk(2,-1)
        basis=basis.tanh()
        ctx,history=temporal_previous_context(basis,self.temporal_mode)
        p,fg=self.fast_map(torch.cat((e,ctx,zr),-1)).chunk(2,-1)
        pf=torch.lerp(ctx,p.tanh(),fg.sigmoid())
        delta,sg=self.slow_map(torch.cat((pf,ctx,zr),-1)).chunk(2,-1)
        ps=torch.lerp(ctx,(ctx+delta.tanh()).tanh(),.25*sg.sigmoid())
        alpha,beta=self.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1)
        alpha=alpha*(1-beta/4.0)
        # Same original parallel 8x8 associative scan and honest old-state reads.
        from .parallel_affine_memory import (
            associative_affine_schedule,work_efficient_affine_prefix)
        P,Q=associative_affine_schedule(
            k,v,alpha,beta,torch.ones_like(alpha,dtype=torch.bool))
        memory=work_efficient_affine_prefix(P,Q)
        prev=torch.cat((memory.new_zeros(B,1,8,8),memory[:,:-1]),1)
        r=(prev@k.unsqueeze(-1)).squeeze(-1)
        f,fg=self.fast_map(torch.cat((e,pf,r),-1)).chunk(2,-1)
        f=torch.lerp(pf,f.tanh(),fg.sigmoid())
        d,g=self.slow_map(torch.cat((f,ps,r),-1)).chunk(2,-1)
        s=torch.lerp(ps,(ps+d.tanh()).tanh(),.25*g.sigmoid())
        correction=self.readout(torch.cat((f,s,r,e),-1)).tanh()
        prediction=obs+self.delta_scale*correction
        if return_trace:
            return prediction,dict(memory=memory,old_read=r,fast=f,
                slow=s,sum_basis=history,persistent_slow=history[:,:,FAST_DIM:])
        return prediction

def temporal_world_stream_step(model, state, obs, action, *,return_trace=False):
    if model.temporal_mode=="additive":
        return world_stream_step(model,state,obs,action,return_trace=return_trace)
    B=state.fast_sum.shape[0]
    if (obs.shape!=(B,model.state_dim) or action.shape!=(B,model.action_dim)
        or state.fast_sum.shape!=(B,88) or state.memory.shape!=(B,8,8)
        or type(state.count) is not int or state.count<0):
        raise ValueError("invalid persistent streaming source tensors")
    e=model.encoder(torch.cat((obs,action),-1)[:,None]).tanh()
    k=F.normalize(model.address(e),dim=-1,eps=1e-6)
    v=F.normalize(model.content(e),dim=-1,eps=1e-6)
    zf=torch.zeros_like(state.fast_sum)[:,None]
    zr=e.new_zeros(B,1,8)
    basis,_=model.fast_map(torch.cat((e,zf,zr),-1)).chunk(2,-1)
    basis=basis.tanh()
    mean_fast=state.fast_sum[:,:FAST_DIM]/max(state.count,1)
    prev_slow=state.fast_sum[:,FAST_DIM:]
    ctx=torch.cat((mean_fast,prev_slow),-1)[:,None]
    p,fg=model.fast_map(torch.cat((e,ctx,zr),-1)).chunk(2,-1)
    pf=torch.lerp(ctx,p.tanh(),fg.sigmoid())
    d,g=model.slow_map(torch.cat((pf,ctx,zr),-1)).chunk(2,-1)
    ps=torch.lerp(ctx,(ctx+d.tanh()).tanh(),.25*g.sigmoid())
    alpha,beta=model.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1)
    alpha=alpha*(1-beta/4.)
    old=state.memory[:,None]
    read=(old@k.unsqueeze(-1)).squeeze(-1)
    memory=(alpha.unsqueeze(-1)*old+
        (beta*(v-alpha*read)).unsqueeze(-1)*k.unsqueeze(-2))[:,0]
    f,g=model.fast_map(torch.cat((e,pf,read),-1)).chunk(2,-1)
    f=torch.lerp(pf,f.tanh(),g.sigmoid())
    d,g=model.slow_map(torch.cat((f,ps,read),-1)).chunk(2,-1)
    slow_feature=torch.lerp(ps,(ps+d.tanh()).tanh(),.25*g.sigmoid())
    correction=model.readout(torch.cat((f,slow_feature,read,e),-1)).tanh()[:,0]
    forecast=obs+model.delta_scale*correction
    next_bank=torch.cat((
        state.fast_sum[:,:FAST_DIM]+basis[:,0,:FAST_DIM],
        (1-ALPHA)*prev_slow+ALPHA*basis[:,0,FAST_DIM:]),-1)
    next_state=WorldStreamState(next_bank,memory,state.count+1)
    if return_trace:
        return forecast,next_state,dict(memory=memory,old_read=read[:,0],
                                       fast=f[:,0],slow=slow_feature[:,0],
                                       persistent_slow=next_bank[:,FAST_DIM:])
    return forecast,next_state

class PersistentSlowSystem1Core(System1ParallelWorld):
    def __init__(self,mode:str):
        super().__init__()
        self.core=PersistentSlowTACWorld(mode)
    def step(self,state,observed,logged_action):
        self._validate_input(observed,logged_action,sequence=False)
        predicted,next_state,trace=temporal_world_stream_step(
            self.core,state,observed,logged_action,return_trace=True)
        return self._project(predicted,trace["fast"],trace["slow"],
                             trace["old_read"]),next_state

class VelocityAnchoredSlowSystem1(VelocityAnchoredSystem1):
    def __init__(self,*,memory_mode:str="additive"):
        super().__init__()
        self.core=PersistentSlowSystem1Core(memory_mode)
        self.memory_mode=memory_mode

    @property
    def temporal_slow_fp32_bytes(self):
        return 0 if self.memory_mode=="additive" else 4*SLOW_DIM
