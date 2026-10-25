"""Fixed-size O(1)-history inference for the continuous robot TAC variant.

This is the STEP evaluator of ParallelTACWorld's parallel model, not a
separate set of learned weights. Its sufficient state is (running_fast_sum
[B,88], associative_matrix[B,8,8], integer event_count) and has 152 float
values/sample regardless of observed history. Each step costs O(8²) for
rank-one memory update, with constant dense-layer arithmetic. No future state,
trajectory cache or real robotic control is required/authorized.

Numerically compared to trained or untrained parallel batch graph using
explicit tolerances; floating point scan regrouping is not bit-identical.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class WorldStreamState:
    fast_sum: torch.Tensor
    memory: torch.Tensor
    count: int


def init_world_stream(model,batch:int):
    if batch<1:raise ValueError("batch>=1 required")
    p=next(model.parameters())
    return WorldStreamState(p.new_zeros(batch,model.width),
                            p.new_zeros(batch,8,8),0)


def world_stream_step(model,state:WorldStreamState,obs:torch.Tensor,
                      action:torch.Tensor,*,return_trace:bool=False):
    B=state.fast_sum.shape[0]
    if (obs.shape!=(B,model.state_dim) or
        action.shape!=(B,model.action_dim) or
        state.memory.shape!=(B,8,8) or state.count<0):
        raise ValueError("invalid stream observation/action/state")
    x=obs.unsqueeze(1);a=action.unsqueeze(1)
    e=model.encoder(torch.cat((x,a),-1)).tanh()
    k=F.normalize(model.address(e),dim=-1,eps=1e-6)
    v=F.normalize(model.content(e),dim=-1,eps=1e-6)
    zeros_f=torch.zeros_like(state.fast_sum).unsqueeze(1)
    zeros_r=e.new_zeros(B,1,8)
    basis,_=model.fast_map(torch.cat((e,zeros_f,zeros_r),-1)).chunk(2,-1)
    basis=basis.tanh()
    ctx=(state.fast_sum/max(state.count,1)).unsqueeze(1)
    p,g=model.fast_map(torch.cat((e,ctx,zeros_r),-1)).chunk(2,-1)
    pf=torch.lerp(ctx,p.tanh(),g.sigmoid())
    d,g=model.slow_map(torch.cat((pf,ctx,zeros_r),-1)).chunk(2,-1)
    ps=torch.lerp(ctx,(ctx+d.tanh()).tanh(),.25*g.sigmoid())
    alpha,beta=model.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1)
    alpha=alpha*(1-beta/4.)
    old=state.memory.unsqueeze(1)
    read=(old@k.unsqueeze(-1)).squeeze(-1)
    updated=alpha.unsqueeze(-1)*old+(
        beta*(v-alpha*read)).unsqueeze(-1)*k.unsqueeze(-2)
    memory=updated[:,0]
    f,g=model.fast_map(torch.cat((e,pf,read),-1)).chunk(2,-1)
    f=torch.lerp(pf,f.tanh(),g.sigmoid())
    d,g=model.slow_map(torch.cat((f,ps,read),-1)).chunk(2,-1)
    s=torch.lerp(ps,(ps+d.tanh()).tanh(),.25*g.sigmoid())
    correction=model.readout(torch.cat((f,s,read,e),-1)).tanh()[:,0]
    prediction=obs+model.delta_scale*correction
    next_state=WorldStreamState(state.fast_sum+basis[:,0],memory,state.count+1)
    if return_trace:
        return prediction,next_state,{"memory":memory,"old_read":read[:,0],
                           "fast":f[:,0],"slow":s[:,0]}
    return prediction,next_state
