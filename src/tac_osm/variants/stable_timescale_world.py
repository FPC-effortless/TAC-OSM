"""PNDS v1.6: bounded multi-timescale event retention for TAC-OSM.

Source-authored NEW CANDIDATE, no changes to original native fast/slow
recurrence or its data. Reuses the exact original 93,324 continuous-model
trainable parameters. The 88D input-only evidence accumulator is changed
from the unbounded running SUM / count to four fixed-width recency bands.

For each dimension d, with fixed 0<gamma[d]<=1:
     H[t+1,d]=(1-gamma[d])*H[t,d]+gamma[d]*tanh(fast_basis[t,d]).
The chronological pair (P,Q) is scalar/elementwise affine; its associative
prefix is computed with an O(T)-work, O(log T)-depth Blelloch tree.
A step computes the identical recurrence with O(88) work and fixed
persistent 88+8x8=152 fp scalars (608 bytes), excluding weights.

This is a separate native-inspired alternative, not a claim that a Mamba
SSM has been implemented. The *fixed* decay is a design parameter, not an
online-learned memory key or demonstrated semantic relevance signal.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .parallel_continuous_world import ParallelTACWorld
from .parallel_affine_memory import associative_affine_schedule,work_efficient_affine_prefix


def decay_bands(width,device,dtype):
    if width<4 or width%4:
        raise ValueError("width must be positive and divisible by 4")
    rates=torch.tensor((.125,.0625,.015625,.00390625),device=device,dtype=dtype)
    return rates.repeat_interleave(width//4)


def balanced_elementwise_affine_prefix(P,Q):
    """Inclusive real-affine prefix over [B,T,D], O(TD) work/O(logT) depth.

    P and Q commute only WITHIN a fixed elementwise coordinate; chronological
    association is maintained to also handle future nonidentical gates.
    This kernel has no sequence-step Python loop, only logarithmic stages.
    """
    if P.ndim!=3 or Q.shape!=P.shape or P.shape[1]<1:
        raise ValueError("P/Q must be matching nonempty B,T,D tensors")
    B,T,D=P.shape
    N=1<<(T-1).bit_length()
    if N>T:
        P=torch.cat((P,P.new_ones(B,N-T,D)),1)
        Q=torch.cat((Q,Q.new_zeros(B,N-T,D)),1)
    levels=[(P,Q)]
    while levels[-1][0].shape[1]>1:
        p,q=levels[-1]
        a,b=p[:,::2],p[:,1::2]
        x,y=q[:,::2],q[:,1::2]
        levels.append((a*b,x*b+y))
    ep=P.new_ones(B,1,D)
    eq=Q.new_zeros(B,1,D)
    for level in range(len(levels)-2,-1,-1):
        lower_p,lower_q=levels[level]
        right_p=ep*lower_p[:,::2]
        right_q=eq*lower_p[:,::2]+lower_q[:,::2]
        ep=torch.stack((ep,right_p),dim=2).reshape(B,-1,D)
        eq=torch.stack((eq,right_q),dim=2).reshape(B,-1,D)
    return (eq*P+Q)[:,:T]


class StableTimescaleTACWorld(ParallelTACWorld):
    """Explicit four-timescale TAC continuous predictor, same learned weights."""
    def forward(self,obs,action,*,return_trace=False):
        if (obs.ndim!=3 or action.ndim!=3 or
            obs.shape[:2]!=action.shape[:2] or
            obs.shape[-1]!=self.state_dim or
            action.shape[-1]!=self.action_dim or obs.shape[1]<1):
            raise ValueError("invalid observed B,T,dimensions")
        B,T,_=obs.shape
        e=self.encoder(torch.cat((obs,action),-1)).tanh()
        k=F.normalize(self.address(e),dim=-1,eps=1e-6)
        v=F.normalize(self.content(e),dim=-1,eps=1e-6)
        zf=e.new_zeros(B,T,self.width)
        zr=e.new_zeros(B,T,8)
        basis,_=self.fast_map(torch.cat((e,zf,zr),-1)).chunk(2,-1)
        basis=basis.tanh()
        gamma=decay_bands(self.width,e.device,e.dtype).reshape(1,1,-1)
        summary=balanced_elementwise_affine_prefix(
             (1-gamma).expand(B,T,-1),
             gamma*basis)
        # Read only PAST observed event summaries, never current/future.
        context=torch.cat((summary.new_zeros(B,1,self.width),
                           summary[:,:-1]),1)
        pf,fg=self.fast_map(torch.cat((e,context,zr),-1)).chunk(2,-1)
        pf=torch.lerp(context,pf.tanh(),fg.sigmoid())
        ds,gs=self.slow_map(torch.cat((pf,context,zr),-1)).chunk(2,-1)
        ps=torch.lerp(context,(context+ds.tanh()).tanh(),.25*gs.sigmoid())
        alpha,beta=self.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1)
        alpha=alpha*(1-beta/4.)
        P,Q=associative_affine_schedule(
            k,v,alpha,beta,torch.ones_like(alpha,dtype=torch.bool))
        memory=work_efficient_affine_prefix(P,Q)
        old=torch.cat((memory.new_zeros(B,1,8,8),memory[:,:-1]),1)
        r=(old@k.unsqueeze(-1)).squeeze(-1)
        f,fg=self.fast_map(torch.cat((e,pf,r),-1)).chunk(2,-1)
        f=torch.lerp(pf,f.tanh(),fg.sigmoid())
        ds,gs=self.slow_map(torch.cat((f,ps,r),-1)).chunk(2,-1)
        s=torch.lerp(ps,(ps+ds.tanh()).tanh(),.25*gs.sigmoid())
        prediction=obs+self.delta_scale*self.readout(
            torch.cat((f,s,r,e),-1)).tanh()
        if return_trace:
            return prediction,{"memory":memory,"context":context,
             "summary":summary,"fast":f,"slow":s,"old_read":r}
        return prediction


@dataclass(frozen=True)
class StableTimescaleState:
    summary:torch.Tensor
    memory:torch.Tensor
    count:int


def init_stable_stream(model,B):
    if B<1:raise ValueError("B must be positive")
    p=next(model.parameters())
    return StableTimescaleState(p.new_zeros(B,model.width),
                                p.new_zeros(B,8,8),0)


def stable_stream_step(model,state,obs,action,*,return_trace=False):
    B=state.summary.shape[0]
    if (obs.shape!=(B,model.state_dim) or action.shape!=(B,model.action_dim)
        or state.memory.shape!=(B,8,8) or state.count<0):
        raise ValueError("invalid single-step observed event/state")
    e=model.encoder(torch.cat((obs,action),-1)).tanh()
    k=F.normalize(model.address(e),dim=-1,eps=1e-6)
    v=F.normalize(model.content(e),dim=-1,eps=1e-6)
    zf=e.new_zeros(B,model.width)
    zr=e.new_zeros(B,8)
    basis,_=model.fast_map(torch.cat((e,zf,zr),-1)).chunk(2,-1)
    basis=basis.tanh()
    context=state.summary
    pf,g=model.fast_map(torch.cat((e,context,zr),-1)).chunk(2,-1)
    pf=torch.lerp(context,pf.tanh(),g.sigmoid())
    ds,g=model.slow_map(torch.cat((pf,context,zr),-1)).chunk(2,-1)
    ps=torch.lerp(context,(context+ds.tanh()).tanh(),.25*g.sigmoid())
    alpha,beta=model.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1)
    alpha=alpha*(1-beta/4.)
    old=state.memory
    r=(old@k.unsqueeze(-1)).squeeze(-1)
    new_m=alpha.unsqueeze(-1)*old + (beta*(v-alpha*r)).unsqueeze(-1)*k.unsqueeze(-2)
    ff,g=model.fast_map(torch.cat((e,pf,r),-1)).chunk(2,-1)
    ff=torch.lerp(pf,ff.tanh(),g.sigmoid())
    d,g=model.slow_map(torch.cat((ff,ps,r),-1)).chunk(2,-1)
    ss=torch.lerp(ps,(ps+d.tanh()).tanh(),.25*g.sigmoid())
    prediction=obs+model.delta_scale*model.readout(
        torch.cat((ff,ss,r,e),-1)).tanh()
    gamma=decay_bands(model.width,e.device,e.dtype)
    new_summary=(1-gamma)*context+gamma*basis
    newstate=StableTimescaleState(new_summary,new_m,state.count+1)
    if return_trace:
        return prediction,newstate,{"memory":new_m,"context":context,
             "summary":new_summary,"fast":ff,"slow":ss,"old_read":r}
    return prediction,newstate
