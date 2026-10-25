"""Continuous world-state forecasting on authenticated real robot records.

Both architectures ingest the SAME observed (state_t,action_t) and predict
the actually recorded next state_t+1. This is NOT DIAGKEY004 classifier
or a claim that its trained weights transfer to robot control.

ParallelTACWorld retains TAC fast/slow proposal, normalized addressed
8x8 associative state, causal evidence summarization and gated writes;
to make memory scan legal, it computes gates independently of old reads.
Its streaming counterpart is separately testable. No entity IDs inferred.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .parallel_affine_memory import associative_affine_schedule,work_efficient_affine_prefix


class ParallelTACWorld(nn.Module):
    def __init__(self,state_dim=14,action_dim=14,width=88,decoder=132):
        super().__init__()
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.width=width
        self.encoder=nn.Linear(state_dim+action_dim,48)
        self.address=nn.Linear(48,8)
        self.content=nn.Linear(48,8)
        self.fast_map=nn.Linear(48+width+8,2*width)
        self.slow_map=nn.Linear(2*width+8,2*width)
        self.memory_gates=nn.Linear(48+2*width,2)
        self.readout=nn.Sequential(nn.Linear(2*width+8+48,decoder),
                                   nn.SiLU(),nn.Linear(decoder,state_dim))
        # Empirical state changes are small. Set this buffer from TRAIN only,
        # never from a held-out outcome or future-state oracle.
        self.register_buffer("delta_scale",torch.full((state_dim,),.02))

    def set_train_delta_scale(self,scale):
        if scale.shape!=(self.state_dim,) or not torch.isfinite(scale).all():
            raise ValueError("TRAIN-only scale must be finite [state_dim]")
        with torch.no_grad():
            self.delta_scale.copy_(scale.clamp_min(1e-5))

    def forward(self,obs,action,*,return_trace=False):
        if (obs.ndim!=3 or action.ndim!=3 or obs.shape[:2]!=action.shape[:2]
            or obs.shape[-1]!=self.state_dim or action.shape[-1]!=self.action_dim
            or obs.shape[1]==0):
            raise ValueError("obs/action require [B,T,observed_dim]")
        B,T,_=obs.shape
        e=self.encoder(torch.cat((obs,action),-1)).tanh()
        k=F.normalize(self.address(e),dim=-1,eps=1e-6)
        v=F.normalize(self.content(e),dim=-1,eps=1e-6)
        zero_fast=e.new_zeros(B,T,self.width)
        zero_read=e.new_zeros(B,T,8)
        basis,_=self.fast_map(torch.cat((e,zero_fast,zero_read),-1)).chunk(2,-1)
        basis=basis.tanh()
        old_sum=F.pad(basis.cumsum(dim=1)[:,:-1],(0,0,1,0))
        ctx=old_sum/torch.arange(T,device=e.device,dtype=e.dtype).clamp_min(1).view(1,T,1)
        p,fg=self.fast_map(torch.cat((e,ctx,zero_read),-1)).chunk(2,-1)
        pf=torch.lerp(ctx,p.tanh(),fg.sigmoid())
        delta,sg=self.slow_map(torch.cat((pf,ctx,zero_read),-1)).chunk(2,-1)
        ps=torch.lerp(ctx,(ctx+delta.tanh()).tanh(),.25*sg.sigmoid())
        alpha,beta=self.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1)
        alpha=alpha*(1-beta/4.0)
        mask=torch.ones_like(alpha,dtype=torch.bool)
        P,Q=associative_affine_schedule(k,v,alpha,beta,mask)
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
            return prediction,{"memory":memory,"old_read":r,"fast":f,
                               "slow":s,"sum_basis":basis.cumsum(dim=1)}
        return prediction


class RealGRUWorld(nn.Module):
    def __init__(self,state_dim=14,action_dim=14,hidden=160):
        super().__init__()
        self.gru=nn.GRU(state_dim+action_dim,hidden,batch_first=True)
        self.readout=nn.Linear(hidden,state_dim)
        self.state_dim=state_dim
        self.register_buffer("delta_scale",torch.full((state_dim,),.02))

    def set_train_delta_scale(self,scale):
        if scale.shape!=(self.state_dim,) or not torch.isfinite(scale).all():
            raise ValueError("TRAIN-only scale must be finite [state_dim]")
        with torch.no_grad():
            self.delta_scale.copy_(scale.clamp_min(1e-5))

    def forward(self,obs,action):
        z=torch.cat((obs,action),-1)
        y,_=self.gru(z)
        return obs+self.delta_scale*self.readout(y).tanh()
