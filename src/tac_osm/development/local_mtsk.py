"""DEV-034 banked MTSK adaptation with an explicitly counted augmented state.

Eight public entity banks, eight experts, rank3: 576 matrix scalars and192
responsibility scalars. Both local/global writer controls read public entity
banks. This is not the unchanged six-expert SwiLAMTSK or discovered addressing.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .memory_baselines import MemoryBaseline

ARMS=('mtsk_global_tied','mtsk_global_multi','mtsk_local_tied','mtsk_local_multi')


@dataclass(frozen=True)
class BankConfig:
    entities:int=8
    experts:int=8
    rank:int=3
    temporal_prior:float=.35


class BankedMTSK(nn.Module):
    def __init__(self,local=True,multiple=True):
        super().__init__();self.config=BankConfig();self.local=local;self.multiple=multiple
        c=self.config;j=c.experts;d=c.rank
        self.bank_embedding=nn.Parameter(torch.randn(c.entities,12)*.02)
        self.key=nn.Linear(12,d,bias=False);self.value=nn.Linear(12,d,bias=False)
        self.key_prior=nn.Linear(12,j*d)
        self.query=nn.Linear(12,d,bias=False);self.query_prior=nn.Linear(12,j*d)
        priors=[.70]*3+[.92]*3+[.99]*2
        betas=[.25]*3+[.08]*3+[.015]*2
        if not multiple:priors=[sum(priors)/j];betas=[sum(betas)/j]
        self.decay_logits=nn.Parameter(torch.tensor([math.log(p/(1-p)) for p in priors]))
        self.beta_logits=nn.Parameter(torch.tensor([math.log(math.expm1(p)) for p in betas]))
        self.input_decay=nn.Linear(12,(j if multiple else 1)*d)

    @property
    def bank_scalars(self):return self.config.experts*self.config.rank*(self.config.rank+1)

    def initial(self,x):
        c=self.config;b=x.shape[0]
        memory=x.new_zeros(b,c.entities,c.experts*c.rank*c.rank)
        responsibility=x.new_full((b,c.entities,c.experts*c.rank),1/c.experts)
        return torch.cat((memory,responsibility),-1)

    def unpack(self,state):
        c=self.config;cut=c.experts*c.rank*c.rank
        return state[...,:cut].reshape(*state.shape[:-1],c.experts,c.rank,c.rank),state[...,cut:].reshape(*state.shape[:-1],c.experts,c.rank)

    def proposal(self,packed,x):
        c=self.config;j=c.experts;d=c.rank
        memory,previous=self.unpack(packed)
        k=self.key(x);v=self.value(x)
        prediction=torch.einsum('...jde,...e->...jd',memory,k)
        delta=v.unsqueeze(-2)-prediction
        prior=self.key_prior(x).reshape(*x.shape[:-1],j,d).softmax(-2)
        posterior=(prior.clamp_min(1e-8).log()-.5*delta.square()).softmax(-2)
        responsibility=(1-c.temporal_prior)*posterior+c.temporal_prior*previous
        responsibility=responsibility/responsibility.sum(-2,keepdim=True).clamp_min(1e-8)
        decay=self.decay_logits.sigmoid().reshape(-1,1,1)
        input_decay=self.input_decay(x).reshape(*x.shape[:-1],j if self.multiple else 1,d,1).sigmoid()
        decay=(.5*decay+.5*input_decay).clamp(1e-4,.9999)
        beta=torch.nn.functional.softplus(self.beta_logits).reshape(-1,1,1)
        updated=decay*memory+beta*responsibility.unsqueeze(-1)*delta.unsqueeze(-1)*k.unsqueeze(-2).unsqueeze(-2)
        return torch.cat((updated.flatten(-3),responsibility.flatten(-2)),-1)

    def update(self,state,x,entities,admit=None):
        c=self.config;b=x.shape[0]
        if state.shape!=(b,c.entities,self.bank_scalars) or entities.shape!=(b,):raise ValueError('invalid bank state/address shape')
        if entities.dtype!=torch.long or ((entities<0)|(entities>=c.entities)).any():raise ValueError('public entity index required')
        if not torch.isfinite(x).all() or not torch.isfinite(state).all():raise ValueError('nonfinite input/state')
        strict=admit is None
        if admit is None:admit=torch.ones(b,dtype=torch.bool,device=x.device)
        if admit.shape!=(b,) or admit.dtype!=torch.bool:raise ValueError('explicit boolean whole-transition admission required')
        if not admit.any():return state
        if self.local:
            index=entities[:,None,None].expand(-1,1,self.bank_scalars)
            old=state.gather(1,index).squeeze(1)
            proposed=self.proposal(old,x+self.bank_embedding[entities])
            if strict and not torch.isfinite(proposed).all():raise FloatingPointError('nonfinite MTSK proposal')
            accepted=admit&torch.isfinite(proposed).all(-1)
            changed=torch.where(accepted[:,None],proposed,old)
            return state.scatter(1,index,changed[:,None])
        proposed=self.proposal(state,x[:,None]+self.bank_embedding)
        if strict and not torch.isfinite(proposed).all():raise FloatingPointError('nonfinite MTSK proposal')
        accepted=admit[:,None]&torch.isfinite(proposed).all(-1)
        return torch.where(accepted[:,:,None],proposed,state)

    def read(self,state,queries):
        # All factorial arms get the same supplied entity read addressing.
        memory,_=self.unpack(state);c=self.config
        q=self.query(queries);weights=self.query_prior(queries).reshape(*queries.shape[:-1],c.experts,c.rank).softmax(-2)
        experts=torch.einsum('bejdv,beav->beajd',memory,q)
        return (weights*experts).sum(-2)


class LocalMTSKPredictor(MemoryBaseline):
    def __init__(self,arm,seed=0):
        if arm not in ARMS:raise ValueError('unregistered banked MTSK arm')
        super().__init__('memoryless',seed)
        self.arm=arm
        self.memory=BankedMTSK(local='_local_' in arm,multiple=arm.endswith('_multi'))
        torch.manual_seed(seed+34000)
        self.adapter=nn.Linear(self.memory.config.rank,32)

    @property
    def state_scalars(self):return self.memory.config.entities*self.memory.bank_scalars

    def encode(self,events,entities,values):
        if events.ndim!=3 or events.shape[-1]!=16 or events.shape[1]<1:raise ValueError('nonempty public event stream required')
        if entities.shape!=events.shape[:2] or values.shape!=entities.shape:raise ValueError('unaligned public stream')
        x=self.event_encoder(events);state=self.memory.initial(x[:,0])
        for t in range(events.shape[1]):state=self.memory.update(state,x[:,t],entities[:,t])
        return state

    def update(self,state,features,entities,values):return self.memory.update(state,self.event_encoder(features),entities)

    def decode(self,state,queries):
        if queries.shape[1:]!=(8,3,14):raise ValueError('public entity/action/time queries required')
        q=self.query_encoder(queries);read=self.adapter(self.memory.read(state,q))
        return self.head(torch.cat((read,q),-1))
