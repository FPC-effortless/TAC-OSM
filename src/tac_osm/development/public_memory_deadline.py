"""DEV-056 memory deadline: public hazard, exact filtering, real write masking."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .learned_stack import LearnedStack, PhysicalWorld, FittedFilter

ARMS = ('mtsk','masked_mtsk','gru_matched','mtsk_reset')
HAZARDS = {'slow':.05,'fast':.25}


class PublicHazardWorld(PhysicalWorld):
    def __init__(self, seed, index, family):
        if family not in HAZARDS: raise ValueError('unknown public hazard')
        super().__init__(seed,index,family)
        self.hazard=HAZARDS[family]

    def step(self, action):
        old_regime, tick=self.regime,self.tick
        event,cost,label=super().step(action)
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.index,tick,771]))
        rng.random(2)
        self.regime=old_regime ^ int(rng.random()<self.hazard)
        return event,cost,label


def public_features(event, hazard):
    values=event.tensor()
    return values[:9]+[hazard]+values[9:]


def collect(seed,count,length=24):
    x,a,y,labels=[],[],[],[]
    for index in range(count):
        w=PublicHazardWorld(seed,index,'slow' if index%2==0 else 'fast')
        rng=np.random.default_rng(np.random.SeedSequence([seed,index,561]))
        es,ac,ys,ls=[],[],[],[]
        for _ in range(length):
            es.append(public_features(w.last,w.hazard))
            action=int(rng.integers(5))
            _,_,label=w.step(action)
            ac.append(action);ys.append(label['after']);ls.append(label)
        x.append(es);a.append(ac);y.append(ys);labels.append(ls)
    return (torch.tensor(x,dtype=torch.float32),torch.tensor(a),torch.tensor(y,dtype=torch.float32)),labels


class DeadlineModel(LearnedStack):
    def __init__(self,seed,arm):
        if arm not in ARMS: raise ValueError('unregistered arm')
        super().__init__(seed,'mtsk' if arm=='masked_mtsk' else arm)
        self.research_arm=arm
        torch.manual_seed(seed+56)
        self.encoder=nn.Sequential(nn.Linear(11,8),nn.Tanh())
        if arm=='masked_mtsk':
            self.write_gate=nn.Linear(8,1)
            nn.init.constant_(self.write_gate.bias,2.)
        self.last_write_count=0

    def initial(self,reference):
        state,resp=super().initial(reference)
        if self.arm!='gru_matched':
            resp=reference.new_full((reference.shape[0],2,3,4),1/3)
        return state,resp

    def update(self,state,public):
        if public.ndim!=2 or public.shape[-1]!=11 or not torch.isfinite(public).all():
            raise ValueError('finite public event plus declared hazard [B,11] required')
        x=self.encoder(public)
        if self.arm=='gru_matched':
            self.last_write_count=len(x)
            return self.memory(x,state[0]),None
        if self.arm=='mtsk_reset': state=self.initial(public)
        if self.research_arm!='masked_mtsk':
            _,memory,_,resp=self.memory._step(x,state[0],state[1])
            self.last_write_count=len(x)
            return memory,resp
        probability=self.write_gate(x).sigmoid().squeeze(-1)
        hard=(probability>=.5).to(x.dtype)
        self.last_write_count=int(hard.detach().sum())
        if self.training:
            _,memory,_,resp=self.memory._step(x,state[0],state[1])
            gate=hard+probability-probability.detach()
            # Mask decay too; a closed write is an exact no-op.
            return (state[0]+gate[:,None,None,None,None]*(memory-state[0]),
                    state[1]+gate[:,None,None,None]*(resp-state[1]))
        ids=hard.nonzero(as_tuple=True)[0]
        if not len(ids): return state
        _,memory,_,resp=self.memory._step(x[ids],state[0][ids],state[1][ids])
        return state[0].index_copy(0,ids,memory),state[1].index_copy(0,ids,resp)


class PublicFilter(FittedFilter):
    def __init__(self,transitions,hazard,independent=False):
        super().__init__(transitions,independent)
        self.hazard=hazard

    def update(self,event,initial=False):
        if not initial:
            prediction=np.einsum('zs,zsn->zn',self.belief,self.transitions[:,event.action])
            self.belief=(1-self.hazard)*prediction+self.hazard*prediction[::-1]
        states=np.array([[s//2,s%2] for s in range(4)])
        likelihood=np.ones(4)
        for e in range(2):
            accuracy=1. if event.exact[e] else .8
            likelihood*=np.where(states[:,e]==event.sensors[e],accuracy,1-accuracy)
        self.belief*=likelihood
        if self.belief.sum()<=0: raise FloatingPointError('impossible public receipt')
        self.belief/=self.belief.sum()
        if self.independent:
            p=self.belief.sum(0)@states
            marginal=np.prod(np.where(states,p,1-p),axis=1)
            self.belief=np.tile(marginal/2,(2,1))


def oracle_transitions():
    """Known probability law only; never a sampled world's state or labels."""
    table=np.empty((2,5,4,4))
    for z in range(2):
        for action in range(5):
            for s in range(4):
                p=np.array([s//2,s%2],dtype=float)
                if action in (1,2):
                    e=action-1
                    p[e]*=1-(.9-.65*z)
                    p[1-e]+=(1-p[1-e])*(.02+.43*z)
                for n in range(4):
                    bits=np.array([n//2,n%2])
                    table[z,action,s,n]=np.prod(np.where(bits,p,1-p))
    return table
