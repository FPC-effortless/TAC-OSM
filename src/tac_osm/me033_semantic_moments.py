"""Factorial semantic adaptation controls and composable checkpoint-fixed moments."""
import torch
from torch import nn
from tac_osm.me032_conditional_memory import NativeMemory,CommonHead,CenteredReadout,split_rms


class SemanticMoments(nn.Module):
    __constants__=['query']
    def __init__(self,query:bool):
        super().__init__();self.query=query
        self.emb=nn.Embedding(16000,8)
        self.pool=nn.Sequential(nn.Linear(11,56),nn.Tanh(),nn.Linear(56,56),nn.Tanh(),nn.Linear(56,64))

    def forward(self,items:torch.Tensor,meta:torch.Tensor,offered:torch.Tensor)->torch.Tensor:
        out=torch.jit.annotate(list[torch.Tensor],[])
        for j in range(4):
            k=(j+1)*32
            identity=self.emb(offered[:,j]) if self.query else self.emb(items[:,:k]).mean(1)
            u=torch.cat((identity,meta[:,:k].mean(1)),-1)
            out.append(split_rms(self.pool(u)))
        return torch.stack(out,1)

    @torch.jit.export
    def empty(self,reference:torch.Tensor)->torch.Tensor:
        return reference.new_zeros((reference.size(0),4 if self.query else 12))

    @torch.jit.export
    def observe(self,state:torch.Tensor,item:torch.Tensor,meta:torch.Tensor)->torch.Tensor:
        u=meta if self.query else torch.cat((self.emb(item),meta),-1)
        return state+torch.cat((u,torch.ones_like(state[:,-1:])), -1)

    @torch.jit.export
    def merge(self,left:torch.Tensor,right:torch.Tensor)->torch.Tensor:
        return left+right

    @torch.jit.export
    def project(self,state:torch.Tensor,offered:torch.Tensor)->torch.Tensor:
        mean=state[:,:-1]/state[:,-1:].clamp_min(1.)
        u=torch.cat((self.emb(offered),mean),-1) if self.query else mean
        return split_rms(self.pool(u))


class NativeSemanticPath(nn.Module):
    def __init__(self):
        super().__init__();self.core=NativeMemory()
    def forward(self,items:torch.Tensor,meta:torch.Tensor,offered:torch.Tensor)->torch.Tensor:
        return self.core(items,meta)


class SemanticStudent(nn.Module):
    def __init__(self,query:bool,native:bool):
        super().__init__()
        self.memory=NativeSemanticPath() if native else SemanticMoments(query)
        self.common=CommonHead();self.correction=CenteredReadout()
        for p in self.common.parameters():p.requires_grad_(False)
    def forward(self,items:torch.Tensor,meta:torch.Tensor,offered:torch.Tensor,
                context:torch.Tensor)->torch.Tensor:
        x=context[...,:26]
        return context[...,26]+self.common(x)+self.correction(self.memory(items,meta,offered),x)


class CommonAdaptation(nn.Module):
    def __init__(self):
        super().__init__();self.common=CommonHead()
    def forward(self,items:torch.Tensor,meta:torch.Tensor,offered:torch.Tensor,
                context:torch.Tensor)->torch.Tensor:
        return context[...,26]+self.common(context[...,:26])


COUNTS={'history_adapt':136772,'query_adapt':136772,
        'history_frozen':8772,'query_frozen':8772,
        'native_adapt':136236,'common_adapt':1205}


def build_model(arm,common_state,embedding,seed):
    if arm not in COUNTS:raise ValueError(arm)
    torch.manual_seed(seed)
    if arm=='common_adapt':
        net=CommonAdaptation()
    else:
        net=SemanticStudent(arm.startswith('query'),arm=='native_adapt')
        e=net.memory.core.emb if arm=='native_adapt' else net.memory.emb
        with torch.no_grad():e.weight.copy_(embedding)
        e.weight.requires_grad_('frozen' not in arm)
        torch.manual_seed(seed+1);net.correction=CenteredReadout()
    net.common.load_state_dict(common_state,strict=True)
    assert sum(p.numel() for p in net.parameters() if p.requires_grad)==COUNTS[arm]
    return net
