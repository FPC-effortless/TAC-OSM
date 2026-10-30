"""HYBGEOM091: trainable causal compact-TAC hybrid/geometry alternatives.

The input is *observed* byte stream only. A step is an ordinal, NOT real elapsed
seconds. Geometry defines a local rank-one SPD metric; this is not the learned
geodesic or human-judgment semantics of the broader proposal. Original TAC core
is imported unchanged, and original 086/087 checkpoints remain untouched.
"""
from __future__ import annotations
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/mathematical-engineering/langmem086'))
from compact_tac import CompactDualClippedTAC086

ARMS = ('original', 'fixed_euclidean', 'contextual_euclidean',
        'contextual_spd', 'hybrid', 'hybrid_contextual_spd')
EXPECTED = {'original': 100521, 'fixed_euclidean': 100522,
            'contextual_euclidean': 100523, 'contextual_spd': 100524,
            'hybrid': 100524, 'hybrid_contextual_spd': 100527}

@dataclass(frozen=True)
class State091:
    core: object
    modes: torch.Tensor  # 0 or 1, per independent batch owner


def logit(p: float) -> float:
    return math.log(p/(1-p))


def inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class HybridGeometryTAC091(nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        if arm not in ARMS: raise ValueError('unregistered arm')
        self.arm=arm
        self.core=CompactDualClippedTAC086()
        self.use_geometry=arm in ('fixed_euclidean','contextual_euclidean','contextual_spd','hybrid_contextual_spd')
        self.use_contextual=arm in ('contextual_euclidean','contextual_spd','hybrid_contextual_spd')
        self.use_spd=arm in ('contextual_spd','hybrid_contextual_spd')
        self.use_hybrid=arm in ('hybrid','hybrid_contextual_spd')
        if self.use_geometry:
            self.raw_strength=nn.Parameter(torch.tensor(logit(.5)))  # strength = 0.02 sigmoid = .01
        if self.use_contextual:
            self.raw_gamma=nn.Parameter(torch.tensor(logit(.15)))
        if self.use_spd:
            self.raw_alpha=nn.Parameter(torch.tensor(inv_softplus(.5)))
        if self.use_hybrid:
            self.raw_rates=nn.Parameter(torch.tensor([inv_softplus(.015),inv_softplus(.15)]))
            self.raw_reset=nn.Parameter(torch.tensor(logit(.9)))
        if self.counts()['trainable']!=EXPECTED[arm]:
            raise ValueError('live model-size mismatch')

    def counts(self):
        return {'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'deployed':sum(p.numel() for p in self.parameters())}

    def start(self,owners: tuple[str,...]):
        return State091(self.core.start(owners),torch.zeros(len(owners),dtype=torch.long,device=self.core.embedding.weight.device))

    def forward(self,tokens: torch.Tensor,*,state: State091|None=None,owners: tuple[str,...]|None=None,
                elapsed: torch.Tensor|None=None):
        if tokens.dtype!=torch.long or tokens.ndim!=2 or tokens.shape[1]<1:
            raise ValueError('observed [B,T] long bytes required')
        b,n=tokens.shape
        if owners is None:owners=tuple(f'fit-row-{i}' for i in range(b))
        if state is None:state=self.start(owners)
        if owners!=state.core.owners or state.modes.shape!=(b,) or not torch.all((state.modes==0)|(state.modes==1)):
            raise ValueError('state owners/modes')
        if elapsed is not None:
            if elapsed.shape!=(b,n) or not torch.isfinite(elapsed).all() or (elapsed<=0).any():
                raise ValueError('strictly positive causal elapsed inputs')
            if not self.use_hybrid: raise ValueError('elapsed only for hybrid arm')
        out=[];core=state.core;modes=state.modes
        for t in range(n):
            byte=tokens[:,t:t+1]
            if self.use_hybrid:
                dt=(elapsed[:,t] if elapsed is not None else core.tensors['upper'].new_ones(b))
                rates=F.softplus(self.raw_rates)
                factor=torch.exp(-rates[modes]*dt)[:,None]
                observed=byte[:,0]
                nxt=torch.where(observed==10,torch.ones_like(modes),
                      torch.where(observed==46,torch.zeros_like(modes),modes))
                reset=torch.where(nxt!=modes,torch.sigmoid(self.raw_reset),factor.new_ones(b))[:,None]
                upper=core.tensors['upper']*factor*reset
                core=replace(core,tensors={**core.tensors,'upper':upper})
                modes=nxt
            logits,core=self.core(byte,state=core,owners=owners)
            logits=logits[:,0,:]
            if self.use_geometry:
                # Context = current causally updated native state; every candidate is a
                # fixed vocabulary embedding. There is no future observed byte here.
                c=F.normalize(core.tensors['upper'],dim=-1,eps=1e-8)
                emb=self.core.embedding.weight
                gamma=torch.sigmoid(self.raw_gamma) if self.use_contextual else emb.new_zeros(())
                z=emb[None]+gamma*(emb[None]*c[:,None,:]).sum(-1,keepdim=True)*c[:,None,:]
                observed_z=z[torch.arange(b,device=byte.device),byte[:,0]]
                diff=z-observed_z[:,None,:]
                d2=diff.square().sum(-1)
                if self.use_spd:
                    # G(c) = I+alpha cc^T, alpha>0, thus strictly SPD on R^32.
                    d2=d2+F.softplus(self.raw_alpha)*(diff*c[:,None,:]).sum(-1).square()
                strength=.02*torch.sigmoid(self.raw_strength)
                logits=logits-strength*(d2-d2.mean(-1,keepdim=True))
            out.append(logits)
        return torch.stack(out,dim=1),State091(core,modes)
