"""Unfitted belief-only prototype. Supplied 4-health x3-relation vocabulary.

No physical repair rules or posterior filter. The graph factor prevents an
information-starved four-state bottleneck from discarding public dependencies.
"""
import math
import torch
from torch import nn
from .belief_agent import BeliefAgent,cosine,mlp


def relative_events(events,present,keys,mode='event_gap',unit=1.):
    if mode not in ('event_gap','relative_unit'):raise ValueError(mode)
    if not math.isfinite(unit) or not unit>0:raise ValueError('Positive public clock unit required')
    x=events.clone();subject=cosine(events[:,:,6:22],keys).argmax(-1);last=events.new_zeros(len(events),2);seen=torch.zeros(len(events),2,dtype=torch.bool,device=events.device);ordinal=present.cumsum(1).to(events.dtype)
    for t in range(events.shape[1]):
        index=subject[:,t,None];before=last.gather(1,index).squeeze(1);known=seen.gather(1,index).squeeze(1);tick=ordinal[:,t] if mode=='event_gap' else events[:,t,42]/unit;active=present[:,t];x[:,t,42]=torch.where(active&known,tick-before,torch.zeros_like(tick));last=last.scatter(1,index,torch.where(active,tick,before)[:,None]);seen=seen.scatter(1,index,(known|active)[:,None])
    return x


class SemanticActor(nn.Module):
    """Only declared semantic belief and current public choices/prices enter."""
    def __init__(self,latent_width,width):
        super().__init__();self.latent_width=latent_width;self.network=mlp(12+3+2+2,width,width)

    def forward(self,features):return self.network(features[...,self.latent_width:])


class BeliefBottleneck(BeliefAgent):
    def __init__(self,arm='tac',width=32,aux=32,seed=0,clock_mode='event_gap',clock_unit=1.):
        super().__init__(arm,width,aux,seed);self.clock_mode=clock_mode;self.register_buffer('clock_unit',torch.tensor(float(clock_unit)));self.belief=mlp(width,width,12);self.actor=SemanticActor(width,width)

    def forward(self,events,present,keys,candidates,prices):
        normalized=relative_events(events,present,keys,self.clock_mode,float(self.clock_unit));out=super().forward(normalized,present,keys,candidates,prices);joint=out['posterior'].reshape(-1,4,3);return {**out,'health_posterior':joint.sum(-1),'relation_posterior':joint.sum(-2)}
