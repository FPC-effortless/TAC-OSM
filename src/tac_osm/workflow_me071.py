"""Persistent native event model with ordered time quantiles and calibration.

Established calibration/quantile methods, integrated with native TAC memory.
No confidence, causal-action or universal coverage guarantee is implied.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .workflow_me069 import EventWorldModel, EventBelief

QUANTILES=(.05,.5,.95)


class QuantileForecast(NamedTuple):
    activity_logits: torch.Tensor
    positive_wait_logit: torch.Tensor
    log_wait_quantiles: torch.Tensor


def detach_belief(belief):
    return EventBelief(*(x.detach() for x in belief))


def assimilate_masked(model,encoded,elapsed_hours,available,belief=None):
    """Skip hidden observations while preserving elapsed time to the next one.

    Missing-input stress support, not evidence of preserved information about
    events the model never observed. Callers must also hide masked prior context.
    """
    belief=model.initial(len(encoded)) if belief is None else belief
    pending=elapsed_hours.new_zeros(len(encoded),1)
    for t in range(encoded.shape[1]):
        pending=pending+elapsed_hours[:,t]
        belief=model.observe(belief,encoded[:,t],pending,available[:,t])
        pending=torch.where(available[:,t,None],torch.zeros_like(pending),pending)
    return belief


class PersistentWorkflowModel(EventWorldModel):
    def __init__(self,preprocessing,width=64,decoder_width=128):
        super().__init__(preprocessing,"native_tac",width,decoder_width)
        n=self.transition_prior.shape[1]
        # Activity residuals, repeat/transition/neural router, history blend,
        # zero/positive occurrence and three ordered unconditional quantiles.
        self.output=nn.Linear(decoder_width,n+8)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        with torch.no_grad():self.output.bias[n:n+3].copy_(torch.tensor([-2.,2.,-2.]))
        self.register_buffer("quantile_prior",torch.tensor(preprocessing["log_wait_quantile_prior"],dtype=torch.float32))

    def forecast(self,belief,encoded,current_activity,history_log_prior):
        query=F.normalize(self.query(encoded),dim=-1,eps=1e-6)
        recall=torch.einsum("bij,bj->bi",belief.memory,query)
        left,right=self.decoder(torch.cat((belief.fast,belief.slow,recall,encoded),-1)).chunk(2,-1)
        out=self.output(F.silu(left)*right)
        n=self.transition_prior.shape[1]
        prior=self.transition_prior[current_activity]
        neural=(prior+out[:,:n]).softmax(-1)
        repeat=F.one_hot(current_activity,num_classes=n).to(encoded.dtype)
        weights=out[:,n:n+3].softmax(-1)
        mixture=(weights[:,:1]*repeat+weights[:,1:2]*prior.exp()+weights[:,2:3]*neural).clamp_min(1e-12)
        blend=out[:,n+3:n+4].sigmoid()
        # Log-linear pooling provides history conditioning, NOT a pointwise
        # guarantee that loss is below either expert's loss.
        logits=(1-blend)*mixture.log()+blend*history_log_prior
        logits=logits-logits.logsumexp(-1,keepdim=True)
        q=self.quantile_prior[current_activity]
        # Stable softplus inverse for the FIT median; no crossing by construction.
        inv=q[:,1]+torch.log(-torch.expm1(-q[:,1].clamp_min(1e-5)))
        median=F.softplus(inv+out[:,n+5])
        ratio=(q[:,0]/q[:,1].clamp_min(1e-5)).clamp(1e-4,1-1e-4)
        low=median*torch.sigmoid(torch.logit(ratio)+out[:,n+6])
        distance=(q[:,2]-q[:,1]).clamp_min(1e-5)
        inv_distance=distance+torch.log(-torch.expm1(-distance))
        high=median+F.softplus(inv_distance+out[:,n+7])
        occurrence=self.time_prior[current_activity,0]+out[:,n+4]
        return QuantileForecast(logits,occurrence,torch.stack((low,median,high),-1))


def predictive_loss(prediction,activity,wait_seconds,available,smoothing=.05):
    mask=available.to(wait_seconds.dtype)
    ce=F.cross_entropy(prediction.activity_logits,activity,reduction="none",label_smoothing=smoothing)
    target=torch.log1p(wait_seconds).unsqueeze(-1)
    delta=target-prediction.log_wait_quantiles
    levels=delta.new_tensor(QUANTILES)
    pinball=torch.maximum(levels*delta,(levels-1)*delta).mean(-1)
    occurrence=F.binary_cross_entropy_with_logits(prediction.positive_wait_logit,(wait_seconds>0).to(mask.dtype),reduction="none")
    return ((ce+.3*pinball+.1*occurrence)*mask).sum()/mask.sum().clamp_min(1.)


def fit_temperature(logits,labels):
    """Bounded deterministic DEV-only search, retaining T=1 if it scores best."""
    logits=logits.detach().double();labels=labels.detach()
    temperatures=torch.cat((torch.logspace(-1,1,161,dtype=torch.float64),torch.ones(1,dtype=torch.float64))).unique()
    losses=torch.stack([F.cross_entropy(logits/t,labels) for t in temperatures])
    return float(temperatures[losses.argmin()])


def cqr_correction(log_quantiles,seconds,alpha=.1):
    """Split CQR in log1p seconds. Marginal guarantee requires exchangeability.

    Nonnegative correction conservatively widens; no promise of smaller widths.
    Each calibration row must represent a distinct case, not dependent prefixes.
    """
    q=np.asarray(log_quantiles,dtype=np.float64);target=np.log1p(seconds)
    scores=np.maximum(q[:,0]-target,target-q[:,2])
    k=math.ceil((len(scores)+1)*(1-alpha))
    if not len(scores) or k>len(scores):raise ValueError("insufficient CQR cases for finite quantile")
    return max(0.,float(np.partition(scores,k-1)[k-1]))


def calibrated_seconds(log_quantiles,correction=0.):
    # Evaluate exponential conversion in float64, retaining log-time outputs
    # for exact quantile-score auditing. Overflow is an error, not silent capping.
    q=log_quantiles.double().clone()
    q[...,0]=(q[...,0]-correction).clamp_min(0.)
    q[...,2]=q[...,2]+correction
    out=torch.expm1(q)
    if not torch.isfinite(out).all():raise ValueError("nonfinite waiting quantiles")
    return out


@dataclass
class EntityMemory:
    """Inference-only entity addressability; IDs are routing keys, not embeddings."""
    model: PersistentWorkflowModel

    def __post_init__(self):self.states={};self.timestamps={}

    @torch.no_grad()
    def observe(self,entity,encoded,event_time_seconds):
        previous=self.timestamps.get(entity,event_time_seconds)
        if event_time_seconds<previous:raise ValueError("out-of-order entity observation")
        state=self.states.get(entity,self.model.initial(1))
        hours=encoded.new_tensor([[(event_time_seconds-previous)/3600.]])
        state=self.model.observe(state,encoded,hours,torch.ones(1,dtype=torch.bool,device=encoded.device))
        self.states[entity]=detach_belief(state);self.timestamps[entity]=event_time_seconds
        return self.states[entity]

    def reset(self,entity):
        self.states.pop(entity,None);self.timestamps.pop(entity,None)
