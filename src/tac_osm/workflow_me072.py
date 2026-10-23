"""Native horizon-conditioned marked event futures with one coherent clock.

Record-boundary mass is not verified business completion or causal stopping.
"""
from typing import NamedTuple
import math
import torch
from torch import nn
from torch.nn import functional as F
from .workflow_me069 import EventWorldModel

HORIZONS=(1,2,4,8)


class MarkedFuture(NamedTuple):
    log_survival: torch.Tensor       # B,H: future recorded event exists
    mark_log_probability: torch.Tensor  # B,H,A: conditional activity
    log_zero: torch.Tensor           # B,H: time atom conditional on existence
    mixture_log_weight: torch.Tensor # B,K, shared over horizons
    log_time_location: torch.Tensor  # B,H,K, monotone per component
    log_time_scale: torch.Tensor     # B,K, log standard deviation


def log_complement(log_probability):
    return torch.log(-torch.expm1(log_probability.clamp_max(-1e-6)))


class MarkedWorldModel(EventWorldModel):
    def __init__(self,preprocessing):
        super().__init__(preprocessing,"native_tac",64,128)
        # Replace the inherited single-event readout; no unused trainable head.
        del self.decoder,self.output
        self.project=nn.Linear(184,96)
        self.horizon_embedding=nn.Embedding(len(HORIZONS),16)
        self.future_update=nn.Linear(112,192)
        self.future_decoder=nn.Linear(112,320)
        activities=self.transition_prior.shape[1]
        self.mark_head=nn.Linear(160,activities)
        self.survival_head=nn.Linear(160,1)
        self.clock_increment=nn.Linear(96,3)
        self.clock_base=nn.Linear(96,7) # two weights, two locations/scales, zero atom
        self.prior_strength=nn.Parameter(torch.zeros(len(HORIZONS)))
        for head in (self.mark_head,self.survival_head,self.clock_increment,self.clock_base):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.register_buffer("future_mark_prior",torch.tensor(preprocessing["future_mark_log_prior"],dtype=torch.float32))
        self.register_buffer("survival_prior",torch.tensor(preprocessing["survival_increment_logit"],dtype=torch.float32))
        self.register_buffer("clock_prior",torch.tensor(preprocessing["coherent_clock_prior"],dtype=torch.float32))

    def forecast(self,belief,encoded,current_activity,history_log_prior):
        query=F.normalize(self.query(encoded),dim=-1,eps=1e-6)
        recall=torch.einsum("bij,bj->bi",belief.memory,query)
        observed=torch.tanh(self.project(torch.cat((belief.fast,belief.slow,recall,encoded),-1)))
        horizon=self.horizon_embedding.weight.unsqueeze(0).expand(len(encoded),-1,-1)
        proposal,gate=self.future_update(torch.cat((observed[:,None,:].expand(-1,len(HORIZONS),-1),horizon),-1)).chunk(2,-1)
        future=observed[:,None,:]+gate.sigmoid()*proposal.tanh()
        left,right=self.future_decoder(torch.cat((future,horizon),-1)).chunk(2,-1)
        decoded=F.silu(left)*right
        priors=self.future_mark_prior[current_activity].clone()
        priors[:,0]=history_log_prior
        marks=(self.mark_head(decoded)+F.softplus(self.prior_strength)[None,:,None]*priors).log_softmax(-1)
        increments=self.survival_head(decoded).squeeze(-1)+self.survival_prior[current_activity]
        survival=F.logsigmoid(increments).cumsum(-1).clamp_max(-1e-6)
        base=self.clock_base(observed);prior=self.clock_prior[current_activity]
        weight=base[:,:2].log_softmax(-1)
        location=prior[:,0,None]+base[:,2:4].tanh()*8.+base.new_tensor([-.75,.75])
        sigma=.1+1.9*torch.sigmoid(prior[:,1,None]+base[:,4:6])
        first_zero=F.logsigmoid(prior[:,2]+base[:,6]).clamp_max(-1e-6)
        extra=F.softplus(self.clock_increment(future[:,1:]))
        zero_prefix=extra.new_zeros(len(encoded),1,1)
        shift=torch.cat((extra.new_zeros(len(encoded),1,2),extra[:,:,:2]),1).cumsum(1)
        zero_decay=torch.cat((zero_prefix,extra[:,:,2:3]),1).cumsum(1).squeeze(-1)
        return MarkedFuture(survival,marks,first_zero[:,None]-zero_decay,weight,
                            location[:,None,:]+shift,sigma.log())


def record_log_probability(forecast):
    """Normalized joint categorical mass: actual activity or source record end."""
    marks=forecast.log_survival[...,None]+forecast.mark_log_probability
    return torch.cat((marks,log_complement(forecast.log_survival)[...,None]),-1)


def factor_scores(forecast,target_activity,seconds,present):
    existence=-torch.where(present,forecast.log_survival,log_complement(forecast.log_survival))
    mark=-forecast.mark_log_probability.gather(-1,target_activity[...,None]).squeeze(-1)
    zero=seconds==0
    occurrence=-torch.where(zero,forecast.log_zero,log_complement(forecast.log_zero))
    log_seconds=seconds.clamp_min(1e-4).log()
    z=(log_seconds[...,None]-forecast.log_time_location)/forecast.log_time_scale[:,None,:].exp()
    component=forecast.mixture_log_weight[:,None,:]-forecast.log_time_scale[:,None,:]-.5*z.square()-.5*math.log(2*math.pi)
    positive_log_density=component.logsumexp(-1)-log_seconds
    return existence,mark,occurrence,-positive_log_density


def world_loss(forecast,target,seconds,present,valid):
    existence,mark,occurrence,positive=factor_scores(forecast,target,seconds,present)
    logp=record_log_probability(forecast)
    classes=forecast.mark_log_probability.shape[-1]
    labels=torch.where(present,target,torch.full_like(target,classes))
    probability=logp.exp()
    brier=probability.square().sum(-1)-2*probability.gather(-1,labels[...,None]).squeeze(-1)+1
    # The log(dt) Jacobian is target-only. Omit it from optimization, retain
    # it in the reported proper likelihood in original seconds.
    positive_logspace=positive-seconds.clamp_min(1e-4).log()
    total=existence+present*(mark+.1*occurrence+.1*(seconds>0)*positive_logspace)+.1*brier
    return (total*valid[:,None]).sum()/(valid.sum()*len(HORIZONS)).clamp_min(1)


@torch.no_grad()
def time_quantiles(forecast,levels=(.05,.5,.95)):
    """Conditional on a future record: invert the same zero/lognormal CDF.

    Shared mixture weights/scales, increasing component locations and decreasing
    zero mass imply nondecreasing quantiles over horizons. Float64 inversion.
    """
    zero=forecast.log_zero.double().exp()[...,None]
    levels=zero.new_tensor(levels)[None,None,:]
    conditional=((levels-zero)/(1-zero).clamp_min(1e-12)).clamp(1e-12,1-1e-12)
    mu=forecast.log_time_location.double();sigma=forecast.log_time_scale.double().exp()[:,None,:]
    low=(mu-12*sigma).min(-1).values[...,None].expand_as(conditional).clone()
    high=(mu+12*sigma).max(-1).values[...,None].expand_as(conditional).clone()
    weights=forecast.mixture_log_weight.double().exp()[:,None,None,:]
    for _ in range(64):
        middle=(low+high)/2
        z=(middle[...,None]-mu[:,:,None,:])/sigma[:,:,None,:]
        cdf=(weights*.5*torch.erfc(-z/math.sqrt(2))).sum(-1)
        low=torch.where(cdf<conditional,middle,low)
        high=torch.where(cdf>=conditional,middle,high)
    seconds=torch.where(levels<=zero,torch.zeros_like(low),((low+high)/2).exp())
    if not torch.isfinite(seconds).all():raise ValueError("nonfinite coherent clock quantile")
    return seconds
