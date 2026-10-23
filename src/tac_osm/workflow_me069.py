"""Native TAC for categorical enterprise event histories and waiting times.

No physical-joint constants, privileged future commands, cost labels or SLA
assumptions. This is an observational predictive-state architecture.
"""
from typing import NamedTuple
import math
import torch
from torch import nn
from torch.nn import functional as F


class EventBelief(NamedTuple):
    fast: torch.Tensor
    slow: torch.Tensor
    memory: torch.Tensor


class WorkflowForecast(NamedTuple):
    activity_logits: torch.Tensor
    positive_wait_logit: torch.Tensor
    log_wait_location: torch.Tensor
    log_wait_scale: torch.Tensor


class EventWorldModel(nn.Module):
    def __init__(self,preprocessing,architecture="native_tac",width=64,decoder_width=96):
        super().__init__()
        if architecture not in ("native_tac","gru_world"):raise ValueError("unknown recurrent architecture")
        self.architecture,self.width=architecture,width
        activities=len(preprocessing["activities"])+1;resources=len(preprocessing["resources"])+1
        self.activity_embedding=nn.Embedding(activities-1,32)
        self.resource_embedding=nn.Embedding(resources-1,8)
        self.encoder=nn.Linear(48,48)
        self.query=nn.Linear(48,8)
        if architecture=="native_tac":
            self.fast_map=nn.Linear(48+width+8,2*width)
            self.slow_map=nn.Linear(2*width+8,2*width)
            self.write_map=nn.Linear(48+2*width,18)
            self.decay=nn.Parameter(torch.tensor([-8.,-10.,-10.]))
            features=2*width+8+48
        else:
            self.cell=nn.GRUCell(48,width)
            features=width+8+48
        self.decoder=nn.Linear(features,2*decoder_width)
        self.output=nn.Linear(decoder_width,activities+6)
        # One output map, not symmetric zero-initialized expert copies.
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            # Begin near the empirical transition expert. Learned router
            # weights are predictive mixture coefficients, not confidence.
            self.output.bias[activities+3:].copy_(torch.tensor([-2.,2.,-2.]))
        self.register_buffer("transition_prior",torch.tensor(preprocessing["transition_log_probability"],dtype=torch.float32))
        self.register_buffer("time_prior",torch.tensor(preprocessing["time_prior"],dtype=torch.float32))

    def initial(self,batch,device=None):
        ref=self.transition_prior
        z=ref.new_zeros(batch,self.width,device=device)
        return EventBelief(z,z.clone(),ref.new_zeros(batch,8,8,device=device))

    @staticmethod
    def embed_known(table,ids):
        # UNK/PAD has a deterministic zero vector, with no unused trained row.
        return table((ids-1).clamp_min(0))*(ids>0).unsqueeze(-1)

    def event_encoding(self,activity,resource,features):
        joined=torch.cat((self.embed_known(self.activity_embedding,activity),
            self.embed_known(self.resource_embedding,resource),features),-1)
        return torch.tanh(self.encoder(joined))

    def observe(self,belief,encoded,elapsed_hours,available):
        if self.architecture=="gru_world":
            fast=self.cell(encoded,belief.fast)
            updated=EventBelief(fast,belief.slow,belief.memory)
        else:
            dt=elapsed_hours.clamp(0.,168.)
            rates=F.softplus(self.decay)
            f=belief.fast*torch.exp(-rates[0]*dt)
            s=belief.slow*torch.exp(-rates[1]*dt)
            retained=belief.memory*torch.exp(-rates[2]*dt)[:,:,None]
            query=F.normalize(f[:,:8]+s[:,:8],dim=-1,eps=1e-6)
            recall=torch.einsum("bij,bj->bi",retained,query)
            value,gate=self.fast_map(torch.cat((encoded,f,recall),-1)).chunk(2,-1)
            f=torch.lerp(f,torch.tanh(value),torch.sigmoid(gate))
            delta,gate=self.slow_map(torch.cat((f,s,recall),-1)).chunk(2,-1)
            s=torch.lerp(s,torch.tanh(s+torch.tanh(delta)),.25*torch.sigmoid(gate))
            k,v,beta,relevance=self.write_map(torch.cat((encoded,f,s),-1)).split((8,8,1,1),-1)
            k=F.normalize(k,dim=-1,eps=1e-6)
            error=torch.tanh(v)-torch.einsum("bij,bj->bi",retained,k)
            memory=retained+(torch.sigmoid(beta)*torch.sigmoid(relevance))[:,:,None]*error[:,:,None]*k[:,None,:]
            norm=memory.square().sum((-2,-1),keepdim=True).sqrt().clamp_min(1.)
            updated=EventBelief(f,s,memory/norm)
        return EventBelief(*(torch.where(available.reshape(-1,*([1]*(old.ndim-1))),new,old)
            for old,new in zip(belief,updated)))

    def forecast(self,belief,encoded,current_activity):
        query=F.normalize(self.query(encoded),dim=-1,eps=1e-6)
        if self.architecture=="native_tac":
            recall=torch.einsum("bij,bj->bi",belief.memory,query)
            inputs=torch.cat((belief.fast,belief.slow,recall,encoded),-1)
        else:
            inputs=torch.cat((belief.fast,query,encoded),-1)
        left,right=self.decoder(inputs).chunk(2,-1)
        output=self.output(F.silu(left)*right)
        n=self.transition_prior.shape[1]
        prior_probability=self.transition_prior[current_activity].exp()
        neural_probability=(self.transition_prior[current_activity]+output[:,:n]).softmax(-1)
        repeated=F.one_hot(current_activity,num_classes=n).to(encoded.dtype)
        weights=output[:,n+3:n+6].softmax(-1)
        mixture=weights[:,:1]*repeated+weights[:,1:2]*prior_probability+weights[:,2:3]*neural_probability
        logits=mixture.clamp_min(1e-12).log()
        prior=self.time_prior[current_activity]
        positive=prior[:,0]+output[:,n]
        location=prior[:,1]+4.*torch.tanh(output[:,n+1]/4.)
        logscale=(prior[:,2]+1.5*torch.tanh(output[:,n+2]/1.5)).clamp(math.log(.1),math.log(5.))
        return WorkflowForecast(logits,positive,location,logscale)


def waiting_quantile(forecast,probability):
    """Hurdle-lognormal quantile in observed wall-clock seconds."""
    p=torch.sigmoid(forecast.positive_wait_logit)
    quantile=(float(probability)-(1-p))/p.clamp_min(1e-8)
    normal=torch.special.ndtri(quantile.clamp(1e-6,1.-1e-6))
    positive=torch.exp((forecast.log_wait_location+forecast.log_wait_scale.exp()*normal).clamp(-20.,22.))
    return torch.where(quantile<=0,torch.zeros_like(positive),positive)


def long_wait_probability(forecast,threshold):
    z=(math.log(threshold)-forecast.log_wait_location)/forecast.log_wait_scale.exp()
    return torch.sigmoid(forecast.positive_wait_logit)*(.5*torch.erfc(z/math.sqrt(2.)))


def workflow_loss(forecast,next_activity,delay,available):
    mask=available.to(delay.dtype)
    positive=delay>0
    ce=F.cross_entropy(forecast.activity_logits,next_activity,reduction="none")
    occurrence=F.binary_cross_entropy_with_logits(forecast.positive_wait_logit,positive.to(delay.dtype),reduction="none")
    target=delay.clamp_min(1e-3).log()
    conditional=.5*((target-forecast.log_wait_location)/forecast.log_wait_scale.exp()).square()+forecast.log_wait_scale
    # Log-seconds Jacobian is target-only; omitting it changes no model ranking
    # or gradient. Timing is an observed positive interval, not a service cost.
    return ((ce+.2*occurrence+.1*conditional*positive)*mask).sum()/mask.sum().clamp_min(1.)
