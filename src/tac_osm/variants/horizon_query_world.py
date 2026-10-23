"""ME081 native horizon-query distributional world model.

Shared nonlinear decoder across known horizons; native fast/slow/delta memory
unchanged. Proper CRPS plus a small log score and factual motion supervision.
Established distributional forecasting, not an architectural novelty claim.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .reusable_execution import BatchedMixtureWorld
from .streaming_mixture_world import log_density


class HorizonQueryWorld(BatchedMixtureWorld):
    architecture='native_horizon_query'
    def __init__(self,identified=None):
        super().__init__(identified)
        self.gated_readout=nn.Linear(174,340)
        self.heads=nn.Linear(170,85)
        nn.init.normal_(self.heads.weight,std=.001);nn.init.zeros_(self.heads.bias)
        with torch.no_grad():
            self.heads.bias[:14].fill_(-2.)
            self.heads.bias[56:70].fill_(-2.)
            self.heads.bias[70:84].fill_(-4.)
            self.heads.bias[84:].fill_(-2.)
        h=self.horizons
        self.register_buffer('horizon_features',torch.stack((h/16,torch.log1p(h)/torch.log(torch.tensor(17.)),h.sqrt()/4,torch.exp(-h/4)),-1))

    def decode(self,x,observed,representation,feature,velocity,command):
        b=len(x);n=len(self.horizons)
        context=torch.cat((observed,representation,feature[:,14:56]),-1)
        query=torch.cat((context[:,None].expand(b,n,170),self.horizon_features[None].expand(b,n,4)),-1)
        value,gate=self.gated_readout(query).chunk(2,-1)
        out=self.heads(value*F.silu(gate))
        mix,raw,b0,b1,vg,sg=out[:,:,:84].reshape(b,n,6,14).unbind(2)
        horizon=self.horizons[None,:,None]
        servo=torch.clamp(self.servo_gain*command,-self.servo_step_limit,self.servo_step_limit)
        bound=torch.maximum(4*self.servo_step_limit,torch.ones_like(self.delta_scale)*.05)*horizon
        delta=bound*torch.tanh((horizon*(vg.sigmoid()*velocity[:,None]+sg.sigmoid()*servo[:,None])+bound*raw.tanh())/bound)
        unit=self.delta_scale*horizon
        quiet=(unit*.1*(.01+F.softplus(b0))).clamp_min(1e-6)
        moving=2*quiet+unit*(.01+F.softplus(b1))
        return {'current':x[:,None].expand_as(delta),'moving_delta':delta,'quiet_scale':quiet,
                'moving_scale':moving,'mixture_logit':mix,'motion_logit':out[:,:,84]}


def mixture_crps(prediction,target):
    """Exact two-Laplace mixture CRPS; b1>=2*b0 avoids singular divided differences."""
    y=target-prediction['current'];m=prediction['moving_delta']
    b=prediction['quiet_scale'];c=prediction['moving_scale'];p=prediction['mixture_logit'].sigmoid()
    d=m.abs()
    expected_cross=d+(c.pow(3)*torch.exp(-d/c)-b.pow(3)*torch.exp(-d/b))/(c.square()-b.square())
    observation=(1-p)*(y.abs()+b*torch.exp(-y.abs()/b))+p*((y-m).abs()+c*torch.exp(-(y-m).abs()/c))
    pair=(1-p).square()*1.5*b+p.square()*1.5*c+2*p*(1-p)*expected_cross
    return observation-.5*pair


def horizon_distribution_loss(prediction,targets,current,valid,importance,physical_scale,normalizer):
    h=targets.new_tensor((1,4,16))[None,None,:,None]
    crps=(mixture_crps(prediction,targets)*physical_scale/(normalizer*h)).mean(-1)
    nll=(-log_density(prediction,targets)+physical_scale.log()).mean(-1)
    factual=(((targets-current[:,:,None]).abs()*physical_scale).mean(-1)>=.025).float()
    bce=F.binary_cross_entropy_with_logits(prediction['motion_logit'],factual,reduction='none')
    score=crps+.02*nll+.2*bce
    per_window=(score*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)
    return (importance*per_window).mean()
