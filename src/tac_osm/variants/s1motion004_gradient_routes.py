"""S1MOTION004 causal gradient-path intervention (untrained source candidate).

Exactly the original 95,907-param shared System1 graph and 664-byte
streaming FP32 state. Auxiliary logits/scales receive identical forward
values but are trained with STOP-GRADIENT inputs in the intervention arm.
No original S1MOTION003 checkpoint is altered or reused.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .system1_parallel import System1ParallelWorld, System1Forecast
from .system1_velocity_anchored import VelocityAnchoredSystem1


class DetachedEvidenceSystem1(System1ParallelWorld):
    """Auxiliary head learns; no gradients from it enter world-state encoder."""
    def _project(self, predicted, fast, slow, old_read):
        evidence=torch.cat((fast, slow, old_read, predicted),dim=-1)
        raw=self.reactive_head(evidence.detach())
        logits=raw[...,0]
        scales=F.softplus(raw[...,1:])+1.e-4
        return System1Forecast(predicted,logits,logits.sigmoid(),scales)


class IsolatedVelocitySystem1(VelocityAnchoredSystem1):
    """Same input-output tensor values as original at identical weights."""
    def __init__(self, state_dim=14, action_dim=14, width=88,
                 decoder=132,head_width=12,correction_gain=.1):
        super().__init__(state_dim,action_dim,width,decoder,
                         head_width,correction_gain)
        self.core=DetachedEvidenceSystem1(state_dim,action_dim,width,
                                          decoder,head_width)


class VelocityGRUControl(nn.Module):
    """Matched 95,869-weight velocity-anchored GRU factorial control."""
    def __init__(self, delta_scale, detach_auxiliary=False):
        super().__init__()
        if type(detach_auxiliary) is not bool:
            raise ValueError("explicit boolean detach_auxiliary required")
        self.recurrent=nn.GRU(28,160,batch_first=True)
        self.readout=nn.Linear(160,29)
        self.register_buffer("delta_scale",delta_scale.detach().clone())
        self.detach_auxiliary=detach_auxiliary

    def forward(self,x,u,previous):
        if (previous.shape!=x[:,0].shape or
            not torch.isfinite(previous).all().item()):
            raise ValueError("observed previous state required")
        hidden,_=self.recurrent(torch.cat((x,u),dim=-1))
        point=F.linear(hidden,self.readout.weight[:14],
                       self.readout.bias[:14])
        aux_in=hidden.detach() if self.detach_auxiliary else hidden
        aux=F.linear(aux_in,self.readout.weight[14:],
                     self.readout.bias[14:])
        hist=torch.cat((previous[:,None],x[:,:-1]),dim=1)
        prediction=2*x-hist+.1*self.delta_scale*point.tanh()
        logit=aux[...,0]
        scales=F.softplus(aux[...,1:])+1.e-4
        return prediction,logit,scales


def outputs(model,x,u,previous):
    if isinstance(model,VelocityAnchoredSystem1):
        forecast=model(x,u,initial_previous_observation=previous)
        return (forecast.next_observation,forecast.adverse_event_logit,
                forecast.residual_scale)
    return model(x,u,previous)


def losses(model,x,u,y,event,previous,*,isolated=False):
    """One forward per real-source batch; same scalar value across modes.

    Isolated: point=MSE still backpropagates through ALL shared memory.
    BCE gradients flow into auxiliary head ONLY (detached evidence).
    NLL receives detached point too; dispersion head remains trainable.
    """
    if type(isolated) is not bool:
        raise ValueError("isolated must be bool")
    point,logit,scale=outputs(model,x,u,previous)
    if ((isinstance(model,IsolatedVelocitySystem1) or
         isinstance(model,VelocityGRUControl) and model.detach_auxiliary)!=isolated):
        raise ValueError("loss gradient route and model branch disagree")
    mse=(point-y).square().mean()
    bce=F.binary_cross_entropy_with_logits(logit,event)
    nll_point=point.detach() if isolated else point
    normal=(.5*((nll_point-y)/scale).square()+scale.log()).mean()
    return mse+.1*bce+.01*normal,dict(point=mse,proxy_bce=bce,dispersion_nll=normal)
