"""Implicit CDF gradients for a coherent mixture median, without root tapes.

Implicit-function derivative dq/dtheta=-(dF/dtheta)/f(q), evaluated at the
fixed converged root. Numerically zero-valued correction carries this gradient;
forward predictions and the original inversion contract are unchanged.
"""
import torch
from .reusable_execution import mixture_quantiles
from .streaming_mixture_world import log_density


def laplace_cdf(z):
    # Safe on both branches and has the correct nonzero derivative at z=0.
    negative=torch.where(z<0,z,torch.zeros_like(z))
    positive=torch.where(z>=0,z,torch.zeros_like(z))
    return torch.where(z<0,.5*torch.exp(negative),1-.5*torch.exp(-positive))


def implicit_median(prediction):
    with torch.no_grad():root=mixture_quantiles(prediction,(.5,))[0]
    p={k:v.double() for k,v in prediction.items()}
    delta=root-p['current'];weight=p['mixture_logit'].sigmoid()
    value=(1-weight)*laplace_cdf(delta/p['quiet_scale'])+weight*laplace_cdf((delta-p['moving_delta'])/p['moving_scale'])
    density=log_density(p,root).exp().detach().clamp_min(1e-12)
    return root+(value.detach()-value)/density
