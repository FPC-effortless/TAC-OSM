"""Proper ordered-scale finite Laplace mixtures and shared quantile inversion."""
import torch
from torch.nn import functional as F
from .implicit_quantile import laplace_cdf


def log_density(prediction,target):
    delta=target.unsqueeze(-2)-prediction['locations'];scale=prediction['scales']
    return torch.logsumexp(F.log_softmax(prediction['component_logits'],dim=-2)-torch.log(2*scale)-delta.abs()/scale,dim=-2)


def cdf(prediction,value):
    probability=prediction['component_logits'].softmax(-2)
    return (probability*laplace_cdf((value.unsqueeze(-2)-prediction['locations'])/prediction['scales'])).sum(-2)


def mixture_crps(prediction,target):
    mean=prediction['locations'];scale=prediction['scales'];w=prediction['component_logits'].softmax(-2)
    delta=(target.unsqueeze(-2)-mean).abs()
    observation=(w*(delta+scale*torch.exp(-delta/scale))).sum(-2)
    pair=(w.square()*1.5*scale).sum(-2)
    for i in range(mean.shape[-2]):
        for j in range(i+1,mean.shape[-2]):
            b=scale[...,i,:];c=scale[...,j,:];distance=(mean[...,i,:]-mean[...,j,:]).abs()
            # Model enforces c>=2b; no near-equal divided difference.
            cross=distance+(c.pow(3)*torch.exp(-distance/c)-b.pow(3)*torch.exp(-distance/b))/(c.square()-b.square())
            pair=pair+2*w[...,i,:]*w[...,j,:]*cross
    return observation-.5*pair


def mixture_quantiles(prediction,probabilities=(.05,.5,.95)):
    probabilities=tuple(probabilities)
    if not probabilities or not all(0<p<1 for p in probabilities):raise ValueError('probabilities must lie in(0,1)')
    mean=prediction['locations'].double();scale=prediction['scales'].double();w=prediction['component_logits'].double().softmax(-2)
    low=(mean-32*scale).amin(-2);high=(mean+32*scale).amax(-2)
    lo=low.unsqueeze(0).expand(len(probabilities),*low.shape);hi=high.unsqueeze(0).expand_as(lo)
    levels=mean.new_tensor(probabilities).reshape(len(probabilities),*([1]*low.ndim))
    for _ in range(40):
        middle=(lo+hi)/2
        value=(w.unsqueeze(0)*laplace_cdf((middle.unsqueeze(-2)-mean.unsqueeze(0))/scale.unsqueeze(0))).sum(-2)
        below=value<levels;lo=torch.where(below,middle,lo);hi=torch.where(below,hi,middle)
    return tuple(((lo+hi)/2).unbind(0))


def implicit_median(prediction):
    with torch.no_grad():root=mixture_quantiles(prediction,(.5,))[0]
    p={k:v.double() for k,v in prediction.items()};value=cdf(p,root)
    density=log_density(p,root).exp().detach().clamp_min(1e-12)
    return root+(value.detach()-value)/density
