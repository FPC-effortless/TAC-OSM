"""Independent complete ME106 forward and training-gradient equations.

Forward reuses the unchanged independent ME105 serial formulation. Losses and
priority projection below are rebuilt without calling the production training
helper, uncertainty view, likelihood, quantiles or reconstruction-loss methods.
"""
import math
import torch
from scripts.me105_graph_reference import serial


def loss_parts(model, batch, rawscale):
    mu, left, right, *_state, trace = serial(model, *batch[:5])
    correction = trace['median_correction']
    uncertainty_mu = mu + (correction.detach() - correction)
    mask, y = batch[7], batch[6]
    z = (torch.log1p(y) - model.target_mean) / model.target_scale

    def average(x, selection=mask):
        return torch.where(selection, x, torch.zeros_like(x)).sum() / selection.sum().clamp_min(1)

    side = torch.where(z < uncertainty_mu, left, right)
    ordinary = (0.5*((z-uncertainty_mu)/side).square() + side.log()
                + .5*math.log(2*math.pi) + model.target_scale.log() + torch.log1p(y))
    threshold = -model.target_mean/model.target_scale
    zero_side = torch.where(threshold < uncertainty_mu, left, right)
    zero = -torch.special.log_ndtr((threshold-uncertainty_mu)/zero_side)
    nll = average(torch.where(y == 0, zero, ordinary))
    location_mse = average((mu-z).square())
    current_variance = trace['current_variance']
    reconstruction = .5*((batch[8]-trace['current_mean']).square()/current_variance
                          + current_variance.log() + math.log(2*math.pi))
    reconstruction_nll = average(reconstruction, batch[9])
    median = torch.expm1(mu*model.target_scale+model.target_mean).clamp_min(0)
    point = average((median-y).abs()/rawscale)
    lower = torch.expm1((uncertainty_mu-1.6448536269514722*left)*model.target_scale+model.target_mean).clamp_min(0)
    upper = torch.expm1((uncertainty_mu+1.6448536269514722*right)*model.target_scale+model.target_mean).clamp_min(0)
    interval = average((upper-lower+20*(lower-y).clamp_min(0)+20*(y-upper).clamp_min(0))/rawscale)
    return dict(nll=nll,point=point,location_mse=location_mse,
                reconstruction_nll=reconstruction_nll,interval=interval)


def priority_gradients(model, batch, rawscale, coefficients):
    parts = loss_parts(model, batch, rawscale)
    parameters = tuple(model.parameters())
    p = torch.autograd.grad(coefficients['point']*parts['point'], parameters,
                            retain_graph=True, allow_unused=True)
    u = torch.autograd.grad(parts['nll']+.1*parts['location_mse']+.3*parts['reconstruction_nll']
                            +coefficients['interval']*parts['interval'],parameters,allow_unused=True)
    flat_p = torch.cat([(torch.zeros_like(w)if g is None else g).reshape(-1).double()for w,g in zip(parameters,p)])
    flat_u = torch.cat([(torch.zeros_like(w)if g is None else g).reshape(-1).double()for w,g in zip(parameters,u)])
    norm = flat_p.dot(flat_p)
    dot = flat_p.dot(flat_u)
    projected = flat_u - dot/norm*flat_p if norm > 0 and dot < 0 else flat_u
    # Production rounds the projected auxiliary to model dtype before addition.
    result=[];offset=0
    for w in parameters:
        size=w.numel();p_slice=flat_p[offset:offset+size].reshape_as(w).to(w.dtype)
        u_slice=projected[offset:offset+size].reshape_as(w).to(w.dtype)
        result.append(p_slice+u_slice);offset+=size
    return result, parts
