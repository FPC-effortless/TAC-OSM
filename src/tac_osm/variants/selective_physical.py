"""Native System1 with a coherent stationary/moving Laplace mixture (ME077).

Observed motion is context, not a future input. The recurrent fast state remains
bounded. Mixture median is the physical-MAE point prediction; quantiles and NLL
come from the same distribution. No policy, causal-control or safety authority.
"""
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F


class MotionForecast(NamedTuple):
    current: torch.Tensor
    moving_delta: torch.Tensor
    stationary_scale: torch.Tensor
    moving_scale: torch.Tensor
    mixture_logit: torch.Tensor
    motion_logit: torch.Tensor
    fast: torch.Tensor


def mixture_cdf(delta, pred):
    def laplace(z):
        return .5 + .5 * z.sign() * (-torch.expm1(-z.abs()))
    p = pred.mixture_logit.sigmoid()
    return (1-p)*laplace(delta/pred.stationary_scale) + p*laplace(
        (delta-pred.moving_delta)/pred.moving_scale)


def mixture_quantile(pred, probability):
    if not 0 < probability < 1:
        raise ValueError("quantile probability must lie inside (0,1)")
    radius = 40 * torch.maximum(pred.stationary_scale, pred.moving_scale)
    low = torch.minimum(pred.moving_delta, torch.zeros_like(pred.moving_delta)) - radius
    high = torch.maximum(pred.moving_delta, torch.zeros_like(pred.moving_delta)) + radius
    for _ in range(26):
        mid = (low+high)*.5
        right = mixture_cdf(mid, pred) < probability
        low = torch.where(right, mid, low)
        high = torch.where(right, high, mid)
    return pred.current + (low+high)*.5


def mixture_nll(pred, target):
    delta = target - pred.current
    log0 = F.logsigmoid(-pred.mixture_logit) - (2*pred.stationary_scale).log() - delta.abs()/pred.stationary_scale
    log1 = F.logsigmoid(pred.mixture_logit) - (2*pred.moving_scale).log() - (delta-pred.moving_delta).abs()/pred.moving_scale
    return -torch.logaddexp(log0, log1).mean()


class SelectivePhysicalModel(nn.Module):
    architecture = "system1_selective_physical"

    def __init__(self, delta_scale=None, delta_bound=None):
        super().__init__()
        width = 96
        self.encoder = nn.Sequential(nn.Linear(43, width), nn.SiLU(), nn.Linear(width, width), nn.Tanh())
        self.fast_candidate_gate = nn.Linear(192, 192)
        self.fast_residual_gate = nn.Linear(192, 192)
        self.readout = nn.Linear(192, 56)
        self.heads = nn.Linear(56, 71)
        # All heads are active. These are initializations, not an oracle motion gate.
        nn.init.zeros_(self.heads.weight)
        nn.init.zeros_(self.heads.bias)
        with torch.no_grad():
            self.heads.bias[:14].fill_(-2.)
            self.heads.bias[-1].fill_(-2.)
        ds = torch.ones(14)*.01 if delta_scale is None else torch.as_tensor(delta_scale).float()
        db = torch.ones(14)*.1 if delta_bound is None else torch.as_tensor(delta_bound).float()
        if ds.shape != (14,) or db.shape != (14,) or not (ds>0).all() or not (db>0).all():
            raise ValueError("FIT-only positive source scales required")
        self.register_buffer("delta_scale", ds.clone())
        self.register_buffer("delta_bound", db.clone())

    def forward(self, current, action, velocity, elapsed_seconds, fast=None):
        if current.ndim != 2 or current.shape[-1] != 14 or action.shape != current.shape or velocity.shape != current.shape:
            raise ValueError("14D observed state, prior-known action and observed velocity required")
        if elapsed_seconds.shape != (len(current),) or not (elapsed_seconds>=0).all():
            raise ValueError("nonnegative observed elapsed time required")
        if not all(torch.isfinite(v).all() for v in (current,action,velocity,elapsed_seconds)):
            raise ValueError("finite observed inputs required")
        if fast is None:
            fast = current.new_zeros(len(current),96)
        if fast.shape != (len(current),96) or not torch.isfinite(fast).all():
            raise ValueError("entity-separated fast state required")
        observed = self.encoder(torch.cat((current,action,velocity,torch.log1p(elapsed_seconds)[:,None]),-1))
        pair = torch.cat((observed,fast),-1)
        candidate, gate = self.fast_candidate_gate(pair).chunk(2,-1)
        residual, rgate = self.fast_residual_gate(pair).chunk(2,-1)
        updated = torch.tanh(torch.lerp(fast,candidate.tanh(),gate.sigmoid()) + .05*rgate.sigmoid()*residual.tanh())
        output = self.heads(F.silu(self.readout(torch.cat((updated,observed),-1))))
        logit, raw_delta, raw_b0, raw_b1, velocity_gate = output[:,:70].chunk(5,-1)
        delta = self.delta_bound * torch.tanh((velocity*velocity_gate.sigmoid() + .1*self.delta_bound*raw_delta.tanh())/self.delta_bound)
        b0 = (self.delta_scale*.1*(F.softplus(raw_b0)+.01)).clamp_min(1e-6)
        b1 = (self.delta_scale*(F.softplus(raw_b1)+.01)).clamp_min(1e-6)
        return MotionForecast(current,delta,b0,b1,logit,output[:,70],updated)
