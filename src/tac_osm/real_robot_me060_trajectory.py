"""Native TAC trajectory experts and an equally informed GRU comparator.

Fast/slow gated updates and an associative delta-rule matrix remain TAC's
native state. The physical decoder predicts all horizons directly and exposes
Laplace forecast scales. Uncertainty does not grant permission to act.
"""
from __future__ import annotations

from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F


class WorldState(NamedTuple):
    fast: torch.Tensor
    slow: torch.Tensor
    memory: torch.Tensor


class Forecast(NamedTuple):
    mean: torch.Tensor
    scale: torch.Tensor
    weights: torch.Tensor


def physical_interval(prediction, state_scale, calibration_quantile):
    """Return normalized-coordinate limits with physical-unit radii."""
    if not calibration_quantile > 0:
        raise ValueError("positive factual calibration quantile required")
    radius=prediction.scale*calibration_quantile
    return prediction.mean-radius,prediction.mean+radius,radius*state_scale


def forecast_requires_review(physical_radius, tolerance):
    """Caller-provided physical tolerance; no autonomous action authorization."""
    limit=torch.as_tensor(tolerance,device=physical_radius.device)
    if limit.shape!=(14,) or not torch.isfinite(limit).all() or not (limit>0).all():
        raise ValueError("14 positive physical error tolerances required")
    return (physical_radius>limit).any(-1)


class TrajectoryWorldModel(nn.Module):
    def __init__(self, identified, architecture="native_tac", width=32,
                 decoder_width=64):
        super().__init__()
        if architecture not in ("native_tac", "gru_world"):
            raise ValueError("unknown architecture")
        self.architecture, self.width = architecture, width
        for name, value in identified.items():
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.shape != (14,) or not torch.isfinite(tensor).all():
                raise ValueError("finite 14D FIT identification required")
            self.register_buffer(name, tensor.clone())
        self.encoder = nn.Linear(43, 24)
        if architecture == "native_tac":
            self.fast_map = nn.Linear(24+width+8, 2*width)
            self.slow_map = nn.Linear(2*width+8, 2*width)
            self.write_map = nn.Linear(24+2*width, 18)
            self.retention = nn.Parameter(torch.full((8,), 4.))
            feature_width = 2*width+8
        else:
            self.cell = nn.GRUCell(24, width)
            feature_width = width
        self.query = nn.Linear(30, 8)
        self.decoder = nn.Linear(feature_width+30, 2*decoder_width)
        # Three learned residual experts, physical-prior mixture, aleatoric scale.
        self.output = nn.Linear(decoder_width, 42+4+14)
        # Start near the known physical prediction; every output can train.
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[42:46].copy_(torch.tensor([0., -1., -1., -1.]))

    def initial(self, batch, device=None):
        ref = self.state_mean
        z = ref.new_zeros(batch, self.width, device=device)
        return WorldState(z, z.clone(), ref.new_zeros(batch, 8, 8, device=device))

    @staticmethod
    def address(value):
        return F.normalize(value, dim=-1, eps=1e-6)

    def servo_step(self, state, action):
        target = (action*self.action_scale+self.action_mean-self.state_mean)/self.state_scale
        delta = self.servo_gain*(target-state)
        return delta.clamp(-self.servo_step_limit, self.servo_step_limit)

    def observe(self, belief, previous, action, current, authority):
        # Innovation uses only the just-arrived real observation, before any
        # future target is requested. tanh(0)=0 guarantees a zero innovation.
        innovation = torch.tanh((current-previous-self.servo_step(previous, action)) /
                                self.delta_scale.clamp_min(1e-5))
        u = torch.tanh(self.encoder(torch.cat((current, action, innovation,
                                              authority.to(current.dtype)), -1)))
        if self.architecture == "gru_world":
            fast = self.cell(u, belief.fast)
            return WorldState(fast, belief.slow, belief.memory)
        fast, slow, memory = belief
        key = self.address(fast[:, :8]+slow[:, :8])
        recall = torch.einsum("bij,bj->bi", memory, key)
        proposal, gate = self.fast_map(torch.cat((u, fast, recall), -1)).chunk(2, -1)
        fast = torch.lerp(fast, torch.tanh(proposal), torch.sigmoid(gate))
        update, slow_gate = self.slow_map(torch.cat((fast, slow, recall), -1)).chunk(2, -1)
        slow = torch.lerp(slow, torch.tanh(slow+torch.tanh(update)),
                          .25*torch.sigmoid(slow_gate))
        k, value, write, relevance = self.write_map(torch.cat((u, fast, slow), -1)).split((8,8,1,1), -1)
        k = self.address(k)
        # Delta-rule write with learned retention and a Frobenius projection.
        # Projection bounds the associative state without repeated tanh erasure.
        retained = memory*(.95+.0499*torch.sigmoid(self.retention))[None, :, None]
        error = torch.tanh(value)-torch.einsum("bij,bj->bi", retained, k)
        beta = torch.sigmoid(write)*torch.sigmoid(relevance)
        memory = retained+beta[:, :, None]*error[:, :, None]*k[:, None, :]
        norm = memory.square().sum((-2,-1), keepdim=True).sqrt().clamp_min(1.)
        return WorldState(fast, slow, memory/norm)

    def forecast(self, belief, current, actions):
        batch, horizon, _ = actions.shape
        if horizon < 1 or horizon > 64:
            raise ValueError("registered horizons are 1 through 64")
        # The fixed physical base consumes only the action prefix at each h.
        state = current
        bases = []
        for h in range(horizon):
            state = state+self.servo_step(state, actions[:, h])
            bases.append(state)
        base = torch.stack(bases, 1)
        h = torch.arange(1, horizon+1, device=current.device, dtype=current.dtype)
        time = torch.stack((h/16., torch.log1p(h)/torch.log(h.new_tensor(17.))), -1)
        displacement = base-current[:, None]
        context = torch.cat((current[:, None].expand(-1,horizon,-1),
                             displacement, time[None].expand(batch,-1,-1)), -1)
        query = self.address(self.query(context))
        if self.architecture == "native_tac":
            recall = torch.einsum("bij,bhj->bhi", belief.memory, query)
            features = torch.cat((belief.fast[:,None].expand(-1,horizon,-1),
                                  belief.slow[:,None].expand(-1,horizon,-1), recall), -1)
        else:
            # Same learned query and its action/horizon information are useful
            # inputs to the GRU comparator, rather than unused parameter padding.
            features = belief.fast[:,None].expand(-1,horizon,-1)
            features = features+F.pad(query, (0,self.width-8))
        left, right = self.decoder(torch.cat((features, context), -1)).chunk(2, -1)
        values = self.output(F.silu(left)*right)
        residual = values[..., :42].reshape(batch,horizon,3,14).tanh()
        weights = values[...,42:46].softmax(-1)
        # The zero residual is the fourth expert. Residual size grows with the
        # factual time horizon, while the convex mixture stays bounded.
        bound = self.delta_scale[None,None]*h[None,:,None]
        correction = (weights[...,1:,None]*residual).sum(-2)*bound
        # Learn when the physical servo is helpful, retaining persistence as
        # a direct path. No predicted observation is recurrently re-ingested.
        mean = current[:,None]+(1.-weights[...,:1])*displacement+correction
        scale = (F.softplus(values[...,46:])+1e-3)*bound.clamp_min(1e-4)
        return Forecast(mean, scale, weights)


def forecast_loss(prediction, actual, alive):
    """Factual normalized absolute error plus a proper Laplace score."""
    mask = alive.to(actual.dtype)
    count = (mask.sum()*actual.shape[-1]).clamp_min(1.)
    error = (prediction.mean-actual).abs()
    nll = error/prediction.scale+torch.log(2.*prediction.scale)
    return ((error+.01*nll)*mask).sum()/count
