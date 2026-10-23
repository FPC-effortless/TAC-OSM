"""ME-063: causal three-regime physical mixture, matched TAC/GRU experiment.

Candidates are last-state persistence, FIT-identified servo dynamics, and
native/GRU residual prediction. Selection features use observed history
and proposed action prefixes only. ALL scores remain unknown until training.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .real_robot_me060_trajectory import Forecast
from .real_robot_me061_reliable import ReliableTrajectoryModel


class RegimeMixtureModel(ReliableTrajectoryModel):
    def __init__(self, identified, architecture="native_tac", width=32, decoder_width=64):
        super().__init__(identified, architecture, width, decoder_width)
        # 14 current + 14 servo displacement + 14 velocity + 14 innovation
        # + 14 current physical scale + 2 horizon clock = 72
        self.router = nn.Sequential(nn.Linear(72, 24), nn.SiLU(), nn.Linear(24, 3))
        # Shared physical base variance calibrated only from original FIT
        self.prior_scale = nn.Sequential(nn.Linear(72, 24), nn.SiLU(), nn.Linear(24, 2))
        nn.init.zeros_(self.router[-1].weight)
        with torch.no_grad():
            self.router[-1].bias.copy_(torch.tensor([2.0, 0.0, 0.0]))
        nn.init.zeros_(self.prior_scale[-1].weight)
        nn.init.zeros_(self.prior_scale[-1].bias)

    def _components(self, belief, current, actions):
        if current.ndim != 2 or current.shape[-1] != 14 or (
                actions.ndim != 3 or actions.shape[0] != current.shape[0]
                or actions.shape[-1] != 14):
            raise ValueError("registered 14D physical action/state adapter required")
        raw = super().forecast(belief, current, actions)
        batch, horizon, _ = actions.shape
        servo = current
        predicted = []
        for t in range(horizon):
            servo = servo + self.servo_step(servo, actions[:, t])
            predicted.append(servo)
        servo = torch.stack(predicted, dim=1)
        h = torch.arange(1, horizon+1, dtype=current.dtype, device=current.device)
        clock = torch.stack((h/16., torch.log1p(h)/torch.log(h.new_tensor(17.))), -1)
        expand = lambda x: x[:, None].expand(-1, horizon, -1)
        features = torch.cat((
            expand(current), servo-current[:, None],
            expand(belief.velocity), expand(belief.innovation),
            torch.log1p(raw.scale.clamp_min(1e-5)),
            clock[None].expand(batch,-1,-1),
        ),-1)
        if features.size(-1) != 72:
            raise RuntimeError("incorrect physical routing feature dimensionality")
        weights = self.router(features).softmax(-1)
        means = torch.stack((expand(current), servo, raw.mean), -2)
        mean = torch.sum(weights[..., None]*means, dim=-2)
        h_factor = h[None, :, None].sqrt()
        natural_step = self.delta_scale[None,None].clamp_min(1e-5)*h_factor
        priors = torch.exp(torch.tanh(self.prior_scale(features)))
        innovation_factor = 1. + .05*belief.innovation[:,None].mean(-1,keepdim=True)
        stay_sigma = natural_step * priors[..., :1] * innovation_factor
        servo_sigma = natural_step * priors[..., 1:2] * innovation_factor
        scales = torch.stack((stay_sigma,servo_sigma,raw.scale.clamp_min(1e-5)),-2)
        # Moment matching makes expert disagreement an uncertainty source.
        variance = (weights[...,None]*(scales.square()+(means-mean[:,:,None]).square())).sum(-2)
        scale = variance.clamp_min(1e-10).sqrt()
        return Forecast(mean, scale, weights)

    def forecast(self, belief, current, actions):
        return self._components(belief, current, actions)

    def regime_weights(self, belief, current, actions):
        """Only interpretive weights; not authorization or causal certainty."""
        return self._components(belief, current, actions).weights


def gaussian_physical_loss(model, prediction, actual, alive):
    mask = alive.to(actual.dtype)
    if actual.shape[-1] != 14:
        raise ValueError("forecast requires real original 14D physical outcomes")
    error = (prediction.mean-actual)*model.state_scale
    sigma = (prediction.scale*model.state_scale).clamp_min(1e-4)
    physical_mae = error.abs()
    gaussian_nll = .5*(error.square()/sigma.square()+2.*sigma.log()+math.log(2.*math.pi))
    reference = (model.delta_scale*model.state_scale).mean().detach()
    count = (mask.sum()*14).clamp_min(1.)
    return ((physical_mae + .01*reference*gaussian_nll)*mask).sum()/count
