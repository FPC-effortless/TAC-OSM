"""ME-062 candidate: selective residual trust over ME-061; not a trained result.

Both native TAC and GRU must use this identical forecasting head. All
probabilistic/abstention claims require fresh source calibration and evaluation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .real_robot_me060_trajectory import Forecast
from .real_robot_me061_reliable import ReliableTrajectoryModel


class SelectiveTrajectoryModel(ReliableTrajectoryModel):
    """A causal, horizon-conditioned blend of learned correction and persistence."""

    def __init__(self, identified, architecture="native_tac", width=32, decoder_width=64):
        super().__init__(identified, architecture, width, decoder_width)
        # Current state, physical servo displacement, observed velocity,
        # observed servo innovation, baseline predicted scale, horizon (72).
        # The entire head is shared in form and features across both arms.
        self.trust_head = nn.Sequential(nn.Linear(72, 24), nn.SiLU(), nn.Linear(24, 1))
        self.log_scale_head = nn.Sequential(nn.Linear(72, 24), nn.SiLU(), nn.Linear(24, 14))
        # A zero-initialized trust logit gives a neutral 0.5. Importantly,
        # the parent's zero residual output still makes the INITIAL point
        # forecast exactly persistence, without multiplying two zero gates.
        nn.init.zeros_(self.trust_head[-1].weight)
        nn.init.zeros_(self.trust_head[-1].bias)
        nn.init.zeros_(self.log_scale_head[-1].weight)
        nn.init.zeros_(self.log_scale_head[-1].bias)

    def _causal_context(self, belief, current, actions, base_scale):
        batch, horizon, width = actions.shape
        if width != 14 or current.shape != (batch, 14):
            raise ValueError("ME-062 physical adapter requires 14 state/action channels")
        state = current
        physical = []
        for step in range(horizon):
            state = state + self.servo_step(state, actions[:, step])
            physical.append(state)
        physical_delta = torch.stack(physical, dim=1) - current[:, None]
        h = torch.arange(1, horizon + 1, dtype=current.dtype, device=current.device)
        clock = torch.stack((h / 16., torch.log1p(h) / torch.log(h.new_tensor(17.))), -1)
        expanded = lambda value: value[:, None].expand(-1, horizon, -1)
        features = torch.cat((
            expanded(current), physical_delta, expanded(belief.velocity),
            expanded(belief.innovation), torch.log1p(base_scale.clamp_min(0.)),
            clock[None].expand(batch, -1, -1),
        ), -1)
        if features.shape[-1] != 72:
            raise RuntimeError("unexpected selective forecast input")
        return features

    def forecast(self, belief, current, actions):
        raw = super().forecast(belief, current, actions)
        features = self._causal_context(belief, current, actions, raw.scale)
        trust = torch.sigmoid(self.trust_head(features))
        correction = raw.mean - current[:, None]
        mean = current[:, None] + trust * correction
        # Heteroscedastic Gaussian standard deviation in normalized units.
        # The correction is bounded to avoid arbitrary exp growth.
        scale_multiplier = torch.exp(torch.tanh(self.log_scale_head(features)))
        scale = raw.scale * scale_multiplier
        # Disagreement with rejected neural motion is an uncertainty source,
        # not a grant of authority to execute an action.
        scale = torch.sqrt(scale.square() + ((1. - trust) * correction).square()).clamp_min(1e-5)
        return Forecast(mean, scale, raw.weights)

    def forecast_with_trust(self, belief, current, actions):
        """Return a forecast and interpretable trust [0,1], NOT approval."""
        raw = super().forecast(belief, current, actions)
        features = self._causal_context(belief, current, actions, raw.scale)
        trust = torch.sigmoid(self.trust_head(features))
        return self.forecast(belief, current, actions), trust


def selective_gaussian_forecast_loss(model, prediction, actual, alive):
    """Physical MAE plus FIT-scaled Gaussian proper score; shared across arms."""
    mask = alive.to(actual.dtype)
    if actual.shape[-1] != 14:
        raise ValueError("expected physical 14-channel targets")
    error = (prediction.mean - actual) * model.state_scale
    sigma = (prediction.scale * model.state_scale).clamp_min(1e-4)
    gaussian_nll = .5 * (error.square() / sigma.square() + 2. * torch.log(sigma) + math.log(2. * math.pi))
    reference = (model.delta_scale * model.state_scale).mean().detach()
    count = (mask.sum() * actual.shape[-1]).clamp_min(1.)
    return ((error.abs() + .01 * reference * gaussian_nll) * mask).sum() / count
