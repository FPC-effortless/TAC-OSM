"""ME104 median-preserving two-sided lognormal sensor forecast.

The inherited ME102 location and sensor-state graph are reused unchanged. A
zero-initialized 102-to-12 head predicts bounded left/right scale multipliers.
On standardized log1p coordinates the two Gaussian half-lines each retain
probability 1/2, so the reported location remains the median at initialization
and after every forward pass. This does not preserve the arithmetic mean.
"""
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .reconstruction_sensor_world import ReconstructionSensorWorld


class MedianSplitForecast(NamedTuple):
    log_location: torch.Tensor
    left_scale: torch.Tensor
    right_scale: torch.Tensor


class MedianSplitSensorWorld(ReconstructionSensorWorld):
    """ME102 world with a coherent asymmetric distribution around its median."""

    def __init__(self, floor_rate=None):
        super().__init__(floor_rate=floor_rate)
        self.width_head = nn.Linear(102, 12)
        nn.init.zeros_(self.width_head.weight)
        nn.init.zeros_(self.width_head.bias)

    def forward(self, values, observed, ages, clock, owners, hours, *, state=None, return_trace=False):
        base, final, old_trace = super().forward(
            values, observed, ages, clock, owners, hours, state=state, return_trace=True
        )
        source_mean = 8 * final.source_sum[:, :13] / final.source_mass[:, :13].clamp_min(1e-6)
        evidence_mean = final.evidence_sum / final.count.clamp_min(1).to(values.dtype)
        # ME102 already built this read from the inclusive post-scan memory.
        # Reuse it so the width head adds no duplicate encoder/address graph.
        shared_write_key = old_trace['shared_write_key']
        belief_read = old_trace['belief_read']
        features = torch.cat((
            source_mean, final.source_mass[:, :13], evidence_mean,
            ages[:, -1, :6], belief_read,
        ), dim=-1)
        raw = self.width_head(features)
        width_factors = torch.exp(math.log(4.0) * raw.tanh())
        left_scale = base.log_scale * width_factors[:, :6]
        right_scale = base.log_scale * width_factors[:, 6:]
        forecast = MedianSplitForecast(base.log_location, left_scale, right_scale)
        if return_trace:
            return forecast, final, dict(
                old_trace,
                base_log_scale=base.log_scale,
                width_factors=width_factors,
                left_scale=left_scale,
                right_scale=right_scale,
                width_features=features,
                belief_read=belief_read,
                shared_write_key=shared_write_key,
            )
        return forecast, final

    def raw_quantiles(self, forecast, probabilities=None):
        """Return censored raw-unit quantiles for probabilities strictly in (0,1)."""
        if probabilities is None:
            probabilities = forecast.log_location.new_tensor([0.05, 0.50, 0.95])
        p = torch.as_tensor(probabilities, dtype=forecast.log_location.dtype,
                            device=forecast.log_location.device)
        if p.ndim != 1 or p.numel() == 0 or not torch.isfinite(p).all() or torch.any((p <= 0) | (p >= 1)):
            raise ValueError('probabilities must be a nonempty vector strictly between zero and one')
        legacy = torch.tensor([-1.6448536269514722, 0.0, 1.6448536269514722],
                              dtype=p.dtype, device=p.device)
        if p.shape == (3,) and torch.equal(p, p.new_tensor([0.05, 0.50, 0.95])):
            z = legacy
        else:
            z = torch.special.ndtri(p)
        scale = torch.where(z[None, None, :] < 0,
                            forecast.left_scale[..., None], forecast.right_scale[..., None])
        standardized = forecast.log_location[..., None] + scale * z
        return torch.expm1(standardized * self.target_scale[None, :, None]
                           + self.target_mean[None, :, None]).clamp_min(0)

    def cdf(self, forecast, raw_values):
        """CDF of the censored raw-unit forecast, including its atom at zero."""
        y = torch.as_tensor(raw_values, dtype=forecast.log_location.dtype,
                            device=forecast.log_location.device)
        if y.shape == forecast.log_location.shape:
            mu, left, right = forecast.log_location, forecast.left_scale, forecast.right_scale
            target_mean, target_scale = self.target_mean, self.target_scale
        elif y.ndim == forecast.log_location.ndim + 1 and y.shape[:-1] == forecast.log_location.shape:
            mu = forecast.log_location[..., None]
            left, right = forecast.left_scale[..., None], forecast.right_scale[..., None]
            target_mean, target_scale = self.target_mean[..., None], self.target_scale[..., None]
        else:
            raise ValueError('raw_values must match [B,6] or [B,6,K] forecast shape')
        if not torch.isfinite(y).all():
            raise ValueError('raw_values must be finite and match [B,6] forecast shape')
        transformed = (torch.log1p(y.clamp_min(0)) - target_mean) / target_scale
        side = torch.where(transformed < mu, left, right)
        positive_cdf = torch.special.ndtr((transformed - mu) / side)
        threshold = -target_mean / target_scale
        threshold_side = torch.where(threshold < mu, left, right)
        zero_atom = torch.special.ndtr((threshold - mu) / threshold_side)
        return torch.where(y < 0, torch.zeros_like(positive_cdf),
                           torch.where(y == 0, zero_atom, positive_cdf))

    def nll(self, forecast, raw_targets, target_mask):
        """Censored split-lognormal NLL; includes the log1p Jacobian for y>0."""
        mu = forecast.log_location
        y = torch.as_tensor(raw_targets, dtype=mu.dtype, device=mu.device)
        if y.shape != mu.shape or not torch.isfinite(y).all() or torch.any(y < 0):
            raise ValueError('raw_targets must be finite nonnegative [B,6] values')
        if target_mask.shape != mu.shape or target_mask.dtype != torch.bool or target_mask.device != mu.device:
            raise ValueError('target_mask must be boolean [B,6] on forecast device')
        z = (torch.log1p(y.clamp_min(0)) - self.target_mean) / self.target_scale
        side = torch.where(z < mu, forecast.left_scale, forecast.right_scale)
        positive = (0.5 * ((z - mu) / side).square() + side.log()
                    + 0.5 * math.log(2 * math.pi) + self.target_scale.log()
                    + torch.log1p(y.clamp_min(0)))
        threshold = -self.target_mean / self.target_scale
        threshold_side = torch.where(threshold < mu, forecast.left_scale, forecast.right_scale)
        zero = -torch.special.log_ndtr((threshold - mu) / threshold_side)
        losses = torch.where(y == 0, zero, positive)
        return torch.where(target_mask, losses, torch.zeros_like(losses)).sum() / target_mask.sum().clamp_min(1)
