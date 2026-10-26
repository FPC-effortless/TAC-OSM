"""ME109 mask-conditioned role residuals over the unchanged ME106 state graph.

Zero output affine maps preserve the entire inherited predictive graph at
initialization, including already trained current, width and median heads.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .belief_sensor_world import BeliefSensorWorld
from .median_split_sensor_world import MedianSplitForecast
from .point_priority_sensor_world import PointPrioritySensorWorld


class RoleResidual(nn.Module):
    """108-to-12-to-output bounded-hidden residual with an exact zero start."""

    def __init__(self, output_size):
        super().__init__()
        self.input = nn.Linear(108, 12)
        self.output = nn.Linear(12, output_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, context):
        return self.output(self.input(context).tanh())


class ResidualRoleSensorWorld(PointPrioritySensorWorld):
    def __init__(self, floor_rate=None):
        super().__init__(floor_rate=floor_rate)
        self.current_residual = RoleResidual(12)
        self.width_residual = RoleResidual(12)
        self.median_residual = RoleResidual(6)

    def forward(self, values, observed, ages, clock, owners, hours, *, state=None, return_trace=False):
        base, final, trace = BeliefSensorWorld.forward(
            self, values, observed, ages, clock, owners, hours,
            state=state, return_trace=True,
        )
        # Use the post-admission source statistics for the current reading.
        # Source sums store values divided by eight, so restore input units.
        mass = final.source_mass[:, :6]
        source_mean = 8 * final.source_sum[:, :6] / mass.clamp_min(1e-6)
        anchor = torch.where(observed[:, -1, :6], values[:, -1, :6], source_mean)
        evidence_mean = final.evidence_sum / final.count.clamp_min(1).to(values.dtype)
        # Re-encode only the current masked event, then query the same address
        # projection used as the write key. The read uses the inclusive
        # post-scan memory and cannot affect the already-computed scan.
        safe_current = torch.where(observed[:, -1], values[:, -1], torch.zeros_like(values[:, -1]))
        station = self.station(owners)
        current_event = self.encoder(torch.cat((
            safe_current,
            observed[:, -1].to(values.dtype),
            ages[:, -1],
            clock[:, -1],
            station,
        ), dim=-1)).tanh()
        shared_write_key = F.normalize(self.address(current_event), dim=-1, eps=1e-6)
        belief_read = (final.memory @ shared_write_key.unsqueeze(-1)).squeeze(-1)
        features = torch.cat((
            8 * final.source_sum[:, :13] / final.source_mass[:, :13].clamp_min(1e-6),
            final.source_mass[:, :13],
            evidence_mean,
            ages[:, -1, :6],
            belief_read,
        ), dim=-1)
        context = torch.cat((features, observed[:, -1, :6].to(values.dtype)), dim=-1)
        raw = self.current_head(features) + self.current_residual(context)
        current_mean = anchor + 2 * raw[:, :6].tanh()
        current_variance = 0.0025 + F.softplus(raw[:, 6:]).square() + trace['latent_state_variance']

        missing = (~observed[:, -1, :6]).to(values.dtype)
        age_hours = torch.expm1(ages[:, -1, :6] * math.log(257.))
        floor_variance = missing * self.floor_rate.to(values.dtype) * (age_hours / 24)
        response = trace['affine_response']
        shift = response * (current_mean - anchor) * missing
        modeled_variance = (
            trace['future_variance']
            + response.square() * current_variance * missing
        )
        total_variance = torch.maximum(modeled_variance, floor_variance)
        base_location = base.log_location
        location = base_location + shift
        base_scale = total_variance.sqrt()
        raw_width = self.width_head(features) + self.width_residual(context)
        width_factors = torch.exp(math.log(4.0) * raw_width.tanh())
        split_left = base_scale * width_factors[:, :6]
        split_right = base_scale * width_factors[:, 6:]
        raw_median = self.median_head(features) + self.median_residual(context)
        correction = 0.5 * raw_median.tanh()
        positive_floor = floor_variance > 0
        safe_floor = torch.where(positive_floor, floor_variance, torch.ones_like(floor_variance))
        floor_scale = torch.where(positive_floor, safe_floor.sqrt(), torch.zeros_like(floor_variance))
        missing_floor = torch.where(observed[:, -1, :6],
                                    torch.zeros_like(split_left), 0.8 * base_scale)
        protected_floor = torch.maximum(floor_scale, missing_floor)
        left = torch.maximum(split_left, protected_floor)
        right = torch.maximum(split_right, protected_floor)
        forecast = MedianSplitForecast(location + correction, left, right)
        if return_trace:
            return forecast, final, dict(
                trace, current_mean=current_mean, current_variance=current_variance,
                current_anchor=anchor, belief_read=belief_read,
                shared_write_key=shared_write_key, base_location=base_location,
                modeled_variance=modeled_variance, floor_variance=floor_variance,
                total_variance=total_variance, base_log_scale=base_scale,
                width_factors=width_factors, width_features=features,
                inherited_log_location=location, median_correction=correction,
                source_floor_scale=floor_scale, missing_floor_scale=missing_floor,
                protected_floor_scale=protected_floor, left_scale=left, right_scale=right,
            )
        return forecast, final
