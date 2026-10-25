"""ME102 current sensor reconstruction with causal uncertainty propagation.

This extends ME101 with a small reconstruction head over the admitted source
state.  Its emission is a conditional diagonal Gaussian; it does not claim an
exact Jacobian for the full neural network.  ``floor_rate`` is a fixed,
FIT-derived per-channel log-variance rate per 24 hours, supplied by the
experiment after fitting that statistic on the training partition.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .belief_sensor_world import BeliefSensorWorld
from .entity_sensor_world import SensorForecast


class ReconstructionSensorWorld(BeliefSensorWorld):
    """Belief world with a masked current-reading reconstruction objective.

    ``floor_rate`` is a non-learned six-channel buffer.  Its units are
    standardized log1p variance per 24 hours and it is applied only when the
    current value is missing.  The default zero rates keep construction
    possible before a real-data FIT statistic has been supplied.
    """

    def __init__(self, floor_rate=None):
        super().__init__()
        self.current_head = nn.Linear(102, 12)
        if floor_rate is None:
            floor_rate = torch.zeros(6)
        floor_rate = torch.as_tensor(floor_rate, dtype=torch.float32)
        if floor_rate.shape != (6,) or not torch.isfinite(floor_rate).all() or torch.any(floor_rate < 0):
            raise ValueError('floor_rate must be six finite nonnegative FIT variances per 24 hours')
        self.register_buffer('floor_rate', floor_rate.clone())

    def forward(self, values, observed, ages, clock, owners, hours, *, state=None, return_trace=False):
        base, final, trace = super().forward(
            values, observed, ages, clock, owners, hours, state=state, return_trace=True
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
        raw = self.current_head(features)
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
        prediction = SensorForecast(base.log_location + shift, total_variance.sqrt())
        if return_trace:
            return prediction, final, dict(
                trace,
                current_mean=current_mean,
                current_variance=current_variance,
                current_anchor=anchor,
                belief_read=belief_read,
                shared_write_key=shared_write_key,
                base_location=base.log_location,
                modeled_variance=modeled_variance,
                floor_variance=floor_variance,
                total_variance=total_variance,
            )
        return prediction, final

    @staticmethod
    def reconstruction_loss(trace, targets_normalized, target_mask):
        """Masked Gaussian NLL for observed held-out current values.

        ``targets_normalized`` uses the model's standardized log1p target
        coordinates. Training callers create ``target_mask`` by hiding one
        pollutant's values across an artificial missing block, then score that
        pollutant's actually recorded current value.
        """
        mean = trace['current_mean']
        variance = trace['current_variance']
        if targets_normalized.shape != mean.shape or target_mask.shape != mean.shape:
            raise ValueError('reconstruction targets and mask must match [B,6] current outputs')
        if target_mask.dtype != torch.bool:
            raise ValueError('reconstruction target mask must be boolean')
        if targets_normalized.device != mean.device or target_mask.device != mean.device:
            raise ValueError('reconstruction targets and mask must share the model device')
        if not torch.isfinite(targets_normalized).all():
            raise ValueError('reconstruction targets must be finite')
        elementwise = 0.5 * (
            (targets_normalized - mean).square() / variance
            + variance.log()
            + math.log(2 * math.pi)
        )
        return torch.where(target_mask, elementwise, torch.zeros_like(elementwise)).sum() / target_mask.sum().clamp_min(1)
