"""ME105 protected source-gap scales and a bounded median innovation.

The causal parallel memory graph is inherited without decoder feedback. This
head preserves the complete FIT-derived scale floor and, for missing current
channels, at least 80% of the inherited scale. Neither bound guarantees coverage.
"""
from torch import nn
import torch

from .median_split_sensor_world import MedianSplitForecast, MedianSplitSensorWorld


class ProtectedSensorWorld(MedianSplitSensorWorld):
    """Source-protected asymmetric law with independently trainable median readout."""

    def __init__(self, floor_rate=None):
        super().__init__(floor_rate=floor_rate)
        self.median_head = nn.Linear(102, 6)
        nn.init.zeros_(self.median_head.weight)
        nn.init.zeros_(self.median_head.bias)

    def forward(self, values, observed, ages, clock, owners, hours, *, state=None, return_trace=False):
        split, final, trace = super().forward(
            values, observed, ages, clock, owners, hours, state=state, return_trace=True
        )
        correction = 0.5 * self.median_head(trace['width_features']).tanh()
        # max(v, 0.0025) is NOT used: retain the inherited channel-specific floor.
        # Select a safe root argument before sqrt: a literal sqrt(0) would
        # poison age gradients through inactive torch.maximum branches.
        floor_var = trace['floor_variance']
        positive_floor = floor_var > 0
        safe_floor = torch.where(positive_floor, floor_var, torch.ones_like(floor_var))
        floor_scale = torch.where(positive_floor, safe_floor.sqrt(), torch.zeros_like(floor_var))
        missing_floor = torch.where(observed[:, -1, :6],
                                    torch.zeros_like(split.left_scale),
                                    0.8 * trace['base_log_scale'])
        protected_floor = torch.maximum(floor_scale, missing_floor)
        left = torch.maximum(split.left_scale, protected_floor)
        right = torch.maximum(split.right_scale, protected_floor)
        forecast = MedianSplitForecast(split.log_location + correction, left, right)
        if return_trace:
            return forecast, final, dict(
                trace, inherited_log_location=split.log_location,
                median_correction=correction, source_floor_scale=floor_scale,
                missing_floor_scale=missing_floor, protected_floor_scale=protected_floor,
                left_scale=left, right_scale=right,
            )
        return forecast, final
