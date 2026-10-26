"""Native TAC with typed, role-specialized post-scan feed-forward heads.

Roles constrain which outputs a branch may produce, not the answer it learns.
No attention, new sequence recurrence, persistent state or decoder-to-scan
feedback is introduced. Shared-ancestor gradient conflicts remain possible.
"""
from enum import Enum

import torch
from torch import nn

from .point_priority_sensor_world import PointPrioritySensorWorld


class ComputationRole(str, Enum):
    RECONSTRUCT = 'current_reconstruction'
    WIDTH = 'uncertainty_shape'
    MEDIAN = 'point_innovation'


class RoleFFN(nn.Module):
    """102 ->16 tanh ->typed output; each role has its own learned parameters."""
    OUTPUTS = {ComputationRole.RECONSTRUCT: 12, ComputationRole.WIDTH: 12,
               ComputationRole.MEDIAN: 6}

    def __init__(self, role):
        super().__init__()
        self.role = ComputationRole(role)
        self.input = nn.Linear(102, 16)
        self.output = nn.Linear(16, self.OUTPUTS[self.role])

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != 102:
            raise ValueError('role features must have shape [batch,102]')
        return self.output(torch.tanh(self.input(features)))


class RoleSpecializedSensorWorld(PointPrioritySensorWorld):
    """Untrained102,713-parameter capability candidate; native TAC core intact.

    Existing outer transforms enforce bounded median/current innovations,
    positive asymmetric scales and source floors. New FFNs are freshly
    initialized, so this is NOT a bit-identical parent warm start. Specialization
    is learned inside fixed output roles; it is not a task-label lookup.
    """
    def __init__(self, floor_rate=None):
        super().__init__(floor_rate=floor_rate)
        self.current_head = RoleFFN(ComputationRole.RECONSTRUCT)
        self.width_head = RoleFFN(ComputationRole.WIDTH)
        self.median_head = RoleFFN(ComputationRole.MEDIAN)
