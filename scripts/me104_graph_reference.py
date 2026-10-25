"""Independent ME104 source evaluator layered on the unchanged ME102 reference.

This deliberately never calls ``MedianSplitSensorWorld.forward``. The parent
serial graph supplies state tensors; the 102 features and width equations are
rebuilt here to provide an executable Python mathematical reference.
"""
import math

import torch
from torch.nn import functional as F

try:
    from scripts.me102_graph_reference import serial as me102_serial
except ImportError:
    from me102_graph_reference import serial as me102_serial


def serial(model, values, observed, ages, clock, owners):
    """Return location, two scales, state payload and independently built trace."""
    (location, base_scale, source_sum, source_mass, evidence_sum, memory,
     inherited) = me102_serial(model, values, observed, ages, clock, owners)
    source_context = 8 * source_sum[:, :13] / source_mass[:, :13].clamp_min(1e-6)
    eligible_count = observed.any(dim=-1).to(values.dtype).sum(dim=1, keepdim=True)
    evidence_context = evidence_sum / eligible_count.clamp_min(1)
    # ME102's independent serial path already computed this inclusive-memory
    # read and its key; the new head only consumes those reference tensors.
    shared_write_key = inherited['shared_write_key']
    belief_read = inherited['belief_read']
    features = torch.cat((
        source_context, source_mass[:, :13], evidence_context,
        ages[:, -1, :6], belief_read,
    ), dim=-1)
    raw = F.linear(features, model.width_head.weight, model.width_head.bias)
    factors = torch.exp(math.log(4.0) * torch.tanh(raw))
    left_scale = base_scale * factors[:, :6]
    right_scale = base_scale * factors[:, 6:]
    trace = dict(inherited)
    trace.update(base_log_scale=base_scale, width_features=features,
                 width_raw=raw, width_factors=factors, left_scale=left_scale,
                 right_scale=right_scale, belief_read=belief_read,
                 shared_write_key=shared_write_key)
    return location, left_scale, right_scale, source_sum, source_mass, evidence_sum, memory, trace
