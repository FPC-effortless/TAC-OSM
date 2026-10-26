"""Independent ME109 retained-affine residual evaluator over the unchanged native serial scan.

This deliberately never calls ``ProtectedSensorWorld.forward``. The parent
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


def ffn(head, x):
    return F.linear(torch.tanh(F.linear(x, head.input.weight, head.input.bias)),
                    head.output.weight, head.output.bias)


class ReferenceView:
    """Dispatch only the new current head to independent tensor equations."""
    def __init__(self, model, available):
        self.model = model
        self.available = available

    def __getattr__(self, name):
        if name == 'current_head':
            return lambda x: (F.linear(x, self.model.current_head.weight, self.model.current_head.bias)
                              + ffn(self.model.current_residual, torch.cat((x, self.available.to(x.dtype)), dim=-1)))
        return getattr(self.model, name)


def serial(model, values, observed, ages, clock, owners):
    """Return location, two scales, state payload and independently built trace."""
    (location, base_scale, source_sum, source_mass, evidence_sum, memory,
     inherited) = me102_serial(ReferenceView(model, observed[:, -1, :6]), values, observed, ages, clock, owners)
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
    context = torch.cat((features, observed[:, -1, :6].to(features.dtype)), dim=-1)
    raw = (F.linear(features, model.width_head.weight, model.width_head.bias)
           + ffn(model.width_residual, context))
    factors = torch.exp(math.log(4.0) * torch.tanh(raw))
    left_scale = base_scale * factors[:, :6]
    right_scale = base_scale * factors[:, 6:]
    median_raw = (F.linear(features, model.median_head.weight, model.median_head.bias)
                  + ffn(model.median_residual, context))
    correction = 0.5 * torch.tanh(median_raw)
    old_location = location
    location = location + correction
    floor_var = inherited['floor_variance']
    positive_floor = floor_var > 0
    root_arg = torch.where(positive_floor, floor_var, torch.ones_like(floor_var))
    floor_scale = torch.where(positive_floor, root_arg.sqrt(), torch.zeros_like(floor_var))
    missing_floor = torch.where(observed[:, -1, :6], torch.zeros_like(base_scale), 0.8 * base_scale)
    protected_floor = torch.maximum(floor_scale, missing_floor)
    left_scale = torch.maximum(left_scale, protected_floor)
    right_scale = torch.maximum(right_scale, protected_floor)
    trace = dict(inherited)
    trace.update(base_log_scale=base_scale, width_features=features,
                 width_raw=raw, median_raw=median_raw, residual_context=context, width_factors=factors, left_scale=left_scale,
                 right_scale=right_scale, belief_read=belief_read,
                 shared_write_key=shared_write_key, inherited_log_location=old_location,
                 median_correction=correction, source_floor_scale=floor_scale,
                 missing_floor_scale=missing_floor, protected_floor_scale=protected_floor)
    return location, left_scale, right_scale, source_sum, source_mass, evidence_sum, memory, trace
