"""Independent ME102 evaluator layered on the ME101 chronological reference.

This computes the new reconstruction head, missing-current location update,
variance maximum, and masked Gaussian NLL without calling the ME102 model's
``forward`` or ``reconstruction_loss``. The inherited ME100/ME101 path remains
the same evaluator used by the corresponding earlier mathematical references.
"""
import math

import torch
from torch.nn import functional as F

try:
    from scripts.me101_graph_reference import serial as belief_serial
except ImportError:
    from belief_sensor_reference import serial as belief_serial


def serial(model, values, observed, ages, clock, owners):
    """Return location, scale, inherited state, and independent ME102 trace."""
    (old_location, _old_scale, source_sum, source_mass, evidence_sum, memory,
     inherited) = belief_serial(model, values, observed, ages, clock, owners)

    source_context = 8 * source_sum[:, :13] / source_mass[:, :13].clamp_min(1e-6)
    # ME100 stores the count separately in the final source state; it is not
    # returned by ME101's reference API. Reconstruct the same eligible-frame
    # count directly from the observation chronology.
    eligible = observed.any(dim=-1).to(values.dtype).sum(dim=1, keepdim=True)
    evidence_context = evidence_sum / eligible.clamp_min(1)
    safe_current = torch.where(
        observed[:, -1], values[:, -1], torch.zeros_like(values[:, -1])
    )
    station = model.station(owners)
    current_event = torch.cat((
        safe_current,
        observed[:, -1].to(values.dtype),
        ages[:, -1],
        clock[:, -1],
        station,
    ), dim=-1)
    current_encoding = torch.tanh(model.encoder(current_event))
    shared_write_key = F.normalize(
        model.address(current_encoding), dim=-1, eps=1e-6
    )
    belief_read = torch.matmul(memory, shared_write_key.unsqueeze(-1)).squeeze(-1)
    features = torch.cat((
        source_context,
        source_mass[:, :13],
        evidence_context,
        ages[:, -1, :6],
        belief_read,
    ), dim=-1)
    raw = model.current_head(features)
    anchor = torch.where(
        observed[:, -1, :6], values[:, -1, :6],
        source_context[:, :6],
    )
    current_mean = anchor + 2 * torch.tanh(raw[:, :6])
    latent_state_variance = inherited["latent_state_variance"]
    current_variance = 0.0025 + F.softplus(raw[:, 6:]).square() + latent_state_variance

    missing = (~observed[:, -1, :6]).to(values.dtype)
    age_hours = torch.expm1(ages[:, -1, :6] * math.log(257.0))
    floor_rate = model.floor_rate.to(dtype=values.dtype, device=values.device)
    floor_variance = missing * floor_rate * age_hours / 24
    response = inherited["affine_response"]
    modeled_variance = (
        inherited["future_variance"]
        + response.square() * current_variance * missing
    )
    total_variance = torch.maximum(modeled_variance, floor_variance)
    location = old_location + response * (current_mean - anchor) * missing
    scale = total_variance.sqrt()
    trace = dict(
        inherited,
        current_mean=current_mean,
        current_variance=current_variance,
        current_anchor=anchor,
        belief_read=belief_read,
        shared_write_key=shared_write_key,
        floor_variance=floor_variance,
        modeled_variance=modeled_variance,
        total_variance=total_variance,
    )
    return location, scale, source_sum, source_mass, evidence_sum, memory, trace


def reconstruction_loss(trace, targets_normalized, target_mask):
    """Independent masked Gaussian NLL on held-out standardized log values."""
    mean = trace["current_mean"]
    variance = trace["current_variance"]
    elementwise = 0.5 * (
        (targets_normalized - mean).square() / variance
        + torch.log(variance)
        + math.log(2 * math.pi)
    )
    weights = target_mask.to(dtype=mean.dtype)
    return (elementwise * weights).sum() / weights.sum().clamp_min(1)
