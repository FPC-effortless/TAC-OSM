"""Independent ME101 evaluator over the pinned ME100 chronological reference.

The inherited serial source evaluator reconstructs the base forecast/state.
This module separately computes the ME101 variance law and its gradients; it
does not call BeliefSensorWorld.forward or its training_loss.
"""
import math

import torch
from torch.nn import functional as F

try:
    from scripts.me100_graph_reference import serial as me100_serial
except ImportError:
    from entity_sensor_reference import serial as me100_serial


def serial(model, values, observed, ages, clock, owners):
    """Return location, scale, inherited state, and ME101 uncertainty trace."""
    location, future_scale, source_sum, source_mass, evidence_sum, memory = me100_serial(
        model, values, observed, ages, clock, owners
    )

    age_hours = torch.expm1(ages[:, -1, :6] * math.log(257.0))
    confidence = source_mass[:, :6].clamp(0, 1)
    coefficients = F.softplus(model.state_noise)
    latent_variance = (
        coefficients[0] + coefficients[1] * age_hours / 24
    ) * (1 + coefficients[2] * (1 - confidence))
    latent_variance = torch.where(
        observed[:, -1, :6], torch.zeros_like(latent_variance), latent_variance
    )
    response = 1 + 0.25 * model.state_response.tanh()
    propagated_variance = response.square() * latent_variance
    future_variance = future_scale.square()
    total_variance = future_variance + propagated_variance
    scale = total_variance.sqrt()

    trace = {
        "latent_state_variance": latent_variance,
        "affine_response": response,
        "future_variance": future_variance,
        "propagated_state_variance": propagated_variance,
        "total_variance": total_variance,
    }
    return location, scale, source_sum, source_mass, evidence_sum, memory, trace
