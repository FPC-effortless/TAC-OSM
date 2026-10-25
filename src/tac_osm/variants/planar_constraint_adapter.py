"""Optional, pure Torch, DIFFERENTIABLE civilian planar single-integrator CBF layer.

This does not modify TAC-OSM's native learned fast/slow/associative state;
it accepts any separate continuous planar command proposal and returns a
conditional halfspace projection. No actuator box is implemented here:
unbounded controls are physically unrealistic and this module MUST NOT be
deployed as a safety controller or used to claim real physical feasibility.

The QP is min_u (1/2)||u-u_des||² subject to
  2r·u >= 2r·w_hat - gamma*(||r||²-R²) + 2||r||*epsilon.
Its analytic solution is u = u_des + max(0,b-n·u_des)*n/||n||².
This is a closed-form differentiable QP in fixed active-set regions.
It does NOT implement a generic OptNet/KKT solver or establish implicit
gradient regularity at the switching boundary.

For known constant velocities within a single sample and a known bound
||w_true-w_hat|| <= epsilon, exact planar single-integrator dynamics imply
conditional barrier preservation for gamma*dt<=1, assuming initial h>=0.
No stochastic confidence bound, accelerated dynamics or independent safety
evaluation is inferred.
"""
from __future__ import annotations

import torch


def project_planar_cbf(
    position: torch.Tensor,
    obstacle: torch.Tensor,
    desired_velocity: torch.Tensor,
    obstacle_velocity_estimate: torch.Tensor,
    *,
    radius: float,
    gamma: float,
    dt: float,
    obstacle_velocity_error_bound: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (projected velocity [B,2], >=0 CBF margin [B]).

    Fail closed via exception if any row is initially unsafe, singular,
    nonfinite, or fails the checked inequality. No actuator bounds here.
    """
    tensors = (position, obstacle, desired_velocity, obstacle_velocity_estimate)
    if not all(isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[-1] == 2
               for x in tensors):
        raise ValueError("inputs must be rank-2 [B,2] tensors")
    if len({tuple(x.shape) for x in tensors}) != 1:
        raise ValueError("batch and coordinate dimensions must match")
    if len({x.dtype for x in tensors}) != 1 or len({x.device for x in tensors}) != 1:
        raise ValueError("all inputs must have equal dtype and device")
    if not all(torch.isfinite(x).all().item() for x in tensors):
        raise ValueError("nonfinite observation or desired command")
    if (not all(isinstance(v,(float,int)) for v in
            (radius,gamma,dt,obstacle_velocity_error_bound)) or
        not (float(radius)>0 and float(gamma)>0 and float(dt)>0) or
        float(gamma)*float(dt)>1 or float(obstacle_velocity_error_bound)<0):
        raise ValueError("invalid radius/gamma/step/robustness contract")

    r=position-obstacle
    norm2=(r*r).sum(-1)
    h=norm2-radius*radius
    if torch.any(h<0).item() or torch.any(norm2<=0).item():
        raise ValueError("pre-existing unsafe or singular state: no safe action returned")

    n=2*r
    floor=(n*obstacle_velocity_estimate).sum(-1) - gamma*h + (
        2*torch.sqrt(norm2)*obstacle_velocity_error_bound)
    gap=floor-(n*desired_velocity).sum(-1)
    multiplier=torch.clamp_min(gap,0)/((n*n).sum(-1))
    action=desired_velocity+multiplier.unsqueeze(-1)*n
    margin=(n*action).sum(-1)-floor
    if not torch.isfinite(action).all().item() or torch.any(margin < -1e-9).item():
        raise ValueError("numerical failure: do not execute candidate")
    return action, margin
