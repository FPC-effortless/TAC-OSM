"""Prospective zero-gain adjoint repair. No fit or execution on import.

Forward values retain the exact frozen zero-gain rule. Backward follows the
smooth candidate update, including at zero. A hard router's straight-through
gain must receive that boundary derivative; an observation mask still multiplies
the gain by zero and therefore blocks learning through a masked observation.
This is a declared surrogate adjoint, not the derivative of hard argmax.
"""
import torch


class _ExactFreeze(torch.autograd.Function):
    @staticmethod
    def forward(ctx, candidate, previous, eligible):
        return torch.where(eligible, candidate, previous)

    @staticmethod
    def backward(ctx, gradient):
        # Candidate already contains every dependence on previous state.
        return gradient, None, None


def write_step(memory, precision, key, value, gain, *, kind="preconditioned", floor=1e-8):
    """Same source equations/forward order; repair only zero-gain backward.

    Shapes: [...,V,K], [...,K,K] or None, [...,K], [...,V], [...].
    Finite valid states and source gain/domain assumptions remain required.
    """
    residual = value - (memory @ key.unsqueeze(-1)).squeeze(-1)
    if kind == "delta":
        direction = key
        candidate_p = None
    else:
        z = (precision @ key.unsqueeze(-1)).squeeze(-1)
        leverage = (key*z).sum(-1)
        direction = z / leverage.clamp_min(floor).unsqueeze(-1)
        candidate_p = precision - gain[..., None, None] * z.unsqueeze(-1)*z.unsqueeze(-2)/(1+gain*leverage)[..., None, None]
        candidate_p = (candidate_p+candidate_p.transpose(-1, -2))*.5
    candidate = memory + gain[..., None, None]*residual.unsqueeze(-1)*direction.unsqueeze(-2)
    eligible = gain != 0
    updated = _ExactFreeze.apply(candidate, memory, eligible[..., None, None])
    if precision is not None:
        candidate_p = _ExactFreeze.apply(candidate_p, precision, eligible[..., None, None])
    return updated, candidate_p
