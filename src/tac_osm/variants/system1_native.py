"""ME075 proposed native one-step fast-state world model — no model trained here.

Physical example contract: available current observation x_t, requested/observed
action a_t, and elapsed observation time since the previous *known* tick.
Predict x_(t+1) and an exploratory event logit. No slow matrix, associative
matrix, imagined trajectories, action authorization or latency claim.

For dimension=14 this model has 100,541 actually trainable/deployed parameters
without unused padding. All new fits require the repository live 95k–105k
guard, authentic data, frozen pre-registered targets and full Git replay.
"""
from __future__ import annotations

from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F


class ImmediateForecast(NamedTuple):
    mean_next: torch.Tensor
    scale_next: torch.Tensor
    event_logit: torch.Tensor
    fast: torch.Tensor


class System1FastModel(nn.Module):
    """Bounded fast recurrence plus measured one-step residual readout."""

    architecture = "system1_native_fast"

    def __init__(self, dimension: int = 14, state_scale=None):
        super().__init__()
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError("positive source-registered state dimension required")
        self.dimension = dimension
        width = 96
        self.encoder = nn.Sequential(
            nn.Linear(2 * dimension + 1, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.Tanh(),
        )
        self.fast_candidate_gate = nn.Linear(2 * width, 2 * width)
        self.fast_residual_gate = nn.Linear(2 * width, 2 * width)
        self.readout = nn.Linear(2 * width, 64)
        self.one_step_heads = nn.Linear(64, 2 * dimension + 1)
        scale = torch.ones(dimension, dtype=torch.float32) if state_scale is None else torch.as_tensor(
            state_scale, dtype=torch.float32
        )
        if scale.shape != (dimension,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("positive finite factual per-feature source scale required")
        self.register_buffer("state_scale", scale.detach().clone())

    def initial(self, batch_size: int, *, device=None, dtype=None) -> torch.Tensor:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("positive batch size required")
        return torch.zeros(batch_size, 96, dtype=dtype or self.state_scale.dtype,
                           device=device or self.state_scale.device)

    def forward(self, current: torch.Tensor, action: torch.Tensor,
                elapsed_seconds: torch.Tensor, fast: torch.Tensor | None = None) -> ImmediateForecast:
        if current.ndim != 2 or current.shape[1] != self.dimension:
            raise ValueError("current state must be [batch, source_dimension]")
        if action.shape != current.shape or current.dtype != action.dtype:
            raise ValueError("action must be same shape and dtype as current state")
        if elapsed_seconds.shape != (current.shape[0],):
            raise ValueError("elapsed_seconds must be [batch]")
        if current.dtype not in (torch.float32, torch.float64):
            raise ValueError("floating-point inputs required")
        if not (torch.isfinite(current).all() and torch.isfinite(action).all()
                and torch.isfinite(elapsed_seconds).all()
                and (elapsed_seconds >= 0).all()):
            raise ValueError("only finite, available-at-time observations accepted")
        if fast is None:
            fast = self.initial(len(current), device=current.device, dtype=current.dtype)
        if fast.shape != (len(current), 96) or not torch.isfinite(fast).all():
            raise ValueError("entity-separated bounded fast state required")
        features = torch.cat((current, action, torch.log1p(elapsed_seconds)[:, None]), dim=-1)
        observed = self.encoder(features)
        pair = torch.cat((observed, fast), dim=-1)
        candidate, admission = self.fast_candidate_gate(pair).chunk(2, dim=-1)
        residual, residual_gate = self.fast_residual_gate(pair).chunk(2, dim=-1)
        # True bounded recurrent state; not a global long-context addressing claim.
        updated = torch.tanh(torch.lerp(fast, candidate.tanh(), admission.sigmoid())
                             + 0.05 * residual_gate.sigmoid() * residual.tanh())
        decoded = F.silu(self.readout(torch.cat((updated, observed), dim=-1)))
        heads = self.one_step_heads(decoded)
        d = self.dimension
        # Residual prediction anchored to the last *observed* current state.
        mean_next = current + self.state_scale * (0.1 * heads[:, :d].tanh())
        scale_next = self.state_scale * (F.softplus(heads[:, d:2*d]) + 1e-4)
        return ImmediateForecast(mean_next, scale_next, heads[:, -1], updated)


def physical_one_step_nll(prediction: ImmediateForecast, actual_next: torch.Tensor,
                          valid: torch.Tensor) -> torch.Tensor:
    """Proper Laplace NLL for actual next real state; no fabricated risk labels."""
    if actual_next.shape != prediction.mean_next.shape or valid.shape != actual_next.shape[:1]:
        raise ValueError("registered real next-state and valid-mask dimensions required")
    if not torch.isfinite(actual_next).all() or valid.dtype != torch.bool or not valid.any():
        raise ValueError("finite authentic future targets and valid rows required")
    scale = prediction.scale_next.clamp_min(1e-5)
    ll = (actual_next - prediction.mean_next).abs() / scale + (2 * scale).log()
    return ll.mean(dim=-1)[valid].mean()
