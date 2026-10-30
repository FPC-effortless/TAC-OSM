"""TAO: experimental, architecture-independent Muon/AdamW optimizer.

Numerical engineering prototype; no TAC-OSM model-quality claims.
Routing is deny-by-default: only explicitly named 2D or 4D tensors use NS5.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.optim import Optimizer


@torch.no_grad()
def ns5(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate rectangular polar factor by quintic Newton--Schulz.

    Float32 computation by default. Specialized BF16 GPU kernels are future work.
    This is an approximation, not an exact SVD or strict spectral constraint.
    """
    if matrix.ndim != 2:
        raise ValueError("NS5 requires a 2D matrix")
    if steps < 1:
        raise ValueError("steps must be positive")
    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (torch.linalg.vector_norm(x) + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transposed else x


class TAO(Optimizer):
    """Safe mixed optimizer; routing is explicit via param-group ``kind``.

    Groups: kind='adamw' (default) or kind='orthogonal'. Orthogonal group
    supports mode='muon', 'adamuon', 'normuon', or 'pre_ns'. The AdaMuon mode
    includes sign-stabilized NS; pre_ns is a new hypothesis, not AdaMuon.

    Both pathways use decoupled WD and a shared scalar LR by default. State
    is serialized by PyTorch's native Optimizer.state_dict mechanism. Dense
    gradients only; nonfinite gradients fail the *whole* step before updating.
    """

    def __init__(
        self, params: Iterable, *, lr: float = 1e-3, weight_decay: float = 0.01,
        mode: str = "muon", momentum: float = 0.95, beta2: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
        ns_steps: int = 5, rms: float = 0.2,
    ) -> None:
        if lr <= 0 or weight_decay < 0 or not 0 <= momentum < 1 or not 0 <= beta2 < 1:
            raise ValueError("invalid optimizer hyperparameters")
        if len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("invalid AdamW betas")
        if eps <= 0 or ns_steps < 1 or rms <= 0:
            raise ValueError("invalid numeric configuration")
        if mode not in {"muon", "adamuon", "normuon", "pre_ns"}:
            raise ValueError("unknown optimizer mode")
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, kind="adamw",
                                       mode=mode, momentum=momentum, beta2=beta2,
                                       betas=betas, eps=eps, ns_steps=ns_steps, rms=rms))
        for group in self.param_groups:
            if group["kind"] not in {"adamw", "orthogonal"}:
                raise ValueError("kind must be adamw or orthogonal")
            if group["mode"] not in {"muon", "adamuon", "normuon", "pre_ns"}:
                raise ValueError("unknown group mode")
            for p in group["params"]:
                if group["kind"] == "orthogonal" and p.ndim not in (2, 4):
                    raise ValueError("orthogonal groups require 2D/4D tensors")
                if not p.is_floating_point():
                    raise ValueError("optimizer requires floating-point weights")

    @torch.no_grad()
    def step(self, closure=None):
        with torch.enable_grad():
            loss = closure() if closure is not None else None
        # Validate before mutating any weight. This prevents partial steps.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.is_sparse or not bool(torch.isfinite(p.grad).all()):
                        raise ValueError("sparse or nonfinite gradient: no step taken")
        for group in self.param_groups:
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            if lr < 0 or wd < 0 or eps <= 0:
                raise ValueError("invalid group configuration")
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach().float()
                state = self.state[p]
                state["step"] = state.get("step", 0) + 1
                t = state["step"]
                if group["kind"] == "adamw":
                    b1, b2 = group["betas"]
                    m = state.setdefault("exp_avg", torch.zeros_like(p, dtype=torch.float32))
                    v = state.setdefault("exp_avg_sq", torch.zeros_like(p, dtype=torch.float32))
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    d = (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
                else:
                    mu, beta2, mode = group["momentum"], group["beta2"], group["mode"]
                    m = state.setdefault("momentum_buffer", torch.zeros_like(p, dtype=torch.float32))
                    m.mul_(mu).add_(g)
                    nesterov = g + mu * m
                    original_shape = nesterov.shape
                    nesterov = nesterov.reshape(nesterov.shape[0], -1)
                    if mode == "pre_ns":
                        v = state.setdefault("second_moment", torch.zeros_like(nesterov))
                        v.mul_(beta2).addcmul_(nesterov, nesterov, value=1 - beta2)
                        nesterov = nesterov / ((v / (1 - beta2 ** t)).sqrt() + eps)
                    if mode == "adamuon":
                        nesterov = nesterov.sign()
                    u = ns5(nesterov, group["ns_steps"], eps)
                    if mode == "adamuon":
                        v = state.setdefault("second_moment", torch.zeros_like(u))
                        v.mul_(beta2).addcmul_(u, u, value=1 - beta2)
                        u = u / (v.sqrt() + eps)
                    elif mode == "normuon":
                        v = state.setdefault("second_moment", torch.zeros((u.shape[0], 1), device=u.device, dtype=torch.float32))
                        v.mul_(beta2).add_(u.square().mean(dim=1, keepdim=True), alpha=1 - beta2)
                        u = u / (v.sqrt() + eps)
                    norm = torch.linalg.vector_norm(u)
                    if not bool(torch.isfinite(norm)) or float(norm) <= eps:
                        if bool(torch.count_nonzero(g)):
                            raise FloatingPointError("orthogonal update collapsed or became nonfinite")
                        d = torch.zeros_like(p, dtype=torch.float32)
                    else:
                        d = (u * (group["rms"] * math.sqrt(p.numel()) / norm)).reshape(original_shape)
                if not bool(torch.isfinite(d).all()):
                    raise FloatingPointError("nonfinite preconditioned update")
                p.mul_(1 - lr * wd).add_(d.to(dtype=p.dtype), alpha=-lr)
        return loss


def build_tao(model: torch.nn.Module, *, orthogonal_names: Iterable[str] = (), **kwargs) -> TAO:
    """Create explicitly routed mixed TAO: all unlisted tensors use AdamW.

    Exact full parameter names are required. Shared/tied weights cannot be
    listed twice. A 4D convolutional kernel is flattened into [out, in*k*k].
    Explicit routing is intentional, not inferred from tensor dimension.
    """
    selected = set(orthogonal_names)
    named = dict(model.named_parameters())
    missing = selected - named.keys()
    if missing:
        raise ValueError(f"unknown orthogonal parameter names: {sorted(missing)}")
    if any(named[name].ndim not in (2, 4) for name in selected):
        raise ValueError("only 2D/4D matrices are orthogonal-eligible")
    orthogonal = [p for n, p in named.items() if n in selected and p.requires_grad]
    standard = [p for n, p in named.items() if n not in selected and p.requires_grad]
    if not orthogonal and not standard:
        raise ValueError("no trainable parameters")
    groups = []
    if orthogonal:
        groups.append({"params": orthogonal, "kind": "orthogonal"})
    if standard:
        groups.append({"params": standard, "kind": "adamw"})
    return TAO(groups, **kwargs)
