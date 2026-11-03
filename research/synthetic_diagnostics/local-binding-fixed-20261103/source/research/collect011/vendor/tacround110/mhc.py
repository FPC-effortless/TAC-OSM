"""Small, UNTRAINED PyTorch mHC residual wrapper for TAC110 experiments.

Source: Xie et al., DeepSeek-AI, *mHC: Manifold-Constrained Hyper-Connections*,
https://arxiv.org/html/2512.24880v2, equations 7--9. For each token, flattened
S x D streams generate sigmoid pre gates, 2*sigmoid post gates, and a positive
Sinkhorn residual mixer. No associative-memory state is expanded or mixed.

Engineering choices relative to the paper: log-domain Sinkhorn, an unweighted
RMS normalization with epsilon=1e-6, near-identity residual bias (diagonal 4),
pre bias logit(1/S), post bias zero, and final arithmetic stream mean. This is
the original current-sublayer coefficient schedule. It does NOT implement
DeepSeek-V4.1 Single-Pass mHC (predecessor-produced pre coefficients), Mega-mHC,
custom CUDA kernels, recomputation, or communication/compute overlap. See the
official V4.1 model card and SGLang implementation account:
https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization

Finite Sinkhorn iterations only APPROXIMATE double stochasticity. The final
normalization is over rows; columns need not sum exactly to one. Ideal-matrix
nonexpansiveness is not a guarantee for this whole content-dependent nonlinear
wrapper or its Jacobian. No quality or runtime advantage has been established.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def sinkhorn_log(logits: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """Column-then-row normalization of exp(logits), without exp overflow.

    The fixed iteration count is part of the graph: this does not test for
    convergence or claim an exact projection. FP16/BF16 coefficients use FP32;
    FP64 stays FP64. Extreme finite logits can underflow entries to zero.
    """
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError("logits must be a floating tensor")
    if logits.ndim < 2 or logits.shape[-1] != logits.shape[-2] or logits.shape[-1] < 1:
        raise ValueError("logits must end in a nonempty square matrix")
    z = logits if logits.dtype == torch.float64 else logits.float()
    for _ in range(iterations):
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)
        z = z - torch.logsumexp(z, dim=-1, keepdim=True)
    return z.exp()


@dataclass(frozen=True)
class MHCContext:
    """Differentiable, ephemeral coefficients from one prepare() invocation.

    This is not persistent per-sequence memory. Keep it attached through the
    corresponding finish() for gradients; do not reuse across modules/inputs.
    """

    pre: torch.Tensor                 # [B,T,S]
    post: torch.Tensor                # [B,T,S]
    mixing: torch.Tensor              # [B,T,output_stream,input_stream]
    stream_shape: tuple[int, ...]
    iterations: int
    module_identity: int

    def constraint_errors(self) -> dict[str, torch.Tensor]:
        """Observed max absolute row/column sum errors (no detached values)."""
        return {
            "row_max_abs": (self.mixing.sum(-1) - 1).abs().amax(),
            "column_max_abs": (self.mixing.sum(-2) - 1).abs().amax(),
        }


class MHCResidual(nn.Module):
    """Token-local residual-stream wrapper; no sequence scan or mutable cache.

    x -> initialize_streams -> h
    branch_input, context = prepare(h)
    delta = branch(branch_input)       # branch returns only its residual delta
    h = finish(h, delta, context)
    output = reduce(h)

    Apply independent wrappers around TAC's memory and FFN branches. Preserve
    each branch's own normalization, residual scaling, and associative state
    protocol. The mixer never sees associative matrices or another token.

    Parameter count: S*D*(S*S+2*S) + (S*S+2*S) + 3. At S=4 this is 96*D+27
    per wrapper, or 192*D+54 for a memory/FFN pair. No padding parameters.

    Mean readout cancels an exactly column-stochastic final residual mixer:
    mean(B @ H) = mean(H). Place a later stream-dependent operation/readout
    when useful activity of a final mixer's parameters is required; tiny
    gradients from finite-iteration column error do not establish that activity.
    """

    def __init__(self, d_model: int, streams: int = 4,
                 sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__()
        for name, value in (("d_model", d_model), ("streams", streams),
                            ("sinkhorn_iters", sinkhorn_iters)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.d_model = d_model
        self.streams = streams
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = float(eps)
        coefficients = streams * streams + 2 * streams
        self.projection = nn.Linear(streams * d_model, coefficients, bias=False)
        self.bias = nn.Parameter(torch.empty(coefficients))
        self.alpha = nn.Parameter(torch.full((3,), 0.01))
        self.reset_parameters()

    def reset_parameters(self):
        """Small nonzero dynamic coefficients, near-identity static transport."""
        nn.init.normal_(self.projection.weight, std=(self.streams * self.d_model) ** -0.5)
        with torch.no_grad():
            self.alpha.fill_(0.01)
            self.bias.zero_()
            # For S=1, sigmoid(0)=.5 is the paper's legal, nonsingular gate.
            if self.streams > 1:
                self.bias[:self.streams].fill_(-math.log(self.streams - 1))
            self.bias[2 * self.streams:].view(self.streams, self.streams).diagonal().fill_(4.0)

    def _validate(self, tensor: torch.Tensor, expanded: bool):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise TypeError("residual activations must be a floating tensor")
        expected = (self.streams, self.d_model) if expanded else (self.d_model,)
        rank = 4 if expanded else 3
        if tensor.ndim != rank or tuple(tensor.shape[-len(expected):]) != expected:
            raise ValueError(f"expected [B,T,{','.join(map(str, expected))}] activations")
        if tensor.shape[0] < 1 or tensor.shape[1] < 1:
            raise ValueError("batch and sequence dimensions must be nonempty")
        if tensor.device != self.bias.device or tensor.dtype != self.bias.dtype:
            raise ValueError("activations must match module parameter device and dtype")

    def initialize_streams(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,D] -> [B,T,S,D] read-only expanded view; never mutated here."""
        self._validate(x, expanded=False)
        return x.unsqueeze(-2).expand(*x.shape[:2], self.streams, self.d_model)

    def prepare(self, streams: torch.Tensor) -> tuple[torch.Tensor, MHCContext]:
        """Compute current-token coefficients and aggregate branch input."""
        self._validate(streams, expanded=True)
        work_dtype = torch.float64 if streams.dtype == torch.float64 else torch.float32
        flat = streams.flatten(-2).to(work_dtype)
        normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        dynamic = F.linear(normalized, self.projection.weight.to(work_dtype)).to(work_dtype)
        s = self.streams
        bias, alpha = self.bias.to(work_dtype), self.alpha.to(work_dtype)
        pre = torch.sigmoid(alpha[0] * dynamic[..., :s] + bias[:s])
        post = 2 * torch.sigmoid(alpha[1] * dynamic[..., s:2*s] + bias[s:2*s])
        residual_logits = alpha[2] * dynamic[..., 2*s:] + bias[2*s:]
        mixing = sinkhorn_log(residual_logits.view(*streams.shape[:2], s, s),
                              self.sinkhorn_iters)
        branch_input = (pre.unsqueeze(-1) * streams.to(work_dtype)).sum(-2)
        context = MHCContext(pre, post, mixing, tuple(streams.shape),
                             self.sinkhorn_iters, id(self))
        return branch_input.to(streams.dtype), context

    def finish(self, streams: torch.Tensor, delta: torch.Tensor,
               context: MHCContext) -> torch.Tensor:
        """h' = mixing @ h + post[..., :, None] * branch_delta[..., None, :]."""
        self._validate(streams, expanded=True)
        self._validate(delta, expanded=False)
        if tuple(delta.shape[:2]) != tuple(streams.shape[:2]):
            raise ValueError("delta batch/sequence dimensions do not match streams")
        if not isinstance(context, MHCContext) or context.module_identity != id(self):
            raise ValueError("context must come from this module's prepare()")
        if context.stream_shape != tuple(streams.shape):
            raise ValueError("context shape does not match streams")
        expected = (*streams.shape[:2], self.streams)
        if context.pre.shape != expected or context.post.shape != expected or context.mixing.shape != (*expected, self.streams):
            raise ValueError("context coefficients have invalid shapes")
        work_dtype = context.mixing.dtype
        mixed = torch.matmul(context.mixing, streams.to(work_dtype))
        output = mixed + context.post.unsqueeze(-1) * delta.to(work_dtype).unsqueeze(-2)
        return output.to(streams.dtype)

    def reduce(self, streams: torch.Tensor) -> torch.Tensor:
        """Return the unlearned arithmetic stream mean, [B,T,D]."""
        self._validate(streams, expanded=True)
        return streams.mean(-2)
