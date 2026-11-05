"""CSL-001: no-fit, depth-sliceable, role-addressed multiscale state research candidate.

No training, calibrated confidence, tiny-device runtime or capability result is
implied by constructing this model. Semantic state is owned per layer and per
stream; callers must not share it across unrelated entities.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CSLConfig:
    vocab_size: int = 256
    width: int = 32
    layers: int = 8
    ff_width: int = 80
    role_count: int = 4
    state_width: int = 16
    conv_kernel: int = 3
    chunk_size: int = 64
    tool_classes: int = 16
    field_classes: int = 16
    rungs: Tuple[int, ...] = (2, 4, 8)

    def __post_init__(self):
        if not self.rungs or self.rungs[-1] != self.layers or tuple(sorted(set(self.rungs))) != self.rungs:
            raise ValueError("rungs must be sorted, unique, and include the final depth")
        if min(self.rungs) < 1 or self.conv_kernel < 2 or self.chunk_size > 64 or self.chunk_size < 1:
            raise ValueError("invalid causal kernel, rung, or stable-scan chunk length")
        if min(self.vocab_size, self.width, self.ff_width, self.role_count, self.state_width) < 1:
            raise ValueError("dimensions must be positive")


def csl20_config() -> CSLConfig:
    """20-block, 2/8/20-rung no-fit design within the existing 100k band.

    This is a separate candidate configuration; it does NOT inherit 8L
    checkpoints or establish a trained 20L deployment.
    """
    return CSLConfig(width=20, layers=20, ff_width=56, state_width=8,
                     rungs=(2, 8, 20))


@dataclass
class LayerState:
    fast: Tensor               # [B, R, E], FP32
    slow: Tensor               # [B, R, E], FP32
    conv_tail: Tensor          # [B, K-1, D], input-normalized hidden dtype

    def detached(self) -> "LayerState":
        return LayerState(self.fast.detach(), self.slow.detach(), self.conv_tail.detach())


@dataclass
class CSLState:
    layers: Tuple[LayerState, ...]
    stream_ids: Optional[Tuple[str, ...]] = None

    def detached(self) -> "CSLState":
        return CSLState(tuple(x.detached() for x in self.layers), self.stream_ids)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Compute the norm in FP32 under mixed precision.
        inv = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (x.float() * inv).to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


def _bounded_scan(z: Tensor, initial: Tensor, decay: Tensor, chunk_size: int) -> Tuple[Tensor, Tensor]:
    """Inclusive EMA, parallel in each <=64-token chunk, serial between chunks.

    z: [B,T,R,E]; initial: [B,R,E]; decay: [R,E]. Scanning is in
    FP32 to avoid activation overflow on mixed-precision hardware.
    """
    if z.ndim != 4 or initial.shape != z.shape[:1] + z.shape[2:]:
        raise ValueError("invalid EMA input/state shape")
    if decay.shape != z.shape[2:]:
        raise ValueError("invalid decay shape")
    if z.shape[1] == 0:
        raise ValueError("zero-length sequence unsupported")
    outputs = []
    current = initial.float()
    a = decay.float().clamp(0.84, 0.995).unsqueeze(0).unsqueeze(0)
    for start in range(0, z.shape[1], chunk_size):
        c = z[:, start:start + chunk_size].float()
        t = torch.arange(c.shape[1], device=c.device, dtype=torch.float32).view(1, -1, 1, 1)
        p = a.pow(t)
        segment = (1.0 - a) * p * torch.cumsum(c / p, dim=1)
        segment = segment + a * p * current.unsqueeze(1)
        outputs.append(segment)
        current = segment[:, -1]
    return torch.cat(outputs, dim=1), current


class RoleStateBlock(nn.Module):
    """Causal depthwise convolution + multi-timescale role-state + gated FFN."""
    def __init__(self, cfg: CSLConfig):
        super().__init__()
        self.cfg = cfg
        d, e, r = cfg.width, cfg.state_width, cfg.role_count
        self.pre = RMSNorm(d)
        self.ffnorm = RMSNorm(d)
        self.conv = nn.Conv1d(d, d, cfg.conv_kernel, groups=d, bias=False)
        self.key = nn.Linear(d, e, bias=False)
        self.value = nn.Linear(d, e, bias=False)
        self.roles = nn.Parameter(torch.randn(r, e) / (e ** 0.5))
        self.read_fast = nn.Linear(e, d, bias=False)
        self.read_slow = nn.Linear(e, d, bias=False)
        self.read_gate = nn.Linear(d, d, bias=False)
        self.fast_decay_logit = nn.Parameter(torch.full((r, e), log(0.9 / 0.1)))
        self.slow_decay_logit = nn.Parameter(torch.full((r, e), log(0.98 / 0.02)))
        self.up = nn.Linear(d, 2 * cfg.ff_width, bias=False)
        self.down = nn.Linear(cfg.ff_width, d, bias=False)
        self.residual_gain = nn.Parameter(torch.tensor(0.1))
        self.ff_gain = nn.Parameter(torch.tensor(0.1))

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> LayerState:
        r, e = self.cfg.role_count, self.cfg.state_width
        return LayerState(
            torch.zeros((batch, r, e), device=device, dtype=torch.float32),
            torch.zeros((batch, r, e), device=device, dtype=torch.float32),
            torch.zeros((batch, self.cfg.conv_kernel - 1, self.cfg.width), device=device, dtype=dtype),
        )

    def forward(self, x: Tensor, state: LayerState) -> Tuple[Tensor, LayerState]:
        b, t, d = x.shape
        if state.fast.shape != (b, self.cfg.role_count, self.cfg.state_width):
            raise ValueError("state batch/role dimensions do not match input")
        if state.conv_tail.shape != (b, self.cfg.conv_kernel - 1, d):
            raise ValueError("invalid convolution-state shape")
        h = self.pre(x)
        with_tail = torch.cat((state.conv_tail, h), dim=1)
        local = self.conv(with_tail.transpose(1, 2)).transpose(1, 2)
        next_tail = with_tail[:, -(self.cfg.conv_kernel - 1):]
        # Soft role addressing; structured head is explicitly ablatable.
        q = self.key(h)
        a = F.softmax(torch.einsum('bte,re->btr', q.float(), self.roles.float()) / (self.cfg.state_width ** 0.5), dim=-1)
        v = torch.tanh(self.value(h)).float()
        writes = a.unsqueeze(-1) * v.unsqueeze(-2)
        fast, fast_end = _bounded_scan(writes, state.fast, torch.sigmoid(self.fast_decay_logit), self.cfg.chunk_size)
        slow, slow_end = _bounded_scan(writes, state.slow, torch.sigmoid(self.slow_decay_logit), self.cfg.chunk_size)
        rf = torch.einsum('btr,btre->bte', a, fast).to(h.dtype)
        rs = torch.einsum('btr,btre->bte', a, slow).to(h.dtype)
        gate = torch.sigmoid(self.read_gate(h))
        memory = gate * self.read_fast(rf) + (1.0 - gate) * self.read_slow(rs)
        x = x + self.residual_gain * (local + memory)
        u, g = self.up(self.ffnorm(x)).chunk(2, dim=-1)
        x = x + self.ff_gain * self.down(F.silu(g) * u)
        return x, LayerState(fast_end, slow_end, next_tail)


class CSL001(nn.Module):
    """One set of weights with explicit 2/4/8-layer prefix exits."""
    def __init__(self, config: CSLConfig = CSLConfig()):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(RoleStateBlock(config) for _ in range(config.layers))
        self.final_norm = RMSNorm(config.width)
        self.tool_head = nn.Linear(config.width, config.tool_classes + 1)
        self.field_head = nn.Linear(config.width, config.field_classes)
        self.accept_head = nn.Linear(config.width, 1)

    def parameter_count(self) -> int:
        return sum(x.numel() for x in self.parameters())

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype,
                      stream_ids: Optional[Tuple[str, ...]] = None,
                      depth: Optional[int] = None) -> CSLState:
        if depth is None:
            depth = self.config.layers
        if depth not in self.config.rungs:
            raise ValueError("invalid depth for state allocation")
        return CSLState(tuple(block.initial_state(batch, device, dtype) for block in self.blocks[:depth]), stream_ids)

    def forward(
        self, tokens: Tensor, *, depth: Optional[int] = None,
        state: Optional[CSLState] = None, all_rungs: bool = False,
        stream_ids: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[Dict[int, Dict[str, Tensor]], CSLState]:
        if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[1] < 1:
            raise ValueError("tokens must be nonempty [B,T] torch.long")
        if depth is None:
            depth = self.config.layers
        if depth not in self.config.rungs:
            raise ValueError("only trained/reregistered rungs may be selected")
        if stream_ids is not None:
            if (len(stream_ids) != tokens.shape[0] or not all(isinstance(s, str) and s for s in stream_ids)
                or len(set(stream_ids)) != len(stream_ids)):
                raise ValueError("stream_ids require one unique, nonempty ID per batch row")
        h = self.embedding(tokens)
        if state is None:
            state = self.initial_state(tokens.shape[0], tokens.device, h.dtype, stream_ids, depth)
        elif stream_ids is None or state.stream_ids is None or stream_ids != state.stream_ids:
            raise ValueError("carried memory requires the original explicit stream_ids; reset on new entities")
        if len(state.layers) != depth:
            raise ValueError("carried state depth differs; reset/replay before switching rungs")
        new_states = list(state.layers)
        outputs: Dict[int, Dict[str, Tensor]] = {}
        for i in range(depth):
            h, new_states[i] = self.blocks[i](h, new_states[i])
            rung = i + 1
            if rung == depth or (all_rungs and rung in self.config.rungs):
                z = self.final_norm(h)
                outputs[rung] = {
                    "token_logits": F.linear(z, self.embedding.weight),
                    "tool_logits": self.tool_head(z[:, -1]),
                    "field_logits": self.field_head(z),
                    # Untrained acceptance logit != calibrated confidence.
                    "accept_logit": self.accept_head(z[:, -1]),
                }
        return outputs, CSLState(tuple(new_states), state.stream_ids)

    def nested_next_token_loss(self, tokens: Tensor, weights: Optional[Dict[int, float]] = None) -> Tensor:
        """All requested rung losses share one forward/backward graph, no separate model copies."""
        if tokens.shape[1] < 2:
            raise ValueError("next-token training needs at least two tokens")
        outs, _ = self.forward(tokens, all_rungs=True)
        weights = weights or {r: 1.0 for r in self.config.rungs}
        if set(weights) != set(self.config.rungs) or any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("weights must cover all trained rungs with nonnegative mass")
        return sum(
            (float(weights[r]) / sum(weights.values())) * F.cross_entropy(
                outs[r]["token_logits"][:, :-1].reshape(-1, self.config.vocab_size),
                tokens[:, 1:].reshape(-1),
            )
            for r in self.config.rungs
        )


    def supervised_task_loss(
        self, tokens: Tensor, tool_labels: Tensor, field_labels: Tensor,
        accept_labels: Tensor, rung_weights: Optional[Dict[int, float]] = None,
    ) -> Tensor:
        """Supervised *gold-label* loss; inputs are predecision tokens only.

        accept_labels are independently provided binary correctness/safety labels,
        NOT calibrated probabilities or model-generated reward. This function
        does no fitting, tool execution, or schema-constrained decoding.
        """
        b, t = tokens.shape
        if tool_labels.shape != (b,) or field_labels.shape != (b, t) or accept_labels.shape != (b,):
            raise ValueError("expected tool[B], field[B,T], accept[B] labels")
        if accept_labels.dtype not in (torch.float32, torch.float64):
            raise ValueError("accept labels must be floating-point binary ground truth")
        if not bool(((accept_labels == 0) | (accept_labels == 1)).all()):
            raise ValueError("accept labels must be binary")
        outs, _ = self.forward(tokens, all_rungs=True)
        weights = rung_weights or {r: 1.0 for r in self.config.rungs}
        if set(weights) != set(self.config.rungs) or any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("invalid rung weights")
        losses = []
        for r in self.config.rungs:
            out = outs[r]
            tool_loss = F.cross_entropy(out['tool_logits'], tool_labels)
            field_loss = F.cross_entropy(out['field_logits'].reshape(-1, self.config.field_classes), field_labels.reshape(-1))
            accept_loss = F.binary_cross_entropy_with_logits(out['accept_logit'].squeeze(-1), accept_labels.float())
            losses.append(float(weights[r]) / sum(weights.values()) * (tool_loss + field_loss + accept_loss))
        return sum(losses)


def theoretical_weight_bytes(count: int, bits_per_weight: float) -> int:
    """Bit-pack lower bound ONLY: excludes tables, scales, indices and runtime buffers."""
    if count < 0 or bits_per_weight <= 0:
        raise ValueError("invalid weight count/precision")
    from math import ceil
    return ceil(count * bits_per_weight / 8)
