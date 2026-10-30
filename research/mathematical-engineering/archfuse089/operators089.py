"""ARCHFUSE089 — no-fit architectural memory operators.

Orientation is the actual LANGMEM086 TAC memory [value,key], rather than
the [key,value] convention in several linear-attention papers.
This is NOT a trained model or language capability evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil, log2
import torch


def channel_delta(state: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  retention: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
    """KDA-style channel decay and error-correcting rank-one update.

    M' = M diag(alpha) + beta [v - M diag(alpha) k] k^T.
    """
    if state.ndim < 2 or state.shape[-1] != key.shape[-1] or state.shape[-2] != value.shape[-1]:
        raise ValueError("TAC value-by-key shape")
    if (not torch.isfinite(state).all() or not torch.isfinite(key).all()
            or not torch.isfinite(value).all() or not torch.isfinite(retention).all()
            or not torch.isfinite(strength).all()):
        raise ValueError("non-finite input")
    if ((retention < 0).any() or (retention > 1).any() or
            (strength < 0).any() or (strength > 1).any()):
        raise ValueError("gates must be in [0,1]")
    aged = state * retention.unsqueeze(-2)
    error = value - (aged @ key.unsqueeze(-1)).squeeze(-1)
    return aged + strength[..., None, None] * error.unsqueeze(-1) * key.unsqueeze(-2)


def scalar_delta(state, key, value, decay, strength):
    """Gated-Delta scalar-retention control in native TAC orientation."""
    return channel_delta(state, key, value, torch.ones_like(key) * decay, strength)


def rotation_ema(state, write, decay, angle):
    """Mamba/Megalodon-like rotary state step represented in real pairs."""
    if state.shape != write.shape or state.shape[-1] != 2:
        raise ValueError("paired real slow-state required")
    if (decay < 0).any() or (decay > 1).any():
        raise ValueError("unbounded decay")
    c, s = torch.cos(angle), torch.sin(angle)
    a = decay * (c * state[..., 0] - s * state[..., 1])
    b = decay * (s * state[..., 0] + c * state[..., 1])
    return torch.stack((a, b), dim=-1) + write


def linear_attention_step(state, key, value):
    """Lightning/mLSTM ungated associative outer-product step."""
    if state.shape[-2:] != (value.shape[-1], key.shape[-1]):
        raise ValueError("matrix dimensions")
    return state + value.unsqueeze(-1) * key.unsqueeze(-2)


def mlstm_step(state, normalizer, key, value, forget, input_gate):
    """xLSTM/mLSTM gated accumulator and independent normalizer."""
    new_state = forget * state + input_gate * value.unsqueeze(-1) * key.unsqueeze(-2)
    new_normalizer = forget * normalizer + input_gate * key
    return new_state, new_normalizer


def mlstm_read(state, normalizer, query):
    denominator = torch.clamp((normalizer * query).sum(-1).abs(), min=1)
    return (state @ query.unsqueeze(-1)).squeeze(-1) / denominator.unsqueeze(-1)


def titans_surprise_step(memory, momentum, key, value, learning_rate,
                         momentum_rate, forget):
    """Test-time local SGD on 1/2||M k-v||², not a neural model-weight fit."""
    if not (0 <= learning_rate <= 1 and 0 <= momentum_rate < 1 and 0 <= forget <= 1):
        raise ValueError("bounded learning, momentum and forget gates")
    surprise = value - memory @ key
    update = momentum_rate * momentum + learning_rate * surprise.unsqueeze(-1) * key.unsqueeze(-2)
    return (1 - forget) * memory + update, update


@dataclass(frozen=True)
class ExactRing:
    owner: str
    capacity: int
    bytes_seen: tuple[int, ...] = ()

    def append(self, owner: str, byte: int) -> "ExactRing":
        if owner != self.owner or not (0 <= byte < 256) or self.capacity < 1:
            raise ValueError("source owner, capacity, or byte violation")
        return ExactRing(self.owner, self.capacity, (self.bytes_seen + (byte,))[-self.capacity:])

    def continuations(self, context: tuple[int, ...]) -> tuple[int, ...]:
        """Query only observed history, before appending the unknown target."""
        n = len(context)
        if not n:
            raise ValueError("nonempty context")
        h = self.bytes_seen
        return tuple(h[i+n] for i in range(len(h)-n) if h[i:i+n] == context)


def selective_indices(prob_a: torch.Tensor, prob_b: torch.Tensor, budget: int):
    """Target-blind budgeted routing. NOT a learned or qualified verifier."""
    if prob_a.shape != prob_b.shape or prob_a.ndim != 2 or not (0 <= budget <= len(prob_a)):
        raise ValueError("shape or budget")
    for p in (prob_a, prob_b):
        if not torch.isfinite(p).all() or (p < 0).any() or not torch.allclose(
                p.sum(-1), torch.ones_like(p[:, 0]), atol=1e-5):
            raise ValueError("probability simplex")
    disagreement = (prob_a - prob_b).abs().sum(-1)
    uncertainty = 1 - .5 * (prob_a.max(-1).values + prob_b.max(-1).values)
    score = (disagreement + uncertainty).detach().tolist()
    return tuple(sorted(range(len(score)), key=lambda i: (-score[i], i))[:budget])


def bounded_attention(keys, values, query, window: int):
    """Exact softmax retrieval over a capped, already-seen memory window."""
    if window < 1 or len(keys) != len(values) or len(keys) < 1:
        raise ValueError("window/source")
    k, v = keys[-window:], values[-window:]
    scores = (k @ query) / (query.numel() ** .5)
    return (scores.softmax(0).unsqueeze(-1) * v).sum(0)


def ternary_absmean(weights):
    """BitNet-style ternary quantizer; not a bitnet.cpp inference kernel."""
    if weights.numel() == 0 or not torch.isfinite(weights).all():
        raise ValueError("empty or nonfinite weights")
    scale = weights.detach().abs().mean().clamp_min(1e-12)
    codes = torch.clamp(torch.round(weights / scale), -1, 1).to(torch.int8)
    return codes, scale


def ternary_ideal_min_bytes(n: int):
    if n < 0:
        raise ValueError("negative weight count")
    return ceil(n * log2(3) / 8)
