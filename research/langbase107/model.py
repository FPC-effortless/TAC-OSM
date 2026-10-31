"""A small conventional causal Transformer control with a real KV cache.

This module contains no TAC associative memory.  It is an independently
implemented pre-norm decoder with RoPE, dense causal self-attention, GELU
feed-forward blocks, tied byte embeddings/readout, and a bounded inference
cache.  It is an untrained engineering control until a separately qualified
training protocol is allowed to run.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig107:
    vocab_size: int = 256
    width: int = 128
    depth: int = 4
    heads: int = 8
    ffn_width: int = 680
    context: int = 128

    def validate(self) -> None:
        if self.vocab_size != 256:
            raise ValueError("LANGBASE107 uses the fixed 256-byte vocabulary")
        if min(self.width, self.depth, self.heads, self.ffn_width, self.context) < 1:
            raise ValueError("all model dimensions must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("head dimension must be positive and even for RoPE")


@dataclass
class KVCache107:
    keys: list[Tensor]
    values: list[Tensor]
    position: int

    def tensor_bytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in self.keys + self.values)


def _apply_rope(x: Tensor, positions: Tensor) -> Tensor:
    """Apply rotary positions to ``[batch, heads, time, head_dim]``."""
    head_dim = x.shape[-1]
    inv = torch.exp(
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device)
        * (-math.log(10_000.0) / head_dim)
    )
    phase = positions.to(torch.float32)[:, None] * inv[None, :]
    cos = phase.cos().to(dtype=x.dtype)[None, None]
    sin = phase.sin().to(dtype=x.dtype)[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class CausalSelfAttention107(nn.Module):
    def __init__(self, config: TransformerConfig107) -> None:
        super().__init__()
        self.width, self.heads = config.width, config.heads
        self.head_dim = config.width // config.heads
        self.context = config.context
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=True)
        self.output = nn.Linear(config.width, config.width, bias=True)

    def forward(
        self,
        x: Tensor,
        *,
        start_position: int,
        cached_key: Tensor | None = None,
        cached_value: Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        batch, time, width = x.shape
        if width != self.width or time < 1:
            raise ValueError("invalid attention input geometry")
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        new_positions = torch.arange(start_position, start_position + time, device=x.device)
        q, k = _apply_rope(q, new_positions), _apply_rope(k, new_positions)

        past = 0
        if cached_key is not None or cached_value is not None:
            if cached_key is None or cached_value is None or cached_key.shape != cached_value.shape:
                raise ValueError("key and value cache must have matching tensors")
            expected = (batch, self.heads, cached_key.shape[2], self.head_dim)
            if cached_key.shape != expected:
                raise ValueError("cache geometry differs from the current stream")
            past = cached_key.shape[2]
            k = torch.cat((cached_key, k), dim=2)
            v = torch.cat((cached_value, v), dim=2)

        key_start = start_position - past
        query_positions = new_positions[:, None]
        key_positions = torch.arange(key_start, start_position + time, device=x.device)[None, :]
        allowed = ((key_positions <= query_positions) &
                   (key_positions > query_positions - self.context))
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        attended = attended.transpose(1, 2).contiguous().view(batch, time, width)
        output = self.output(attended)
        if not use_cache:
            return output, None, None
        return output, k[:, :, -self.context :].detach(), v[:, :, -self.context :].detach()


class TransformerBlock107(nn.Module):
    def __init__(self, config: TransformerConfig107) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = CausalSelfAttention107(config)
        self.ffn_norm = nn.LayerNorm(config.width)
        self.ffn = nn.Sequential(
            nn.Linear(config.width, config.ffn_width, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.ffn_width, config.width, bias=True),
        )

    def forward(
        self,
        x: Tensor,
        *,
        start_position: int,
        cached_key: Tensor | None,
        cached_value: Tensor | None,
        use_cache: bool,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        attended, key, value = self.attention(
            self.attention_norm(x),
            start_position=start_position,
            cached_key=cached_key,
            cached_value=cached_value,
            use_cache=use_cache,
        )
        x = x + attended
        return x + self.ffn(self.ffn_norm(x)), key, value


class CausalTransformer107(nn.Module):
    """Four-layer, approximately one-million-parameter Transformer control."""

    def __init__(self, config: TransformerConfig107 | None = None) -> None:
        super().__init__()
        self.config = config or TransformerConfig107()
        self.config.validate()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.width)
        self.blocks = nn.ModuleList(TransformerBlock107(self.config) for _ in range(self.config.depth))
        self.final_norm = nn.LayerNorm(self.config.width)
        self.lm_head = nn.Linear(self.config.width, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def start(self) -> KVCache107:
        return KVCache107(keys=[], values=[], position=0)

    def forward(
        self,
        tokens: Tensor,
        *,
        cache: KVCache107 | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache107 | None]:
        if tokens.dtype != torch.long or tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("tokens must be nonempty [batch,time] int64 byte IDs")
        if bool(((tokens < 0) | (tokens >= self.config.vocab_size)).any()):
            raise ValueError("token is outside the fixed byte vocabulary")
        if cache is not None and not use_cache:
            raise ValueError("a cache may only be supplied when use_cache=True")
        if not use_cache and tokens.shape[1] > self.config.context:
            raise ValueError("independent training windows cannot exceed the registered context")
        if cache is None:
            cache = self.start()
        if cache.position < 0 or len(cache.keys) not in (0, self.config.depth) or len(cache.values) != len(cache.keys):
            raise ValueError("invalid cache identity")

        x = self.token_embedding(tokens)
        keys: list[Tensor] = []
        values: list[Tensor] = []
        for index, block in enumerate(self.blocks):
            old_key = None if not cache.keys else cache.keys[index]
            old_value = None if not cache.values else cache.values[index]
            x, key, value = block(
                x,
                start_position=cache.position,
                cached_key=old_key,
                cached_value=old_value,
                use_cache=use_cache,
            )
            if use_cache:
                keys.append(key)
                values.append(value)
        logits = self.lm_head(self.final_norm(x))
        next_cache = KVCache107(keys=keys, values=values, position=cache.position + tokens.shape[1]) if use_cache else None
        return logits, next_cache


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

