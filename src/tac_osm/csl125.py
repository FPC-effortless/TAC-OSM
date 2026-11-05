"""CSL-125: 125M-capped research-only Compositional State Ladder.

Retains CSL-001's bounded multiscale role-state, adds four local causal GQA
readers, and shares one 32,768-token embedding/output matrix. This module
constructs an UNTRAINED architecture. No task/quality/quantization claims.

All positions and state belong to one explicitly identified input stream;
changing exit depth requires resetting/replaying the state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tac_osm.csl001 import CSLConfig, LayerState, RMSNorm, RoleStateBlock


CAP = 125_000_000


@dataclass(frozen=True)
class CSL125Config(CSLConfig):
    vocab_size: int = 32_768
    width: int = 768
    layers: int = 20
    ff_width: int = 1_408
    role_count: int = 8
    state_width: int = 256
    conv_kernel: int = 3
    chunk_size: int = 64
    tool_classes: int = 16
    field_classes: int = 16
    rungs: Tuple[int, ...] = (2, 8, 20)
    attention_layers: Tuple[int, ...] = (4, 8, 14, 20)  # one-based layer IDs
    query_heads: int = 12
    kv_heads: int = 3
    attention_window: int = 128
    max_parameters: int = CAP

    def __post_init__(self):
        super().__post_init__()
        if self.query_heads < 1 or self.kv_heads < 1 or self.width % self.query_heads != 0:
            raise ValueError('invalid query-head geometry')
        if self.query_heads % self.kv_heads != 0:
            raise ValueError('query heads must divide into KV groups')
        if self.attention_window < 2:
            raise ValueError('local attention must include more than one token')
        if self.attention_layers != tuple(sorted(set(self.attention_layers))) or any(
            x < 1 or x > self.layers for x in self.attention_layers
        ):
            raise ValueError('attention layers must be unique ordered valid depths')
        if self.max_parameters < 1 or self.max_parameters > CAP:
            raise ValueError('maximum learned parameters cannot exceed 125M')


@dataclass
class LocalKVState:
    k: Tensor        # [B, kv_heads, <=window-1, head_dim], pre-RoPE positions not assumed
    v: Tensor        # same shape
    position: int    # global count of tokens already processed

    def detached(self) -> 'LocalKVState':
        return LocalKVState(self.k.detach(), self.v.detach(), self.position)


@dataclass
class CSL125State:
    layers: Tuple[LayerState, ...]
    attention: Tuple[Optional[LocalKVState], ...]
    stream_ids: Optional[Tuple[str, ...]]

    def detached(self) -> 'CSL125State':
        return CSL125State(tuple(s.detached() for s in self.layers),
                           tuple(s.detached() if s is not None else None for s in self.attention),
                           self.stream_ids)


class LocalGQA(nn.Module):
    """Causal bounded-window attention, with explicit chunk/stream KV ownership.

    No full-history KV or full T-by-T attention matrix. The reference uses
    <=64-query blocks with <=window+63 keys. Fused device kernels remain OPEN.
    """
    def __init__(self, cfg: CSL125Config):
        super().__init__()
        self.d = cfg.width
        self.heads = cfg.query_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.width // cfg.query_heads
        self.window = cfg.attention_window
        self.pre = RMSNorm(cfg.width)
        self.q = nn.Linear(cfg.width, cfg.width, bias=False)
        self.kv = nn.Linear(cfg.width, 2 * cfg.kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(cfg.width, cfg.width, bias=False)
        self.gain = nn.Parameter(torch.tensor(0.1))
        self.register_buffer('inv_freq', 1.0 / (10000.0 ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)),
            persistent=False)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> LocalKVState:
        empty = torch.empty((batch, self.kv_heads, 0, self.head_dim),
                            device=device, dtype=dtype)
        return LocalKVState(empty, empty.clone(), 0)

    def _rope(self, x: Tensor, positions: Tensor) -> Tensor:
        """Nonparametric RoPE; the frequency vector is a nonpersistent buffer."""
        half = x.shape[-1] // 2
        if 2 * half != x.shape[-1]:
            raise ValueError('rotary head dimension must be even')
        phase = positions.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        c = phase.cos().to(x.dtype).view(1, 1, -1, half)
        s = phase.sin().to(x.dtype).view(1, 1, -1, half)
        a, b = x[..., :half], x[..., half:]
        return torch.cat([a * c - b * s, a * s + b * c], dim=-1)

    def forward(self, x: Tensor, prior: LocalKVState) -> Tuple[Tensor, LocalKVState]:
        b, t, d = x.shape
        if d != self.d or t < 1 or prior.position < 0:
            raise ValueError('bad local-attention input')
        if prior.k.shape != prior.v.shape or prior.k.shape[:2] != (b, self.kv_heads):
            raise ValueError('attention batch/KV-state ownership mismatch')
        if prior.k.shape[-1] != self.head_dim or prior.k.shape[-2] > self.window - 1:
            raise ValueError('invalid retained KV dimensions')
        h = self.pre(x)
        q = self.q(h).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        newkv = self.kv(h).view(b, t, 2, self.kv_heads, self.head_dim)
        k = newkv[:, :, 0].transpose(1, 2)
        v = newkv[:, :, 1].transpose(1, 2)
        qpos = torch.arange(prior.position, prior.position + t, device=x.device)
        q = self._rope(q, qpos)
        k = self._rope(k, qpos)
        all_k = torch.cat((prior.k, k), dim=2)
        all_v = torch.cat((prior.v, v), dim=2)
        old = prior.k.shape[2]
        kpos = torch.arange(prior.position - old, prior.position + t, device=x.device)
        # Bounded exact local attention: no global T-by-T attention matrix.
        # Each call has at most `window - 1 + block` candidate KV positions.
        block = min(64, self.window)
        chunks = []
        for start in range(0, t, block):
            stop = min(t, start + block)
            key_start = max(0, old + start - self.window + 1)
            key_stop = old + stop
            local_k = all_k[:, :, key_start:key_stop, :]
            local_v = all_v[:, :, key_start:key_stop, :]
            local_kpos = kpos[key_start:key_stop]
            local_qpos = qpos[start:stop]
            legal = (local_kpos.unsqueeze(0) <= local_qpos.unsqueeze(1)) & (
                local_kpos.unsqueeze(0) >= (local_qpos - self.window + 1).unsqueeze(1))
            chunks.append(F.scaled_dot_product_attention(
                q[:, :, start:stop], local_k, local_v,
                attn_mask=legal.unsqueeze(0).unsqueeze(0),
                dropout_p=0.0, enable_gqa=True,
            ))
        attn = torch.cat(chunks, dim=2)
        y = self.out(attn.transpose(1, 2).contiguous().reshape(b, t, d))
        kept = max(0, self.window - 1)
        # Keep only PRECEDING tokens. The next call adds the new token(s).
        nk = all_k[:, :, -kept:] if kept else all_k[:, :, :0]
        nv = all_v[:, :, -kept:] if kept else all_v[:, :, :0]
        return x + self.gain * y, LocalKVState(nk, nv, prior.position + t)


class CSL125(nn.Module):
    """Shared weights, explicit registered depth exits, hard param ceiling.

    The declared SentencePiece byte-fallback tokenizer is NOT shipped or
    pretrained. Its vocabulary is represented by the counted embedding.
    """
    def __init__(self, config: CSL125Config = CSL125Config(), *, device=None):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width, device=device)
        # Construct directly in meta device for parameter-budget and shape checks
        # without allocating hundreds of megabytes of trainable weights.
        if device is not None:
            with torch.device(device):
                self.blocks = nn.ModuleList(RoleStateBlock(config) for _ in range(config.layers))
                self.attention = nn.ModuleDict({str(n): LocalGQA(config) for n in config.attention_layers})
                self.final_norm = RMSNorm(config.width)
                self.tool_head = nn.Linear(config.width, config.tool_classes + 1)
                self.field_head = nn.Linear(config.width, config.field_classes)
                self.accept_head = nn.Linear(config.width, 1)
        else:
            self.blocks = nn.ModuleList(RoleStateBlock(config) for _ in range(config.layers))
            self.attention = nn.ModuleDict({str(n): LocalGQA(config) for n in config.attention_layers})
            self.final_norm = RMSNorm(config.width)
            self.tool_head = nn.Linear(config.width, config.tool_classes + 1)
            self.field_head = nn.Linear(config.width, config.field_classes)
            self.accept_head = nn.Linear(config.width, 1)
        actual = self.parameter_count()
        if actual > config.max_parameters:
            raise ValueError(f'CSL-125 ceiling exceeded: {actual} > {config.max_parameters}')

    def parameter_count(self) -> int:
        # unique parameters: embedding and tied output projection count ONCE
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> Dict[str, int]:
        names = {
            'embedding_and_tied_lm_head': sum(p.numel() for p in self.embedding.parameters()),
            'role_state_conv_ffn': sum(p.numel() for p in self.blocks.parameters()),
            'local_attention': sum(p.numel() for p in self.attention.parameters()),
            'task_and_calibration_heads': sum(p.numel() for p in self.tool_head.parameters()) +
                                          sum(p.numel() for p in self.field_head.parameters()) +
                                          sum(p.numel() for p in self.accept_head.parameters()),
            'final_norm': sum(p.numel() for p in self.final_norm.parameters()),
        }
        if sum(names.values()) != self.parameter_count():
            raise AssertionError('parameter categorization must reconcile')
        return names

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype,
                      *, depth: int, stream_ids: Optional[Tuple[str, ...]] = None) -> CSL125State:
        if depth not in self.config.rungs:
            raise ValueError('unregistered exit depth')
        return CSL125State(
            tuple(block.initial_state(batch, device, dtype) for block in self.blocks[:depth]),
            tuple(self.attention[str(i)].initial_state(batch, device, dtype)
                  if str(i) in self.attention else None for i in range(1, depth + 1)),
            stream_ids,
        )

    def forward(self, tokens: Tensor, *, depth: Optional[int] = None,
                state: Optional[CSL125State] = None, all_rungs: bool = False,
                stream_ids: Optional[Tuple[str, ...]] = None,
                emit_token_logits: bool = True) -> Tuple[Dict[int, Dict[str, Tensor]], CSL125State]:
        if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[1] < 1:
            raise ValueError('tokens must be nonempty [B,T] integer IDs')
        # PyTorch embedding checks token-ID bounds. Avoid GPU synchronization
        # from an explicit bool(tensor.all()) on every streaming decode step.
        d = depth or self.config.layers
        if d not in self.config.rungs:
            raise ValueError('only preregistered exit depths supported')
        if stream_ids is not None and (len(stream_ids) != tokens.shape[0] or
            any(not isinstance(s, str) or not s for s in stream_ids) or
            len(set(stream_ids)) != len(stream_ids)):
            raise ValueError('stream_ids must uniquely name every batch row')
        h = self.embedding(tokens)
        if state is None:
            state = self.initial_state(tokens.shape[0], tokens.device, h.dtype, depth=d,
                                       stream_ids=stream_ids)
        elif stream_ids is None or state.stream_ids is None or state.stream_ids != stream_ids:
            raise ValueError('carried state requires unchanged explicit stream IDs')
        if len(state.layers) != d or len(state.attention) != d:
            raise ValueError('changing active depth requires reset/replay')
        layers, caches = list(state.layers), list(state.attention)
        outs: Dict[int, Dict[str, Tensor]] = {}
        for i in range(d):
            h, layers[i] = self.blocks[i](h, layers[i])
            layer_num = i + 1
            if str(layer_num) in self.attention:
                if caches[i] is None:
                    raise ValueError('missing local KV cache for active attention layer')
                h, caches[i] = self.attention[str(layer_num)](h, caches[i])
            if layer_num == d or (all_rungs and layer_num in self.config.rungs):
                z = self.final_norm(h)
                o: Dict[str, Tensor] = {
                    'tool_logits': self.tool_head(z[:, -1]),
                    'field_logits': self.field_head(z),
                    # UNCALIBRATED. External action approval is forbidden.
                    'accept_logit': self.accept_head(z[:, -1]),
                }
                if emit_token_logits:
                    o['token_logits'] = F.linear(z, self.embedding.weight)
                outs[layer_num] = o
        return outs, CSL125State(tuple(layers), tuple(caches), state.stream_ids)

    def nested_next_token_loss(self, tokens: Tensor, weights: Optional[Dict[int, float]] = None) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] < 2:
            raise ValueError('next-token loss requires at least two token positions')
        weights = weights or {r: 1.0 for r in self.config.rungs}
        if set(weights) != set(self.config.rungs) or any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError('invalid all-rung loss weights')
        outs, _ = self.forward(tokens, all_rungs=True)
        denom = sum(weights.values())
        return sum(
            weights[r] / denom * F.cross_entropy(
                outs[r]['token_logits'][:, :-1].reshape(-1, self.config.vocab_size),
                tokens[:, 1:].reshape(-1)) for r in self.config.rungs
        )


def absolute_storage_lower_bound_bytes(parameter_count: int, bits_per_weight: float) -> int:
    """Parameter-only ideal packing; actual per-matrix scales/tokenizer/kernel overhead excluded."""
    from math import ceil
    if parameter_count < 0 or bits_per_weight <= 0:
        raise ValueError('invalid storage input')
    return ceil(parameter_count * bits_per_weight / 8)
