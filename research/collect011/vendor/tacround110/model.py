"""New TAC graphs, distinct from the frozen execution-only TACOPT109 release.

mHC wraps memory and FFN residual branches independently. SSA is invoked once,
after physical block L/2, with residual scale 1/sqrt(2L). All computations are
causal and state remains attached unless the caller explicitly detaches it.
This is an untrained research implementation, not a validated SSA008 replica.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from tacopt109.optimized import OptimizedBlock104, OptimizedTAC104, OptimizationOptions
from tacopt109.reference.model import Config104, State104, LayerState104
from tacopt109.reference.kernels import rotary_scan
from tacopt109.paired_banks import read_bank_pair
from .mhc import MHCResidual


@dataclass(frozen=True)
class RoundConfig:
    depth: int = 4
    d_model: int = 120
    ffn_dim: int = 248
    mhc: bool = False
    ssa: bool = False
    hebbian: bool = True
    nsf: bool = True
    streams: int = 4
    n_heads: int = 4
    key_dim: int = 16
    value_dim: int = 16
    write_rank: int = 2
    window: int = 320
    chunk_size: int = 32
    rotary: bool = True
    local_rope: bool = True
    tied_embedding: bool = True
    slots: int = 32
    topk: int = 4

    def __post_init__(self):
        if self.depth not in (4, 8) or type(self.depth) is not int:
            raise ValueError("registered depths are 4 and 8")
        for name in ("mhc", "ssa", "hebbian", "nsf"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(name + " must be bool")
        if type(self.streams) is not int or self.streams < 2:
            raise ValueError("at least two streams required")
        if not self.ssa and (not self.hebbian or not self.nsf):
            raise ValueError("SSA ablation flags require SSA")
        if type(self.slots) is not int or type(self.topk) is not int or not 1 <= self.topk <= self.slots:
            raise ValueError("1 <= topk <= slots required")
        self.native_config()

    def native_config(self):
        return Config104(d_model=self.d_model, n_layers=self.depth,
                         n_heads=self.n_heads, key_dim=self.key_dim,
                         value_dim=self.value_dim, write_rank=self.write_rank,
                         ffn_dim=self.ffn_dim, window=self.window,
                         chunk_size=self.chunk_size, rotary=self.rotary,
                         local_rope=self.local_rope, tied_embedding=self.tied_embedding)


@dataclass(frozen=True)
class RoundState:
    native: State104
    ssa: object | None
    config_signature: str

    @property
    def owners(self):
        return self.native.owners

    @property
    def clock(self):
        return self.native.clock

    @property
    def layers(self):
        return self.native.layers

    def detach(self):
        return RoundState(self.native.detach(),
                          None if self.ssa is None else self.ssa.detach(),
                          self.config_signature)

    def state_bytes(self):
        total = self.native.state_bytes()
        if self.ssa is not None:
            total += self.ssa.state_bytes()
        return total

    tensor_bytes = state_bytes

    def storage_bytes(self):
        """Retained backing storage, deduplicated across all state tensors."""
        tensors = [v for layer in self.layers for v in layer.tensors().values()]
        if self.ssa is not None:
            tensors.extend(getattr(self.ssa, name) for name in ("keys", "values", "survival")
                           if getattr(self.ssa, name) is not None)
        seen, total = set(), 0
        for value in tensors:
            storage = value.untyped_storage()
            key = (value.device, storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key); total += storage.nbytes()
        return total

    def payload(self):
        return {"schema": "tacround110.state.v1", "native": self.native.payload(),
                "ssa": None if self.ssa is None else self.ssa.payload(),
                "config_signature": self.config_signature}

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict) or set(payload) != {"schema", "native", "ssa", "config_signature"}:
            raise ValueError("complete round state payload required")
        if payload["schema"] != "tacround110.state.v1":
            raise ValueError("unsupported round state schema")
        sparse = None
        if payload["ssa"] is not None:
            from .ssa import SSAState
            sparse = SSAState.from_payload(payload["ssa"])
        return cls(State104.from_payload(payload["native"]), sparse, payload["config_signature"])


class ResearchBlock110(OptimizedBlock104):
    def __init__(self, config, options, streams):
        super().__init__(config, options)
        self.memory_mhc = MHCResidual(config.d_model, streams)
        self.ffn_mhc = MHCResidual(config.d_model, streams)

    def forward(self, streams, state, write_mask, mode, clock, *, rope=None):
        c = self.config
        x, memory_context = self.memory_mhc.prepare(streams)
        b, t, _ = x.shape
        normalized = self.memory_norm(x)
        projected = self._input_projections(normalized)
        fast_inputs = self._bank_inputs(self.fast, "fast", normalized,
            state.fast_memory, state.fast_mass, write_mask, projected)
        slow_inputs = self._bank_inputs(self.slow, "slow", normalized,
            state.slow_memory, state.slow_mass, write_mask, projected)
        fast_result, slow_result = read_bank_pair(fast_inputs, slow_inputs,
                                                chunk_size=c.chunk_size, mode=mode)
        fast, fast_memory, fast_mass, fast_evidence = fast_result
        slow, slow_memory, slow_mass, slow_evidence = slow_result
        recent, recent_key, recent_value = self.local_attention(normalized,
            state.recent_key, state.recent_value, clock, rope=rope, projected=projected)
        pieces = [fast, slow, fast-slow, fast*slow]
        rotary_final = state.rotary
        if c.rotary:
            alpha = projected["rotary_retention"].sigmoid()[..., None]
            a = torch.where(write_mask[..., None, None], alpha, torch.ones_like(alpha))
            theta = math.pi * projected["rotary_angle"].tanh().reshape(
                b, t, c.n_heads, c.value_dim // 2)
            theta = torch.where(write_mask[..., None, None], theta, torch.zeros_like(theta))
            offer = projected["rotary_offer"].tanh().reshape(b, t, c.n_heads, c.value_dim)
            trajectory = rotary_scan(a, theta, offer, state.rotary,
                                     mode="parallel" if mode == "chunk" else "recurrent")
            pieces.append(trajectory)
            rotary_final = trajectory[:, -1].clone()
        pieces.extend((recent, fast_evidence[..., None], slow_evidence[..., None]))
        delta = self.residual_scale * self.perspective_out(torch.cat(pieces, -1).flatten(-2))
        streams = self.memory_mhc.finish(streams, delta, memory_context)
        x, ffn_context = self.ffn_mhc.prepare(streams)
        normalized = self.ffn_norm(x)
        value, gate = F.linear(normalized, torch.cat((self.ffn_value.weight,
                                self.ffn_gate.weight), dim=0)).split(c.ffn_dim, dim=-1)
        delta = self.residual_scale * self.ffn_out(value * F.silu(gate))
        streams = self.ffn_mhc.finish(streams, delta, ffn_context)
        return streams, LayerState104(fast_memory, fast_mass, slow_memory, slow_mass,
                                     rotary_final, recent_key, recent_value)


class ResearchTAC110(nn.Module):
    def __init__(self, config: RoundConfig):
        super().__init__()
        if type(config) is not RoundConfig:
            raise ValueError("exact RoundConfig required")
        self.config = config
        self.config_signature = hashlib.sha256(json.dumps(asdict(config),
                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        options = OptimizationOptions(pack_projections=True, share_rope=True, fuse_banks=True)
        self.core = OptimizedTAC104(config.native_config(), options)
        if config.mhc:
            self.core.blocks = nn.ModuleList(ResearchBlock110(config.native_config(),
                            options, config.streams) for _ in range(config.depth))
            # Fixed orthonormal zero-sum contrast basis: no unused common mode.
            s = config.streams
            basis = torch.zeros(s, s - 1)
            for j in range(s - 1):
                basis[:j+1, j] = 1 / math.sqrt((j+1)*(j+2))
                basis[j+1, j] = -(j+1) / math.sqrt((j+1)*(j+2))
            self.register_buffer("stream_basis", basis)
            self.stream_contrasts = nn.Parameter(torch.randn(s-1, config.d_model) * .01)
            # Nonuniform content readout avoids cancellation of the final
            # doubly-stochastic residual transport by an arithmetic mean.
            self.stream_readout = nn.Parameter(torch.randn(config.d_model) * .02)
        if config.ssa:
            from .ssa import SparseStructureMemory
            self.sparse_memory = SparseStructureMemory(config.d_model,
                slots=config.slots, key_dim=config.key_dim, value_dim=config.value_dim,
                topk=config.topk, hebbian=config.hebbian, nsf=config.nsf)

    def start(self, owners):
        owners = self.core._owners(owners)
        sparse = self.sparse_memory.start(owners) if self.config.ssa else None
        return RoundState(self.core.start(owners), sparse, self.config_signature)

    def reduce_streams(self, x):
        logits = torch.einsum("btsd,d->bts", x, self.stream_readout) / math.sqrt(self.config.d_model)
        return (x * logits.softmax(-1).unsqueeze(-1)).sum(-2)

    def forward(self, tokens, *, state=None, owners=None, write_mask=None, mode="chunk"):
        if mode not in ("chunk", "recurrent"):
            raise ValueError("mode must be chunk or recurrent")
        if owners is None:
            if state is not None:
                raise ValueError("explicit owners required for continuing state")
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
                raise ValueError("Long[B,T] required")
            owners = tuple(f"fresh-{i}" for i in range(tokens.shape[0]))
        owners = self.core._owners(owners)
        fresh = state is None
        if fresh:
            state = self.start(owners)
        if not isinstance(state, RoundState) or state.config_signature != self.config_signature:
            raise ValueError("same-variant RoundState required")
        if not isinstance(state.native, State104):
            raise ValueError("typed native state required")
        if self.config.ssa != (state.ssa is not None):
            raise ValueError("complete sparse state required")
        if self.config.ssa:
            from .ssa import SSAState
            if not isinstance(state.ssa, SSAState) or state.ssa.clock != state.native.clock:
                raise ValueError("typed native and sparse states must describe the same observation clock")
        self.core._validate(tokens, state.native, owners, fresh=fresh)
        if write_mask is None:
            write_mask = torch.ones_like(tokens, dtype=torch.bool)
        if (not isinstance(write_mask, torch.Tensor) or write_mask.dtype != torch.bool or
                write_mask.shape != tokens.shape or write_mask.device != tokens.device):
            raise ValueError("write_mask must be Bool[B,T] on source device")
        x = self.core.embedding(tokens)
        if self.config.mhc:
            offsets = self.stream_basis @ self.stream_contrasts
            x = x.unsqueeze(-2) + offsets
        ropes = self.core._shared_rope(state.clock, tokens.shape[1], x.dtype)
        layers, sparse = [], state.ssa
        for i, (block, layer, rope) in enumerate(zip(self.core.blocks, state.layers, ropes)):
            x, final = block(x, layer, write_mask, mode, state.clock, rope=rope)
            layers.append(final)
            if self.config.ssa and i + 1 == self.config.depth // 2:
                query = self.reduce_streams(x) if self.config.mhc else x
                residual, sparse = self.sparse_memory(query, state=sparse,
                                                     write_mask=write_mask, owners=owners)
                residual = residual / math.sqrt(2 * self.config.depth)
                x = x + (residual.unsqueeze(-2) if self.config.mhc else residual)
        if self.config.mhc:
            x = self.reduce_streams(x)
        logits = self.core.output(self.core.output_norm(x))
        native = State104(owners, state.clock + tokens.shape[1], tuple(layers), self.core.config_signature)
        return logits, RoundState(native, sparse, self.config_signature)

    def step(self, token, *, state, owners, write_mask=None):
        if isinstance(token, torch.Tensor) and token.ndim == 1:
            token = token[:, None]
        if not isinstance(token, torch.Tensor) or token.ndim != 2 or token.shape[1] != 1:
            raise ValueError("step requires exactly one observed byte per stream")
        return self.forward(token, state=state, owners=owners,
                            write_mask=write_mask, mode="recurrent")
