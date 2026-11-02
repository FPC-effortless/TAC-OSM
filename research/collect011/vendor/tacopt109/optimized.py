"""Source-equivalent execution candidates for the frozen native TAC104 graph.

No fitted weights or new trainable parameters are introduced.  Options change
execution only; old state_dict keys and the immutable State104 interface remain
valid.  Speed is a measurement question, not a property asserted by this file.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from .reference.model import (Block104, Config104, DeepCacheTAC104, LayerState104,
                              RMSNorm104, State104)
from .reference.kernels import associative_read, rotary_scan
from .paired_banks import read_bank_pair


@dataclass(frozen=True)
class OptimizationOptions:
    single_sdpa: bool = True
    aggregate_validation: bool = True
    pack_projections: bool = False
    share_rope: bool = False
    fuse_banks: bool = False

    def __post_init__(self):
        if any(type(getattr(self, f.name)) is not bool for f in fields(self)):
            raise ValueError("optimization options must be bool")


def _options(value):
    if value is None:
        return OptimizationOptions()
    if isinstance(value, OptimizationOptions):
        return value
    if isinstance(value, dict):
        try:
            return OptimizationOptions(**value)
        except TypeError as error:
            raise ValueError("unknown optimization option") from error
    raise ValueError("OptimizationOptions, dict, or None required")


def rope_table(frequencies, clock, length, dtype):
    """Exactly the reference's integer clock / FP64 phase calculation."""
    if type(clock) is not int or clock < 0 or clock + length > 2**53:
        raise ValueError("RoPE observation positions must remain in exact float64 integer range")
    positions = torch.arange(clock, clock + length, device=frequencies.device,
                             dtype=torch.long).to(torch.float64)
    phase = positions[:, None] * frequencies.to(torch.float64)[None, :]
    return phase.cos().to(dtype)[None, None], phase.sin().to(dtype)[None, None]


class OptimizedBlock104(Block104):
    def __init__(self, config, options=None):
        if type(config) is not Config104:
            raise ValueError("Exact native Config104 required; architectural variants need their own model")
        super().__init__(config)
        self.options = _options(options)

    def _input_projections(self, x):
        modules = [("fast." + name, getattr(self.fast, name))
                   for name in ("query", "key", "value", "retention", "eligibility")]
        modules += [("slow." + name, getattr(self.slow, name))
                    for name in ("query", "key", "value", "retention", "eligibility")]
        modules += [(name, getattr(self, name))
                    for name in ("local_query", "local_key", "local_value")]
        if self.config.rotary:
            modules += [(name, getattr(self, name)) for name in
                        ("rotary_retention", "rotary_angle", "rotary_offer")]
        # Packing stays connected to the original Parameters.  Constant zeros
        # for bias-free slices are NOT parameters and cannot learn dummy biases.
        weight = torch.cat([module.weight for _, module in modules], dim=0)
        bias = torch.cat([module.bias if module.bias is not None else
                          x.new_zeros(module.out_features) for _, module in modules])
        result = F.linear(x, weight, bias)
        sizes = [module.out_features for _, module in modules]
        return dict(zip((name for name, _ in modules), result.split(sizes, dim=-1)))

    def _bank_inputs(self, bank, name, x, memory, mass, write_mask, projected):
        c = bank.config
        b, t, _ = x.shape
        def p(key):
            return getattr(bank, key)(x) if projected is None else projected[name + "." + key]
        q = p("query").reshape(b, t, c.n_heads, c.key_dim).softmax(-1)
        k = p("key").reshape(b, t, c.n_heads, c.write_rank, c.key_dim).softmax(-1)
        v = p("value").reshape(b, t, c.n_heads, c.write_rank, c.value_dim).tanh()
        alpha = p("retention").sigmoid()
        a = torch.where(write_mask[..., None], alpha, torch.ones_like(alpha))
        w = (1 - a) * p("eligibility").sigmoid()
        return q, k, v, a, w, memory, mass

    def _bank(self, bank, name, x, memory, mass, write_mask, mode, projected):
        if projected is None:
            return bank(x, memory, mass, write_mask, mode)
        arguments = self._bank_inputs(bank, name, x, memory, mass, write_mask, projected)
        return associative_read(*arguments, chunk_size=bank.config.chunk_size, mode=mode)

    def local_attention(self, x, previous_key, previous_value, clock,
                        *, rope=None, projected=None):
        c = self.config
        b, t, _ = x.shape
        def projection(name):
            return getattr(self, name)(x) if projected is None else projected[name]
        q = projection("local_query").reshape(b, t, c.n_heads, c.key_dim).transpose(1, 2)
        new_key = projection("local_key").reshape(b, t, c.n_heads, c.key_dim).transpose(1, 2)
        new_value = projection("local_value").reshape(b, t, c.n_heads, c.value_dim).transpose(1, 2)
        if c.local_rope:
            # Validate even if a caller supplies a precomputed table.
            if type(clock) is not int or clock < 0 or clock + t > 2**53:
                raise ValueError("RoPE observation positions must remain in exact float64 integer range")
            cosine, sine = rope if rope is not None else rope_table(
                self.local_frequencies, clock, t, x.dtype)
            def rotate(z):
                pairs = z.reshape(b, c.n_heads, t, c.key_dim // 2, 2)
                u, v = pairs.unbind(-1)
                return torch.stack((cosine*u - sine*v, sine*u + cosine*v), -1).flatten(-2)
            q, new_key = rotate(q), rotate(new_key)
        keys = torch.cat((previous_key, new_key), dim=2)
        values = torch.cat((previous_value, new_value), dim=2)
        past = previous_key.shape[2]
        if self.options.single_sdpa and past == 0 and t <= c.window:
            # Every query sees precisely its causal prefix. No dense mask.
            recent = F.scaled_dot_product_attention(q, keys, values,
                dropout_p=0., is_causal=True).transpose(1, 2)
        elif self.options.single_sdpa and t == 1 and keys.shape[2] <= c.window:
            # A single continuing query sees all available past + current keys.
            # is_causal=True would incorrectly align it with the FIRST key.
            recent = F.scaled_dot_product_attention(q, keys, values,
                dropout_p=0., is_causal=False).transpose(1, 2)
        elif self.options.single_sdpa and t <= c.window:
            # T <= W and past <= W-1 in the public state contract. This mask is
            # bounded by W*(2W-1), independent of an arbitrarily long stream.
            query_pos = torch.arange(past, past+t, device=x.device)
            key_pos = torch.arange(keys.shape[2], device=x.device)
            legal = ((key_pos[None] <= query_pos[:, None]) &
                     (key_pos[None] > query_pos[:, None] - c.window))
            recent = F.scaled_dot_product_attention(q, keys, values,
                attn_mask=legal[None, None], dropout_p=0., is_causal=False).transpose(1, 2)
        else:
            # Preserve the native bounded-mask route for arbitrarily long
            # prefills; never replace it by an unbounded T-by-T mask.
            outputs = []
            for begin in range(0, t, c.chunk_size):
                end = min(begin + c.chunk_size, t)
                left, right = max(0, past + begin - c.window + 1), past + end
                query_pos = torch.arange(past + begin, past + end, device=x.device)
                key_pos = torch.arange(left, right, device=x.device)
                legal = ((key_pos[None] <= query_pos[:, None]) &
                         (key_pos[None] > query_pos[:, None] - c.window))
                result = F.scaled_dot_product_attention(q[:, :, begin:end],
                    keys[:, :, left:right], values[:, :, left:right],
                    attn_mask=legal[None, None], dropout_p=0., is_causal=False)
                outputs.append(result.transpose(1, 2))
            recent = torch.cat(outputs, dim=1)
        keep = c.window - 1
        final_key = keys[:, :, -keep:].clone() if keep else keys[:, :, :0].clone()
        final_value = values[:, :, -keep:].clone() if keep else values[:, :, :0].clone()
        return recent, final_key, final_value

    def forward(self, x, state, write_mask, mode, clock, *, rope=None):
        c = self.config
        b, t, _ = x.shape
        normalized = self.memory_norm(x)
        projected = self._input_projections(normalized) if self.options.pack_projections else None
        if self.options.fuse_banks:
            fast_inputs = self._bank_inputs(self.fast, "fast", normalized,
                state.fast_memory, state.fast_mass, write_mask, projected)
            slow_inputs = self._bank_inputs(self.slow, "slow", normalized,
                state.slow_memory, state.slow_mass, write_mask, projected)
            fast_result, slow_result = read_bank_pair(fast_inputs, slow_inputs,
                chunk_size=c.chunk_size, mode=mode)
            fast, fast_memory, fast_mass, fast_evidence = fast_result
            slow, slow_memory, slow_mass, slow_evidence = slow_result
        else:
            fast, fast_memory, fast_mass, fast_evidence = self._bank(
                self.fast, "fast", normalized, state.fast_memory, state.fast_mass,
                write_mask, mode, projected)
            slow, slow_memory, slow_mass, slow_evidence = self._bank(
                self.slow, "slow", normalized, state.slow_memory, state.slow_mass,
                write_mask, mode, projected)
        recent, recent_key, recent_value = self.local_attention(normalized,
            state.recent_key, state.recent_value, clock, rope=rope, projected=projected)
        pieces = [fast, slow, fast-slow, fast*slow]
        rotary_final = state.rotary
        if c.rotary:
            def projection(name):
                return getattr(self, name)(normalized) if projected is None else projected[name]
            alpha = projection("rotary_retention").sigmoid()[..., None]
            a = torch.where(write_mask[..., None, None], alpha, torch.ones_like(alpha))
            theta = math.pi * projection("rotary_angle").tanh().reshape(
                b, t, c.n_heads, c.value_dim // 2)
            theta = torch.where(write_mask[..., None, None], theta, torch.zeros_like(theta))
            offer = projection("rotary_offer").tanh().reshape(b, t, c.n_heads, c.value_dim)
            trajectory = rotary_scan(a, theta, offer, state.rotary,
                mode="parallel" if mode == "chunk" else "recurrent")
            pieces.append(trajectory)
            rotary_final = trajectory[:, -1].clone()
        pieces.extend((recent, fast_evidence[..., None], slow_evidence[..., None]))
        x = x + self.residual_scale * self.perspective_out(torch.cat(pieces, -1).flatten(-2))
        normalized = self.ffn_norm(x)
        if self.options.pack_projections:
            value, gate = F.linear(normalized, torch.cat((self.ffn_value.weight,
                self.ffn_gate.weight), dim=0)).split(c.ffn_dim, dim=-1)
        else:
            value, gate = self.ffn_value(normalized), self.ffn_gate(normalized)
        x = x + self.residual_scale * self.ffn_out(value * F.silu(gate))
        return x, LayerState104(fast_memory, fast_mass, slow_memory, slow_mass,
                               rotary_final, recent_key, recent_value)


class OptimizedTAC104(DeepCacheTAC104):
    def __init__(self, config: Config104 | None = None, options=None):
        # Same creation order and initializers as the immutable source, so the
        # same seed also creates the same initial parameters and buffers.
        nn.Module.__init__(self)
        self.config = Config104() if config is None else config
        if type(self.config) is not Config104:
            raise ValueError("Exact native Config104 required; architectural variants need their own model")
        self.options = _options(options)
        self.config_signature = hashlib.sha256(json.dumps(asdict(self.config),
            sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        c = self.config
        self.embedding = nn.Embedding(256, c.d_model)
        nn.init.normal_(self.embedding.weight, mean=0., std=0.02)
        self.blocks = nn.ModuleList(OptimizedBlock104(c, self.options) for _ in range(c.n_layers))
        self.output_norm = RMSNorm104(c.d_model)
        self.output = nn.Linear(c.d_model, 256, bias=False)
        if c.tied_embedding:
            self.output.weight = self.embedding.weight

    def _validate(self, tokens, state, owners, *, fresh):
        if not self.options.aggregate_validation:
            return super().validate(tokens, state, owners)
        if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long or
                tokens.ndim != 2 or min(tokens.shape) < 1):
            raise ValueError("observed nonempty Long[B,T] byte IDs required")
        if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            raise ValueError("autocast is unsupported until mixed-precision correspondence is qualified")
        weight = self.embedding.weight
        if weight.dtype not in (torch.float32, torch.float64):
            raise ValueError("FP32/FP64 only until mixed-precision correspondence is qualified")
        if tokens.device != weight.device:
            raise ValueError("tokens and model must share device")
        if (not isinstance(state, State104) or state.owners != owners or
                type(state.clock) is not int or state.clock < 0 or
                tokens.shape[0] != len(owners) or len(state.layers) != self.config.n_layers):
            raise ValueError("complete same-owner layer state and legal clock required")
        if state.config_signature != self.config_signature:
            raise ValueError("carried state configuration does not match model")
        c, b = self.config, len(owners)
        if c.local_rope and state.clock + tokens.shape[1] > 2**53:
            raise ValueError("RoPE observation positions must remain in exact float64 integer range")
        cache_length = min(state.clock, c.window-1)
        shapes = {"fast_memory": (b,c.n_heads,c.value_dim,c.key_dim),
                  "slow_memory": (b,c.n_heads,c.value_dim,c.key_dim),
                  "fast_mass": (b,c.n_heads,c.key_dim), "slow_mass": (b,c.n_heads,c.key_dim),
                  "rotary": (b,c.n_heads,c.value_dim),
                  "recent_key": (b,c.n_heads,cache_length,c.key_dim),
                  "recent_value": (b,c.n_heads,cache_length,c.value_dim)}
        checks = [((tokens >= 0) & (tokens <= 255)).all()]
        for layer in state.layers:
            if not isinstance(layer, LayerState104):
                raise ValueError("typed layer state required")
            for name, shape in shapes.items():
                value = getattr(layer, name)
                if (not isinstance(value, torch.Tensor) or value.shape != shape or
                        value.device != weight.device or value.dtype != weight.dtype):
                    raise ValueError("state shape/precision/device mismatch: " + name)
                # Fresh state was just constructed from zeros by self.start;
                # its finiteness follows by construction. External state must
                # still pass every native finite-value condition.
                if not fresh:
                    checks.append(torch.isfinite(value).all())
        if not bool(torch.stack(checks).all()):
            raise ValueError("invalid byte range or nonfinite carried state")

    def validate(self, tokens, state, owners):
        return self._validate(tokens, state, owners, fresh=False)

    def _shared_rope(self, clock, length, dtype):
        if not self.config.local_rope or not self.options.share_rope:
            return [None] * len(self.blocks)
        # Batch the phase/trigonometry kernels across layers.  Unlike caching a
        # supposedly common table this is correct even for distinct, replaced,
        # .data-mutated or inference-mode frequency buffers.  It needs no GPU
        # equality check or unreliable tensor version-counter assumption.
        frequencies = torch.stack([block.local_frequencies for block in self.blocks]).to(torch.float64)
        positions = torch.arange(clock, clock + length, device=frequencies.device,
                                 dtype=torch.long).to(torch.float64)
        phase = positions[None, :, None] * frequencies[:, None, :]
        cosine, sine = phase.cos().to(dtype), phase.sin().to(dtype)
        return [(cosine[i][None, None], sine[i][None, None])
                for i in range(len(self.blocks))]

    def forward(self, tokens, *, state=None, owners=None, write_mask=None, mode="chunk"):
        if mode not in ("chunk", "recurrent"):
            raise ValueError("mode must be chunk or recurrent")
        if owners is None:
            if state is not None:
                raise ValueError("explicit owners required when continuing carried state")
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
                raise ValueError("Long[B,T] byte IDs required")
            owners = tuple(f"fresh-{i}" for i in range(tokens.shape[0]))
        owners = self._owners(owners)
        fresh = state is None
        if fresh:
            state = self.start(owners)
        self._validate(tokens, state, owners, fresh=fresh)
        if write_mask is None:
            write_mask = torch.ones_like(tokens, dtype=torch.bool)
        if (not isinstance(write_mask, torch.Tensor) or write_mask.dtype != torch.bool or
                write_mask.shape != tokens.shape or write_mask.device != tokens.device):
            raise ValueError("write_mask must be Bool[B,T] on source device")
        x = self.embedding(tokens)
        ropes = self._shared_rope(state.clock, tokens.shape[1], x.dtype)
        layers = []
        for block, layer, rope in zip(self.blocks, state.layers, ropes):
            x, final = block(x, layer, write_mask, mode, state.clock, rope=rope)
            layers.append(final)
        logits = self.output(self.output_norm(x))
        return logits, State104(owners, state.clock + tokens.shape[1],
                                tuple(layers), self.config_signature)

