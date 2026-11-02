"""Vision-grounded TAC / cached Transformer / residual GRU candidates.

ZERO TRAINING by construction scripts. Frozen encoder features are inputs,
never text logits. All arms use the same UTF-8 vocabulary and visual bridge.
TAC reuses the actual LANGCACHE104 blocks, not an architecture re-creation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import uuid

import torch
from torch import nn
from torch.nn import functional as F

from research.langcache104.model import Block104, Config104, LayerState104, RMSNorm104
from .tokenizer import VOCAB_SIZE


@dataclass(frozen=True)
class OCRConfig:
    family: str = "tac"
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 384
    feature_dim: int = 384
    reference_budget: int = 128
    window: int = 128
    chunk_size: int = 32
    key_dim: int = 16
    value_dim: int = 16
    write_rank: int = 1

    def __post_init__(self):
        if self.family not in ("tac", "transformer", "gru"):
            raise ValueError("unknown decoder family")
        for name, value in asdict(self).items():
            if name != "family" and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("complete even head widths required")
        if self.key_dim % 2:
            raise ValueError("TAC local RoPE requires even key_dim")


@dataclass(frozen=True)
class VisualReferences:
    owners: tuple[str, ...]
    page_ids: tuple[str, ...]
    model_tag: str
    keys: torch.Tensor
    values: torch.Tensor

    def tensor_bytes(self):
        return sum(x.numel() * x.element_size() for x in (self.keys, self.values))


@dataclass(frozen=True)
class DecoderState:
    owners: tuple[str, ...]
    signature: str
    clock: int
    layers: tuple
    model_tag: str = ""

    def tensors(self):
        result = {}
        for i, value in enumerate(self.layers):
            items = value.tensors() if isinstance(value, LayerState104) else (
                {"hidden": value} if isinstance(value, torch.Tensor) else
                {"key": value[0], "value": value[1]})
            result.update({f"layer.{i}.{key}": tensor for key, tensor in items.items()})
        return result

    def tensor_bytes(self):
        return sum(x.numel() * x.element_size() for x in self.tensors().values())

    def detach(self):
        layers = tuple(x.detach() if hasattr(x, "detach") else
                       tuple(y.detach() for y in x) for x in self.layers)
        return replace(self, layers=layers)


def owners_tuple(owners):
    owners = tuple(owners)
    if not owners or any(type(x) is not str or not x for x in owners) or len(set(owners)) != len(owners):
        raise ValueError("unique, nonempty explicit document owners required")
    return owners


def compress_references(features, boxes, budget):
    """Label-free raster-order pooling. Not learned relevance/SSA routing.

    Dense/very small text must be encoded from line/region crops, not resized
    whole pages. Original images/features remain external exact evidence.
    """
    if features.ndim != 3 or boxes.shape != (*features.shape[:2], 4) or features.shape[1] < 1:
        raise ValueError("features[B,N,F] and normalized boxes[B,N,4] required")
    if boxes.device != features.device or boxes.dtype != features.dtype:
        raise ValueError("boxes/features must share precision and device")
    if features.shape[1] > budget:
        features = F.adaptive_avg_pool1d(features.transpose(1, 2), budget).transpose(1, 2)
        boxes = F.adaptive_avg_pool1d(boxes.transpose(1, 2), budget).transpose(1, 2)
    # Own compact storage; a slice of a full document would retain it.
    return features.contiguous().clone(), boxes.contiguous().clone()


class FeedForward(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.norm = RMSNorm104(width)
        self.gate_value = nn.Linear(width, hidden * 2, bias=False)
        self.out = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        gate, value = self.gate_value(self.norm(x)).chunk(2, -1)
        return self.out(F.silu(gate) * value)


class WindowAttention(nn.Module):
    """Fused QKV + SDPA tiles; no full-history or T-by-T attention matrix."""
    def __init__(self, config):
        super().__init__()
        self.c = config
        self.norm = RMSNorm104(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        head = config.d_model // config.n_heads
        self.register_buffer("frequencies", torch.exp(-math.log(10000.) * torch.arange(0, head, 2) / head))

    def forward(self, x, state, clock):
        c, (b, t, d) = self.c, x.shape
        q, k, v = (z.reshape(b, t, c.n_heads, d // c.n_heads).transpose(1, 2)
                   for z in self.qkv(self.norm(x)).chunk(3, -1))
        # Strict double phase for long clocks, matching LANGCACHE's policy.
        phase = torch.arange(clock, clock+t, device=x.device, dtype=torch.float64)[:, None] * self.frequencies.double()[None]
        co, si = phase.cos().to(x.dtype)[None, None], phase.sin().to(x.dtype)[None, None]
        def rotate(z):
            a, z_b = z.reshape(b, c.n_heads, t, -1, 2).unbind(-1)
            return torch.stack((co*a-si*z_b, si*a+co*z_b), -1).flatten(-2)
        q, k = rotate(q), rotate(k)
        old_k, old_v = state
        keys, values = torch.cat((old_k, k), 2), torch.cat((old_v, v), 2)
        past, outputs = old_k.shape[2], []
        for start in range(0, t, c.chunk_size):
            end = min(start+c.chunk_size, t)
            left, right = max(0, past+start-c.window+1), past+end
            qi = torch.arange(past+start, past+end, device=x.device)
            ki = torch.arange(left, right, device=x.device)
            legal = (ki[None] <= qi[:, None]) & (ki[None] > qi[:, None]-c.window)
            outputs.append(F.scaled_dot_product_attention(q[:, :, start:end], keys[:, :, left:right],
                           values[:, :, left:right], attn_mask=legal[None, None], dropout_p=0.))
        keep = c.window-1
        next_state = (keys[:, :, -keep:].clone(), values[:, :, -keep:].clone()) if keep else (
            keys[:, :, :0].clone(), values[:, :, :0].clone())
        return self.out(torch.cat(outputs, 2).transpose(1, 2).reshape(b, t, d)), next_state


class TransformerBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention, self.ffn = WindowAttention(c), FeedForward(c.d_model, c.ffn_dim)
        self.scale = 1 / math.sqrt(2*c.n_layers)

    def forward(self, x, state, clock):
        delta, state = self.attention(x, state, clock)
        x = x + self.scale*delta
        return x + self.scale*self.ffn(x), state


class GRUBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = RMSNorm104(c.d_model)
        self.gru = nn.GRU(c.d_model, c.d_model, batch_first=True)
        self.ffn = FeedForward(c.d_model, c.ffn_dim)
        self.scale = 1 / math.sqrt(2*c.n_layers)

    def forward(self, x, state):
        delta, hidden = self.gru(self.norm(x), state)
        x = x + self.scale*delta
        return x + self.scale*self.ffn(x), hidden.clone()


class GroundedOCRDecoder(nn.Module):
    def __init__(self, config: OCRConfig):
        super().__init__()
        self.config = config
        self.signature = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
        # References cannot silently cross model instances/checkpoint loads.
        self.model_tag = self.signature + ":" + uuid.uuid4().hex
        c = config
        self.embedding = nn.Embedding(VOCAB_SIZE, c.d_model)
        nn.init.normal_(self.embedding.weight, std=.02)
        self.visual_projection = nn.Linear(c.feature_dim, c.d_model, bias=False)
        self.box_projection = nn.Linear(4, c.d_model, bias=False)
        self.visual_norm = RMSNorm104(c.d_model)
        self.visual_kv = nn.Linear(c.d_model, 2*c.d_model, bias=False)
        self.cross_norms = nn.ModuleList(RMSNorm104(c.d_model) for _ in range(c.n_layers))
        self.cross_queries = nn.ModuleList(nn.Linear(c.d_model, c.d_model, bias=False) for _ in range(c.n_layers))
        self.cross_outputs = nn.ModuleList(nn.Linear(c.d_model, c.d_model, bias=False) for _ in range(c.n_layers))
        if c.family == "tac":
            self.tac_config = Config104(d_model=c.d_model, n_layers=c.n_layers, n_heads=c.n_heads,
                key_dim=c.key_dim, value_dim=c.value_dim, write_rank=c.write_rank,
                ffn_dim=c.ffn_dim, window=c.window, chunk_size=c.chunk_size, rotary=False)
            self.layers = nn.ModuleList(Block104(self.tac_config) for _ in range(c.n_layers))
        else:
            block = GRUBlock if c.family == "gru" else TransformerBlock
            self.layers = nn.ModuleList(block(c) for _ in range(c.n_layers))
        self.output_norm = RMSNorm104(c.d_model)
        self.output = nn.Linear(c.d_model, VOCAB_SIZE, bias=False)
        self.output.weight = self.embedding.weight

    def __setstate__(self, state):
        # Module copy/unpickle restores a distinct Python model instance. Do not
        # inherit its source instance's trusted local cache/state identity.
        super().__setstate__(state)
        self.model_tag = self.signature + ":" + uuid.uuid4().hex

    def _load_from_state_dict(self, *args, **kwargs):
        # A strict load may fail after copying some tensors. Invalidate first,
        # including when this decoder is loaded as a child of another module.
        self.model_tag = self.signature + ":" + uuid.uuid4().hex
        return super()._load_from_state_dict(*args, **kwargs)

    def counts(self, *, frozen_encoder_parameters=0):
        if type(frozen_encoder_parameters) is not int or frozen_encoder_parameters < 0:
            raise ValueError("measured nonnegative encoder count required")
        return {"trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "decoder_deployed": sum(p.numel() for p in self.parameters()),
                "frozen_encoder": frozen_encoder_parameters,
                "total_deployed": sum(p.numel() for p in self.parameters()) + frozen_encoder_parameters}

    def prepare_visual(self, features, boxes, *, owners, page_ids):
        owners = owners_tuple(owners)
        if (features.shape[0] != len(owners) or features.shape[-1] != self.config.feature_dim or
                len(page_ids) != len(owners) or any(type(x) is not str or not x for x in page_ids)):
            raise ValueError("same-batch named visual regions required")
        if features.device != self.embedding.weight.device or features.dtype != self.embedding.weight.dtype:
            raise ValueError("features must share model precision/device")
        if not torch.isfinite(features).all() or not torch.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any():
            raise ValueError("finite features and normalized box coordinates required")
        features, boxes = compress_references(features, boxes, self.config.reference_budget)
        x = self.visual_norm(self.visual_projection(features) + self.box_projection(boxes))
        b, n, _ = x.shape
        keys, values = (z.reshape(b, n, self.config.n_heads, -1).transpose(1, 2).contiguous()
                        for z in self.visual_kv(x).chunk(2, -1))
        return VisualReferences(owners, tuple(page_ids), self.model_tag, keys, values)

    def start(self, owners):
        owners = owners_tuple(owners)
        c, b, new = self.config, len(owners), self.embedding.weight.new_zeros
        layers = []
        for _ in range(c.n_layers):
            if c.family == "tac":
                layers.append(LayerState104(new((b,c.n_heads,c.value_dim,c.key_dim)), new((b,c.n_heads,c.key_dim)),
                    new((b,c.n_heads,c.value_dim,c.key_dim)), new((b,c.n_heads,c.key_dim)),
                    new((b,c.n_heads,c.value_dim)), new((b,c.n_heads,0,c.key_dim)), new((b,c.n_heads,0,c.value_dim))))
            elif c.family == "gru":
                layers.append(new((1,b,c.d_model)))
            else:
                layers.append((new((b,c.n_heads,0,c.d_model//c.n_heads)), new((b,c.n_heads,0,c.d_model//c.n_heads))))
        return DecoderState(owners, self.signature, 0, tuple(layers), self.model_tag)

    def forward(self, tokens, references, *, owners, state=None, mode="chunk"):
        owners = owners_tuple(owners)
        if mode not in ("chunk", "recurrent"):
            raise ValueError("chunk or recurrent mode required")
        if (tokens.ndim != 2 or tokens.dtype != torch.long or min(tokens.shape) < 1 or
                tokens.shape[0] != len(owners) or (tokens < 0).any() or (tokens >= VOCAB_SIZE).any()):
            raise ValueError("nonempty legal UTF-8 token tensor[B,T] required")
        if (not isinstance(references, VisualReferences) or references.owners != owners or
                references.model_tag != self.model_tag):
            raise ValueError("visual references belong to another model/document")
        state = self.start(owners) if state is None else state
        if state.owners != owners or state.signature != self.signature or state.model_tag != self.model_tag or len(state.layers) != self.config.n_layers:
            raise ValueError("decoder state belongs to another document/model/load generation")
        if type(state.clock) is not int or state.clock < 0 or state.clock+tokens.shape[1] > 2**53:
            raise ValueError("legal exact-integer stream clock required")
        if tokens.device != self.embedding.weight.device:
            raise ValueError("token/model devices differ")
        # LANGCACHE104 is qualified only for strict FP32/FP64. Do not quietly
        # advertise AMP/INT8/Triton before correspondence and timings exist.
        if self.embedding.weight.dtype not in (torch.float32, torch.float64) or torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            raise ValueError("strict FP32/FP64 only; mixed precision remains unqualified")
        x, final = self.embedding(tokens), []
        c, (b,t) = self.config, tokens.shape
        for i, (block, previous) in enumerate(zip(self.layers, state.layers)):
            q = self.cross_queries[i](self.cross_norms[i](x)).reshape(b,t,c.n_heads,-1).transpose(1,2)
            vision = F.scaled_dot_product_attention(q, references.keys, references.values, dropout_p=0.)
            x = x + self.cross_outputs[i](vision.transpose(1,2).reshape(b,t,c.d_model)) / math.sqrt(c.n_layers)
            if c.family == "tac":
                x, next_layer = block(x, previous, torch.ones_like(tokens, dtype=torch.bool), mode, state.clock)
            elif c.family == "transformer":
                x, next_layer = block(x, previous, state.clock)
            else:
                x, next_layer = block(x, previous)
            final.append(next_layer)
        return self.output(self.output_norm(x)), DecoderState(owners, self.signature, state.clock+t, tuple(final), self.model_tag)

    def step(self, token, references, *, owners, state):
        if token.ndim != 2 or token.shape[1] != 1:
            raise ValueError("step requires [B,1]")
        return self(token, references, owners=owners, state=state, mode="recurrent")
