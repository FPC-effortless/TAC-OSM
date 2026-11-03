"""Independent time-loop equations for the actual TAC104/TAC109 graph.

This module does not call any source model forward, associative kernel, rotary
kernel or SDPA implementation. It uses the source's typed state containers and
the actual parameter/buffer tensors. Valid FP32/FP64 input/state is an explicit
precondition; the production boundary validation is tested separately.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from tacopt109.reference.model import LayerState104, State104



def affine_sequential(a, offer, initial):
    """Time-loop s[t] = a[t]*s[t-1]+offer[t], with time on axis 1."""
    state, history = initial, []
    for time in range(a.shape[1]):
        coefficient = a[:, time]
        coefficient = coefficient.reshape(*coefficient.shape,
                                          *((1,) * (state.ndim - coefficient.ndim)))
        state = coefficient * state + offer[:, time]
        history.append(state)
    return torch.stack(history, 1) if history else offer


def associative_sequential(q, k, v, a, w, memory, mass, epsilon=1e-4):
    """Scalar-time outer-product reference, independent of source GEMM scans."""
    if q.shape[1] == 0:
        return q.new_empty((*q.shape[:-1], v.shape[-1])), memory, mass, a
    current_memory, current_mass = memory, mass
    reads, evidences = [], []
    for time in range(q.shape[1]):
        # Mean over rank, with explicit multiply/reduce rather than source @.
        keys, values = k[:, time], v[:, time]
        write = (values.unsqueeze(-1) * keys.unsqueeze(-2)).mean(dim=2)
        current_memory = (a[:, time, :, None, None] * current_memory +
                          w[:, time, :, None, None] * write)
        current_mass = (a[:, time, :, None] * current_mass +
                        w[:, time, :, None] * keys.mean(dim=2))
        query = q[:, time]
        evidence = (current_mass * query).sum(-1)
        numerator = (current_memory * query.unsqueeze(-2)).sum(-1)
        reads.append(numerator / (evidence.unsqueeze(-1) + epsilon))
        evidences.append(evidence)
    return torch.stack(reads, 1), current_memory, current_mass, torch.stack(evidences, 1)


def rotary_sequential(a, theta, offer, initial):
    """Explicit time-loop over fixed 2D rotation planes; no prefix kernel."""
    state = initial.reshape(*initial.shape[:-1], initial.shape[-1] // 2, 2)
    history = []
    for time in range(offer.shape[1]):
        real = a[:, time] * theta[:, time].cos()
        imaginary = a[:, time] * theta[:, time].sin()
        first, second = state.unbind(-1)
        state = torch.stack((real * first - imaginary * second,
                             imaginary * first + real * second), -1)
        addition = ((1 - a[:, time]) * offer[:, time]).reshape_as(state)
        state = state + addition
        history.append(state.flatten(-2))
    return torch.stack(history, 1) if history else offer


def _linear(value, module):
    return F.linear(value, module.weight, module.bias)


def _rms(value, module):
    return module.weight * value * torch.rsqrt(
        value.square().mean(-1, keepdim=True) + module.epsilon)


def _bank(value, module, memory, mass, mask):
    config = module.config
    batch, length = value.shape[:2]
    q = _linear(value, module.query).reshape(
        batch, length, config.n_heads, config.key_dim).softmax(-1)
    k = _linear(value, module.key).reshape(
        batch, length, config.n_heads, config.write_rank, config.key_dim).softmax(-1)
    v = _linear(value, module.value).reshape(
        batch, length, config.n_heads, config.write_rank, config.value_dim).tanh()
    alpha = _linear(value, module.retention).sigmoid()
    retention = torch.where(mask[..., None], alpha, torch.ones_like(alpha))
    weight = (1 - retention) * _linear(value, module.eligibility).sigmoid()
    return associative_sequential(q, k, v, retention, weight, memory, mass)


def _local_attention(value, block, previous_key, previous_value, clock):
    """Per-query softmax oracle with exact recent-window/time semantics."""
    c = block.config
    batch, length = value.shape[:2]
    q = _linear(value, block.local_query).reshape(batch, length, c.n_heads, c.key_dim)
    k = _linear(value, block.local_key).reshape_as(q)
    v = _linear(value, block.local_value).reshape(batch, length, c.n_heads, c.value_dim)
    if c.local_rope:
        positions = torch.arange(clock, clock + length, device=value.device,
                                 dtype=torch.long).to(torch.float64)
        phase = positions[:, None] * block.local_frequencies.to(torch.float64)[None]
        cosine = phase.cos().to(value.dtype)[None, :, None]
        sine = phase.sin().to(value.dtype)[None, :, None]
        def rotate(tensor):
            first, second = tensor.reshape(batch, length, c.n_heads,
                                            c.key_dim // 2, 2).unbind(-1)
            return torch.stack((cosine * first - sine * second,
                                sine * first + cosine * second), -1).flatten(-2)
        q, k = rotate(q), rotate(k)
    past = previous_key.shape[2]
    keys = torch.cat((previous_key.transpose(1, 2), k), 1)
    values = torch.cat((previous_value.transpose(1, 2), v), 1)
    reads = []
    for time in range(length):
        end = past + time + 1
        begin = max(0, end - c.window)
        scores = (keys[:, begin:end] * q[:, time, None]).sum(-1) / math.sqrt(c.key_dim)
        probabilities = scores.softmax(dim=1)
        reads.append((probabilities[..., None] * values[:, begin:end]).sum(1))
    keep = c.window - 1
    if keep:
        final_key = keys[:, -keep:].transpose(1, 2).contiguous().clone()
        final_value = values[:, -keep:].transpose(1, 2).contiguous().clone()
    else:
        final_key = keys[:, :0].transpose(1, 2).clone()
        final_value = values[:, :0].transpose(1, 2).clone()
    return torch.stack(reads, 1), final_key, final_value


def _fresh_state(model, owners):
    c, batch = model.config, len(owners)
    is109 = hasattr(c, "associative_key_dim")
    key = c.associative_key_dim if is109 else c.key_dim
    value = c.associative_value_dim if is109 else c.value_dim
    field = c.value_dim if (c.rotary or not is109) else 0
    layer_type, state_type = (LayerState109, State109) if is109 else (LayerState104, State104)
    zeros = model.embedding.weight.new_zeros
    layers = tuple(layer_type(
        zeros((batch, c.n_heads, value, key)), zeros((batch, c.n_heads, key)),
        zeros((batch, c.n_heads, value, key)), zeros((batch, c.n_heads, key)),
        zeros((batch, c.n_heads, field)), zeros((batch, c.n_heads, 0, c.key_dim)),
        zeros((batch, c.n_heads, 0, c.value_dim))) for _ in range(c.n_layers))
    return state_type(owners, 0, layers, model.config_signature)


def model_reference(model, tokens, *, state=None, owners=None, write_mask=None):
    """Whole typed input-to-logits graph, re-expressed from equations.

    Source weight tensors are shared so parameter gradients can be compared.
    Production validation, source checkpoint authenticity, floating-point error
    proofs, learning quality and external policy qualification are not supplied
    by this mathematical reference.
    """
    c = model.config
    owners = tuple(f"fresh-{i}" for i in range(tokens.shape[0])) if owners is None else tuple(owners)
    state = _fresh_state(model, owners) if state is None else state
    write_mask = torch.ones_like(tokens, dtype=torch.bool) if write_mask is None else write_mask
    value = F.embedding(tokens, model.embedding.weight)
    layers = []
    attention = getattr(c, "attention", True)
    for block, previous in zip(model.blocks, state.layers):
        normalized = _rms(value, block.memory_norm)
        fast, fm, fz, fe = _bank(normalized, block.fast, previous.fast_memory,
                                previous.fast_mass, write_mask)
        slow, sm, sz, se = _bank(normalized, block.slow, previous.slow_memory,
                                previous.slow_mass, write_mask)
        pieces = [fast, slow, fast - slow, fast * slow]
        final_field = previous.rotary
        if c.rotary:
            alpha = _linear(normalized, block.rotary_retention).sigmoid()[..., None]
            alpha = torch.where(write_mask[..., None, None], alpha, torch.ones_like(alpha))
            angles = math.pi * _linear(normalized, block.rotary_angle).tanh().reshape(
                *tokens.shape, c.n_heads, c.value_dim // 2)
            angles = torch.where(write_mask[..., None, None], angles, torch.zeros_like(angles))
            offered = _linear(normalized, block.rotary_offer).tanh().reshape(
                *tokens.shape, c.n_heads, c.value_dim)
            trajectory = rotary_sequential(alpha, angles, offered, previous.rotary)
            final_field = trajectory[:, -1].clone()
            pieces.append(trajectory)
        final_key, final_value = previous.recent_key, previous.recent_value
        if attention:
            recent, final_key, final_value = _local_attention(
                normalized, block, final_key, final_value, state.clock)
            pieces.append(recent)
        pieces.extend((fe[..., None], se[..., None]))
        compared = torch.cat(pieces, -1).flatten(-2)
        value = value + block.residual_scale * _linear(compared, block.perspective_out)
        normalized = _rms(value, block.ffn_norm)
        hidden = _linear(normalized, block.ffn_value) * F.silu(_linear(normalized, block.ffn_gate))
        value = value + block.residual_scale * _linear(hidden, block.ffn_out)
        layers.append(type(previous)(fm, fz, sm, sz, final_field, final_key, final_value))
    logits = _linear(_rms(value, model.output_norm), model.output)
    final = type(state)(owners, state.clock + tokens.shape[1], tuple(layers), model.config_signature)
    return logits, final
