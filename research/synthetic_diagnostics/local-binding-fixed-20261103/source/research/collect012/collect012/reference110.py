"""Independent explicit-time equations for valid TACROUND110 graphs.

Source parameters and typed state containers are reused, but no source model,
block, mHC, SSA, associative kernel, rotary scan, or SDPA forward is called.
This is finite-fixture executable evidence, not source/formal qualification.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from tacopt109.reference.model import LayerState104, State104
from .reference104 import (
    _linear, _rms, _local_attention, _fresh_state,
    associative_sequential, rotary_sequential,
)


def mhc_prepare_reference(module, streams):
    """Positive-domain Sinkhorn oracle, independent of source log iteration."""
    s = module.streams
    dtype = torch.float64 if streams.dtype == torch.float64 else torch.float32
    value = streams.to(dtype)
    flat = value.flatten(-2)
    normalized = flat / torch.sqrt(flat.square().mean(-1, keepdim=True) + module.eps)
    raw = F.linear(normalized, module.projection.weight.to(dtype))
    alpha, bias = module.alpha.to(dtype), module.bias.to(dtype)
    pre = (alpha[0] * raw[..., :s] + bias[:s]).sigmoid()
    post = 2 * (alpha[1] * raw[..., s:2*s] + bias[s:2*s]).sigmoid()
    logits = (alpha[2] * raw[..., 2*s:] + bias[2*s:]).reshape(*streams.shape[:2], s, s)
    mixing = (logits - logits.amax((-2, -1), keepdim=True)).exp()
    for _ in range(module.sinkhorn_iters):
        mixing = mixing / mixing.sum(-2, keepdim=True)
        mixing = mixing / mixing.sum(-1, keepdim=True)
    branch = (pre[..., None] * value).sum(-2)
    return branch.to(streams.dtype), (pre, post, mixing)


def mhc_finish_reference(streams, delta, coefficients):
    _, post, mixing = coefficients
    value = streams.to(mixing.dtype)
    transported = (mixing[..., None] * value.unsqueeze(-3)).sum(-2)
    return (transported + post[..., None] * delta.to(mixing.dtype).unsqueeze(-2)).to(streams.dtype)


def content_readout_reference(streams, vector):
    logits = (streams * vector).sum(-1) / math.sqrt(streams.shape[-1])
    weights = (logits - logits.amax(-1, keepdim=True)).exp()
    weights = weights / weights.sum(-1, keepdim=True)
    return (streams * weights[..., None]).sum(-2)


def _normalize(value,epsilon):
    return value / torch.linalg.vector_norm(value,dim=-1,keepdim=True).clamp_min(epsilon)


def fresh_ssa_reference(module,owners):
    from tacround110.ssa import SSAState
    keys=_normalize(module.initial_keys,module.epsilon)
    values=module.initial_values.tanh()
    count=len(owners)
    return SSAState(tuple(owners),0,keys.unsqueeze(0).repeat(count,1,1),
        values.unsqueeze(0).repeat(count,1,1),
        keys.new_full((count,module.slots),0.5) if module.nsf else None,
        module.config_signature)


def ssa_reference(module,hidden,*,state=None,owners=None,write_mask=None):
    """Explicit-time prior-read/new-write oracle, including omission arms.

    Index selection uses Python's ordered sort on detached finite fixture
    scores; score/value gradients follow the selected fixed indices, as in the
    source top-k graph. This slow CPU-friendly oracle makes no speed claim.
    """
    from tacround110.ssa import SSAState
    owners=tuple(f"ssa-fresh-{i}" for i in range(hidden.shape[0])) if owners is None else tuple(owners)
    state=fresh_ssa_reference(module,owners) if state is None else state
    mask=torch.ones(hidden.shape[:2],dtype=torch.bool,device=hidden.device) if write_mask is None else write_mask
    prior_key=_normalize(module.initial_keys,module.epsilon)
    prior_value=module.initial_values.tanh()
    keys,values,strength=state.keys,state.values,state.survival
    reads=[]
    for time in range(hidden.shape[1]):
        h=hidden[:,time]
        query=_normalize(_linear(h,module.query),module.epsilon)
        scores=(_normalize(keys,module.epsilon)*query[:,None]).sum(-1)/module.temperature
        if module.nsf:
            scores=scores+(strength+module.epsilon).log()
        order=[sorted(range(module.slots),key=lambda j:(-float(row[j]),j))[:module.topk]
               for row in scores.detach().cpu()]
        indices=torch.tensor(order,dtype=torch.long,device=hidden.device)
        selected_scores=torch.stack([row[index] for row,index in zip(scores,indices)])
        selected_values=torch.stack([row[index] for row,index in zip(values,indices)])
        positive=(selected_scores-selected_scores.amax(-1,keepdim=True)).exp()
        probabilities=positive/positive.sum(-1,keepdim=True)
        reads.append((probabilities[...,None]*selected_values).sum(1))
        sparse=torch.zeros_like(scores).scatter(1,indices,probabilities)
        if module.nsf:
            gate=_linear(h,module.survival_query)[:,None]+_linear(keys,module.survival_key)
            alpha=module.fixed_retention+(1-module.fixed_retention)*gate.sigmoid().squeeze(-1)
            proposed_strength=alpha*strength+(1-alpha)*sparse
        else:
            alpha=module.fixed_retention
        if module.hebbian:
            offered_key=_normalize(_linear(h,module.key),module.epsilon)
            offered_value=_linear(h,module.value).tanh()
            eligibility=(_linear(h,module.admission).sigmoid()*
                         _linear(h,module.plasticity).sigmoid())
            rate=eligibility*sparse
            keep=alpha[...,None] if isinstance(alpha,torch.Tensor) else alpha
            base_key=_normalize(keep*keys+(1-keep)*prior_key,module.epsilon)
            base_value=keep*values+(1-keep)*prior_value
            proposed_key=_normalize((1-rate[...,None])*base_key+
                                    rate[...,None]*offered_key[:,None],module.epsilon)
            proposed_value=(1-rate[...,None])*base_value+rate[...,None]*offered_value[:,None]
            keys=torch.where(mask[:,time,None,None],proposed_key,keys)
            values=torch.where(mask[:,time,None,None],proposed_value,values)
        if module.nsf:
            strength=torch.where(mask[:,time,None],proposed_strength,strength)
    residual=_linear(torch.stack(reads,1),module.readout)
    return residual,SSAState(owners,state.clock+hidden.shape[1],keys.clone(),values.clone(),
                            None if strength is None else strength.clone(),module.config_signature)


def _bank_reference(query_source, writer_source, bank, memory, mass, mask):
    c = bank.config
    b, t = query_source.shape[:2]
    q = _linear(query_source, bank.query).reshape(b,t,c.n_heads,c.key_dim).softmax(-1)
    k = _linear(writer_source, bank.key).reshape(b,t,c.n_heads,c.write_rank,c.key_dim).softmax(-1)
    v = _linear(writer_source, bank.value).reshape(b,t,c.n_heads,c.write_rank,c.value_dim).tanh()
    alpha = _linear(writer_source, bank.retention).sigmoid()
    a = torch.where(mask[..., None], alpha, torch.ones_like(alpha))
    w = (1-a) * _linear(writer_source, bank.eligibility).sigmoid()
    return associative_sequential(q,k,v,a,w,memory,mass)


def memory_delta_reference(value, block, previous, mask, clock, *, writer=None):
    """Raw bank equations plus per-query local-attention/time-loop field."""
    c = block.config
    normalized = _rms(value, block.memory_norm)
    write_source = normalized if writer is None else _rms(writer, block.memory_norm)
    fast, fm, fz, fe = _bank_reference(normalized, write_source, block.fast,
        previous.fast_memory, previous.fast_mass, mask)
    slow, sm, sz, se = _bank_reference(normalized, write_source, block.slow,
        previous.slow_memory, previous.slow_mass, mask)
    local, key, local_value = _local_attention(normalized, block,
        previous.recent_key, previous.recent_value, clock)
    pieces = [fast, slow, fast-slow, fast*slow]
    field = previous.rotary
    if c.rotary:
        a = _linear(normalized, block.rotary_retention).sigmoid()[..., None]
        a = torch.where(mask[..., None,None], a, torch.ones_like(a))
        theta = math.pi * _linear(normalized, block.rotary_angle).tanh().reshape(
            *value.shape[:2], c.n_heads, c.value_dim//2)
        theta = torch.where(mask[..., None,None], theta, torch.zeros_like(theta))
        offered = _linear(normalized, block.rotary_offer).tanh().reshape(
            *value.shape[:2], c.n_heads, c.value_dim)
        trajectory = rotary_sequential(a, theta, offered, previous.rotary)
        pieces.append(trajectory)
        field = trajectory[:, -1].clone()
    pieces.extend((local, fe[...,None], se[...,None]))
    delta = block.residual_scale * _linear(torch.cat(pieces,-1).flatten(-2), block.perspective_out)
    return delta, LayerState104(fm,fz,sm,sz,field,key,local_value)


def ffn_delta_reference(value, block):
    normalized = _rms(value, block.ffn_norm)
    return block.residual_scale * _linear(
        _linear(normalized, block.ffn_value) * F.silu(_linear(normalized, block.ffn_gate)),
        block.ffn_out)


def _fresh_ced(model, owners):
    from .ced_model import CEDState
    c, b = model.config, len(owners)
    zeros = model.embedding.weight.new_zeros
    layers = tuple(LayerState104(
        zeros((b,c.n_heads,c.value_dim,c.key_dim)),zeros((b,c.n_heads,c.key_dim)),
        zeros((b,c.n_heads,c.value_dim,c.key_dim)),zeros((b,c.n_heads,c.key_dim)),
        zeros((b,c.n_heads,c.value_dim)),zeros((b,c.n_heads,0,c.key_dim)),
        zeros((b,c.n_heads,0,c.value_dim))) for _ in model.schedule)
    return CEDState(owners,0,layers,model.config_signature,model.schedule)


def ced_reference(model, tokens, *, state=None, owners=None, write_mask=None):
    from .ced_model import CEDState
    owners = tuple(f"fresh-{i}" for i in range(tokens.shape[0])) if owners is None else tuple(owners)
    state = _fresh_ced(model, owners) if state is None else state
    mask = torch.ones_like(tokens,dtype=torch.bool) if write_mask is None else write_mask
    value = F.embedding(tokens,model.embedding.weight)
    context, layers = None, []
    for physical, previous in zip(model.schedule,state.layers):
        block = model.blocks[physical-1]
        writer = context if physical > model.encoder_boundary else None
        delta, updated = memory_delta_reference(value,block,previous,mask,state.clock,writer=writer)
        value = value + delta
        value = value + ffn_delta_reference(value,block)
        layers.append(updated)
        if physical == model.encoder_boundary:
            context = value
    logits = _linear(_rms(value,model.output_norm),model.output)
    return logits,CEDState(owners,state.clock+tokens.shape[1],tuple(layers),
                          model.config_signature,model.schedule)


def round_reference(model, tokens, *, state=None, owners=None, write_mask=None):
    from tacround110.model import RoundState
    owners = tuple(f"fresh-{i}" for i in range(tokens.shape[0])) if owners is None else tuple(owners)
    if state is None:
        native = _fresh_state(model.core,owners)
        sparse = fresh_ssa_reference(model.sparse_memory,owners) if model.config.ssa else None
        state = RoundState(native,sparse,model.config_signature)
    mask = torch.ones_like(tokens,dtype=torch.bool) if write_mask is None else write_mask
    value = F.embedding(tokens,model.core.embedding.weight)
    if model.config.mhc:
        # Explicit contrast expansion instead of the source matrix product.
        offsets = (model.stream_basis[...,None] * model.stream_contrasts[None]).sum(1)
        value = value.unsqueeze(-2) + offsets
    layers, sparse = [], state.ssa
    for index,(block,previous) in enumerate(zip(model.core.blocks,state.layers)):
        if model.config.mhc:
            branch,coeff = mhc_prepare_reference(block.memory_mhc,value)
            delta,updated = memory_delta_reference(branch,block,previous,mask,state.clock)
            value = mhc_finish_reference(value,delta,coeff)
            branch,coeff = mhc_prepare_reference(block.ffn_mhc,value)
            value = mhc_finish_reference(value,ffn_delta_reference(branch,block),coeff)
        else:
            delta,updated = memory_delta_reference(value,block,previous,mask,state.clock)
            value = value+delta
            value = value+ffn_delta_reference(value,block)
        layers.append(updated)
        if model.config.ssa and index+1 == model.config.depth//2:
            query = content_readout_reference(value,model.stream_readout) if model.config.mhc else value
            residual,sparse = ssa_reference(model.sparse_memory,query,state=sparse,
                                            owners=owners,write_mask=mask)
            residual = residual/math.sqrt(2*model.config.depth)
            value = value+(residual.unsqueeze(-2) if model.config.mhc else residual)
    if model.config.mhc:
        value = content_readout_reference(value,model.stream_readout)
    logits = _linear(_rms(value,model.core.output_norm),model.core.output)
    native = State104(owners,state.clock+tokens.shape[1],tuple(layers),model.core.config_signature)
    return logits,RoundState(native,sparse,model.config_signature)
