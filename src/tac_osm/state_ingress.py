"""Explicit, read-only restoration checks, never source/checkpoint authority.

Call while owning/quiescing the state, before resuming a stream. This is not an
atomic snapshot, serializer, authentication check, proof of causality or security
certificate. It may synchronize tensor checks; never installed in hot forward.
Original model code/configuration remains unchanged. Invalid state is rejected,
not repaired. Learned FP32 weights only in this first source-reviewed boundary.
"""
from math import isfinite

import torch
from tac_osm.csl125 import CSL125State, LocalKVState
from tac_osm.csl125_r1 import CSL125R1, R1LayerState, R1RoleStateBlock
from tac_osm.csl125_r2 import CSL125R2, R2RoleStateBlock
from tac_osm.csl125_r3 import CSL125R3
from tac_osm.csl125_tool_model import CSL125R4
from tac_osm.csl125_memory_precision import PreciseMemoryState, PreciseRoleStateBlock


class StateIngressError(ValueError):
    """Stable code only; no raw values/source secrets in exception text."""


def validate_restored_stream_state(model, state, *, depth, stream_ids, observed_tokens):
    """Check existing memory/conv/KV contract against caller's trusted context.

    Does not infer identity/cutoff from the untrusted state. A passing result
    cannot establish that values actually came from the claimed source/model.
    No model call, cast, detach, reset, repair, mutation, RNG or file/network I/O.
    """
    def require(condition, code):
        if not condition:
            raise StateIngressError(code)

    require(type(model) in (CSL125R1, CSL125R2, CSL125R3, CSL125R4), 'model_version')
    cfg=model.config
    require(type(depth) is int and depth in cfg.rungs, 'depth')
    require(type(observed_tokens) is int and observed_tokens>=0, 'observed_tokens')
    require(type(stream_ids) is tuple and len(stream_ids)>0 and
            all(type(s) is str and s for s in stream_ids) and
            len(set(stream_ids))==len(stream_ids), 'stream_ids')
    require(type(state) is CSL125State and type(state.layers) is tuple and
            type(state.attention) is tuple and len(state.layers)==depth and
            len(state.attention)==depth, 'state_layout')
    require(state.stream_ids==stream_ids and type(state.stream_ids) is tuple, 'stream_ownership')
    require(model.embedding.weight.dtype==torch.float32 and not model.embedding.weight.is_meta,
            'learned_dtype_or_device')
    require(len(model.blocks)>=depth, 'model_layout')
    device=model.embedding.weight.device
    batch=len(stream_ids)
    tensors=0

    def tensor(t, shape, dtype, name):
        nonlocal tensors
        require(type(t) is torch.Tensor and t.layout==torch.strided and not t.is_quantized and
                t.shape==shape and t.dtype==dtype and t.device==device and not t.is_meta,
                name+'_layout')
        require(bool(torch.isfinite(t).all()), name+'_nonfinite')
        tensors+=1

    for i,s in enumerate(state.layers):
        block=model.blocks[i]
        require(type(block) in (R1RoleStateBlock,R2RoleStateBlock,PreciseRoleStateBlock), 'block_version')
        precise=type(block) is PreciseRoleStateBlock
        expected=PreciseMemoryState if precise else R1LayerState
        dtype=torch.float64 if precise else torch.float32
        require(type(s) is expected, 'memory_version')
        eps=cfg.denominator_epsilon
        require(type(eps) in (int,float) and isfinite(eps) and
                torch.finfo(dtype).tiny<=eps<=1e-3, 'epsilon_positive_normal')
        shape=(batch,cfg.role_count,cfg.state_width)
        for name in ('fast','slow','fast_mass','slow_mass'):
            tensor(getattr(s,name),shape,dtype,'memory')
        for n,d in ((s.fast,s.fast_mass),(s.slow,s.slow_mass)):
            require(bool((d>=0).all()), 'mass_negative')
            require(bool((n.abs()<=d).all()), 'numerator_outside_mass')
        tensor(s.conv_tail,(batch,cfg.conv_kernel-1,cfg.width),torch.float32,'conv')
        cache=state.attention[i]
        if i+1 not in cfg.attention_layers:
            require(cache is None, 'unexpected_KV')
            continue
        require(type(cache) is LocalKVState, 'KV_version')
        require(type(cache.position) is int and cache.position==observed_tokens, 'KV_cutoff')
        retained=min(observed_tokens,cfg.attention_window-1)
        kv_shape=(batch,cfg.kv_heads,retained,cfg.width//cfg.query_heads)
        tensor(cache.k,kv_shape,torch.float32,'KV')
        tensor(cache.v,kv_shape,torch.float32,'KV')
    return {'status':'PASS_SCOPED_RESTORATION_CONTRACT','checked_tensors':tensors,
            'depth':depth,'observed_tokens':observed_tokens,
            'source_authentication':'NOT_ESTABLISHED','full_IEEE_AD':'OPEN',
            'independently_qualified':False}
