"""Caller-pinned in-memory integrity, NOT source/independent authentication.

Capture with the generating model/state under exclusive ownership; freeze the
receipt hash outside the mutable restore payload. Regenerating a receipt during
restore cannot prove provenance. No checkpoint/pickle loader, file/network I/O,
signing key, neural call or installation hook. CPU only; O(model+state bytes).
Quiesce all model/state writers throughout capture/verification and continuation.
Hashes bind declared classes/config/tensors, NOT executable-code authenticity,
optimizer/RNG history, raw observations, signatures or arbitrary Python hooks.
"""
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import re
import sys

import torch
from tac_osm.state_ingress import validate_restored_stream_state

SCHEMA='tac-osm.stream-binding.v1'
SCOPE='tensor_config_integrity_only'


class StreamBindingError(ValueError):
    pass


def binding_sha256(payload):
    """Canonical public fingerprint, never a signature/authority receipt."""
    try:
        raw=json.dumps(payload,ensure_ascii=False,sort_keys=True,
                       separators=(',',':'),allow_nan=False).encode('utf-8')
    except (TypeError,ValueError,RecursionError) as exc:
        raise StreamBindingError('binding_format') from exc
    return hashlib.sha256(raw).hexdigest()


def tensor_fingerprint(t):
    """Logical native bytes, excluding strides/backing allocation and AD graph."""
    if (type(t) is not torch.Tensor or t.device.type!='cpu' or
        t.layout!=torch.strided or t.is_quantized or t.is_meta):
        raise StreamBindingError('binding_tensor_layout')
    # Temporary read view only; live state/weights retain their original graph.
    raw=t.detach().contiguous().reshape(-1).view(torch.uint8)
    return {'dtype':str(t.dtype),'shape':list(t.shape),
            'bytes_sha256':hashlib.sha256(memoryview(raw.numpy())).hexdigest()}


def _class_name(x):
    return type(x).__module__+'.'+type(x).__qualname__


def _state_values(node):
    if isinstance(node,torch.Tensor):
        return {'tensor':tensor_fingerprint(node)}
    if is_dataclass(node):
        return {'class':_class_name(node),
                'fields':{f.name:_state_values(getattr(node,f.name)) for f in fields(node)}}
    if type(node) is tuple:
        return [_state_values(v) for v in node]
    if node is None or type(node) in (str,int):
        return node
    raise StreamBindingError('binding_state_type')


def capture_stream_binding(model,state,*,depth,stream_ids,observed_tokens):
    """Fingerprint currently valid owned state; does not establish its origin."""
    if model.embedding.weight.device.type!='cpu':
        raise StreamBindingError('binding_CPU_only')
    validate_restored_stream_state(model,state,depth=depth,stream_ids=stream_ids,
                                   observed_tokens=observed_tokens)
    descriptor={'class':_class_name(model),'training':model.training,
                'blocks':[_class_name(b) for b in model.blocks],
                'config_sha256':binding_sha256(asdict(model.config)),
                'named_parameters_sha256':binding_sha256({n:tensor_fingerprint(p.detach())
                    for n,p in model.named_parameters()}),
                # state_dict omits non-persistent buffers (including RoPE here).
                'named_buffers_sha256':binding_sha256({n:tensor_fingerprint(t)
                    for n,t in model.named_buffers()})}
    return {'schema':SCHEMA,'scope':SCOPE,'byteorder':sys.byteorder,
            'context':{'depth':depth,'stream_ids':list(stream_ids),'observed_tokens':observed_tokens},
            'model':descriptor,'state_sha256':binding_sha256(_state_values(state))}


def verify_stream_binding(model,state,expected,*,expected_binding_sha256,
                          depth,stream_ids,observed_tokens):
    """Reject accidental model/state/context swaps against an external pin.

    Pin custody and generating-model/source authority are caller obligations.
    This does not grant protected TEST/reward/safety/promotion authorization.
    """
    if type(expected_binding_sha256) is not str or not re.fullmatch('[0-9a-f]{64}',expected_binding_sha256):
        raise StreamBindingError('trusted_binding_pin_required')
    if binding_sha256(expected)!=expected_binding_sha256:
        raise StreamBindingError('binding_hash_mismatch')
    keys={'schema','scope','byteorder','context','model','state_sha256'}
    if (type(expected) is not dict or set(expected)!=keys or expected['schema']!=SCHEMA or
        expected['scope']!=SCOPE or expected['byteorder']!=sys.byteorder):
        raise StreamBindingError('binding_format')
    context={'depth':depth,'stream_ids':list(stream_ids),'observed_tokens':observed_tokens}
    if expected['context']!=context:
        raise StreamBindingError('binding_context_mismatch')
    actual=capture_stream_binding(model,state,depth=depth,stream_ids=stream_ids,
                                 observed_tokens=observed_tokens)
    if actual['model']!=expected['model']:
        raise StreamBindingError('model_binding_mismatch')
    if actual['state_sha256']!=expected['state_sha256']:
        raise StreamBindingError('state_binding_mismatch')
    return {'status':'PASS_PINNED_TENSOR_STATE_BINDING','source_authentication':'NOT_ESTABLISHED',
            'executable_code_authentication':'NOT_ESTABLISHED','independently_qualified':False}
