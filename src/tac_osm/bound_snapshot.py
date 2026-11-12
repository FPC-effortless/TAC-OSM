"""CPU inference-only snapshot, never a training-resume/source certificate.

Safetensors only; fixed filenames, no pickle/dynamic imports/model construction
from metadata. Caller provides trusted prototype/context and raw manifest SHA
from outside the mutable folder. Files/model/state must be owned and quiescent.
This is not a hostile-filesystem/atomicity/security qualification. Original
checkpoint formats and model/ingress/binding implementations stay unchanged.
"""
from copy import deepcopy
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import re

import torch
from safetensors.torch import save as encode_tensors, load as decode_tensors
from safetensors import SafetensorError
from tac_osm.state_ingress import validate_restored_stream_state
from tac_osm.compact_stream_state import compact_stream_state
from tac_osm.stream_binding import capture_stream_binding, binding_sha256, verify_stream_binding

SCHEMA='tac-osm.bound-inference-snapshot.v1'
PURPOSE='inference_only_no_optimizer_rng_no_qualification'
MAX_TENSOR_BYTES=64*1024*1024
MAX_MANIFEST_BYTES=1024*1024


class BoundSnapshotError(ValueError):
    pass


def _model_tensors(model):
    return {**{'parameter.'+n:p for n,p in model.named_parameters()},
            **{'buffer.'+n:b for n,b in model.named_buffers()}}


def _state_tensors(state):
    out={f'state.layer.{i}.{f.name}':getattr(s,f.name)
         for i,s in enumerate(state.layers) for f in fields(s)}
    for i,c in enumerate(state.attention):
        if c is not None:
            out[f'state.KV.{i}.k']=c.k;out[f'state.KV.{i}.v']=c.v
    return out


def save_bound_snapshot(folder,model,state,*,depth,stream_ids,observed_tokens):
    """Save a NEW owned folder, return RAW manifest-file SHA; never overwrite.

    Snapshot has parameters, all buffers and inference state only. No optimizer,
    RNG, AD history or observation/source authentication. Freeze the returned pin
    separately before later restore. Memory/file budget is a format limit, not
    permission to train/deploy a model or weaken the operator's parameter policy.
    """
    folder=Path(folder)
    if folder.exists() or folder.is_symlink():
        raise BoundSnapshotError('snapshot_exists')
    if model.training:
        raise BoundSnapshotError('snapshot_requires_eval')
    binding=capture_stream_binding(model,state,depth=depth,stream_ids=stream_ids,observed_tokens=observed_tokens)
    tensors={**_model_tensors(model),**_state_tensors(state)}
    if sum(t.numel()*t.element_size() for t in tensors.values())>MAX_TENSOR_BYTES:
        raise BoundSnapshotError('snapshot_budget')
    # Own each logical value; neither shared storage nor transient backing bytes
    # are serialized. Detachment affects exported views only, not live state.
    payload=encode_tensors({n:t.detach().contiguous().clone() for n,t in tensors.items()})
    if len(payload)>MAX_TENSOR_BYTES:
        raise BoundSnapshotError('snapshot_budget')
    manifest={'schema':SCHEMA,'purpose':PURPOSE,'binding':binding,
              'tensor_sha256':hashlib.sha256(payload).hexdigest(),'tensor_bytes':len(payload)}
    raw=json.dumps(manifest,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>MAX_MANIFEST_BYTES:
        raise BoundSnapshotError('snapshot_budget')
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'tensors.safetensors').write_bytes(payload)
    (folder/'manifest.json').write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _read(path,budget):
    if path.is_symlink():
        raise BoundSnapshotError('snapshot_symlink')
    try:
        with path.open('rb') as f:
            raw=f.read(budget+1)
    except FileNotFoundError as exc:
        raise BoundSnapshotError('snapshot_missing') from exc
    if len(raw)>budget:
        raise BoundSnapshotError('snapshot_budget')
    return raw


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise BoundSnapshotError('manifest_duplicate_key')
        result[key]=value
    return result


def restore_bound_snapshot(folder,prototype,*,expected_manifest_sha256,depth,stream_ids,observed_tokens):
    """Restore to a NEW deepcopy of trusted prototype; caller remains unchanged.

    Raw outer manifest pin authenticates integrity of its nested binding pin,
    not publisher/source authority. Context/config/mode/code come from caller;
    never instantiate a metadata-selected class. CPU only. Uses identical bytes
    for hash verification and Safetensors parsing, with bounded reads.
    """
    if type(expected_manifest_sha256) is not str or not re.fullmatch('[0-9a-f]{64}',expected_manifest_sha256):
        raise BoundSnapshotError('trusted_manifest_pin_required')
    folder=Path(folder)
    if folder.is_symlink():
        raise BoundSnapshotError('snapshot_symlink')
    raw=_read(folder/'manifest.json',MAX_MANIFEST_BYTES)
    if hashlib.sha256(raw).hexdigest()!=expected_manifest_sha256:
        raise BoundSnapshotError('manifest_hash_mismatch')
    try:
        manifest=json.loads(raw,object_pairs_hook=_unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(BoundSnapshotError('manifest_nonfinite')))
    except (ValueError,UnicodeDecodeError) as exc:
        raise BoundSnapshotError('manifest_format') from exc
    if (type(manifest) is not dict or set(manifest)!={'schema','purpose','binding','tensor_sha256','tensor_bytes'} or
        manifest['schema']!=SCHEMA or manifest['purpose']!=PURPOSE or
        type(manifest['tensor_bytes']) is not int or not 0<manifest['tensor_bytes']<=MAX_TENSOR_BYTES):
        raise BoundSnapshotError('manifest_format')
    ctx={'depth':depth,'stream_ids':stream_ids,'observed_tokens':observed_tokens}
    if (type(manifest['binding']) is not dict or manifest['binding'].get('context')!=
        {'depth':depth,'stream_ids':list(stream_ids),'observed_tokens':observed_tokens}):
        raise BoundSnapshotError('snapshot_context')
    payload=_read(folder/'tensors.safetensors',MAX_TENSOR_BYTES)
    if len(payload)!=manifest['tensor_bytes'] or hashlib.sha256(payload).hexdigest()!=manifest['tensor_sha256']:
        raise BoundSnapshotError('tensor_hash_mismatch')
    # Do not allow a large prototype allocation merely because file is small.
    if sum(t.numel()*t.element_size() for t in _model_tensors(prototype).values())>MAX_TENSOR_BYTES:
        raise BoundSnapshotError('snapshot_budget')
    template=prototype.initial_state(len(stream_ids),torch.device('cpu'),torch.float32,
                                     depth=depth,stream_ids=stream_ids)
    validate_restored_stream_state(prototype,template,depth=depth,stream_ids=stream_ids,observed_tokens=0)
    targets={**_model_tensors(prototype),**_state_tensors(template)}
    layouts={n:(tuple(t.shape),t.dtype) for n,t in targets.items()}
    kvshape=(len(stream_ids),prototype.config.kv_heads,
             min(observed_tokens,prototype.config.attention_window-1),
             prototype.config.width//prototype.config.query_heads)
    for i,c in enumerate(template.attention):
        if c is not None:
            layouts[f'state.KV.{i}.k']=(kvshape,c.k.dtype)
            layouts[f'state.KV.{i}.v']=(kvshape,c.v.dtype)
    try:
        data=decode_tensors(payload)  # Standard Safetensors decoder, never pickle/code.
    except SafetensorError as exc:
        raise BoundSnapshotError('tensor_format') from exc
    if set(data)!=set(layouts) or any((tuple(t.shape),t.dtype)!=layouts[n] for n,t in data.items()):
        raise BoundSnapshotError('tensor_layout')
    model=deepcopy(prototype)
    with torch.no_grad():
        for n,t in _model_tensors(model).items():
            t.copy_(data[n])
    layers=tuple(replace(s,**{f.name:data[f'state.layer.{i}.{f.name}'] for f in fields(s)})
                 for i,s in enumerate(template.layers))
    attention=tuple(None if c is None else replace(c,k=data[f'state.KV.{i}.k'],
        v=data[f'state.KV.{i}.v'],position=observed_tokens) for i,c in enumerate(template.attention))
    state=compact_stream_state(replace(template,layers=layers,attention=attention))
    verify_stream_binding(model,state,manifest['binding'],
                          expected_binding_sha256=binding_sha256(manifest['binding']),**ctx)
    return model,state
