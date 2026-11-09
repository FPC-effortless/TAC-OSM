"""Explicit storage ownership for existing CSL states; no learned-law change.

Tensor.detach shares storage. Tensor.clone copies values and preserves AD. This
helper deliberately does NOT detach: attached graphs can retain old allocations,
so the compact-storage benefit is an inference/no-grad contract only. Drop the
old state after replacement to release its backing allocations. This is neither
a checkpoint serializer nor validation of mass/finiteness/source authenticity.
"""
from dataclasses import fields, replace

import torch

from tac_osm.csl001 import CSLState, LayerState
from tac_osm.csl125 import CSL125State, LocalKVState
from tac_osm.csl125_r1 import R1LayerState
from tac_osm.csl125_memory_precision import PreciseMemoryState

_STATES = (CSLState, LayerState, CSL125State, LocalKVState, R1LayerState,
           PreciseMemoryState)


def _map(node, tensor_fn):
    if isinstance(node, torch.Tensor):
        if node.layout != torch.strided or node.device.type == 'meta' or node.is_quantized:
            raise TypeError('only materialized dense unquantized state tensors supported')
        return tensor_fn(node)
    if type(node) in _STATES:
        return replace(node, **{f.name: _map(getattr(node, f.name), tensor_fn) for f in fields(node)})
    if type(node) is tuple:
        return tuple(_map(item, tensor_fn) for item in node)
    if node is None or type(node) in (str, int):
        return node
    raise TypeError('unregistered state type; explicit source review required')


def _check_root(state):
    if type(state) not in (CSLState, CSL125State):
        raise TypeError('expected registered top-level CSL state')


def compact_stream_state(state):
    """Return an independent contiguous copy, preserving values/dtype/device/AD.

    Example, for inference only: ``state = compact_stream_state(state)`` inside
    torch.no_grad(). No global mode, default model behavior or Parameter changes.
    Known state versions/ownership metadata are preserved; unknown subtypes fail
    rather than silently downgrade precision/state version. Strides and storage
    aliases intentionally change. This function performs no finite-value repair.
    """
    _check_root(state)
    return _map(state, lambda t: t.clone(memory_format=torch.contiguous_format))


def stream_state_storage(state):
    """Logical element bytes and unique backing storage, NOT process/peak RAM.

    Shared backing storage is counted once, including bytes outside each view.
    No claim about allocator reservations, autograd, weights or RSS follows.
    """
    _check_root(state)
    tensors = []
    _map(state, lambda t: tensors.append(t) or t)
    storage = {}
    for t in tensors:
        s = t.untyped_storage()
        storage[(str(t.device), s.data_ptr())] = s.nbytes()
    return {'tensor_count': len(tensors),
            'logical_bytes': sum(t.numel() * t.element_size() for t in tensors),
            'retained_storage_bytes': sum(storage.values()),
            'unique_nonempty_storages': sum(size > 0 for size in storage.values()),
            'requires_grad_tensors': sum(t.requires_grad for t in tensors)}
