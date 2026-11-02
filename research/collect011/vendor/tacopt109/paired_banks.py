"""Head-axis batching for the unchanged TAC104 associative-bank equations.

This helper batches two independent bank calls. It adds no learned parameters,
changes no recurrence or normalization, and makes no unmeasured speed claim.
Finite values and legal gates remain the calling model's boundary contract;
only tensor metadata are checked here, without device-to-host value reads.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .reference.kernels import associative_read


BankInputs = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
BankOutputs = tuple[Tensor, Tensor, Tensor, Tensor]


def _geometry(inputs: BankInputs) -> tuple[int, int, int, int, int, int]:
    if not isinstance(inputs, (tuple, list)) or len(inputs) != 7:
        raise ValueError("bank inputs must be (q,k,v,a,w,memory,mass)")
    if any(not isinstance(value, Tensor) for value in inputs):
        raise ValueError("all bank inputs must be tensors")
    q, k, v, a, w, memory, mass = inputs
    if q.ndim != 4 or k.ndim != 5 or v.ndim != 5:
        raise ValueError("expected q[B,T,H,K], k[B,T,H,R,K], v[B,T,H,R,V]")
    batch, length, heads, key_dim = q.shape
    rank, value_dim = k.shape[-2], v.shape[-1]
    if min(batch, heads, key_dim, rank, value_dim) < 1:
        raise ValueError("batch, heads, rank, key and value dimensions must be positive")
    if k.shape != (batch, length, heads, rank, key_dim):
        raise ValueError("key geometry does not match query geometry")
    if v.shape != (batch, length, heads, rank, value_dim):
        raise ValueError("value geometry does not match key geometry")
    if a.shape != (batch, length, heads) or w.shape != a.shape:
        raise ValueError("gate geometry does not match query geometry")
    if memory.shape != (batch, heads, value_dim, key_dim):
        raise ValueError("memory geometry does not match features")
    if mass.shape != (batch, heads, key_dim):
        raise ValueError("mass geometry does not match features")
    if not q.is_floating_point() or any(
        value.dtype != q.dtype or value.device != q.device for value in inputs
    ):
        raise ValueError("all inputs must share a floating dtype and device")
    return batch, length, heads, key_dim, rank, value_dim


def read_bank_pair(
    fast_inputs: BankInputs,
    slow_inputs: BankInputs,
    *,
    chunk_size: int,
    mode: str,
) -> tuple[BankOutputs, BankOutputs]:
    """Return two independent bank outputs using one larger batched call.

    Each input tuple is ``(q,k,v,a,w,M,z)`` in native104 geometry. The banks
    share batch/time/key/value/rank geometry, dtype and device; their head
    counts may differ. Head concatenation never mixes streams or introduces
    interaction between banks. Outputs are ``(reads,M,z,evidence)`` per bank.

    Final M/z tensors are independently cloned into compact storage, retaining
    their gradient paths. Splitting alone would leave state views pinning the
    other bank's allocation, contrary to the functional carried-state contract.
    """
    fast_geometry = _geometry(fast_inputs)
    slow_geometry = _geometry(slow_inputs)
    if (fast_geometry[:2] + fast_geometry[3:] !=
            slow_geometry[:2] + slow_geometry[3:]):
        raise ValueError("banks must share batch/time/key/value/rank geometry")
    if (fast_inputs[0].dtype != slow_inputs[0].dtype or
            fast_inputs[0].device != slow_inputs[0].device):
        raise ValueError("banks must share floating dtype and device")
    fast_heads, slow_heads = fast_geometry[2], slow_geometry[2]
    joined = tuple(torch.cat((fast, slow), dim=2 if index < 5 else 1)
                   for index, (fast, slow) in enumerate(zip(fast_inputs, slow_inputs)))
    outputs = associative_read(*joined, chunk_size=chunk_size, mode=mode)
    fast_output, slow_output = [], []
    for index, value in enumerate(outputs):
        head_dim = 2 if index in (0, 3) else 1
        fast, slow = value.split((fast_heads, slow_heads), dim=head_dim)
        if index in (1, 2):
            fast = fast.clone(memory_format=torch.contiguous_format)
            slow = slow.clone(memory_format=torch.contiguous_format)
        fast_output.append(fast)
        slow_output.append(slow)
    return tuple(fast_output), tuple(slow_output)
