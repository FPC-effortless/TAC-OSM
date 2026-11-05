"""CSL-125 independent numpy references: bounded causal GQA and EMA.

Verification here is only at the *operator* level. It is not a whole-model
formal/source-equivalence proof, trained-memory test or learned key binding.
"""
from __future__ import annotations

import math
import numpy as np


def legal_keys(query_position: int, key_positions: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError('window must be positive')
    return (key_positions <= query_position) & (key_positions >= query_position - window + 1)


def grouped_local_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                            query_positions: np.ndarray, key_positions: np.ndarray,
                            window: int) -> np.ndarray:
    """Float64 stable reference for [B,Hq,T,D], [B,Hkv,S,D].

    Inputs are already RoPE-transformed. Models with 12 Q / 3 KV heads
    replicate one KV group across four adjacent query heads.
    """
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError('invalid q/k/v shapes')
    b,hq,t,d = q.shape
    if k.shape[0] != b or k.shape[3] != d or hq % k.shape[1] or len(query_positions) != t or len(key_positions) != k.shape[2]:
        raise ValueError('grouped head/time geometry mismatch')
    hkv = k.shape[1]
    heads_per_group = hq // hkv
    out = np.zeros_like(q, dtype=np.float64)
    for bi in range(b):
        for head in range(hq):
            group = head // heads_per_group
            for ti, qp in enumerate(query_positions):
                mask = legal_keys(int(qp), key_positions, window)
                if not bool(mask.any()):
                    raise ValueError('query has no legal keys')
                scores = np.asarray(k[bi, group, mask], np.float64) @ np.asarray(q[bi,head,ti],np.float64) / math.sqrt(d)
                p = np.exp(scores - scores.max())
                p /= p.sum()
                out[bi,head,ti] = p @ np.asarray(v[bi,group,mask],np.float64)
    return out


def serial_ema(writes: np.ndarray, initial: np.ndarray, decay: np.ndarray) -> np.ndarray:
    """[B,T,R,E] exact-recurrence reference at float64 arithmetic precision."""
    if writes.ndim != 4 or initial.shape != (writes.shape[0],)+writes.shape[2:] or decay.shape != writes.shape[2:]:
        raise ValueError('invalid serial EMA layout')
    out = []
    current = np.asarray(initial, np.float64).copy()
    a = np.asarray(decay, np.float64)
    for ti in range(writes.shape[1]):
        current = a[None] * current + (1.0-a[None]) * np.asarray(writes[:,ti],np.float64)
        out.append(current.copy())
    return np.stack(out,axis=1)
