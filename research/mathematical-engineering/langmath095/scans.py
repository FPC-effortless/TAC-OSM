"""Research-only right-affine scans; original model operators stay unchanged.

For matrix states S_next = S @ P + Q, chronological pair composition is
(P1, Q1) followed by (P2, Q2) = (P1 @ P2, Q1 @ P2 + Q2).
The diagonal analogue replaces matrix products by elementwise products.
This is associative over exact reals, not bitwise associative in IEEE floats.

``tree`` recursively scans adjacent pairs and reconstructs intermediate
prefixes: O(T) pair compositions, O(log T) composition depth. ``chunked``
uses the same work-efficient scan inside chunks and on their summaries.
These are reference algorithms, not a claim of faster kernels or full-model
mathematical certification. Input storage and returned prefix states are O(T).
"""
from __future__ import annotations

import importlib
import numpy as np


class _Ops:
    def __init__(self, p, backend):
        if backend not in ("auto", "numpy", "torch"):
            raise ValueError("backend must be auto, numpy or torch")
        self.torch = None
        if backend == "torch" or (backend == "auto" and
                                  type(p).__module__.split(".")[0] == "torch"):
            self.torch = importlib.import_module("torch")
        elif not isinstance(p, np.ndarray):
            raise TypeError("NumPy inputs must be ndarrays")
        self.is_torch = self.torch is not None

    def check_array(self, x, name):
        kind = self.torch.Tensor if self.is_torch else np.ndarray
        if not isinstance(x, kind):
            raise TypeError(name + " must use the selected backend")

    def floating(self, x):
        return x.is_floating_point() if self.is_torch else np.issubdtype(x.dtype, np.floating)

    def zeros(self, shape, like):
        return like.new_zeros(shape) if self.is_torch else np.zeros(shape, dtype=like.dtype)

    def ones(self, shape, like):
        return like.new_ones(shape) if self.is_torch else np.ones(shape, dtype=like.dtype)

    def cat(self, xs, axis=1):
        return self.torch.cat(xs, dim=axis) if self.is_torch else np.concatenate(xs, axis=axis)

    def stack(self, xs, axis=1):
        return self.torch.stack(xs, dim=axis) if self.is_torch else np.stack(xs, axis=axis)

    def where(self, mask, a, b):
        return self.torch.where(mask, a, b) if self.is_torch else np.where(mask, a, b)

    def identity(self, p, shape, matrix):
        if not matrix:
            return self.ones(shape, p)
        d = p.shape[-1]
        if self.is_torch:
            return self.torch.eye(d, dtype=p.dtype, device=p.device).expand(shape)
        return np.broadcast_to(np.eye(d, dtype=p.dtype), shape)


def _validate(p, q, initial, write_mask, matrix, ops):
    ops.check_array(p, "p")
    ops.check_array(q, "q")
    ndim = 4 if matrix else 3
    if p.ndim != ndim or q.ndim != ndim or not ops.floating(p) or not ops.floating(q):
        raise ValueError("floating rank-4 matrix or rank-3 diagonal inputs required")
    if p.dtype != q.dtype or p.shape[:2] != q.shape[:2]:
        raise ValueError("p and q must share dtype, batch and time dimensions")
    if matrix:
        if p.shape[-1] != p.shape[-2] or p.shape[-1] != q.shape[-1]:
            raise ValueError("matrix transition geometry differs")
        state_shape = (p.shape[0], q.shape[-2], p.shape[-1])
    else:
        if p.shape != q.shape:
            raise ValueError("diagonal p and q shapes must match")
        state_shape = (p.shape[0], p.shape[-1])
    if any(d <= 0 for d in state_shape[1:]):
        raise ValueError("state dimensions must be positive")
    if ops.is_torch and p.device != q.device:
        raise ValueError("p and q devices differ")
    if initial is None:
        initial = ops.zeros(state_shape, q)
    else:
        ops.check_array(initial, "initial")
        if initial.shape != state_shape or initial.dtype != p.dtype:
            raise ValueError("initial shape or dtype differs")
        if ops.is_torch and initial.device != p.device:
            raise ValueError("initial device differs")
    if write_mask is not None:
        ops.check_array(write_mask, "write_mask")
        bool_dtype = ops.torch.bool if ops.is_torch else np.dtype(bool)
        if write_mask.shape != p.shape[:2] or write_mask.dtype != bool_dtype:
            raise ValueError("write_mask must be boolean [B,T]")
        if ops.is_torch and write_mask.device != p.device:
            raise ValueError("write_mask device differs")
        mask = write_mask.reshape(*write_mask.shape, *([1] * (ndim - 2)))
        p = ops.where(mask, p, ops.identity(p, p.shape, matrix))
        q = ops.where(mask, q, ops.zeros(q.shape, q))
    return p, q, initial


def _combine(lp, lq, rp, rq, matrix):
    if matrix:
        return lp @ rp, lq @ rp + rq
    return lp * rp, lq * rp + rq


def _pair_prefix(p, q, matrix, ops):
    """Inclusive chronological prefix pairs without power-of-two padding."""
    n = p.shape[1]
    if n < 2:
        return p, q
    paired = n // 2
    pp, pq = _combine(p[:, :2*paired:2], q[:, :2*paired:2],
                      p[:, 1:2*paired:2], q[:, 1:2*paired:2], matrix)
    oddp, oddq = _pair_prefix(pp, pq, matrix, ops)
    ep, eq = _combine(oddp[:, :(n-1)//2], oddq[:, :(n-1)//2],
                      p[:, 2::2], q[:, 2::2], matrix)
    evenp = ops.cat((p[:, :1], ep))
    evenq = ops.cat((q[:, :1], eq))

    def interleave(even, odd):
        if n % 2:
            body = ops.stack((even[:, :-1], odd), axis=2)
            body = body.reshape(p.shape[0], n-1, *even.shape[2:])
            return ops.cat((body, even[:, -1:]))
        return ops.stack((even, odd), axis=2).reshape(p.shape[0], n, *even.shape[2:])
    return interleave(evenp, oddp), interleave(evenq, oddq)


def _apply(initial, p, q, matrix):
    return initial[:, None] @ p + q if matrix else initial[:, None] * p + q


def _chunked(p, q, initial, matrix, chunk_size, ops):
    b, t = p.shape[:2]
    size = min(chunk_size, t)
    chunks = (t + size - 1) // size
    padding = chunks * size - t
    if padding:
        p = ops.cat((p, ops.identity(p, (b, padding, *p.shape[2:]), matrix)))
        q = ops.cat((q, ops.zeros((b, padding, *q.shape[2:]), q)))
    localp, localq = _pair_prefix(p.reshape(b*chunks, size, *p.shape[2:]),
                                  q.reshape(b*chunks, size, *q.shape[2:]), matrix, ops)
    summaryp = localp[:, -1].reshape(b, chunks, *p.shape[2:])
    summaryq = localq[:, -1].reshape(b, chunks, *q.shape[2:])
    prefixp, prefixq = _pair_prefix(summaryp, summaryq, matrix, ops)
    ends = _apply(initial, prefixp, prefixq, matrix)
    incoming = ops.cat((initial[:, None], ends[:, :-1]))
    incoming = incoming.reshape(b*chunks, *initial.shape[1:])
    result = _apply(incoming, localp, localq, matrix)
    return result.reshape(b, chunks*size, *q.shape[2:])[:, :t]


def _scan(p, q, initial, write_mask, algorithm, chunk_size, backend, matrix):
    if algorithm not in ("sequential", "tree", "chunked"):
        raise ValueError("algorithm must be sequential, tree or chunked")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    ops = _Ops(p, backend)
    p, q, initial = _validate(p, q, initial, write_mask, matrix, ops)
    if p.shape[1] == 0:
        return q + initial[:, None] * 0
    if algorithm == "sequential":
        state, result = initial, []
        for i in range(p.shape[1]):
            state = state @ p[:, i] + q[:, i] if matrix else state * p[:, i] + q[:, i]
            result.append(state)
        return ops.stack(result)
    if algorithm == "tree":
        pp, pq = _pair_prefix(p, q, matrix, ops)
        return _apply(initial, pp, pq, matrix)
    return _chunked(p, q, initial, matrix, chunk_size, ops)


def matrix_scan(p, q, *, initial=None, write_mask=None, algorithm="chunked",
                chunk_size=32, backend="auto"):
    """Return all inclusive states for P[B,T,D,D], Q[B,T,R,D].

    ``initial`` is [B,R,D], zero by default. A false [B,T] ``write_mask``
    applies the identity transition; source time is not changed. Inputs must
    share floating dtype and backend (and device for torch). Nothing is
    mutated. An empty sequence returns [B,0,R,D]. Finite data is assumed;
    no data-dependent synchronization/finite checking is performed.
    """
    return _scan(p, q, initial, write_mask, algorithm, chunk_size, backend, True)


def diagonal_scan(p, q, *, initial=None, write_mask=None, algorithm="chunked",
                  chunk_size=32, backend="auto"):
    """Elementwise version: P,Q[B,T,D], initial[B,D], result[B,T,D]."""
    return _scan(p, q, initial, write_mask, algorithm, chunk_size, backend, False)
