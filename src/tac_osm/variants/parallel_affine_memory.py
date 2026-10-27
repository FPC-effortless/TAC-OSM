"""Exact-real algebra and torch-autograd prefix implementation for TAC 8x8
gated-delta memory, given an EXTERNALLY FIXED chronological gate schedule.

This module DOES NOT bypass the source's state dependence or radial projection.
For one step, with M shaped [B,8,8] and 8-vector k,v:

    M_next = M @ P + Q
    P = alpha*(I - beta*k@k.T)
    Q = beta*v@k.T.

A read-only update has P=I and Q=0. The chronology-preserving
composition (Pa,Qa) THEN (Pb,Qb) is:
    (Pa @ Pb, Qa @ Pb + Qb).

This algebra is associative over exact reals; scan is parallel in time
*conditional on knowing all gate/key/value schedules in advance* and with
global Frobenius clipping DISABLED/INACTIVE. In real native TAC, retention
and strength depend on f/s and previously read M. Precomputing those gates
requires the native sequential pass, so this scan is NOT a faster end-to-end
exact time-parallel implementation of the whole original model.

This is an independently testable parallelizable subgraph and a mathematical
kernel for a possible future *explicit-gate, no-clip or budget-aware* variant.
No model training, checkpoint inference, verified real performance, or
physical control is performed here.
"""
from __future__ import annotations
import torch


def _shapes(key, value, retention, strength, writable):
    if key.ndim != 3 or key.shape[-1] != 8 or value.shape != key.shape:
        raise ValueError("key/value must both have shape [batch,time,8]")
    if retention.shape != (*key.shape[:2], 1) or strength.shape != retention.shape:
        raise ValueError("retention and strength must be [batch,time,1]")
    if writable.shape != retention.shape or writable.dtype is not torch.bool:
        raise ValueError("writable must be boolean [batch,time,1]")
    if not all(x.device == key.device and x.dtype == key.dtype
               for x in (value, retention, strength)):
        raise ValueError("floating tensors must share dtype and device")
    if not writable.device == key.device or key.shape[1] == 0:
        raise ValueError("writable device mismatch or empty history")
    if not all(torch.isfinite(x).all().item()
               for x in (key, value, retention, strength)):
        raise ValueError("nonfinite associative schedule")
    if torch.any(retention < 0).item() or torch.any(retention > 1).item() or (
        torch.any(strength < 0).item() or torch.any(strength > 1).item()):
        raise ValueError("gates require [0,1]")


def associative_affine_schedule(key, value, retention, strength, writable):
    """Return (P,Q), [B,T,8,8] for chronology-preserving M@P+Q.

    Gate/mask semantics are exactly source-shaped; address is not secretly
    renormalized here. No projection is applied. Gradient flows through all
    floating inputs; write masking means gradient-zero for inactive steps.
    """
    _shapes(key, value, retention, strength, writable)
    keep=torch.where(writable,retention,torch.ones_like(retention))
    beta=torch.where(writable,strength,torch.zeros_like(strength))
    eye=torch.eye(8,dtype=key.dtype,device=key.device).view(1,1,8,8)
    outer_key=key.unsqueeze(-1)*key.unsqueeze(-2)
    P=keep.unsqueeze(-1)*(eye-beta.unsqueeze(-1)*outer_key)
    Q=beta.unsqueeze(-1)*value.unsqueeze(-1)*key.unsqueeze(-2)
    return P,Q


def parallel_affine_prefix(P,Q, initial=None):
    """Inclusive logarithmic-depth Hillis--Steele associative prefix.

    Parallel *time* work uses batched 8x8 matrix multiplications; O(T log T)
    total small-matrix work and O(log T) reduction stages (not an observed
    hardware speedup guarantee). No per-timestep Python loop exists except
    across log2(T) stages. Output [B,T,8,8].
    """
    if P.ndim != 4 or P.shape[-2:] != (8,8) or Q.shape != P.shape or P.shape[1] < 1:
        raise ValueError("P/Q must be [B,T,8,8] with nonempty T")
    if not all(x.device==P.device and x.dtype==P.dtype for x in (Q,)):
        raise ValueError("P/Q dtype or device mismatch")
    count=P.shape[1]
    gap=1
    while gap<count:
        # New right prefix = old prefix [t-gap] then the current length-gap
        # suffix. Use out-of-place updates to preserve autograd.
        rightP=P[:,gap:]
        rightQ=Q[:,gap:]
        mergedP=P[:,:-gap]@rightP
        mergedQ=Q[:,:-gap]@rightP+rightQ
        P=torch.cat((P[:,:gap],mergedP),1)
        Q=torch.cat((Q[:,:gap],mergedQ),1)
        gap*=2
    if initial is None:
        return Q
    if initial.shape != (P.shape[0],8,8) or initial.dtype != P.dtype or initial.device != P.device:
        raise ValueError("initial state must be [B,8,8] matching P")
    return initial.unsqueeze(1)@P+Q


def sequential_affine_reference(P,Q, initial=None):
    """Independent time-ordered affine reference, projection absent."""
    if P.ndim != 4 or Q.shape != P.shape or P.shape[-2:] != (8,8):
        raise ValueError("P/Q dimension mismatch")
    memory=P.new_zeros(P.shape[0],8,8) if initial is None else initial
    out=[]
    for t in range(P.shape[1]):
        memory=memory@P[:,t]+Q[:,t]
        out.append(memory)
    return torch.stack(out,1)



def work_efficient_affine_prefix(P,Q,initial=None):
    """Inclusive Blelloch-style hierarchical scan for right-affine pairs.

    O(T) 8x8 matrix-multiplication *work* and O(log T) parallel depth over
    chronological affine transforms, with out-of-place batched tensors
    supporting autograd. Time is padded to a power of two with identity
    operators and cropped afterward. The ordering of noncommuting matrices
    is always preserved; only parentheses are regrouped.

    Real-number equivalence to sequential follows from affine monoid
    associativity. Float operations can be differently rounded. Exactly
    preserving read-only source bytes in a full model still requires the
    separate last-written-prefix gather used by full_time_parallel_native.
    """
    if (P.ndim != 4 or P.shape[-2:] != (8,8) or
        Q.shape != P.shape or P.shape[1] < 1 or
        P.dtype != Q.dtype or P.device != Q.device):
        raise ValueError("P/Q must be same dtype/device [B,T,8,8] with T>=1")
    B,T=P.shape[:2]
    padded=1 << (T-1).bit_length()
    eye=torch.eye(8,dtype=P.dtype,device=P.device).view(1,1,8,8)
    if padded>T:
        P=torch.cat((P,eye.expand(B,padded-T,8,8)),1)
        Q=torch.cat((Q,Q.new_zeros(B,padded-T,8,8)),1)

    # Parallel UPSWEEP: each next level summarizes pairs of contiguous
    # chronological blocks. Total blocks processed < T across all levels.
    levels=[(P,Q)]
    while levels[-1][0].shape[1]>1:
        prevP,prevQ=levels[-1]
        evenP,oddP=prevP[:,0::2],prevP[:,1::2]
        evenQ,oddQ=prevQ[:,0::2],prevQ[:,1::2]
        levels.append((evenP@oddP,evenQ@oddP+oddQ))

    # Parallel DOWNSWEEP: incoming transform for left child is its parent
    # prefix, for right child it is (parent prefix THEN left block).
    exclP=eye.expand(B,1,8,8)
    exclQ=Q.new_zeros(B,1,8,8)
    for level in range(len(levels)-2,-1,-1):
        childP,childQ=levels[level]
        leftP,leftQ=childP[:,0::2],childQ[:,0::2]
        rightExclP=exclP@leftP
        rightExclQ=exclQ@leftP+leftQ
        exclP=torch.stack((exclP,rightExclP),dim=2).reshape(B,-1,8,8)
        exclQ=torch.stack((exclQ,rightExclQ),dim=2).reshape(B,-1,8,8)

    inclP=exclP@P
    inclQ=exclQ@P+Q
    if initial is None:
        return inclQ[:,:T]
    if initial.shape!=(B,8,8) or initial.dtype!=P.dtype or initial.device!=P.device:
        raise ValueError("initial must be matching [B,8,8]")
    return (initial[:,None]@inclP+inclQ)[:,:T]

def scan_source_schedules(key,value,retention,strength,writable,initial=None):
    P,Q=associative_affine_schedule(key,value,retention,strength,writable)
    return parallel_affine_prefix(P,Q,initial)


def preserve_nonwrite_prefix(memory, writable):
    """Reuse the last admitted prefix bytes after a zero-initialized scan.

    Identity operators alone give an exact-real identity. Different scan
    groupings can round the same prefix differently at a nonwrite position.
    Gathering the last admitted result makes nonwrites literal copies while
    preserving autograd through admitted prefixes. This helper is for ZERO
    initial memory, as used by the two PAR-MEM-010 batch memory paths.
    """
    if (memory.ndim < 3 or writable.shape != memory.shape[:2] or
        writable.dtype != torch.bool or writable.device != memory.device or
        memory.shape[1] < 1):
        raise ValueError("memory requires nonempty [B,T,...] and boolean [B,T] mask")
    position=torch.arange(memory.shape[1],device=memory.device).expand_as(writable)
    last=torch.where(writable,position,torch.full_like(position,-1)).cummax(1).values
    extra=(1,)*(memory.ndim-2)
    index=last.clamp_min(0).reshape(*last.shape,*extra).expand_as(memory)
    reused=memory.gather(1,index)
    return torch.where((last>=0).reshape(*last.shape,*extra),reused,torch.zeros_like(reused))
