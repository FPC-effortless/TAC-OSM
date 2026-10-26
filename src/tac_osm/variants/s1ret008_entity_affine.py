"""S1RET008: source-slot factorized 8x8 bounded observational memory.

All identities and relevance grades must be *known at prediction time* and
authenticated by an independent source reader. A source-assigned fixed slot
(0..7) addresses one 8-vector; the other seven rows do not change at all.
Admission policy uses ONLY observed relevance, never future labels/outcomes.

The chronological row update is affine: M_t[s] = (1-b_t) M_{t-1}[s] +
b_t v_t when source-identity s is currently addressed and admission is
authorized. A segment tree gives O(T*64) work / O(log T) parallel scan depth;
one streaming step uses O(8) row work with fixed 64 float state.

Modes:
 - global: prior standard 8x8 key-addressed TAC (ignores source slot)
 - entity_all: source fixed entity identity, always admits valid observations
 - entity_priority: same identity/features, admits only source-known priority
 - entity_none: source fixed entity identity but never writes (memory ablation)

This is observational only; neither drive replacement effects nor candidate
maintenance actions are identifiable from Backblaze failure snapshots.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from .stable_timescale_world import balanced_elementwise_affine_prefix

SLOTS=8
VALUE_DIM=8
MODES=("global","entity_all","entity_priority","entity_none")
THRESHOLD=0.75

def check_entity_inputs(slot,priority,reference,*,sequence:bool):
    expected=reference.shape[:-1]
    if (slot.shape!=expected or priority.shape!=expected or
        slot.dtype!=torch.int64 or priority.dtype!=reference.dtype or
        slot.device!=reference.device or priority.device!=reference.device or
        not torch.isfinite(priority).all().item() or
        torch.any((priority<0)|(priority>1)).item() or
        torch.any((slot<0)|(slot>=SLOTS)).item()):
        raise ValueError("bad source-authenticated entity slot/observed priority")
    if reference.ndim!=(3 if sequence else 2):
        raise ValueError("event source clock rank mismatch")

def admitted_mask(slot,priority,mode):
    if mode not in MODES:
        raise ValueError("unknown source entity admission policy")
    if mode=="entity_none":
        return priority.new_zeros(slot.shape,dtype=torch.bool)
    if mode=="entity_priority":
        return priority>=THRESHOLD
    return torch.ones_like(slot,dtype=torch.bool)

def entity_affine_prefix(values,gate,slot,priority,mode):
    """Return current inclusive bank [B,T,8,8] and P/Q for independent audit."""
    check_entity_inputs(slot,priority,values,sequence=True)
    if (values.shape[-1]!=VALUE_DIM or gate.shape!=slot.shape or
        gate.dtype!=values.dtype or not torch.isfinite(values).all().item()
        or not torch.isfinite(gate).all().item() or
        torch.any(gate<0).item() or torch.any(gate>1).item()):
        raise ValueError("invalid bounded current entity write proposal")
    B,T,_=values.shape
    admit=admitted_mask(slot,priority,mode)
    one=F.one_hot(slot,num_classes=SLOTS).to(values.dtype)
    frac=one*admit.unsqueeze(-1)*gate.unsqueeze(-1)
    # When source admits nothing, P=1 and Q=0 exactly (strict isolation).
    P=(1-frac[...,None]).expand(-1,-1,-1,VALUE_DIM)
    Q=frac[...,None]*values[:,:,None,:]
    trace=balanced_elementwise_affine_prefix(
        P.reshape(B,T,SLOTS*VALUE_DIM),
        Q.reshape(B,T,SLOTS*VALUE_DIM))
    return trace.reshape(B,T,SLOTS,VALUE_DIM), P, Q

def source_row_read(old_bank,slot):
    if old_bank.ndim==4:
        B,T,S,D=old_bank.shape
        if (S,D)!=(SLOTS,VALUE_DIM) or slot.shape!=(B,T):
            raise ValueError("wrong source row gather")
        choose=F.one_hot(slot,num_classes=SLOTS).to(old_bank.dtype)
        return (old_bank*choose[...,None]).sum(dim=-2)
    if old_bank.ndim==3:
        B,S,D=old_bank.shape
        if (S,D)!=(SLOTS,VALUE_DIM) or slot.shape!=(B,):
            raise ValueError("wrong streaming source row gather")
        choose=F.one_hot(slot,num_classes=SLOTS).to(old_bank.dtype)
        return (old_bank*choose[...,None]).sum(dim=-2)
    raise ValueError("invalid entity memory state rank")

def entity_row_step(bank,values,gate,slot,priority,mode):
    """Strictly preserve all non-addressed rows, including their gradients."""
    if bank.ndim!=3 or bank.shape[-2:]!=(SLOTS,VALUE_DIM):
        raise ValueError("invalid entity bank [B,8,8]")
    check_entity_inputs(slot,priority,values,sequence=False)
    if (values.shape!=(bank.shape[0],VALUE_DIM) or
        gate.shape!=slot.shape or gate.dtype!=values.dtype or
        not torch.isfinite(gate).all().item() or
        torch.any((gate<0)|(gate>1)).item() or
        not torch.isfinite(bank).all().item()):
        raise ValueError("invalid source update")
    admit=admitted_mask(slot,priority,mode)
    mask=F.one_hot(slot,num_classes=SLOTS).to(values.dtype)*admit[:,None]
    frac=mask*gate[:,None]
    next_bank=bank*(1-frac[:,:,None])+frac[:,:,None]*values[:,None,:]
    return next_bank
