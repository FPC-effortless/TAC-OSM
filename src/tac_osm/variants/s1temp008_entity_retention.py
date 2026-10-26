"""S1TEMP008 source-authorized entity-indexed fixed 8x8 associative memory.

The column index is supplied by a VERIFIED original event-entity identifier
within a CASE. Slot0 = application/global event, slots1..7 = distinct offer IDs
assigned IN ORDER OF FIRST PAST OBSERVATION. No string hashes, modulo collision,
future label, test-set identifier or inferred true action intervention enters
the model.

Same 64-FP32-float 8x8 matrix as the original TAC architecture.
Input entity slots are trusted only after external source-time auditing.

The associative memory admits information using the existing learned gates,
where strength = beta*(1-alpha); ONE indexed column is modified and all
unaddressed columns are unchanged in exact arithmetic.

This is a source-indexed memory KERNEL, not yet a trained comparative model or
proof of the full BPI process predictor. The user must not confuse event IDs
with discrete future intervention authority or a safety guarantee.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from .parallel_affine_memory import work_efficient_affine_prefix

ENTITY_SLOTS=8
SLOT_CAPACITY=7
def validate_event_slots(slot:torch.Tensor,shape:tuple[int,...])->None:
    if (slot.shape!=shape or slot.dtype!=torch.int64
        or slot.ndim not in (1,2)
        or not torch.all((slot>=0)&(slot<ENTITY_SLOTS)).item()):
        raise ValueError("requires trusted integer source offer slots in [0,7]")
    # Python caller must attest as-of source identity; do NOT infer it from
    # a future offer, true label, target, or a global hashed slot.

def indexed_affine_schedule(
    entity_slot:torch.Tensor, value:torch.Tensor,
    raw_retention:torch.Tensor, raw_strength:torch.Tensor
):
    """Create per-time local-column affine schedules for causal parallel scan.

    value [B,T,8], gates [B,T,1], actual source slot [B,T].
    M_next = M + gamma*(v - M[:, :, entity_slot]) e_slot.T
    gamma=beta*(1-alpha) in [0,1]. Other columns EXACTLY unchanged.
    """
    if value.ndim!=3 or value.shape[-1]!=ENTITY_SLOTS or value.shape[1]<1:
        raise ValueError("nonempty observed source values [B,T,8] required")
    B,T,_=value.shape
    validate_event_slots(entity_slot,(B,T))
    if (entity_slot.device!=value.device
        or raw_retention.shape!=(B,T,1)
        or raw_strength.shape!=(B,T,1)
        or any(x.dtype!=value.dtype or x.device!=value.device or
               not torch.isfinite(x).all().item()
               for x in (raw_retention,raw_strength,value))):
        raise ValueError("shape/dtype/source mismatch for indexed events")
    if (torch.any(raw_retention<0).item() or torch.any(raw_retention>1).item()
        or torch.any(raw_strength<0).item() or torch.any(raw_strength>1).item()):
        raise ValueError("retention/strength must be bounded real gates")
    k=F.one_hot(entity_slot,ENTITY_SLOTS).to(dtype=value.dtype)
    gamma=raw_strength*(1-raw_retention)
    identity=torch.eye(ENTITY_SLOTS,dtype=value.dtype,device=value.device)
    outer=k.unsqueeze(-1)*k.unsqueeze(-2)
    P=identity.view(1,1,8,8)-gamma.unsqueeze(-1)*outer
    Q=gamma.unsqueeze(-1)*value.unsqueeze(-1)*k.unsqueeze(-2)
    return P,Q,k

def indexed_affine_parallel(
    entity_slot:torch.Tensor, value:torch.Tensor,
    retention:torch.Tensor,strength:torch.Tensor):
    P,Q,k=indexed_affine_schedule(entity_slot,value,retention,strength)
    return work_efficient_affine_prefix(P,Q),k

def indexed_affine_step(
    memory:torch.Tensor, entity_slot:torch.Tensor, value:torch.Tensor,
    retention:torch.Tensor,strength:torch.Tensor):
    """Read ONLY past memory, then update one source-authorized indexed slot."""
    if memory.ndim!=3 or memory.shape[-2:]!=(8,8):
        raise ValueError("invalid original fixed memory state")
    B=memory.shape[0]
    if (value.shape!=(B,8) or retention.shape!=(B,1)
        or strength.shape!=(B,1) or entity_slot.shape!=(B,)):
        raise ValueError("original indexed step shape mismatch")
    P,Q,k=indexed_affine_schedule(
        entity_slot[:,None],value[:,None],
        retention[:,None],strength[:,None])
    previous_read=(memory@k[:,0,:,None]).squeeze(-1)
    updated=(memory@P[:,0])+Q[:,0]
    return updated,previous_read
