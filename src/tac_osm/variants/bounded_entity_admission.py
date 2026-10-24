"""Bounded, evidence-gated entity memory admission for TAC model experiments.

Separate optional entity-identity memory, NOT the trained 8x8 source memory
and NOT automatically incorporated into the parallel candidate.

Stable entity_id, observed value, and score must be genuinely known at the
event time. No future outcome, oracle label or implied do(a) effects allowed.

One row is reserved for each currently admitted entity. Insertion preserves
all unrelated rows exactly; if full, optionally replace the lowest unprotected
score only when a strictly higher certified priority clears the threshold.
This budget rule solves the prior global-projection cross-key *mechanism*
but necessarily trades off admission and loss of old identities.

Updates are O(B*S*D), stored values O(B*S*D), independent of episode length.
Slot selection is dynamically state-dependent and so NOT assumed to have
the source candidate's affine time-parallel scan. For a trainable combined
model, use a separately justified parallel-compatible routing schedule or
accept serial dynamic admission: don't silently claim both.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class EntityMemoryState:
    entity_ids: torch.Tensor    # [B,S], -1 means never allocated
    values: torch.Tensor        # [B,S,D]
    priority: torch.Tensor      # [B,S], bounded [0,1]
    protected: torch.Tensor     # [B,S] externally authorized protection


def init_entity_state(batch: int,slots: int=8,width: int=8,*,device=None,dtype=torch.float32):
    if batch<1 or slots<1 or width<1:
        raise ValueError("positive batch, slots, width required")
    return EntityMemoryState(
        torch.full((batch,slots),-1,dtype=torch.int64,device=device),
        torch.zeros((batch,slots,width),dtype=dtype,device=device),
        torch.zeros((batch,slots),dtype=dtype,device=device),
        torch.zeros((batch,slots),dtype=torch.bool,device=device))


def entity_read(state: EntityMemoryState,entity_id: torch.Tensor):
    if entity_id.shape!=(state.entity_ids.shape[0],) or entity_id.dtype!=torch.int64:
        raise ValueError("entity ID must be int64 [B]")
    found=(state.entity_ids==entity_id[:,None])&(entity_id[:,None]>=0)
    output=(state.values*found.unsqueeze(-1)).sum(dim=1)
    return output,found.any(dim=1)


def entity_admit_step(
    state:EntityMemoryState,entity_id:torch.Tensor,observed_value:torch.Tensor,
    relevance:torch.Tensor,writable:torch.Tensor, *,
    beta:float=1.,replacement_margin:float=.1
):
    """Returns (new_state, read_value_after, accepted, replaced).

    No source authentication can be inferred from this API. Caller must
    provide observation-time relevance and evidence verification. Writable
    False has bitwise no state changes. Any missing ID/priority is REJECTED.
    """
    B,S=state.entity_ids.shape
    D=state.values.shape[-1]
    if (entity_id.shape!=(B,) or entity_id.dtype!=torch.int64 or
        observed_value.shape!=(B,D) or relevance.shape!=(B,) or
        writable.shape!=(B,) or writable.dtype!=torch.bool):
        raise ValueError("bad observed entity event shape")
    if (observed_value.device!=state.values.device or
        relevance.device!=state.values.device or
        entity_id.device!=state.values.device or
        writable.device!=state.values.device):
        raise ValueError("device mismatch")
    if not 0<=beta<=1 or replacement_margin<0:
        raise ValueError("invalid beta/replacement margin")
    if (not torch.isfinite(observed_value).all().item()
        or not torch.isfinite(relevance).all().item()
        or torch.any(relevance<0).item() or torch.any(relevance>1).item()
        or torch.any(observed_value.abs()>1).item()):
        raise ValueError("value/relevance requires finite bounded observed data")

    matches=(state.entity_ids==entity_id[:,None])&(entity_id[:,None]>=0)
    found=matches.any(dim=1)
    exists=state.entity_ids.eq(-1)
    has_empty=exists.any(dim=1)
    empty_slot=exists.int().argmax(dim=1)
    match_slot=matches.int().argmax(dim=1)
    # Protected rows are never evicted, however big new priority.
    candidate_score=state.priority.masked_fill(state.protected,float("inf"))
    min_score,min_slot=candidate_score.min(dim=1)
    can_evict=(~state.protected.all(dim=1))&(
        relevance>min_score+replacement_margin)
    accepted=writable & (entity_id>=0) & (
        found | has_empty | can_evict)
    replaced=accepted & (~found) & (~has_empty)
    selected=torch.where(found,match_slot,
                 torch.where(has_empty,empty_slot,min_slot))
    choose=torch.nn.functional.one_hot(selected,num_classes=S).bool()
    write_mask=choose & accepted[:,None]
    new_id=state.entity_ids.clone()
    new_id=torch.where(write_mask,entity_id[:,None].expand_as(new_id),new_id)
    existing_mask=write_mask & found[:,None]
    updated_value=(1-beta)*state.values+beta*observed_value[:,None,:]
    new_values=torch.where(write_mask.unsqueeze(-1),updated_value,state.values)
    new_scores=torch.where(
        write_mask,torch.where(existing_mask,
            torch.maximum(state.priority,relevance[:,None]),
            relevance[:,None]),state.priority)
    # New admissions are not automatically protected.
    new_protected=torch.where(write_mask & (~found[:,None]),
                              torch.zeros_like(state.protected),
                              state.protected)
    after=EntityMemoryState(new_id,new_values,new_scores,new_protected)
    return after,entity_read(after,entity_id)[0],accepted,replaced


def entity_protect(state:EntityMemoryState,entity_id:torch.Tensor,
                   allow:torch.Tensor) -> EntityMemoryState:
    """Protection may be granted/revoked only by an authorized caller."""
    B,S=state.entity_ids.shape
    if (entity_id.shape!=(B,) or allow.shape!=(B,) or allow.dtype!=torch.bool
        or entity_id.dtype!=torch.int64):
        raise ValueError("invalid requested protection")
    hit=state.entity_ids==entity_id[:,None]
    mask=hit & entity_id[:,None].ge(0)
    flags=torch.where(mask,allow[:,None],state.protected)
    return EntityMemoryState(state.entity_ids,state.values,state.priority,flags)
