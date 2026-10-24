"""Batched masked role retrieval and deterministic collision-refusing admission.

Standalone reference implementation, not a drop-in full ES-WAM model rewrite.
"""
import torch
import torch.nn.functional as F

def retrieve(ids, keys, values, occupied, query, role, requested_ids=None):
    valid = occupied[:, :, role] & (ids >= 0)
    if requested_ids is not None:
        valid = valid & (ids == requested_ids[:, None])
    similarity = torch.einsum('bsd,bd->bs', keys, F.normalize(query, dim=-1, eps=1e-6))
    logits = similarity.masked_fill(~valid, -torch.inf)
    hit = valid.any(dim=1)
    logits = torch.where(hit[:, None], logits, torch.zeros_like(logits))
    weights = torch.softmax(logits, dim=-1) * valid.to(query.dtype)
    return torch.einsum('bs,bsd->bd', weights, values[:, :, role, :]), hit

def admission(ids, requested_ids, enabled):
    matches = ids == requested_ids[:, None]
    free = ids == -1
    has_match = matches.any(-1)
    has_free = free.any(-1)
    admitted = enabled & (has_match | has_free)
    slot = torch.where(has_match, matches.int().argmax(-1), free.int().argmax(-1))
    slot_mask = F.one_hot(slot, ids.shape[1]).bool() & admitted[:, None]
    fresh = slot_mask & ~has_match[:, None]
    updated_ids = torch.where(fresh, requested_ids[:, None], ids)
    return updated_ids, slot_mask, admitted
