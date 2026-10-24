"""Fixed-slot batched memory: original ES-WAM forward semantics, no new weights."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from tac_osm.variants.exchange_world_action import (
    ExchangeMemory, ExchangeState, ExchangeWorldActionModel,
)


class VectorizedExchangeMemory(ExchangeMemory):
    """One actor ID per batch row; roles, collisions and keys are preserved."""

    def retrieve(self, state: ExchangeState, query: torch.Tensor, role: int,
                 requested_ids: torch.Tensor | None = None):
        if role not in range(self.roles) or query.ndim != 2 or query.shape[-1] != self.width:
            raise ValueError("invalid query shape or evidence role")
        b = len(query)
        if state.ids.shape != (b, self.slots):
            raise ValueError("batch/address mismatch")
        if requested_ids is not None and (requested_ids.shape != (b,) or requested_ids.dtype != torch.long):
            raise ValueError("requested identities must be observed integer IDs")
        valid = state.occupied[:, :, role] & (state.ids >= 0)
        if requested_ids is not None:
            valid = valid & (state.ids == requested_ids[:, None])
        hits = valid.any(dim=1)
        sims = torch.einsum("bsd,bd->bs", state.keys,
                            F.normalize(query, dim=-1, eps=1e-6))
        logits = sims.masked_fill(~valid, -torch.inf)
        logits = torch.where(hits[:, None], logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1) * valid.to(sims.dtype)
        result = torch.einsum("bs,bsd->bd", weights, state.values[:, :, role])
        return result, hits

    def write(self, state: ExchangeState, ids: torch.Tensor, role: int,
              keys: torch.Tensor, values: torch.Tensor, enabled: torch.Tensor):
        b, d = values.shape
        if role not in range(self.roles):
            raise ValueError("unknown evidence role")
        if (ids.shape != (b,) or ids.dtype != torch.long or (ids < 0).any()
                or keys.shape != (b, d) or d != self.width
                or enabled.shape != (b,) or enabled.dtype != torch.bool
                or state.ids.shape != (b, self.slots)
                or state.values.shape != (b, self.slots, self.roles, d)
                or state.keys.shape != (b, self.slots, d)
                or state.occupied.shape != (b, self.slots, self.roles)):
            raise ValueError("invalid memory state, identity or write shape")
        if not (torch.isfinite(keys).all() and torch.isfinite(values).all()):
            raise ValueError("nonfinite memory write")
        matched = state.ids == ids[:, None]
        vacant = state.ids == -1
        has_match, has_vacant = matched.any(1), vacant.any(1)
        admitted = enabled & (has_match | has_vacant)
        slot = torch.where(has_match, matched.long().argmax(1), vacant.long().argmax(1))
        selected = F.one_hot(slot, self.slots).bool() & admitted[:, None]
        allocated = selected & ~has_match[:, None]
        new_ids = torch.where(allocated, ids[:, None], state.ids)
        new_keys = torch.where(allocated[..., None],
                               F.normalize(keys, dim=-1, eps=1e-6)[:, None, :],
                               state.keys)
        old = state.values[torch.arange(b, device=values.device), slot, role]
        present = state.occupied[torch.arange(b, device=values.device), slot, role]
        # A fully empty merge batch must NOT connect exchange parameters to
        # the autograd graph: original loop leaves their grad=None. Explicit
        # zero gradients change AdamW/weight-decay and momentum semantics.
        # One batch-level decision replaces O(batch) host-side branches.
        if bool((present & admitted).any()):
            symmetric, antisymmetric = self.exchange(old, values)
            gate = self.merge_gate(torch.cat((old + symmetric, values + antisymmetric), -1)).sigmoid()
            merged = torch.lerp(old, values, gate)
            proposed = torch.where(present[:, None], merged, values)
        else:
            proposed = values
        role_mask = F.one_hot(torch.tensor(role, device=values.device), self.roles).bool()
        touched = selected[:, :, None] & role_mask[None, None, :]
        new_values = torch.where(touched[..., None], proposed[:, None, None, :], state.values)
        new_occupied = state.occupied | touched
        return ExchangeState(new_ids, new_keys, new_values, new_occupied), admitted


class VectorizedExchangeWorldActionModel(ExchangeWorldActionModel):
    """Entire original world/action model with only memory tensor ops replaced."""
    architecture = "eswam011_original_graph_vectorized_memory_candidate"

    def __init__(self, dimension=14, action_dim=14, width=90,
                 memory_width=16, slots=8, state_scale=None):
        super().__init__(dimension, action_dim, width, memory_width, slots, state_scale)
        self.memory = VectorizedExchangeMemory(memory_width, slots)
