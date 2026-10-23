"""ME075 mathematical engineering: selective Gated Delta fast-weight associative state.

Mechanism/test implementation ONLY. Do not call it learned content retrieval or
real-world intelligence before training a complete legal, source-qualified model.
Uses the published gated-delta idea; not a claim of original invention.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def delta_read(memory: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Read value-space state from normalized known query key.

    Memory is [B,V,K], keys are [B,K]. Caller projects event/query encoding.
    """
    if memory.ndim != 3 or key.shape != (memory.shape[0], memory.shape[-1]):
        raise ValueError("expected memory [batch,value,key], key [batch,key]")
    if not torch.isfinite(memory).all() or not torch.isfinite(key).all():
        raise ValueError("nonfinite fast-weight state or key")
    normalized = F.normalize(key, dim=-1, eps=1e-6)
    return torch.bmm(memory, normalized.unsqueeze(-1)).squeeze(-1)


def gated_delta_update(memory: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       retention: torch.Tensor, write: torch.Tensor) -> torch.Tensor:
    """M' = alpha M + beta (v - (alpha M) k) outer k.

    Distinguishes global retention alpha from targeted corrective write beta.
    On a unit key and beta=1, M'k=v regardless of prior stored value.
    Does NOT guarantee learning separable bindings for 16 keys in only 8 dims.
    """
    if memory.ndim != 3 or value.shape != memory.shape[:2] or (
            key.shape != (memory.shape[0], memory.shape[-1])):
        raise ValueError("mismatched fast-weight key/value widths")
    batch = memory.shape[0]
    if retention.shape not in ((batch,), (batch, 1)) or write.shape not in (
            (batch,), (batch, 1)):
        raise ValueError("retention and write gate must be per-entity scalars")
    if (not torch.isfinite(memory).all() or not torch.isfinite(key).all()
            or not torch.isfinite(value).all()
            or not torch.isfinite(retention).all() or not torch.isfinite(write).all()):
        raise ValueError("finite original memory, key, value and gates required")
    if ((retention < 0).any() or (retention > 1).any()
            or (write < 0).any() or (write > 1).any()):
        raise ValueError("gate values must be probabilities in [0,1]")
    k = F.normalize(key, dim=-1, eps=1e-6)
    decayed = memory * retention.reshape(batch, 1, 1)
    error = value - delta_read(decayed, k)
    return decayed + write.reshape(batch, 1, 1) * (error.unsqueeze(-1) * k.unsqueeze(1))


class GatedDelta8x8(nn.Module):
    """Optional learned 8x8 memory-write module, independent from System1."""

    def __init__(self, input_width: int):
        super().__init__()
        if input_width < 1:
            raise ValueError("positive input projection width required")
        self.key = nn.Linear(input_width, 8)
        self.value = nn.Linear(input_width, 8)
        self.control = nn.Linear(input_width, 2)

    def initial(self, batch_size: int, *, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch_size, 8, 8, device=device, dtype=dtype)

    def write(self, memory: torch.Tensor, observed: torch.Tensor,
              write_enabled: torch.Tensor) -> torch.Tensor:
        if observed.ndim != 2 or observed.shape[-1] != self.key.in_features:
            raise ValueError("known observed feature dimension required")
        if write_enabled.shape != (len(observed),) or write_enabled.dtype != torch.bool:
            raise ValueError("per-entity WRITE mask is required")
        if memory.shape != (len(observed), 8, 8):
            raise ValueError("8x8 state required")
        controls = self.control(observed).sigmoid()
        updated = gated_delta_update(memory, self.key(observed), self.value(observed).tanh(),
                                     controls[:, 0], controls[:, 1])
        # Mask query/no-write events AFTER the candidate update so query is never written.
        return torch.where(write_enabled[:, None, None], updated, memory)

    def query(self, memory: torch.Tensor, observed_query: torch.Tensor) -> torch.Tensor:
        if observed_query.ndim != 2 or observed_query.shape[-1] != self.key.in_features:
            raise ValueError("observed query feature shape required")
        return delta_read(memory, self.key(observed_query))
