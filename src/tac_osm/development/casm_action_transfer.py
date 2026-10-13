"""DEV-014: action-rule transfer diagnostic through TAC-OSM's real CASM.

The supplied bit distribution is a *controlled diagnostic interface*. In
oracle arms the historical bit is evaluator-provided at BOTH train and test;
this is not a realistic TAC-OSM memory read or production inference.

Non-oracle arms receive ONLY bit probabilities produced by a frozen
DEV-013 TAC-OSM writer + aligned bit head from public past writes/query.
In both cases the ORIGINAL ConditionalExecutor and answer layer act;
there is no hard-coded XOR and no production code replacement.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .casm_readout_alignment import CASMAlignedModel
from .conditional_compute import ConditionalExecutor


class CASMActionAdapter(nn.Module):
    """Same parameter budget for top-1 sparse and top-4 dense execution."""

    def __init__(self, parent: CASMAlignedModel, top_k: int=1):
        super().__init__()
        if top_k not in (1,4):
            raise ValueError("top_k must be 1 or 4")
        d=parent.config.dim
        self.bit_to_query=nn.Linear(4,d)
        self.bit_to_routed=nn.Linear(4,d)
        self.compute=ConditionalExecutor(d,parent.config.operators,top_k,
                                         parent.config.depth,
                                         parent.config.halt_threshold)
        self.compute.load_state_dict(parent.compute.state_dict(),strict=True)
        self.answer=nn.Linear(2*d,2)
        self.answer.load_state_dict(parent.answer.state_dict(),strict=True)
        self.top_k=top_k
        self.last_operator_sample_calls=0

    def forward(self, bit_distribution: Tensor, current: Tensor, *,\n                ablate_compute: bool=False) -> Tensor:
        if (bit_distribution.ndim!=2 or bit_distribution.shape[1]!=2
                or current.shape!=(bit_distribution.shape[0],)):
            raise ValueError("bit probability must be [B,2] and current [B]")
        if (not torch.isfinite(bit_distribution).all()
                or not bool(((current==0)|(current==1)).all())):
            raise ValueError("nonfinite bit or invalid current")
        if ((bit_distribution<0)|(bit_distribution>1)).any() or not torch.allclose(
                bit_distribution.sum(-1),torch.ones_like(current,dtype=bit_distribution.dtype),
                atol=1e-5,rtol=1e-5):
            raise ValueError("invalid bit probability simplex")
        visible=torch.cat((bit_distribution,F.one_hot(current.long(),2).float()),dim=-1)
        q=self.bit_to_query(visible)
        routed=self.bit_to_routed(visible)
        if ablate_compute:
            # Only linear input projections and linear answer remain. With
            # balanced XOR labels this pathway cannot implement the rule.
            computed=routed
            self.last_operator_sample_calls=0
        else:
            computed,_=self.compute(routed,q)
            self.last_operator_sample_calls=sum(
                t.operator_sample_calls for t in self.compute.last_trace)
        return self.answer(torch.cat((computed,q),dim=-1))


@torch.no_grad()
def frozen_memory_posterior(parent: CASMAlignedModel, addresses: Tensor,
                            values: Tensor, address: Tensor) -> Tensor:
    """No oracle role/index/target can enter; only public event stream/query."""
    parent.eval()
    state=parent.encode(addresses,values,trace=False)
    if state.ndim!=4 or state.shape[1:]!=(4,8,24):
        raise AssertionError("persistent state capacity is not 768")
    logits=parent.aligned_historical_logits(state,address)
    return logits.softmax(-1).detach()


def teacher_bit_distribution(bit: Tensor) -> Tensor:
    """Explicit oracle control only; MUST NOT be called by non-oracle arms."""
    if not bool(((bit==0)|(bit==1)).all()):
        raise ValueError("training/evaluation teacher bit must be binary")
    return F.one_hot(bit.long(),2).float()
