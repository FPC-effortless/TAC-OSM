"""DEV-013: supervise actual CASM-path readouts rather than an isolated probe.

Development-only: all 5 arms share inherited TAC-OSM 768-scalar PLM,
learned expert selection, conditional structural execution, and answer head.
The extra shared auxiliary head taps CASM outputs, not a bypass decision.
No oracle target/target slot is passed into the online query/state transition.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .conditional_compute import selected_weights
from .noisy_memory import MemoryConfig
from .rehearsal_writer import RehearsalWriterModel

ARMS = ("task_only", "separate_aux", "aligned_aux",
        "aligned_shuffled", "aligned_detached")


class CASMAlignedModel(RehearsalWriterModel):
    """Inherited 768-scalar recurrent state and actual native CASM executor."""

    def __init__(self, config: MemoryConfig = MemoryConfig()) -> None:
        super().__init__(mode="always_update", config=config)
        self.shared_head = nn.Linear(2 * config.dim, 2)
        self.last_alignment_dispatch_calls = 0

    def casm_decision_features(self, state: Tensor, address: Tensor,
                               current_bit: Tensor) -> tuple[Tensor, Tensor]:
        c = self.config
        if (state.ndim != 4 or state.shape[1:] !=
                (c.experts, c.value_dim, c.key_dim)
                or address.shape != (state.shape[0], c.address_dim)
                or current_bit.shape != (state.shape[0],)):
            raise ValueError("invalid memory/public query input")
        if not torch.isfinite(state).all() or not torch.isfinite(address).all():
            raise ValueError("nonfinite memory or query")
        if not bool(((current_bit == 0) | (current_bit == 1)).all()):
            raise ValueError("binary current bit required")
        onehot=F.one_hot(current_bit.long(),2).to(address.dtype)
        query=self.query(torch.cat((address,onehot),dim=-1))
        reads=torch.einsum("bevk,bek->bev",state,self.memory.keys(address))
        views=self.read_projection(reads)
        scores=self.read_selector(torch.cat((views,query[:,None].expand_as(views)),-1)).squeeze(-1)
        indices,weights=selected_weights(scores,c.read_top_k,self.training)
        selected=views.gather(1,indices[...,None].expand(-1,-1,views.shape[-1]))
        routed=(selected*weights.gather(1,indices)[...,None]).sum(1)
        computation,_=self.compute(routed,query)
        self.last_alignment_dispatch_calls=sum(
            t.operator_sample_calls for t in self.compute.last_trace)
        return computation,query

    def predict(self, state: Tensor, address: Tensor, current_bit: Tensor) -> Tensor:
        computation,query=self.casm_decision_features(state,address,current_bit)
        return self.answer(torch.cat((computation,query),dim=-1))

    def aligned_historical_logits(self, state: Tensor, address: Tensor,
                                  *, detach_features: bool = False) -> Tensor:
        """Historical-bit auxiliary CE must pass through actual CASM.

        Training-only association queries contain *only* observed noisy
        addresses. Current bit is neutral and constant (0), because the
        historical bit (not XOR task answer) is the supervised target.
        This head is not the user-facing TAC-OSM answer head.
        """
        zeros=torch.zeros(state.shape[0],device=state.device,dtype=torch.long)
        comp,query=self.casm_decision_features(state,address,zeros)
        features=torch.cat((comp,query),dim=-1)
        if detach_features:
            features=features.detach()
        return self.shared_head(features)


def matching_models(seed: int) -> dict[str,CASMAlignedModel]:
    first=None
    out={}
    for arm in ARMS:
        torch.manual_seed(seed)
        model=CASMAlignedModel()
        current=model.state_dict()
        if first is None:
            first={k:v.detach().clone() for k,v in current.items()}
        elif set(current)!=set(first) or any(
                not torch.equal(v,first[k]) for k,v in current.items()):
            raise AssertionError("arms are not byte-identically initialized")
        out[arm]=model
    return out
