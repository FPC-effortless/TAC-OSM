"""DEV-015 — TAC-OSM PLM-to-CASM differentiable decision bridge.

Development-only controlled intervention: the online forward receives ONLY
public sequential address/value observations, noisy query and current bit.
The learned 4 x 8 x 24 = 768-scalar PLM is the sole recurrent memory. No
teacher bit, reference slot, task identity, answer or full-history lookup
enters this forward pass. The original TAC-OSM ConditionalExecutor,
operators and answer structure are reused in the action adapter.

The bridge adds a separately trainable input translation and a second
conditional-executor call; it is NOT the original native predict() and is
NOT cost/parameter matched to the unmodified TAC-OSM model.
"""
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from .casm_readout_alignment import CASMAlignedModel
from .casm_action_transfer import CASMActionAdapter

ARMS = ("frozen_task", "reader_aux", "joint_task", "joint_aux",
        "joint_shuffled")


class PLMCASMBridge(nn.Module):
    def __init__(self, reference: CASMAlignedModel, *,
                 mode: str = "joint_aux", top_k: int = 1):
        super().__init__()
        if mode not in ARMS:
            raise ValueError("unregistered experimental arm")
        self.mode = mode
        self.reader = copy.deepcopy(reference)
        self.actor = CASMActionAdapter(reference, top_k=top_k)
        self._set_trainability()

    def _set_trainability(self) -> None:
        # Preserve all pretrained weights, including the fixed synthetic
        # verifier. The original parent is never mutated or optimized.
        for p in self.reader.parameters():
            p.requires_grad_(False)
        if self.mode in ("reader_aux","joint_task","joint_aux","joint_shuffled"):
            for name,p in self.reader.named_parameters():
                if not name.startswith(("memory.", "verifier.")):
                    p.requires_grad_(True)
        if self.mode in ("joint_task","joint_aux","joint_shuffled"):
            for p in self.reader.memory.parameters():
                p.requires_grad_(True)
            # ProtectedMemory in always_update mode has deliberately frozen
            # protection parameters. Keep that parameter contract.
            for p in self.reader.memory.protection.parameters():
                p.requires_grad_(False)

    def forward(self, addresses: Tensor, values: Tensor,
                query: Tensor, current_bit: Tensor,
                *, ablate_compute: bool = False
                ) -> tuple[Tensor,Tensor,Tensor]:
        if (addresses.ndim!=3 or addresses.shape[-1]!=12
                or values.shape!=addresses.shape[:2]
                or query.shape!=(addresses.shape[0],12)
                or current_bit.shape!=(addresses.shape[0],)
                or addresses.shape[1]<1):
            raise ValueError("bad public streaming input shape")
        if not bool(torch.isfinite(addresses).all()
                    and torch.isfinite(query).all()):
            raise ValueError("nonfinite public observations")
        # No oracle target at inference. End-to-end task gradients can
        # propagate through the probability bridge to the PLM writer.
        state=self.reader.encode(addresses,values,trace=False)
        if state.shape!=(addresses.shape[0],4,8,24):
            raise AssertionError("recurrent PLM state changed capacity")
        bit_logits=self.reader.aligned_historical_logits(state,query)
        posterior=bit_logits.softmax(-1)
        action_logits=self.actor(posterior,current_bit,
                                 ablate_compute=ablate_compute)
        return action_logits,bit_logits,posterior

    @torch.no_grad()
    def with_neutral_bit(self, addresses: Tensor, values: Tensor,
                         query: Tensor, current_bit: Tensor) -> Tensor:
        self.eval()
        posterior=torch.full((current_bit.shape[0],2),0.5,
                             device=current_bit.device,dtype=query.dtype)
        return self.actor(posterior,current_bit)
