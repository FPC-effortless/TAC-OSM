"""DEV-012: writer rehearsal with unchanged 768-scalar TAC-OSM state.

This is a development-only self-supervised *training* intervention, not a
Gated DeltaNet-2 implementation or evidence of solved persistent memory.
All evaluation uses visible (address,value) write streams; evaluator targets
enter losses and scoring only, not the recurrent state transition or query.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .protected_memory import ProtectionModel
from .noisy_memory import MemoryConfig


class RehearsalWriterModel(ProtectionModel):
    """Existing TAC-OSM PLM writer/readout/CASM + auxiliary historical-bit head.

    The auxiliary head reads four expert responses from the *bounded* PLM.
    No extra persistent state or externally stored history is consulted at
    inference. Offline training may revisit **observed** past training
    address/value pairs to supervise retrieval; all of these training-only
    reads and their cost must be reported separately.
    """

    def __init__(self, *, mode: str = "always_update",
                 config: MemoryConfig = MemoryConfig()) -> None:
        super().__init__(config=config, mode=mode)
        self.rehearsal_head = nn.Sequential(
            nn.Linear(config.experts * config.value_dim + config.address_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def rehearsal_logits(self, state: Tensor, addresses: Tensor) -> Tensor:
        c=self.config
        if (state.ndim!=4 or state.shape[1:]!=(c.experts,c.value_dim,c.key_dim)
                or addresses.shape!=(state.shape[0],c.address_dim)):
            raise ValueError("state/address dimensions incompatible")
        if not torch.isfinite(state).all() or not torch.isfinite(addresses).all():
            raise ValueError("nonfinite memory probe inputs")
        reads=torch.einsum("bevk,bek->bev",state,self.memory.keys(addresses))
        features=torch.cat((reads.flatten(1),addresses),dim=-1)
        return self.rehearsal_head(features)


def make_matching_models(seed: int) -> dict[str,RehearsalWriterModel]:
    """Byte-identical common initial tensors across all writer variants."""
    modes={
        "task_only":"always_update",
        "rehearsal":"always_update",
        "shuffled_rehearsal":"always_update",
        "fixed_rehearsal":"fixed_protection",
        "learned_rehearsal":"learned_protection",
    }
    arms={}
    first=None
    for name,mode in modes.items():
        torch.manual_seed(seed)
        model=RehearsalWriterModel(mode=mode)
        current=model.state_dict()
        if first is None:
            first={k:v.detach().clone() for k,v in current.items()}
        elif set(current)!=set(first) or any(
                not torch.equal(v,first[k]) for k,v in current.items()):
            raise AssertionError("writer variants start with different weights")
        arms[name]=model
    return arms
