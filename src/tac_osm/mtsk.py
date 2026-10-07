"""Multi-timescale adaptive state mechanisms for the PLM research lane."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SingleAdaptiveTimescaleState(nn.Module):
    """Parameter-matched control: one effective decay/gate shared by 3 banks."""

    def __init__(self, hidden: int, entity_count: int, banks: int = 3) -> None:
        super().__init__()
        if banks != 3 or hidden % banks:
            raise ValueError("MTSK-001 requires three banks and hidden divisible by three")
        self.hidden = hidden
        self.entity_count = entity_count
        self.banks = banks
        self.bank_hidden = hidden // banks
        self.write_value = nn.Linear(hidden, hidden)
        self.write_gate = nn.Linear(hidden, banks)
        self.decay_logits = nn.Parameter(torch.zeros(banks))

    def initial(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.entity_count, self.banks, self.bank_hidden, device=device)

    def effective_decay(self) -> Tensor:
        return torch.sigmoid(self.decay_logits.mean())

    def effective_gate(self, z: Tensor) -> Tensor:
        return torch.sigmoid(self.write_gate(z)).mean(dim=-1, keepdim=True).expand(-1, self.banks)

    def write(
        self, memory: Tensor, z: Tensor, entity: Tensor, strength: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        probs = F.one_hot(entity.long(), num_classes=self.entity_count).to(z.dtype)
        candidate = (self.write_value(z) + z).reshape(z.shape[0], self.banks, self.bank_hidden)
        gate = self.effective_gate(z)
        write_strength = gate * (1.0 - self.effective_decay())
        if strength is not None:
            write_strength = write_strength * strength.view(-1, 1)
        rows = torch.arange(z.shape[0], device=z.device)
        old = memory[rows, entity.long()]
        new = (1.0 - write_strength.unsqueeze(-1)) * old + write_strength.unsqueeze(-1) * candidate
        updated = memory.clone()
        updated[rows, entity.long()] = new
        return updated, probs

    def read(self, memory: Tensor, entity: Tensor) -> tuple[Tensor, Tensor]:
        probs = F.one_hot(entity.long(), num_classes=self.entity_count).to(memory.dtype)
        rows = torch.arange(memory.shape[0], device=memory.device)
        value = memory[rows, entity.long()].reshape(memory.shape[0], self.hidden)
        return value, probs


class MTSKEntityState(nn.Module):
    """Three independently gated, ordered-retention adaptive timescales."""

    def __init__(self, hidden: int, entity_count: int, banks: int = 3) -> None:
        super().__init__()
        if banks != 3 or hidden % banks:
            raise ValueError("MTSK-001 requires three banks and hidden divisible by three")
        self.hidden = hidden
        self.entity_count = entity_count
        self.banks = banks
        self.bank_hidden = hidden // banks
        self.write_value = nn.Linear(hidden, hidden)
        self.write_gate = nn.Linear(hidden, banks)
        self.decay_logits = nn.Parameter(torch.tensor([-1.3862944, 0.0, 1.3862944]))

    def initial(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.entity_count, self.banks, self.bank_hidden, device=device)

    def ordered_decays(self) -> Tensor:
        increments = torch.sigmoid(self.decay_logits)
        d0 = increments[0]
        d1 = d0 + (1.0 - d0) * increments[1]
        d2 = d1 + (1.0 - d1) * increments[2]
        return torch.stack((d0, d1, d2))

    def write(
        self, memory: Tensor, z: Tensor, entity: Tensor, strength: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        probs = F.one_hot(entity.long(), num_classes=self.entity_count).to(z.dtype)
        candidate = (self.write_value(z) + z).reshape(z.shape[0], self.banks, self.bank_hidden)
        gate = torch.sigmoid(self.write_gate(z))
        decay = self.ordered_decays().to(z.dtype).view(1, self.banks)
        write_strength = gate * (1.0 - decay)
        if strength is not None:
            write_strength = write_strength * strength.view(-1, 1)
        rows = torch.arange(z.shape[0], device=z.device)
        old = memory[rows, entity.long()]
        new = (1.0 - write_strength.unsqueeze(-1)) * old + write_strength.unsqueeze(-1) * candidate
        updated = memory.clone()
        updated[rows, entity.long()] = new
        return updated, probs

    def read(self, memory: Tensor, entity: Tensor) -> tuple[Tensor, Tensor]:
        probs = F.one_hot(entity.long(), num_classes=self.entity_count).to(memory.dtype)
        rows = torch.arange(memory.shape[0], device=memory.device)
        value = memory[rows, entity.long()].reshape(memory.shape[0], self.hidden)
        return value, probs
