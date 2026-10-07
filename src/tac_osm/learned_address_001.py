"""Learned content-addressed PLM state for the isolated addressing lane."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .integrated_e2e_005 import (
    AudioEncoder,
    FixedCASM,
    ImageEncoder,
    SharedRepresentation,
    TextEncoder,
)


class LearnedAddressState(nn.Module):
    """Store observations in anonymous slots and retrieve by learned key match."""

    def __init__(
        self,
        hidden: int,
        address_dim: int = 16,
        slots: int = 3,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.address_dim = address_dim
        self.slots = slots
        self.key_encoder = nn.Sequential(
            nn.Linear(address_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(address_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.write_value = nn.Linear(hidden, hidden)

    def initial(self, batch: int, device: torch.device):
        keys = torch.zeros(
            batch, self.slots, self.address_dim, device=device
        )
        values = torch.zeros(batch, self.slots, self.hidden, device=device)
        return keys, values

    def write(
        self,
        memory,
        address_key: Tensor,
        z: Tensor,
        slot: int,
    ):
        keys, values = memory
        slot_key = address_key.to(keys.dtype).view(-1, self.address_dim)
        slot_value = self.write_value(z)
        key_update = keys.clone()
        value_update = values.clone()
        key_update[:, slot, :] = slot_key
        value_update[:, slot, :] = slot_value
        return (key_update, value_update)

    def read(self, memory, address_key: Tensor):
        keys, values = memory
        q = self.query_encoder(address_key.view(-1, self.address_dim))
        k = self.key_encoder(keys)
        scores = torch.einsum("bd,bsd->bs", q, k) / math.sqrt(self.hidden)

        # Unwritten all-zero slots are masked. A slot is written exactly once
        # in this benchmark, so the key norm is a safe occupancy indicator.
        occupied = keys.abs().sum(dim=-1) > 0
        scores = scores.masked_fill(~occupied, float("-inf"))
        attention = scores.softmax(dim=-1)
        read = torch.einsum("bs,bsh->bh", attention, values)
        return read, attention

    def read_oracle(self, memory, slot: Tensor):
        keys, values = memory
        idx = slot.long().view(-1, 1, 1).expand(-1, 1, self.hidden)
        read = values.gather(1, idx).squeeze(1)
        attention = torch.zeros(
            values.shape[0], self.slots, device=values.device
        )
        attention.scatter_(1, slot.long().view(-1, 1), 1.0)
        return read, attention


class LearnedAddressPLM(nn.Module):
    """Multimodal encoder + anonymous state slots + learned key addressing."""

    def __init__(
        self,
        hidden_dim: int = 64,
        address_dim: int = 16,
        slots: int = 3,
        vocab_size: int = 40,
        latent_bits: int = 12,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.address_dim = address_dim
        self.slots = slots
        self.text = TextEncoder(vocab_size, hidden_dim)
        self.image = ImageEncoder(hidden_dim)
        self.audio = AudioEncoder(hidden_dim)
        self.rep = SharedRepresentation(hidden_dim)
        self.state = LearnedAddressState(hidden_dim, address_dim, slots)
        self.casm = FixedCASM(hidden_dim, latent_bits)

    def encode(self, text: Tensor, image: Tensor, audio: Tensor) -> Tensor:
        return self.rep(
            self.text(text),
            self.image(image),
            self.audio(audio),
        )

    def write_observation(
        self,
        memory,
        address_key: Tensor,
        text: Tensor,
        image: Tensor,
        audio: Tensor,
        slot: int,
    ):
        z = self.encode(text, image, audio)
        return self.state.write(memory, address_key, z, slot)

    def query(
        self,
        memory,
        address_key: Tensor,
        i: Tensor,
        j: Tensor,
        op: Tensor,
    ) -> dict[str, Tensor]:
        state_read, attention = self.state.read(memory, address_key)
        action, logits, bits = self.casm(state_read, i, j, op)
        return {
            "read": state_read,
            "attention": attention,
            "action": action,
            "logits": logits,
            "bits": bits,
        }

    def oracle_query(
        self,
        memory,
        slot: Tensor,
        i: Tensor,
        j: Tensor,
        op: Tensor,
    ) -> dict[str, Tensor]:
        state_read, attention = self.state.read_oracle(memory, slot)
        action, logits, bits = self.casm(state_read, i, j, op)
        return {
            "read": state_read,
            "attention": attention,
            "action": action,
            "logits": logits,
            "bits": bits,
        }
