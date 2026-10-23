"""Native fast/slow TAC with shared addresses and a gated corrective 8x8 store.

An integrated candidate, not a claim of architectural novelty or real skill.
The independent retention and error-write equations follow Gated DeltaNet.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def gated_write(memory, key, value, retention, strength, writable, radius=4.0):
    """Value-by-key orientation; a read-only event preserves the exact store."""
    keep = torch.where(writable, retention, torch.ones_like(retention))
    beta = torch.where(writable, strength, torch.zeros_like(strength))
    decayed = memory * keep.unsqueeze(-1)
    innovation = value - torch.bmm(decayed, key.unsqueeze(-1)).squeeze(-1)
    updated = decayed + (beta * innovation).unsqueeze(-1) * key.unsqueeze(1)
    # Radius four, unlike the old unit Frobenius projection, accommodates four
    # unit values at orthogonal keys without attenuating a correctly stored pair.
    if radius is not None:
        norm = updated.square().sum((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        projected = updated / (norm / radius).clamp_min(1.0)
        updated = torch.where(writable.unsqueeze(-1), projected, memory)
    return updated


class AlignedAssociativeTAC(nn.Module):
    def __init__(self, width=88, decoder=132):
        super().__init__()
        self.width = width
        self.kind_embed = nn.Embedding(3, 48)
        self.key = nn.Embedding(65, 48)
        self.value = nn.Embedding(33, 48)
        self.encoder = nn.Linear(48, 48)
        # Address uses only the observed key ID. The same learned code serves
        # writing and lookup; value and event role cannot rotate it independently.
        self.address = nn.Embedding(65, 8)
        self.content = nn.Embedding(33, 8)
        self.fast_map = nn.Linear(48 + width + 8, 2 * width)
        self.slow_map = nn.Linear(2 * width + 8, 2 * width)
        self.memory_gates = nn.Linear(48 + 2 * width, 2)
        nn.init.zeros_(self.memory_gates.weight)
        with torch.no_grad():
            self.memory_gates.bias.copy_(torch.tensor([math.log(99.0), -2.0]))
        self.read_gain = nn.Parameter(torch.tensor(math.log(10.0)))
        self.readout = nn.Sequential(
            nn.Linear(2 * width + 8 + 48, decoder), nn.SiLU(), nn.Linear(decoder, 32)
        )

    def forward(self, kind, keys, values, return_trace=False):
        encoded = torch.tanh(self.encoder(self.kind_embed(kind) + self.key(keys) + self.value(values)))
        addresses = F.normalize(self.address(keys), dim=-1, eps=1e-6)
        contents = F.normalize(self.content(values), dim=-1, eps=1e-6)
        f = encoded.new_zeros(len(encoded), self.width)
        s = f.clone()
        memory = encoded.new_zeros(len(encoded), 8, 8)
        states, gates = [], []
        for j in range(encoded.shape[1]):
            event, address = encoded[:, j], addresses[:, j]
            recall = torch.bmm(memory, address.unsqueeze(-1)).squeeze(-1)
            proposal, gate = self.fast_map(torch.cat((event, f, recall), -1)).chunk(2, -1)
            f = torch.lerp(f, proposal.tanh(), gate.sigmoid())
            delta, gate = self.slow_map(torch.cat((f, s, recall), -1)).chunk(2, -1)
            s = torch.lerp(s, (s + delta.tanh()).tanh(), .25 * gate.sigmoid())
            retention, strength = self.memory_gates(torch.cat((event, f, s), -1)).sigmoid().chunk(2, -1)
            retention = retention.clamp(max=1.0 - 1e-4)
            memory = gated_write(memory, address, contents[:, j], retention, strength, kind[:, j:j+1] != 2)
            if return_trace:
                states.append(memory)
                gates.append(torch.cat((retention, strength), -1))
        read = torch.bmm(memory, addresses[:, -1].unsqueeze(-1)).squeeze(-1)
        codebook = F.normalize(self.content.weight[1:], dim=-1, eps=1e-6)
        logits = self.readout(torch.cat((f, s, read, encoded[:, -1]), -1))
        logits = logits + self.read_gain.clamp(-2.0, 4.0).exp() * (read @ codebook.T)
        if return_trace:
            return logits, {"addresses": addresses, "memory": torch.stack(states, 1),
                            "gates": torch.stack(gates, 1), "read": read}
        return logits
