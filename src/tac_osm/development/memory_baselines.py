"""DEV-018: common public boundary/head; memory interventions only.

TAC here means its native DeltaMemory writer with a controlled common head,
not the full CASM system. Standard gated delta is an equation-level baseline,
not a reproduction of an optimized Gated DeltaNet implementation.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
from .noisy_memory import DeltaMemory, MemoryConfig

ARMS = ("memoryless", "gru768", "delta768", "schema41", "schema41_frozen",
        "tac_joint", "tac_encoder", "tac_memory", "tac_reader", "transformer")


class GatedDelta(nn.Module):
    """S'=alpha*S + beta*(v-alpha*S*k)*k^T; four 8x24 heads."""
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(12, 96)
        self.value = nn.Linear(12, 32)
        self.gates = nn.Linear(12, 8)
        nn.init.zeros_(self.gates.weight)
        with torch.no_grad():
            self.gates.bias[:4].fill_(5.)
            self.gates.bias[4:].zero_()

    def forward(self, state, x):
        k = F.normalize(self.key(x).reshape(-1, 4, 24), dim=-1)
        v = self.value(x).reshape(-1, 4, 8)
        alpha, beta = self.gates(x).sigmoid().chunk(2, -1)
        decayed = alpha[:, :, None, None] * state
        residual = v - torch.einsum("bevk,bek->bev", decayed, k)
        return decayed + beta[:, :, None, None] * residual[..., None] * k[:, :, None]


class MemoryBaseline(nn.Module):
    def __init__(self, arm, seed=0):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("unregistered DEV-018 arm")
        self.arm = arm
        # These tensors are byte-identical across every arm at initialization.
        torch.manual_seed(seed)
        self.event_encoder = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 12))
        self.query_encoder = nn.Linear(14, 12)
        self.head = nn.Sequential(nn.Linear(44, 64), nn.Tanh(), nn.Linear(64, 2))
        self.read_key = nn.Linear(12, 96)
        self.memory = nn.Identity()
        if arm == "memoryless":
            self.adapter = nn.Linear(12, 32)
        elif arm == "gru768":
            self.memory = nn.GRU(12, 768, batch_first=True)
            self.adapter = nn.Linear(768, 32)
        elif arm.startswith("schema"):
            self.memory = nn.ModuleDict({"slots": nn.GRUCell(12, 5), "global": nn.GRUCell(12, 1)})
            self.adapter = nn.Linear(6, 32)
        elif arm == "delta768":
            self.memory = GatedDelta()
        elif arm.startswith("tac"):
            self.memory = DeltaMemory(MemoryConfig())
        else:
            self.memory = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                32, 4, 64, dropout=0., batch_first=True), 1, enable_nested_tensor=False)
            self.token_projection = nn.Linear(12, 32)
            self.cross_query = nn.Linear(12, 32)
            self.cross_attention = nn.MultiheadAttention(32, 4, dropout=0., batch_first=True)
        if arm in ("schema41_frozen", "tac_encoder", "tac_reader"):
            for p in self.memory.parameters():
                p.requires_grad_(False)
        if arm in ("tac_memory", "tac_reader"):
            for p in self.event_encoder.parameters():
                p.requires_grad_(False)
        # read_key exists to make head/query initialization identical; avoid
        # charging inactive trainable parameters or giving them optimizer state.
        if not (arm.startswith("tac") or arm == "delta768"):
            for p in self.read_key.parameters():
                p.requires_grad_(False)

    @property
    def state_scalars(self):
        if self.arm == "transformer":
            return None
        if self.arm == "memoryless":
            return 0  # 12 current-observation features, no historical state
        return 41 if self.arm.startswith("schema") else 768

    def encode(self, events, entities, values):
        if events.ndim != 3 or events.shape[-1] != 16 or events.shape[1] < 1:
            raise ValueError("nonempty public stream required")
        if entities.shape != events.shape[:2] or values.shape != entities.shape:
            raise ValueError("mismatched public stream")
        x = self.event_encoder(events)
        b = events.shape[0]
        if self.arm == "memoryless":
            return x[:, -1]
        if self.arm == "transformer":
            return x  # full encoded history retained; explicitly unbounded
        if self.arm == "gru768":
            return self.memory(x)[1][0]
        state = x.new_zeros((b, 41) if self.arm.startswith("schema") else (b, 4, 8, 24))
        for t in range(x.shape[1]):
            state = self._update_encoded(state, x[:, t], entities[:, t], values[:, t])
        return state

    def _update_encoded(self, state, x, entities, values):
        if self.arm == "memoryless":
            return x
        if self.arm == "transformer":
            return torch.cat((state, x[:, None]), 1)
        if self.arm == "gru768":
            return self.memory(x[:, None], state[None])[1][0]
        if self.arm.startswith("schema"):
            slots = state[:, :40].reshape(-1, 8, 5)
            index = entities.long()[:, None, None].expand(-1, 1, 5)
            previous = slots.gather(1, index).squeeze(1)
            new = self.memory["slots"](x, previous)
            slots = slots.scatter(1, index, new[:, None])
            return torch.cat((slots.flatten(1), self.memory["global"](x, state[:, 40:])), -1)
        if self.arm.startswith("tac"):
            return self.memory.update(state, x, values)
        return self.memory(state, x)

    def update(self, state, features, entities, values):
        return self._update_encoded(state, self.event_encoder(features), entities, values)

    def decode(self, state, queries):
        if queries.shape[1:] != (8, 3, 14):
            raise ValueError("public entity/action/time queries required")
        b = state.shape[0]
        q = self.query_encoder(queries).reshape(b, 24, 12)
        if self.arm in ("memoryless", "gru768"):
            read = self.adapter(state)[:, None].expand(-1, 24, -1)
        elif self.arm.startswith("schema"):
            slots = state[:, :40].reshape(b, 8, 5)
            global_state = state[:, 40:, None].transpose(1, 2).expand(-1, 8, -1)
            read = self.adapter(torch.cat((slots, global_state), -1))
            read = read[:, :, None].expand(-1, -1, 3, -1).reshape(b, 24, 32)
        elif self.arm == "transformer":
            tokens = self.token_projection(state)
            t = torch.arange(tokens.shape[1], device=tokens.device, dtype=tokens.dtype)[:, None]
            frequencies = torch.exp(torch.arange(0, 32, 2, device=tokens.device) * (-math.log(10000.) / 32))
            pos = torch.zeros_like(tokens[0])
            pos[:, 0::2], pos[:, 1::2] = torch.sin(t * frequencies), torch.cos(t * frequencies)
            # Only the observed prefix exists. Bidirectional prefix encoding is
            # allowed for this full-history reference; future tokens never exist.
            context = self.memory(tokens + pos)
            read = self.cross_attention(self.cross_query(q), context, context, need_weights=False)[0]
        else:
            keys = F.normalize(self.read_key(q).reshape(b, 24, 4, 24), dim=-1)
            read = torch.einsum("bevk,bqek->bqev", state, keys).flatten(2)
        return self.head(torch.cat((read, q), -1)).reshape(b, 8, 3, 2)
