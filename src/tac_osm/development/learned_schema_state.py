"""DEV-017 fully learned recurrent state, no semantic state assignments.

Schema has eight public entity-indexed 5D GRU slots and a 1D GRU global state.
TAC uses the actual 768-scalar delta PLM and sparse CASM. Only returned state
persists; raw histories and evaluator objects are never consulted by decode.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .casm_readout_alignment import CASMAlignedModel

ARMS = ("schema41", "schema41_encoder", "tac_reader", "tac_encoder",
        "tac_memory", "tac_joint")


class MaintenancePredictor(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        if mode not in ARMS:
            raise ValueError("unregistered DEV-017 arm")
        self.mode = mode
        self.schema = mode.startswith("schema")
        self.event_encoder = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 12))
        self.query_encoder = nn.Linear(14, 12)
        if self.schema:
            self.slot_gru = nn.GRUCell(12, 5)
            self.global_gru = nn.GRUCell(12, 1)
            self.readout = nn.Sequential(nn.Linear(18, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
            if mode == "schema41_encoder":
                for module in (self.slot_gru, self.global_gru):
                    for p in module.parameters():
                        p.requires_grad_(False)
        else:
            self.reader = CASMAlignedModel()  # fresh, no privileged pretrained transfer
            for p in self.reader.parameters():
                p.requires_grad_(False)
            for module in (self.reader.query, self.reader.read_projection,
                           self.reader.read_selector, self.reader.compute):
                for p in module.parameters():
                    p.requires_grad_(True)
            if mode in ("tac_memory", "tac_joint"):
                for name, p in self.reader.memory.named_parameters():
                    if not name.startswith("protection."):
                        p.requires_grad_(True)
            if mode in ("tac_reader", "tac_memory"):
                for p in self.event_encoder.parameters():
                    p.requires_grad_(False)
        self.future_head = nn.Linear(64, 2)
        self.last_calls_per_episode: list[int] = []

    @property
    def state_scalars(self):
        return 41 if self.schema else 768

    def initial(self, batch: int, reference: Tensor) -> Tensor:
        return reference.new_zeros((batch, 41) if self.schema else (batch, 4, 8, 24))

    def update(self, state: Tensor, features: Tensor, entities: Tensor, values: Tensor) -> Tensor:
        b = features.shape[0]
        if (features.shape != (b, 16) or entities.shape != (b,) or values.shape != (b,) or
                not torch.isfinite(features).all() or not ((values == 0) | (values == 1)).all() or
                not ((entities >= 0) & (entities < 8)).all()):
            raise ValueError("invalid public observation")
        encoded = self.event_encoder(features)
        if self.schema:
            if state.shape != (b, 41):
                raise ValueError("invalid 41-scalar state")
            slots = state[:, :40].reshape(b, 8, 5)
            index = entities.long()[:, None, None].expand(-1, 1, 5)
            previous = slots.gather(1, index).squeeze(1)
            # Nothing here knows bits, due dates, generations or pending status.
            replacement = self.slot_gru(encoded, previous)
            slots = slots.scatter(1, index, replacement[:, None])
            global_state = self.global_gru(encoded, state[:, 40:])
            return torch.cat((slots.flatten(1), global_state), -1)
        return self.reader.memory.update(state, encoded, values)

    def encode(self, events: Tensor, entities: Tensor, values: Tensor) -> Tensor:
        if (events.ndim != 3 or events.shape[2] != 16 or events.shape[1] < 1 or
                entities.shape != events.shape[:2] or values.shape != entities.shape):
            raise ValueError("nonempty exact-length public stream required")
        state = self.initial(events.shape[0], events)
        for t in range(events.shape[1]):
            state = self.update(state, events[:, t], entities[:, t], values[:, t])
        return state

    def decode(self, state: Tensor, queries: Tensor) -> Tensor:
        b = state.shape[0]
        if queries.shape != (b, 8, 3, 14) or not torch.isfinite(queries).all():
            raise ValueError("eight public entities and three candidate queries required")
        calls = [0] * b
        output = []
        for candidate in range(3):
            projected = self.query_encoder(queries[:, :, candidate].reshape(b * 8, 14))
            if self.schema:
                slots = state[:, :40].reshape(b, 8, 5)
                shared = state[:, 40:, None].transpose(1, 2).expand(-1, 8, -1)
                features = self.readout(torch.cat((slots, shared, projected.reshape(b, 8, 12)), -1))
            else:
                repeated = state[:, None].expand(-1, 8, -1, -1, -1).reshape(b * 8, 4, 8, 24)
                zeros = torch.zeros(b * 8, dtype=torch.long, device=state.device)
                comp, query = self.reader.casm_decision_features(repeated, projected, zeros)
                features = torch.cat((comp, query), -1).reshape(b, 8, 64)
                for trace in self.reader.compute.last_trace:
                    for _, samples in trace.dispatched:
                        for sample in samples:
                            calls[sample // 8] += 1
            output.append(self.future_head(features))
        self.last_calls_per_episode = calls
        result = torch.stack(output, dim=2)  # [B,entity,wait/service/current,bit]
        if not torch.isfinite(result).all():
            raise AssertionError("nonfinite forecast")
        return result

    def forward(self, events: Tensor, entities: Tensor, values: Tensor, queries: Tensor) -> Tensor:
        return self.decode(self.encode(events, entities, values), queries)
