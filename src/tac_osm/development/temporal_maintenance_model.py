"""DEV-016 learned temporal transfer with real TAC-OSM PLM and CASM.

Adapted interface, not the unmodified native model or hardened planner.
Forward accepts only public features, values, padding mask and action queries.
The sole recurrent policy tensor is [B,4,8,24], 768 scalars per episode.
"""
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from .casm_readout_alignment import CASMAlignedModel

ARMS = ("frozen_writer", "joint_task", "joint_predictive", "joint_shuffled")


class TemporalMaintenanceModel(nn.Module):
    def __init__(self, reference: CASMAlignedModel, mode: str):
        super().__init__()
        if mode not in ARMS:
            raise ValueError("unregistered arm")
        self.mode = mode
        self.reader = copy.deepcopy(reference)
        self.event_map = nn.Linear(14, 12)
        self.query_map = nn.Linear(14, 12)
        self.action_head = nn.Linear(64, 2)
        self.future_head = nn.Linear(64, 2)
        self.predictive_aux_head = nn.Linear(64, 2)
        for p in self.reader.parameters():
            p.requires_grad_(False)
        for module in (self.reader.query, self.reader.read_projection,
                       self.reader.read_selector, self.reader.compute):
            for p in module.parameters():
                p.requires_grad_(True)
        if mode != "frozen_writer":
            for name, p in self.reader.memory.named_parameters():
                if not name.startswith("protection."):
                    p.requires_grad_(True)
        else:
            for p in self.event_map.parameters():
                p.requires_grad_(False)
        self.last_operator_calls = 0
        self.last_state_scalars = 0

    def encode(self, events: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        if (events.ndim != 3 or events.shape[2] != 14 or events.shape[1] < 1 or
                values.shape != events.shape[:2] or mask.shape != values.shape or
                mask.dtype != torch.bool):
            raise ValueError("invalid public event tensors")
        if not torch.isfinite(events).all() or not ((values == 0) | (values == 1)).all():
            raise ValueError("nonfinite features or nonbinary payload")
        addresses = self.event_map(events)
        state = self.reader.memory.initial(events.shape[0], addresses)
        for step in range(events.shape[1]):
            proposed = self.reader.memory.update(state, addresses[:, step], values[:, step])
            state = torch.where(mask[:, step, None, None, None], proposed, state)
        if state.shape[1:] != (4, 8, 24):
            raise AssertionError("persistent state exceeds declared shape")
        self.last_state_scalars = state[0].numel()
        return state

    def decode(self, state: Tensor, queries: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if queries.shape != (state.shape[0], 2, 14) or not torch.isfinite(queries).all():
            raise ValueError("invalid public candidate queries")
        outputs = []
        calls = 0
        for candidate in range(2):
            address = self.query_map(queries[:, candidate])
            zeros = torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
            comp, query = self.reader.casm_decision_features(state, address, zeros)
            outputs.append(torch.cat((comp, query), -1))
            calls += self.reader.last_alignment_dispatch_calls
        self.last_operator_calls = calls
        # Primary future prediction is for waiting, with a second candidate
        # service prediction. Actual action classification is a separate head.
        action = self.action_head(outputs[0])
        # Every arm trains the evaluated future readout on true outcomes.
        # Detaching its features during training prevents this common decoder
        # supervision from becoming a writer intervention. A separate matched
        # auxiliary head provides true/shuffled predictive feature gradients.
        future = torch.stack([self.future_head(x.detach() if self.training else x)
                              for x in outputs], dim=1)
        auxiliary = torch.stack([self.predictive_aux_head(x) for x in outputs], dim=1)
        if not torch.isfinite(action).all() or not torch.isfinite(future).all():
            raise AssertionError("nonfinite model output")
        return action, future, auxiliary

    def forward(self, events: Tensor, values: Tensor, mask: Tensor,
                queries: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.decode(self.encode(events, values, mask), queries)
