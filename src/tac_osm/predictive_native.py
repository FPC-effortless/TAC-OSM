"""Optional Torch native TAC research core with bounded query-conditioned readout.

Importing tac_osm does not import this optional numerical module. This preserves
the established nonlinear fast/slow recurrence; it is not a linear SSM or LLM.
"""
from __future__ import annotations

import math
import torch
from torch import nn


class PredictiveNative(nn.Module):
    __constants__ = ["history", "action_query", "gain"]

    def __init__(self, action_query: bool, duration_mean: float,
                 duration_scale: float, history: int = 128):
        super().__init__()
        if history != 128 or duration_scale <= 0:
            raise ValueError("registered native geometry/normalization required")
        self.history = history
        self.action_query = action_query
        self.gain = 4.0
        # Order matches the original ActionMemory initializer exactly.
        self.emb = nn.Embedding(16000, 8)
        self.enc = nn.Linear(11, 32)
        self.transition = nn.Linear(32, 32, bias=False)
        self.gate = nn.Linear(43, 32)
        self.slow = nn.Linear(64, 32)
        self.slow_gate = nn.Linear(64, 32)
        self.head = nn.Linear(73, 1)
        nn.init.constant_(self.gate.bias, math.log(1/7))
        nn.init.constant_(self.slow_gate.bias, math.log(1/23))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.interaction = nn.Parameter(torch.zeros(64, 8))
        self.register_buffer("duration_mean", torch.tensor(duration_mean, dtype=torch.float32))
        self.register_buffer("duration_scale", torch.tensor(duration_scale, dtype=torch.float32))

    def bounded_matrix(self) -> torch.Tensor:
        return self.gain*torch.tanh(self.interaction)/math.sqrt(512.)

    @torch.jit.export
    def readout(self, state: torch.Tensor, action: torch.Tensor,
                duration: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        a = self.emb(action)
        q = torch.tanh(a if self.action_query else state[:, :8])/math.sqrt(8.)
        residual = (((state/8.) @ self.bounded_matrix())*q).sum(-1, keepdim=True)
        d = (duration-self.duration_mean)/self.duration_scale
        return self.head(torch.cat((state, a, d), -1)) + torch.logit(prior) + residual

    @torch.jit.export
    def encode(self, items: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.emb(items), metadata), -1)
        z = x.new_zeros((items.size(0), 32))
        s = torch.zeros_like(z)
        for t in range(self.history):
            u = x[:, t]
            proposal = torch.tanh(self.enc(u)+self.transition(z))
            g = torch.sigmoid(self.gate(torch.cat((u, z), -1)))
            z = (1.-g)*z+g*proposal
            ctx = torch.cat((z, s), -1)
            gs = torch.sigmoid(self.slow_gate(ctx))
            s = (1.-gs)*s+gs*torch.tanh(self.slow(ctx))
        return torch.cat((z, s), -1)

    def forward(self, items: torch.Tensor, metadata: torch.Tensor,
                action: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        state = self.encode(items, metadata)
        prior = (metadata[:, -32:, 1].sum(-1, keepdim=True)+1.)/34.
        return self.readout(state, action, duration, prior)

    @torch.jit.export
    def prefix_logits(self, items: torch.Tensor, metadata: torch.Tensor,
                      action: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        """Predict observations32/64/96 BEFORE consuming them, then the target."""
        x = torch.cat((self.emb(items), metadata), -1)
        z = x.new_zeros((items.size(0), 32))
        s = torch.zeros_like(z)
        out = torch.jit.annotate(list[torch.Tensor], [])
        for t in range(self.history):
            u = x[:, t]
            proposal = torch.tanh(self.enc(u)+self.transition(z))
            g = torch.sigmoid(self.gate(torch.cat((u, z), -1)))
            z = (1.-g)*z+g*proposal
            ctx = torch.cat((z, s), -1)
            gs = torch.sigmoid(self.slow_gate(ctx))
            s = (1.-gs)*s+gs*torch.tanh(self.slow(ctx))
            depth = t+1
            if depth == 32 or depth == 64 or depth == 96:
                prior = (metadata[:, depth-32:depth, 1].sum(-1, keepdim=True)+1.)/34.
                out.append(self.readout(torch.cat((z, s), -1), items[:, depth],
                                        metadata[:, depth, 0:1], prior))
        prior = (metadata[:, -32:, 1].sum(-1, keepdim=True)+1.)/34.
        out.append(self.readout(torch.cat((z, s), -1), action, duration, prior))
        return torch.cat(out, -1)
