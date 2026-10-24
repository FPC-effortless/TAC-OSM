"""ESWAM013 — source-causal motion evidence gates relevance of long memory.

Native TAC-OSM parallel fast/slow/associative observed-history experiment:
- fast, slow and observed key/value slots are concurrently computed by dense
  tensor operators over original 256 real frames and source-authorized actors;
- recent physical acceleration/velocity changes enrich fast cognition;
- long-term information is conditionally admitted by learned relevance;
- residual correction is physically closed when measured velocity is zero.
No intervention data, action authority or source-independent success claim.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from tac_osm.variants.eswam012_multiagent import Native


class MotionRelevanceNative(Native):
    architecture = "eswam013_kinetic_relevance_parallel_native"

    def __init__(self):
        super().__init__(width=66, memory_width=32, identity_protected=True)
        self.motion_code = nn.Sequential(
            nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 32))
        self.motion_to_fast = nn.Linear(32, 66)
        self.long_trust = nn.Sequential(
            nn.Linear(2 * 66 + 32, 8), nn.SiLU(), nn.Linear(8, 1))
        with torch.no_grad():
            self.long_trust[-1].bias.fill_(-3.0)

    def forward(self, x, prior, active):
        b, t, n, features = x.shape
        if t != 256 or features != 8 or n != 4:
            raise ValueError("original 256 recorded frames and four masked source-ID slots required")
        if prior.shape != (b, n, 2, 2) or active.shape != (b, n):
            raise ValueError("source factual prior or observed actor mask dimension violation")
        z = self.embed(x)
        recent = z[:, -8:]
        fast = (F.softmax(self.fast_score(recent), 1) * recent).sum(1)
        motion = self.motion_code(x[:, -1] - x[:, -2])
        fast = fast + .25 * torch.tanh(self.motion_to_fast(motion))
        time_fraction = torch.arange(t, device=x.device, dtype=x.dtype).view(1, t, 1, 1) / t
        slow = (F.softmax(self.slow_score(z) + 2 * time_fraction, 1) * z).sum(1)
        source_slots = z.reshape(b, 8, 32, n, self.width).mean(2).permute(0, 2, 1, 3)
        keys, values, query = (
            self.key(source_slots), self.value(source_slots), self.query(fast))
        attention = F.softmax(
            (keys * query.unsqueeze(2)).sum(-1) / math.sqrt(keys.shape[-1]), dim=2)
        memory = (attention.unsqueeze(-1) * values).sum(2)
        trust = torch.sigmoid(self.long_trust(torch.cat((fast, slow, motion), -1)))
        slow_gated = fast + trust * (slow - fast)
        evidence = trust * memory
        candidate = self.fuse(torch.cat((fast, slow_gated, evidence), -1))
        previous = torch.cat((fast, fast), -1)
        gate = torch.sigmoid(self.fast_slow(previous))
        state = gate * torch.tanh(candidate) + (1 - gate) * previous
        core = torch.tanh(self.compress(state))
        proposed, scale = self.head(core, x, prior, active)
        physically_measured_speed = torch.linalg.vector_norm(
            x[:, -1, :, 2:4] * 10, dim=-1)
        source_motion_open = (
            physically_measured_speed / (physically_measured_speed + .5)
        )[:, :, None, None]
        mean_future = prior + (proposed - prior) * source_motion_open
        return mean_future, scale
