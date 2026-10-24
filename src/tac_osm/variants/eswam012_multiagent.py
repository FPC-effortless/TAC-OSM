"""Native parallel fast/slow/associative 256-history observed-world predictor.

Matched capacity GRU and causal attention rival models. No observed action
authority or causal policy claim. Source IDs define entity axes; no identity
inference and no future target, velocity or timestamp enters input.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class PhysicalHead(nn.Module):
    """Shared, learned bounded correction to the strong measured velocity prior."""
    def __init__(self, width: int):
        super().__init__()
        self.actor_fusion = nn.Sequential(
            nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width))
        self.proj = nn.Sequential(
            nn.Linear(width + 8, width), nn.SiLU(), nn.Linear(width, 12))
        # All architectures initially predict EXACT measured-velocity forecast.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, core, features, prior, active):
        b, n, _ = core.shape
        core = core * active[..., None]
        denom = (active.sum(1, keepdim=True) - 1).clamp_min(1)[..., None]
        others = (core.sum(1, keepdim=True) - core) / denom
        fused = core + .25 * torch.tanh(
            self.actor_fusion(torch.cat((core, others.expand_as(core)), -1)))
        output = self.proj(torch.cat((fused, features[:, -1]), -1))
        residual = torch.tanh(output[..., :4]).reshape(b, n, 2, 2)
        gate = torch.sigmoid(output[..., 4:8]).reshape(b, n, 2, 2)
        scale = F.softplus(output[..., 8:12]).reshape(b, n, 2, 2) + .01
        cap_meters = prior.new_tensor([2., 6.]).view(1, 1, 2, 1)
        return prior + residual * gate * cap_meters, scale


class Native(nn.Module):
    """TAC-OSM native observed-evidence exchange-memory experimental variant.

    All real 256-frame actor histories are source-recorded and entity-axis
    indexed. Fast last-eight selection, long-span slow selection and eight
    slot key/value reads are computed in parallel. Fast/slow gated fusion,
    source-actor information exchange, and physical residual all train jointly.

    identity_protected=False disables SAME-source-actor slot read masking for
    a controlled identity-addressing ablation (NOT a physical collision).
    """
    def __init__(self, width: int = 68, memory_width: int = 32,
                 identity_protected: bool = True):
        super().__init__()
        self.width = width
        self.identity_protected = identity_protected
        self.embed = nn.Sequential(nn.Linear(8, width), nn.SiLU(),
                                   nn.Linear(width, width), nn.SiLU())
        self.fast_score = nn.Linear(width, 1)
        self.slow_score = nn.Linear(width, 1)
        self.key = nn.Linear(width, memory_width)
        self.value = nn.Linear(width, memory_width)
        self.query = nn.Linear(width, memory_width)
        self.fuse = nn.Sequential(
            nn.Linear(2 * width + memory_width, 2 * width),
            nn.SiLU(), nn.Linear(2 * width, 2 * width))
        self.fast_slow = nn.Linear(2 * width, 2 * width)
        self.compress = nn.Linear(2 * width, width)
        self.head = PhysicalHead(width)

    def forward(self, x, prior, active):
        b, t, n, features = x.shape
        if features != 8 or t not in (8, 256) or n != 4:
            raise ValueError("actual source actor features must use 8/256 observed frames and four maskable slots")
        z = self.embed(x)
        recent = z[:, -8:]
        fast = (torch.softmax(self.fast_score(recent), 1) * recent).sum(1)
        past_fraction = torch.arange(t, device=z.device, dtype=z.dtype).view(1, t, 1, 1) / t
        slow = (torch.softmax(self.slow_score(z) + 2 * past_fraction, 1) * z).sum(1)
        # Source-grounded time blocks: 256 frames -> 8x32; 8-frame ablation -> 8x1.
        observed_slots = z.reshape(b, 8, t // 8, n, self.width).mean(2).permute(0, 2, 1, 3)
        keys = self.key(observed_slots)
        values = self.value(observed_slots)
        query = self.query(fast)
        if self.identity_protected:
            # Each real actor reads its own observed history only.
            weights = torch.softmax(
                (keys * query.unsqueeze(2)).sum(-1) / math.sqrt(keys.shape[-1]), dim=2)
            remembered = (weights.unsqueeze(-1) * values).sum(2)
        else:
            # Controlled, explicit unscoped retrieval; still excludes padded actors.
            flattened_keys = keys.reshape(b, n * 8, -1)
            flattened_values = values.reshape(b, n * 8, -1)
            logits = torch.einsum("bnm,bkm->bnk", query, flattened_keys) / math.sqrt(keys.shape[-1])
            source_valid = active.bool()[:, :, None].expand(b, n, 8).reshape(b, n * 8)
            weights = torch.softmax(logits.masked_fill(~source_valid[:, None], -torch.inf), -1)
            remembered = torch.einsum("bnk,bkm->bnm", weights, flattened_values)
        candidate = self.fuse(torch.cat((fast, slow, remembered), -1))
        prior_state = torch.cat((fast, slow), -1)
        gate = torch.sigmoid(self.fast_slow(prior_state))
        state = gate * torch.tanh(candidate) + (1 - gate) * prior_state
        core = torch.tanh(self.compress(state))
        return self.head(core, x, prior, active)


class GRU(nn.Module):
    """Equally source-informed standard two-layer recurrent control."""
    def __init__(self, width: int = 78):
        super().__init__()
        self.embed = nn.Linear(8, width)
        self.gru = nn.GRU(width, width, num_layers=2, batch_first=True)
        self.head = PhysicalHead(width)

    def forward(self, x, prior, active):
        b, t, n, _ = x.shape
        z = self.embed(x).permute(0, 2, 1, 3).reshape(b * n, t, -1)
        out, _ = self.gru(z)
        core = out[:, -1].reshape(b, n, -1)
        return self.head(core, x, prior, active)


class Transformer(nn.Module):
    """Genuine causal time attention over all identical 256 source observations."""
    def __init__(self, width: int = 68, feedforward: int = 155):
        super().__init__()
        if width % 4:
            raise ValueError("attention width must divide 4 heads")
        self.embed = nn.Linear(8, width)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, dim_feedforward=feedforward,
                                       dropout=0, batch_first=True),
            num_layers=2, enable_nested_tensor=False)
        self.head = PhysicalHead(width)

    def forward(self, x, prior, active):
        b, t, n, _ = x.shape
        original = x
        z = self.embed(x).permute(0, 2, 1, 3).reshape(b * n, t, -1)
        times = torch.arange(t, device=z.device, dtype=z.dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, z.shape[-1], 2, device=z.device, dtype=z.dtype)
                        * (-math.log(10000) / z.shape[-1]))
        positions = torch.zeros(t, z.shape[-1], device=z.device, dtype=z.dtype)
        positions[:, 0::2] = torch.sin(times * div)
        positions[:, 1::2] = torch.cos(times * div)
        future_mask = torch.triu(torch.ones(t, t, device=z.device, dtype=torch.bool), 1)
        encoded = self.enc(z + positions, mask=future_mask, is_causal=True)
        return self.head(encoded[:, -1].reshape(b, n, -1), original, prior, active)
