"""Optional Torch models for bounded, full-horizon real-data forecasting.

Compact baselines are independent implementations, not official paper recipes.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('tac_patch', 'ar', 'context', 'gru', 'dlinear', 'patchtst_small', 'itransformer_small')


class NativePatchMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(8, 32)
        self.transition = nn.Linear(32, 32, bias=False)
        self.gate = nn.Linear(40, 32)
        self.slow = nn.Linear(64, 32)
        self.slow_gate = nn.Linear(64, 32)
        nn.init.orthogonal_(self.transition.weight, gain=.9)
        nn.init.normal_(self.slow.weight, std=.01)
        nn.init.zeros_(self.slow.bias)
        with torch.no_grad():
            self.gate.bias.copy_(-torch.log(torch.logspace(math.log10(2), math.log10(16), 32)-1))
            self.slow_gate.bias.copy_(-torch.log(torch.logspace(math.log10(8), math.log10(64), 32)-1))

    def forward(self, patches):
        f = patches.new_zeros((patches.size(0), 32))
        s = torch.zeros_like(f)
        for t in range(12):
            u = patches[:, t]
            g = torch.sigmoid(self.gate(torch.cat((u, f), -1)))
            f = f + g * (torch.tanh(self.enc(u) + self.transition(f)) - f)
            v = torch.sigmoid(self.slow_gate(torch.cat((f, s), -1)))
            innovation = torch.tanh(self.slow(torch.cat((f-s, s), -1)))
            s = s + v * (torch.tanh(s + innovation) - s)
        h = torch.cat((f, s), -1)
        return h / torch.sqrt(h.square().mean(-1, keepdim=True) + .01)


class PatchForecaster(nn.Module):
    def __init__(self, arm, horizon):
        super().__init__()
        if arm not in ARMS or horizon not in (96, 192, 336, 720):
            raise ValueError('unregistered model or horizon')
        self.arm, self.horizon = arm, horizon
        if arm in ('tac_patch', 'ar', 'context', 'gru'):
            self.ar = nn.Linear(96, horizon)
            nn.init.constant_(self.ar.weight, 1/96)
            nn.init.zeros_(self.ar.bias)
            if arm != 'ar':
                if arm == 'tac_patch':
                    self.memory = NativePatchMemory()
                elif arm == 'context':
                    self.memory = nn.Sequential(nn.Linear(96, 42), nn.SiLU(), nn.Linear(42, 64))
                else:
                    self.gru = nn.GRU(8, 34, batch_first=True)
                    self.projection = nn.Linear(34, 64)
                self.mix = nn.Linear(80, 128)
                self.basis = nn.Linear(64, 32)
                self.out = nn.Linear(32, horizon)
                nn.init.normal_(self.out.weight, std=.001)
                nn.init.zeros_(self.out.bias)
        elif arm == 'dlinear':
            self.trend = nn.Linear(96, horizon)
            self.season = nn.Linear(96, horizon)
            for layer in (self.trend, self.season):
                nn.init.constant_(layer.weight, 1/96)
                nn.init.zeros_(layer.bias)
        else:
            self.embedding = nn.Linear(8 if arm == 'patchtst_small' else 96, 32)
            layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0., activation='gelu', batch_first=True, norm_first=True)
            self.attention = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            if arm == 'patchtst_small':
                self.position = nn.Parameter(torch.zeros(1, 12, 32))
                nn.init.normal_(self.position, std=.02)
            self.out = nn.Linear(384 if arm == 'patchtst_small' else 32, horizon)

    def forward(self, x):
        if x.ndim != 3 or x.size(1) != 96:
            raise ValueError('expected B x 96 x channels history')
        b, _, c = x.shape
        mean = x.mean(1, keepdim=True)
        scale = torch.sqrt((x-mean).square().mean(1, keepdim=True)+1e-5)
        z = ((x-mean)/scale).transpose(1, 2)
        flat = z.reshape(b*c, 96)
        if self.arm == 'dlinear':
            trend = F.avg_pool1d(F.pad(flat[:, None], (12, 12), mode='replicate'), 25, stride=1)[:, 0]
            pred = self.trend(trend) + self.season(flat-trend)
        elif self.arm == 'patchtst_small':
            h = self.embedding(flat.reshape(-1, 12, 8)) + self.position
            pred = self.out(self.attention(h).flatten(1))
        elif self.arm == 'itransformer_small':
            h = self.attention(self.embedding(z))
            pred = self.out(h).reshape(b*c, self.horizon)
        else:
            pred = self.ar(flat)
            if self.arm != 'ar':
                patches = flat.reshape(-1, 12, 8)
                if self.arm == 'gru':
                    _, state = self.gru(patches)
                    h = self.projection(state[0])
                elif self.arm == 'context':
                    h = self.memory(flat)
                else:
                    h = self.memory(patches)
                context = torch.cat((patches.mean(-1), flat[:, -1:],
                    flat[:, -1:]-flat[:, -8:].mean(-1, keepdim=True),
                    mean.transpose(1, 2).reshape(-1, 1), scale.log().transpose(1, 2).reshape(-1, 1)), -1)
                a, gate = self.mix(torch.cat((context, h), -1)).chunk(2, -1)
                pred = pred + self.out(F.silu(self.basis(F.silu(a)*gate)))
        return pred.reshape(b, c, self.horizon).transpose(1, 2)*scale + mean


def build_model(arm, horizon, seed):
    torch.manual_seed(seed)
    return PatchForecaster(arm, horizon)
