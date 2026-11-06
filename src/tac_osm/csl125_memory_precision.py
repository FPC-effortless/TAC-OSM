"""Opt-in R4 memory precision island; no new parameters or exact-real state law.

Large learned projections remain in their existing execution dtype. Only the
numerator/mass recurrence, carried memory, normalization and role reduction
use FP64. This is NOT a qualified mobile/GPU kernel or learned capability.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor
from torch.nn import functional as F
from tac_osm.csl125_r1 import R1LayerState
from tac_osm.csl125_r2 import R2RoleStateBlock


@dataclass
class PreciseMemoryState(R1LayerState):
    """Versioned FP64 memory; convolution history keeps its original dtype."""
    def detached(self):
        return PreciseMemoryState(self.fast.detach(), self.slow.detach(),
            self.conv_tail.detach(), self.fast_mass.detach(), self.slow_mass.detach())


def precise_affine_scan(z: Tensor, carry: Tensor, rate: Tensor, gate: Tensor,
                        chunk_size: int) -> tuple[Tensor, Tensor]:
    """Original ordered affine prefix, FP64 only inside the memory operator."""
    if (z.ndim != 5 or z.shape[-1] != 2 or z.shape[1] < 1 or
        carry.shape != z.shape[:1] + z.shape[2:] or
        rate.shape != z.shape[2:4] or gate.shape != z.shape[:3] or
        type(chunk_size) is not int or not 1 <= chunk_size <= 64):
        raise ValueError('invalid precise scan shape/chunk')
    if not (torch.isfinite(rate).all() and (rate > 0).all() and
            torch.isfinite(gate).all() and (gate >= 0).all() and (gate <= 1).all()):
        raise ValueError('rate must be positive and gate finite in [0,1]')
    if any(a.device != z.device for a in (carry, rate, gate)):
        raise ValueError('memory inputs must share a device')
    with torch.autocast(device_type=z.device.type, enabled=False):
        r = rate.double().view(1, 1, *rate.shape)
        state, outputs = carry.double(), []
        for begin in range(0, z.shape[1], chunk_size):
            src = z[:, begin:begin+chunk_size].double()
            rg = r * gate[:, begin:begin+chunk_size].double().unsqueeze(-1)
            eta, beta = -torch.expm1(-rg), torch.exp(-rg)
            a, b = beta, eta.unsqueeze(-1) * src
            shift = 1
            while shift < src.shape[1]:
                old_a, old_b = a, b
                a = torch.cat((old_a[:, :shift], old_a[:, shift:] * old_a[:, :-shift]), 1)
                b = torch.cat((old_b[:, :shift], old_b[:, shift:] +
                    old_a[:, shift:].unsqueeze(-1) * old_b[:, :-shift]), 1)
                shift *= 2
            result = a.unsqueeze(-1) * state.unsqueeze(1) + b
            outputs.append(result)
            state = result[:, -1]
        return torch.cat(outputs, 1), state


def precise_normalized_read(pair: Tensor, role_p: Tensor, epsilon: float) -> Tensor:
    if pair.dtype != torch.float64 or not isfinite(epsilon) or epsilon <= 0:
        raise ValueError('normalization needs FP64 memory and positive epsilon')
    with torch.autocast(device_type=pair.device.type, enabled=False):
        mean = pair[..., 0] / (pair[..., 1] + epsilon)
        return torch.einsum('btr,btre->bte', role_p.double(), mean)


class PreciseRoleStateBlock(R2RoleStateBlock):
    """Same parameters/names as R2; explicit precision boundary and state type."""
    def initial_state(self, batch, device, dtype):
        s = super().initial_state(batch, device, dtype)
        return PreciseMemoryState(s.fast.double(), s.slow.double(), s.conv_tail,
                                  s.fast_mass.double(), s.slow_mass.double())

    def forward(self, x: Tensor, state: PreciseMemoryState):
        cfg = self.cfg
        b, t, d = x.shape
        shape = (b, cfg.role_count, cfg.state_width)
        keys = ('fast', 'slow', 'fast_mass', 'slow_mass')
        if (not isinstance(state, PreciseMemoryState) or
            any(getattr(state, k).shape != shape or
                getattr(state, k).dtype != torch.float64 for k in keys) or
            state.conv_tail.shape != (b, cfg.conv_kernel - 1, d)):
            raise ValueError('precision-state mismatch: reset/replay required')
        h = self.pre(x)
        with_tail = torch.cat((state.conv_tail, h), 1)
        local = self.conv(with_tail.transpose(1, 2)).transpose(1, 2)
        next_tail = with_tail[:, -(cfg.conv_kernel-1):]
        q = self.key(h)
        role_p = F.softmax(torch.einsum('bte,re->btr', q.float(), self.roles.float()) /
                           cfg.state_width**0.5, dim=-1)
        value = torch.tanh(self.value(h)).float()
        numerator = role_p.unsqueeze(-1) * value.unsqueeze(-2)
        mass = role_p.unsqueeze(-1).expand_as(numerator)
        writes = torch.stack((numerator, mass), -1)
        gate = torch.sigmoid(self.write_gate(h).float())
        fast_rate, slow_rate = self.rates()
        f, f_end = precise_affine_scan(writes,
            torch.stack((state.fast, state.fast_mass), -1), fast_rate, gate, cfg.chunk_size)
        s, s_end = precise_affine_scan(writes,
            torch.stack((state.slow, state.slow_mass), -1), slow_rate, gate, cfg.chunk_size)
        rf = precise_normalized_read(f, role_p, cfg.denominator_epsilon).to(h.dtype)
        rs = precise_normalized_read(s, role_p, cfg.denominator_epsilon).to(h.dtype)
        read_gate = torch.sigmoid(self.read_gate(h))
        memory = read_gate * self.read_fast(rf) + (1. - read_gate) * self.read_slow(rs)
        z = x + self.residual_gain * (local + memory)
        u, g = self.up(self.ffnorm(z)).chunk(2, -1)
        out = z + self.ff_gain * self.down(F.silu(g) * u)
        return out, PreciseMemoryState(f_end[..., 0], s_end[..., 0], next_tail,
                                      f_end[..., 1], s_end[..., 1])


def enable_precise_memory_(model):
    """Opt-in installation with identical Parameter objects and checkpoint keys.

    Install before creating any carried stream. Existing FP32 carried states
    are rejected, not silently converted. Default source model is untouched.
    No module-global patching; parallel independent models remain independent.
    """
    if not hasattr(model, 'blocks') or not model.blocks:
        raise ValueError('expected CSL R4 role-state blocks')
    if all(type(b) is PreciseRoleStateBlock for b in model.blocks):
        return model
    if any(type(b) is not R2RoleStateBlock for b in model.blocks):
        raise ValueError('only unmodified R2 blocks may opt into this precision policy')
    before = {name: id(p) for name, p in model.named_parameters()}
    keys = tuple(model.state_dict())
    replacements = []
    for block in model.blocks:
        with torch.device('meta'):
            replacement = PreciseRoleStateBlock(block.cfg)
        replacement.load_state_dict(block.state_dict(keep_vars=True), strict=True, assign=True)
        replacement.train(block.training)
        replacements.append(replacement)
    for i, replacement in enumerate(replacements):
        model.blocks[i] = replacement
    after = {name: id(p) for name, p in model.named_parameters()}
    if before != after or keys != tuple(model.state_dict()):
        raise AssertionError('precision installation changed learned weights or names')
    return model
