"""CSL-125-R2: isolated NO-FIT selective-write research candidate.

Mathematical change relative to immutable CSL-125-R1: each role obtains a
learned, predecision, token-dependent admission coefficient g in [0,1].
The numerator and occupancy share the same gate-dependent effective update
rate eta = -expm1(-g / tau), so g=0 preserves the full memory state exactly
and g=1 recovers the R1 EMA recurrence (up to FP rounding). Gating is NOT
claimed to learn useful keys, deletion, absence or identity from no-fit tests.

No inference-time oracle mask, hidden answer information, tool execution, fitted
model, approved GPU budget or accuracy claim is included.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tac_osm.csl001 import RMSNorm
from tac_osm.csl125 import CSL125
from tac_osm.csl125_r1 import (
    CSL125R1, CSL125R1Config, R1LayerState, R1LocalGQA, R1RoleStateBlock,
)


@dataclass(frozen=True)
class CSL125R2Config(CSL125R1Config):
    write_bias_initial: float = -1.0

    def __post_init__(self):
        super().__post_init__()
        if not isfinite(self.write_bias_initial) or abs(self.write_bias_initial) > 5.0:
            raise ValueError('write gate bias must be finite in [-5,5]')


def selective_affine_scan(z: Tensor, carry: Tensor, rate: Tensor, gate: Tensor,
                          chunk_size: int) -> tuple[Tensor, Tensor]:
    """Gradient-compatible, FP32 stable associative prefix of affine updates.

    Shapes: z[B,T,R,E,2]=[address*value,address]; carry[B,R,E,2];
    rate[R,E]>0; gate[B,T,R] in [0,1]; chunk length 1..64.
    eta=1-exp(-rate*gate) (computed via expm1) and beta=exp(-rate*gate).
    S_t = beta_t*S_{t-1} + eta_t*z_t applies equally to value and mass.

    Per-chunk prefix pairs (A_i,B_i): x -> A_i*x+B_i. Composition is
    (A2,B2) o (A1,B1) = (A2*A1, B2 + A2*B1). Ordered doubling has
    O(log chunk) parallel stages, O(T log chunk) work, no division by
    vanishing prefix products. CPU/PyTorch performance is NOT asserted.
    """
    if (z.ndim != 5 or z.shape[-1] != 2 or z.shape[1] < 1 or
        carry.shape != z.shape[:1]+z.shape[2:] or
        rate.shape != z.shape[2:4] or gate.shape != z.shape[:3] or
        not (1 <= chunk_size <= 64)):
        raise ValueError('invalid selective scan shape/chunk')
    if not (torch.isfinite(rate).all() and (rate > 0).all() and
            torch.isfinite(gate).all() and (gate >= 0).all() and (gate <= 1).all()):
        raise ValueError('rate must be positive and gate must be finite [0,1]')
    r = rate.float().view(1, 1, *rate.shape)
    outputs = []
    state = carry.float()
    for begin in range(0, z.shape[1], chunk_size):
        src = z[:, begin:begin+chunk_size].float()
        g = gate[:, begin:begin+chunk_size].float().unsqueeze(-1)
        rg = r * g
        # Avoids cancellation at near-zero gates; both are exactly zero/one
        # for a gate that is exactly closed.
        eta = -torch.expm1(-rg)
        beta = torch.exp(-rg)
        A, B = beta, eta.unsqueeze(-1)*src
        shift = 1
        while shift < src.shape[1]:
            prev_A, prev_B = A, B
            A = torch.cat([prev_A[:,:shift],
                prev_A[:,shift:] * prev_A[:,:-shift]], dim=1)
            B = torch.cat([prev_B[:,:shift],
                prev_B[:,shift:] + prev_A[:,shift:].unsqueeze(-1)*prev_B[:,:-shift]], dim=1)
            shift *= 2
        result = A.unsqueeze(-1)*state.unsqueeze(1) + B
        outputs.append(result)
        state = result[:,-1]
    return torch.cat(outputs, dim=1), state


class R2RoleStateBlock(R1RoleStateBlock):
    """R1 fast/slow roles with learned per-role write admission."""
    def __init__(self, cfg: CSL125R2Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.write_gate = nn.Linear(cfg.width, cfg.role_count, bias=True)
        if not self.write_gate.bias.is_meta:
            nn.init.zeros_(self.write_gate.weight)
            nn.init.constant_(self.write_gate.bias, cfg.write_bias_initial)

    def forward(self, x: Tensor, state: R1LayerState) -> tuple[Tensor, R1LayerState]:
        cfg=self.cfg
        b,t,d=x.shape
        shape=(b,cfg.role_count,cfg.state_width)
        if (not isinstance(state,R1LayerState) or
            any(getattr(state,k).shape!=shape for k in ('fast','slow','fast_mass','slow_mass')) or
            state.conv_tail.shape!=(b,cfg.conv_kernel-1,d)):
            raise ValueError('R2 carried state invalid')
        h=self.pre(x)
        with_tail=torch.cat((state.conv_tail,h),dim=1)
        local=self.conv(with_tail.transpose(1,2)).transpose(1,2)
        next_tail=with_tail[:,-(cfg.conv_kernel-1):]

        q=self.key(h)
        role_p=F.softmax(torch.einsum('bte,re->btr',q.float(),self.roles.float())/
                         cfg.state_width**0.5,dim=-1)
        value=torch.tanh(self.value(h)).float()
        numerator=role_p.unsqueeze(-1)*value.unsqueeze(-2)
        mass=role_p.unsqueeze(-1).expand_as(numerator)
        writes=torch.stack([numerator,mass],dim=-1)
        gate=torch.sigmoid(self.write_gate(h).float())
        fast_rate,slow_rate=self.rates()
        f,f_end=selective_affine_scan(writes,torch.stack([state.fast,state.fast_mass],-1),
                                      fast_rate,gate,cfg.chunk_size)
        s,s_end=selective_affine_scan(writes,torch.stack([state.slow,state.slow_mass],-1),
                                      slow_rate,gate,cfg.chunk_size)
        rf=torch.einsum('btr,btre->bte',role_p,
                         f[...,0]/(f[...,1]+cfg.denominator_epsilon)).to(h.dtype)
        rs=torch.einsum('btr,btre->bte',role_p,
                         s[...,0]/(s[...,1]+cfg.denominator_epsilon)).to(h.dtype)
        read_gate=torch.sigmoid(self.read_gate(h))
        memory=read_gate*self.read_fast(rf)+(1.-read_gate)*self.read_slow(rs)
        z=x+self.residual_gain*(local+memory)
        u,g=self.up(self.ffnorm(z)).chunk(2,-1)
        out=z+self.ff_gain*self.down(F.silu(g)*u)
        return out,R1LayerState(f_end[...,0],s_end[...,0],next_tail,
                                f_end[...,1],s_end[...,1])


class CSL125R2(CSL125R1):
    """20-layer R2, same existing 2/8/20 exits; adds 20 small gate heads."""
    def __init__(self, config: CSL125R2Config=CSL125R2Config(), *, device=None):
        nn.Module.__init__(self)
        self.config=config
        with torch.device('cpu' if device is None else device):
            self.embedding=nn.Embedding(config.vocab_size,config.width)
            self.blocks=nn.ModuleList(R2RoleStateBlock(config) for _ in range(config.layers))
            self.attention=nn.ModuleDict({str(i):R1LocalGQA(config) for i in config.attention_layers})
            self.final_norm=RMSNorm(config.width)
            self.tool_head=nn.Linear(config.width,config.tool_classes+1)
            self.field_head=nn.Linear(config.width,config.field_classes)
            self.accept_head=nn.Linear(config.width,1)
        if self.parameter_count()>config.max_parameters:
            raise ValueError('R2 learned parameters exceed hard 125M cap')
