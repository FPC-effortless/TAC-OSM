"""CSL-125-R1: research-only revision of the untrained CSL-125 memory math.

Preserves all original CSL-125 source/checkpoints. Changes only the role-state
operator, numerical parameterization and strictly checked training-objective
interface. The grouped local-attention implementation and weight geometry are
inherited without altering semantics. There is NO learned-capability claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tac_osm.csl001 import LayerState, RMSNorm, RoleStateBlock
from tac_osm.csl125 import CAP, CSL125, CSL125Config, LocalGQA


@dataclass(frozen=True)
class CSL125R1Config(CSL125Config):
    # Both role initialization and the retention law are part of the *new*
    # CSL-125-R1 variant, not a silent change to CSL-125's previous experiment.
    role_init_std: float = 1.0
    fast_tau_min: float = 8.0
    fast_tau_max: float = 128.0
    slow_tau_min: float = 256.0
    slow_tau_max: float = 8192.0
    tau_hard_min: float = 2.0
    tau_hard_max: float = 32768.0
    denominator_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        super().__post_init__()
        for n in ('role_init_std', 'fast_tau_min', 'fast_tau_max', 'slow_tau_min',
                  'slow_tau_max', 'tau_hard_min', 'tau_hard_max', 'denominator_epsilon'):
            v = getattr(self, n)
            if not isfinite(v) or v <= 0:
                raise ValueError(f'{n} must be finite and positive')
        if not (self.tau_hard_min < self.fast_tau_min < self.fast_tau_max < self.tau_hard_max):
            raise ValueError('fast timescale bounds invalid')
        if not (self.tau_hard_min < self.slow_tau_min < self.slow_tau_max < self.tau_hard_max):
            raise ValueError('slow timescale bounds invalid')
        if self.width // self.query_heads % 2:
            raise ValueError('RoPE needs an even head dimension')
        if self.denominator_epsilon > 1e-3:
            raise ValueError('normalization epsilon too large for initial writes')


@dataclass
class R1LayerState(LayerState):
    """[B,R,E] FP32 numerators + matching masses, and causal conv tail."""
    fast_mass: Tensor
    slow_mass: Tensor

    def detached(self) -> 'R1LayerState':
        return R1LayerState(self.fast.detach(), self.slow.detach(),
                            self.conv_tail.detach(), self.fast_mass.detach(),
                            self.slow_mass.detach())


def bounded_weighted_mass_scan(z: Tensor, initial: Tensor, rate: Tensor,
                               chunk_size: int) -> tuple[Tensor, Tensor]:
    """Stable exact-real decayed prefix recurrence; FP32 implementation.

    z[B,T,R,E,2] is [address*value, address]. The two channels must share
    decay. `rate=1/tau` is positive and *not* 1-alpha, avoiding cancellation
    for long timescales. Per chunk: p_i=exp(-rate*i), w=-expm1(-rate),
    S_i=exp(-rate*(i+1))*S_prev + w*p_i*cumsum(z_i/p_i).
    Finite-precision parity, not bitwise serial equality, is expected.
    """
    if (z.ndim != 5 or z.shape[-1] != 2 or z.shape[1] < 1 or
        initial.shape != z.shape[:1] + z.shape[2:] or rate.shape != z.shape[2:4] or
        not (1 <= chunk_size <= 64)):
        raise ValueError('invalid normalized scan shape or chunk')
    out = []
    carry = initial.float()
    r = rate.float().view(1, 1, *rate.shape, 1)
    w = -torch.expm1(-r)  # accurate even when tau >> 1
    for start in range(0, z.shape[1], chunk_size):
        block = z[:, start:start+chunk_size].float()
        t = torch.arange(block.shape[1], dtype=torch.float32,
                         device=z.device).view(1, -1, 1, 1, 1)
        p = torch.exp(-r*t)
        result = (torch.exp(-r*(t+1)) * carry.unsqueeze(1)
                  + w*p*torch.cumsum(block/p, dim=1))
        out.append(result)
        carry = result[:, -1]
    return torch.cat(out, dim=1), carry


class R1RoleStateBlock(RoleStateBlock):
    """Role-addressed bounded weighted-value + weighted-occupancy memory."""
    def __init__(self, cfg: CSL125R1Config):
        super().__init__(cfg)
        self.cfg = cfg
        # Legacy role initialization N(0,1/sqrt(E)) and a subsequent /sqrt(E)
        # yields nearly uniform softmax logits; R1 uses N(0,1) *before* /sqrt(E).
        if not self.roles.is_meta:
            nn.init.normal_(self.roles, mean=0., std=cfg.role_init_std)
        del self.fast_decay_logit
        del self.slow_decay_logit
        self.fast_tau_logits = nn.Parameter(self._init_logit(cfg.fast_tau_min, cfg.fast_tau_max))
        self.slow_tau_logits = nn.Parameter(self._init_logit(cfg.slow_tau_min, cfg.slow_tau_max))

    def _init_logit(self, minimum: float, maximum: float) -> Tensor:
        cfg = self.cfg
        x = torch.linspace(log(minimum), log(maximum), cfg.state_width).exp()
        u = ((x.log() - log(cfg.tau_hard_min)) /
             (log(cfg.tau_hard_max) - log(cfg.tau_hard_min)))
        logits = torch.logit(u.clamp(1e-4, 1-1e-4)).view(1, -1).expand(cfg.role_count, -1).clone()
        return logits.to(self.roles.device)

    def rates(self) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        lo, span = log(cfg.tau_hard_min), log(cfg.tau_hard_max/cfg.tau_hard_min)
        fast_rate = torch.exp(-(lo + span * torch.sigmoid(self.fast_tau_logits.float())))
        slow_rate = torch.exp(-(lo + span * torch.sigmoid(self.slow_tau_logits.float())))
        return fast_rate, slow_rate

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> R1LayerState:
        legacy = super().initial_state(batch, device, dtype)
        return R1LayerState(legacy.fast, legacy.slow, legacy.conv_tail,
                            torch.zeros_like(legacy.fast), torch.zeros_like(legacy.slow))

    def forward(self, x: Tensor, state: R1LayerState) -> tuple[Tensor, R1LayerState]:
        if not isinstance(state, R1LayerState):
            raise ValueError('CSL-125-R1 requires its own memory-state version')
        b, t, d = x.shape
        cfg = self.cfg
        shape = (b, cfg.role_count, cfg.state_width)
        if (state.fast.shape != shape or state.slow.shape != shape or
            state.fast_mass.shape != shape or state.slow_mass.shape != shape or
            state.conv_tail.shape != (b, cfg.conv_kernel-1, d)):
            raise ValueError('R1 carried state layout does not match input')
        h = self.pre(x)
        with_tail = torch.cat((state.conv_tail, h), dim=1)
        local = self.conv(with_tail.transpose(1,2)).transpose(1,2)
        next_tail = with_tail[:, -(cfg.conv_kernel-1):]

        query = self.key(h)
        role_p = F.softmax(torch.einsum('bte,re->btr', query.float(), self.roles.float()) /
                           (cfg.state_width**0.5), dim=-1)
        val = torch.tanh(self.value(h)).float()
        # For every role j and channel e: z=[a_j * v_e, a_j].
        numerator_write = role_p.unsqueeze(-1) * val.unsqueeze(-2)
        mass_write = role_p.unsqueeze(-1).expand_as(numerator_write)
        writes = torch.stack((numerator_write, mass_write), dim=-1)
        fast_rate, slow_rate = self.rates()
        f, f_end = bounded_weighted_mass_scan(writes,
            torch.stack((state.fast, state.fast_mass), dim=-1), fast_rate, cfg.chunk_size)
        s, s_end = bounded_weighted_mass_scan(writes,
            torch.stack((state.slow, state.slow_mass), dim=-1), slow_rate, cfg.chunk_size)

        # Constrained weighted mean: abs(numerator)<=mass, hence abs(read)<=1.
        # The positive epsilon prevents NaNs for empty slots, without changing
        # the read of a populated slot by an uncontrolled 1/occupancy factor.
        rf = torch.einsum('btr,btre->bte',role_p,
                          f[...,0] / (f[...,1] + cfg.denominator_epsilon)).to(h.dtype)
        rs = torch.einsum('btr,btre->bte',role_p,
                          s[...,0] / (s[...,1] + cfg.denominator_epsilon)).to(h.dtype)
        gate = torch.sigmoid(self.read_gate(h))
        memory = gate*self.read_fast(rf) + (1.-gate)*self.read_slow(rs)
        x = x + self.residual_gain * (local + memory)
        u,g = self.up(self.ffnorm(x)).chunk(2,dim=-1)
        x = x + self.ff_gain * self.down(F.silu(g) * u)
        return x, R1LayerState(f_end[...,0],s_end[...,0],next_tail,
                                f_end[...,1],s_end[...,1])


class R1LocalGQA(LocalGQA):
    """Keep RoPE phase distinguishable for long-lived persistent streams.

    Original CSL-125 casts integer positions to FP32; at positions >=2**24,
    neighboring integers may map to the same float. Fixed-frequency rotary
    angles are computed in FP64 and cast back to the activation dtype. This
    has NO learned parameters. A fused low-power implementation remains OPEN.
    """
    def __init__(self, cfg: CSL125R1Config):
        super().__init__(cfg)
        # Recompute from the exact mathematical formula in FP64; converting
        # the original FP32 frequency buffer to FP64 would keep its error.
        self.inv_freq = 1. / (10000. ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float64,
                         device=self.inv_freq.device) / self.head_dim))

    def _rope(self, x: Tensor, positions: Tensor) -> Tensor:
        half = x.shape[-1] // 2
        if half * 2 != x.shape[-1]:
            raise ValueError('rotary head width must be even')
        phase = positions.to(torch.float64).unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        c = phase.cos().to(dtype=x.dtype).view(1,1,-1,half)
        s = phase.sin().to(dtype=x.dtype).view(1,1,-1,half)
        a,b = x[..., :half],x[..., half:]
        return torch.cat([a*c-b*s,a*s+b*c],dim=-1)


class CSL125R1(CSL125):
    """Drop-in depth-ladder interface with a scientifically distinct state law."""
    def __init__(self, config: CSL125R1Config = CSL125R1Config(), *, device=None):
        nn.Module.__init__(self)
        self.config = config
        with torch.device('cpu' if device is None else device):
            self.embedding = nn.Embedding(config.vocab_size,config.width)
            self.blocks = nn.ModuleList(R1RoleStateBlock(config) for _ in range(config.layers))
            self.attention = nn.ModuleDict({str(i):R1LocalGQA(config) for i in config.attention_layers})
            self.final_norm = RMSNorm(config.width)
            self.tool_head = nn.Linear(config.width, config.tool_classes+1)
            self.field_head = nn.Linear(config.width, config.field_classes)
            self.accept_head = nn.Linear(config.width,1)
        if self.parameter_count() > config.max_parameters:
            raise ValueError('CSL-125-R1 learned parameters exceed hard cap')

    def forward(self, tokens: Tensor, *, depth: Optional[int] = None, **kwargs):
        """Reject invalid exit requests instead of silently evaluating full depth.

        The original CSL-125 uses `depth or layers`, which maps requested depth
        zero to the full model. R1 fails closed; this does not alter valid-rung
        logits, trained weights, or the mathematical memory recurrence.
        """
        if depth is not None and (type(depth) is not int or depth not in self.config.rungs):
            raise ValueError('requested depth is not a registered CSL125R1 exit')
        return super().forward(tokens, depth=depth, **kwargs)

    def nested_next_token_loss(self, tokens: Tensor,
            weights: Optional[Dict[int, float]] = None,
            loss_mask: Optional[Tensor] = None) -> Tensor:
        """Masked multi-exit next-token objective with fail-closed weights.

        `loss_mask` labels valid *next-token targets*, shape [B,T-1]. It is
        NOT a document-boundary reset operator: do not pack independent
        documents in one stream without an independent reset/partition adapter.
        """
        if tokens.ndim != 2 or tokens.shape[1] < 2:
            raise ValueError('next-token training needs at least two tokens')
        if weights is None:
            weights = {r: 1. for r in self.config.rungs}
        if (set(weights) != set(self.config.rungs) or
            any(not isfinite(float(w)) or w < 0 for w in weights.values()) or
            sum(weights.values()) <= 0):
            raise ValueError('loss weights must be finite, nonnegative and positive in total')
        if loss_mask is not None:
            if loss_mask.shape != (tokens.shape[0],tokens.shape[1]-1) or loss_mask.dtype != torch.bool:
                raise ValueError('loss_mask must be boolean [B,T-1]')
            if not loss_mask.any().item():
                raise ValueError('empty valid-target set')
        outs, _ = self.forward(tokens,all_rungs=True)
        targets=tokens[:,1:].reshape(-1)
        mask=torch.ones_like(targets,dtype=torch.bool) if loss_mask is None else loss_mask.reshape(-1)
        count=mask.sum()
        denom=sum(weights.values())
        return sum((float(weights[r])/denom) *
            F.cross_entropy(outs[r]['token_logits'][:,:-1].reshape(-1,self.config.vocab_size)[mask],targets[mask])
                for r in self.config.rungs if weights[r]>0)
