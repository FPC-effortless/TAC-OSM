"""COLLECT011: causal, differentiable translation of COLLECT010's operator.

This is a NEW LM address frontend, not the validated three-character frontend.
Only the analytic normalized preconditioned write is transferred unchanged.
State belongs to an ordered batch of owners. No target or task-role label enters.
FP32/64 only; memory/precision are attached to autograd until explicit detach.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PhaseState:
    owners: tuple[str, ...]
    clock: int
    memory: torch.Tensor
    precision: torch.Tensor | None
    signature: str

    def detach(self):
        return PhaseState(self.owners, self.clock, self.memory.detach(),
                          None if self.precision is None else self.precision.detach(), self.signature)

    def state_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.memory, self.precision) if t is not None)

    def payload(self):
        return dict(schema="collect011.phase.v1", owners=self.owners, clock=self.clock,
                    memory=self.memory.clone(), precision=None if self.precision is None else self.precision.clone(),
                    signature=self.signature)

    @classmethod
    def from_payload(cls, p):
        if set(p) != {"schema", "owners", "clock", "memory", "precision", "signature"} or p["schema"] != "collect011.phase.v1":
            raise ValueError("complete phase payload required")
        return cls(tuple(p["owners"]), p["clock"], p["memory"].clone(),
                   None if p["precision"] is None else p["precision"].clone(), p["signature"])


def write_step(memory, precision, key, value, gain, *, kind="preconditioned", floor=1e-8):
    """Shapes [...,V,K], [...,K,K], [...,K], [...,V], [...].

    Floor is numerical protection, not a proof that arbitrary corrupted P is SPD.
    Callers validate incoming P; zero gain freezes every carried tensor exactly.
    """
    residual = value - (memory @ key.unsqueeze(-1)).squeeze(-1)
    if kind == "delta":
        direction = key
        candidate_p = None
    else:
        z = (precision @ key.unsqueeze(-1)).squeeze(-1)
        leverage = (key*z).sum(-1)
        direction = z / leverage.clamp_min(floor).unsqueeze(-1)
        candidate_p = precision - gain[..., None, None] * z.unsqueeze(-1)*z.unsqueeze(-2)/(1+gain*leverage)[..., None, None]
        candidate_p = (candidate_p+candidate_p.transpose(-1, -2))*.5
    candidate = memory + gain[..., None, None]*residual.unsqueeze(-1)*direction.unsqueeze(-2)
    eligible = gain != 0
    updated = torch.where(eligible[..., None, None], candidate, memory)
    if precision is not None:
        candidate_p = torch.where(eligible[..., None, None], candidate_p, precision)
    return updated, candidate_p


class PhaseMemory(nn.Module):
    def __init__(self, width, key_dim=16, value_dim=16, heads=1, *,
                 kind="preconditioned", hybrid=False, ridge=.1, support_gate=False, hard_routing=False):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (width, key_dim, value_dim, heads)) or key_dim % 2:
            raise ValueError("positive geometry and even phase key dimension required")
        if kind not in ("delta", "preconditioned") or not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("legal operator and positive finite ridge required")
        if support_gate and kind == "delta":
            raise ValueError("support gate requires precision state")
        if hard_routing and not hybrid:raise ValueError("hard routing requires hybrid banks")
        self.width, self.key_dim, self.value_dim, self.heads = width, key_dim, value_dim, heads
        self.kind, self.hybrid, self.ridge, self.support_gate = kind, hybrid, ridge, support_gate
        self.hard_routing=hard_routing
        self.content = nn.Linear(width, heads*key_dim//2, bias=False)
        self.role = nn.Linear(width, heads*key_dim//2, bias=False)
        self.value = nn.Linear(width, heads*value_dim, bias=False)
        self.admission = nn.Linear(width, heads)
        # Conservative but nonzero initial writes; never silently zero a residual.
        nn.init.constant_(self.admission.bias, -2.)
        if hybrid:
            self.router = nn.Linear(width, heads)
        self.signature = hashlib.sha256(json.dumps(dict(width=width,key_dim=key_dim,
            value_dim=value_dim,heads=heads,kind=kind,hybrid=hybrid,ridge=ridge,
            support_gate=support_gate,hard_routing=hard_routing), sort_keys=True).encode()).hexdigest()

    @staticmethod
    def owners(owners):
        if not isinstance(owners, (list, tuple)) or not owners or any(type(x) is not str or not x for x in owners) or len(set(owners)) != len(owners):
            raise ValueError("ordered unique nonempty owners required")
        return tuple(owners)

    def start(self, owners):
        owners = self.owners(owners)
        w = self.content.weight
        memory = w.new_zeros(len(owners), self.heads, self.value_dim, self.key_dim)
        precision = None if self.kind == "delta" else torch.eye(self.key_dim, device=w.device, dtype=w.dtype).expand(len(owners),self.heads,-1,-1).clone()/self.ridge
        return PhaseState(owners, 0, memory, precision, self.signature)

    def validate(self, x, state, owners, mask):
        w = self.content.weight
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or min(x.shape) < 1 or x.shape[-1] != self.width or x.dtype not in (torch.float32,torch.float64) or x.dtype != w.dtype or x.device != w.device:
            raise ValueError("same precision/device nonempty Float[B,T,D] required")
        if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            raise ValueError("mixed precision unqualified")
        if not isinstance(state, PhaseState) or state.owners != owners or len(owners) != x.shape[0] or state.signature != self.signature or type(state.clock) is not int or state.clock < 0:
            raise ValueError("complete same-owner/config phase state required")
        shapes = [(state.memory,(x.shape[0],self.heads,self.value_dim,self.key_dim))]
        if self.kind == "preconditioned":
            shapes.append((state.precision,(x.shape[0],self.heads,self.key_dim,self.key_dim)))
        elif state.precision is not None:
            raise ValueError("delta has no precision state")
        for tensor, shape in shapes:
            if not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != x.dtype or tensor.device != x.device or not bool(torch.isfinite(tensor).all()):
                raise ValueError("finite same-geometry state required")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("finite inputs required")
        if state.precision is not None:
            if not torch.allclose(state.precision,state.precision.transpose(-1,-2),atol=1e-5,rtol=1e-5) or bool((torch.linalg.cholesky_ex(state.precision.detach()).info != 0).any()):
                raise ValueError("symmetric positive definite precision required")
        if not isinstance(mask,torch.Tensor) or mask.shape != x.shape[:2] or mask.device != x.device or mask.dtype != torch.bool:
            raise ValueError("Bool[B,T] mask required")

    def forward(self, x, state=None, write_mask=None, *, owners=None):
        if owners is None:
            if state is not None:
                raise ValueError("explicit owners required for continuing state")
            owners = tuple(f"fresh-{i}" for i in range(x.shape[0]))
        owners = self.owners(owners)
        state = self.start(owners) if state is None else state
        mask = torch.ones(x.shape[:2],device=x.device,dtype=torch.bool) if write_mask is None else write_mask
        self.validate(x,state,owners,mask)
        shape = (*x.shape[:2],self.heads,self.key_dim//2)
        # Content/role phases are relative to input features, never absolute time.
        theta = math.pi*(self.content(x).reshape(shape).tanh()+self.role(x).reshape(shape).tanh().roll(1,-1))
        key = torch.stack((theta.cos(),theta.sin()),-1).flatten(-2)/math.sqrt(self.key_dim//2)
        values = self.value(x).reshape(*x.shape[:2],self.heads,self.value_dim).tanh()
        gain = self.admission(x).sigmoid()*mask[...,None]
        route = self.router(x).softmax(-1) if self.hybrid else torch.ones_like(gain)
        if self.hard_routing:
            discrete=F.one_hot(route.argmax(-1),self.heads).to(route.dtype)
            route=discrete+(route-route.detach())
        # Soft mixture and explicitly separate straight-through discrete control.
        # Both compute dense projections/updates; no sparse-kernel gain is claimed.
        gain = gain*route
        memory, precision = state.memory,state.precision
        reads=[]
        for q,v,g,r in zip(key.unbind(1),values.unbind(1),gain.unbind(1),route.unbind(1)):
            read = (memory @ q.unsqueeze(-1)).squeeze(-1)
            if self.support_gate:
                novelty = self.ridge*(q*(precision @ q.unsqueeze(-1)).squeeze(-1)).sum(-1)
                # Smooth evidence scales ONLY the memory residual; no NONE token.
                read = read*(1-novelty.clamp(0,1)).unsqueeze(-1)
            reads.append(read*r.unsqueeze(-1))
            memory,precision=write_step(memory,precision,q,v,g,kind=self.kind)
        return torch.stack(reads,1),PhaseState(owners,state.clock+x.shape[1],memory.contiguous(),
            None if precision is None else precision.contiguous(),self.signature)
