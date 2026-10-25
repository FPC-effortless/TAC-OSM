"""Shared-core candidate: causal query-conditioned retrieval, no new model family.

All schedule inputs precede memory reads. Parallel time scan and carried streaming
state implement the same graph; valid=False is an identity transition. No claim
that normalization learns angular alignment or that retained state is useful.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .retention_controlled_world import RetentionControlledTACWorld
from .parallel_affine_memory import associative_affine_schedule, work_efficient_affine_prefix


@dataclass(frozen=True)
class EvidenceState:
    basis_sum: torch.Tensor
    count: torch.Tensor
    memory: torch.Tensor


class EvidenceRetrievalWorld(RetentionControlledTACWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query = nn.Linear(48 + 2*self.width, 8)
        self.retrieved_delta = nn.Sequential(nn.Linear(8, 32), nn.SiLU(),
                                             nn.Linear(32, self.state_dim))

    def initial_state(self, batch):
        p = next(self.parameters())
        return EvidenceState(p.new_zeros(batch, self.width),
                             torch.zeros(batch, 1, device=p.device, dtype=torch.int64),
                             p.new_zeros(batch, 8, 8))

    def forward(self, obs, action, *, valid=None, state=None, return_state=False,
                return_trace=False):
        if (obs.ndim != 3 or action.shape != (*obs.shape[:2], self.action_dim)
                or obs.shape[-1] != self.state_dim or obs.shape[1] == 0):
            raise ValueError('nonempty matching [B,T,D] observation/action required')
        B,T,_ = obs.shape
        if obs.dtype != action.dtype or obs.device != action.device:
            raise ValueError('observation/action dtype/device mismatch')
        if valid is None:
            valid = torch.ones(B,T, device=obs.device, dtype=torch.bool)
        if valid.shape != (B,T) or valid.dtype != torch.bool or valid.device != obs.device:
            raise ValueError('valid must be matching bool [B,T]')
        if not (torch.isfinite(obs).all() and torch.isfinite(action).all()):
            raise ValueError('inputs must be finite, including padding')
        state = self.initial_state(B) if state is None else state
        if (state.basis_sum.shape != (B,self.width) or state.count.shape != (B,1)
            or state.memory.shape != (B,8,8) or state.count.dtype != torch.int64
            or torch.any(state.count < 0)):
            raise ValueError('invalid carried state shape/count')
        if any(x.device != obs.device for x in (state.basis_sum,state.count,state.memory)):
            raise ValueError('state device mismatch')
        if any(x.dtype != obs.dtype or not torch.isfinite(x).all()
               for x in (state.basis_sum,state.memory)):
            raise ValueError('state floating dtype/nonfinite mismatch')
        e = self.encoder(torch.cat((obs,action),-1)).tanh()
        k = F.normalize(self.address(e),dim=-1,eps=1e-6)
        v = F.normalize(self.content(e),dim=-1,eps=1e-6)
        zr = e.new_zeros(B,T,8)
        basis = self.fast_map(torch.cat((e,e.new_zeros(B,T,self.width),zr),-1)).chunk(2,-1)[0].tanh()
        basis = torch.where(valid[...,None],basis,torch.zeros_like(basis))
        sums = state.basis_sum[:,None] + basis.cumsum(1)
        counts = state.count[:,None] + valid.to(torch.int64).cumsum(1)[...,None]
        ctx = (sums-basis)/(counts-valid.to(torch.int64)[...,None]).clamp_min(1).to(obs.dtype)
        p,g = self.fast_map(torch.cat((e,ctx,zr),-1)).chunk(2,-1)
        pf = torch.lerp(ctx,p.tanh(),g.sigmoid())
        d,g = self.slow_map(torch.cat((pf,ctx,zr),-1)).chunk(2,-1)
        ps = torch.lerp(ctx,(ctx+d.tanh()).tanh(),.25*g.sigmoid())
        alpha,beta = self.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1)
        alpha = alpha*(1-beta/4)
        P,Q = associative_affine_schedule(k,v,alpha,beta,valid[...,None])
        memory = work_efficient_affine_prefix(P,Q,state.memory)
        previous = torch.cat((state.memory[:,None],memory[:,:-1]),1)
        # Read BEFORE current write, with a separately learned context query.
        q = F.normalize(k + self.query(torch.cat((e,pf,ps),-1)),dim=-1,eps=1e-6)
        r = (previous @ q[...,None]).squeeze(-1)
        f,g = self.fast_map(torch.cat((e,pf,r),-1)).chunk(2,-1)
        f = torch.lerp(pf,f.tanh(),g.sigmoid())
        d,g = self.slow_map(torch.cat((f,ps,r),-1)).chunk(2,-1)
        s = torch.lerp(ps,(ps+d.tanh()).tanh(),.25*g.sigmoid())
        delta = self.readout(torch.cat((f,s,r,e),-1)) + self.retrieved_delta(r)
        prediction = obs + self.delta_scale*delta.tanh()
        prediction = torch.where(valid[...,None],prediction,obs)
        final = EvidenceState(sums[:,-1],counts[:,-1],memory[:,-1])
        if return_trace:
            return prediction,final,dict(query=q,read=r,memory=memory,context=ctx,
                                          key=k,value=v,alpha=alpha,beta=beta)
        return (prediction,final) if return_state else prediction
