"""PAR-TIME-001: source-time / sequence-position TAC with post-scan FFN.

Prospective, UNTRAINED variant of the existing six-family parallel model.
No new family. Every timestamp delta is an observed-at, caller-authorized
NONNEGATIVE interval in domain-specific units; no secret target timestamps.

Seven deterministic input-only features:
  log1p(sequence_index), sin(index/16), cos(index/16),
  log1p(elapsed_delta), log1p(elapsed_since_sequence_start),
  log1p(age_of_most_recent_prior_admitted_write), prior_write_seen.
The age is NOT the exact age of every fact in an 8x8 superposed matrix.
An explicit source-provided boolean mask says which observations WRITE.
The current write is excluded from the current read/age features. Writing
uses the same precomputed input-only affine schedule as ParallelTACWorld.

Running state: fast sum[88], associative memory[8,8], elapsed clock[1],
last admitted write clock[1], prior-write-seen[bool], count[int].
This is 154 FP32 values + one bool + an integer in default geometry,
excluding model weights and all external source/entity metadata.

Synthetic numeric tests are engineering diagnostics, not model ability.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from .parallel_continuous_world import ParallelTACWorld
from .parallel_affine_memory import (
    associative_affine_schedule, work_efficient_affine_prefix,
    preserve_nonwrite_prefix,
)
from .parallel_residual_world import ParallelResidualWorld


@dataclass(frozen=True)
class TimeAwareStreamState:
    fast_sum: torch.Tensor
    memory: torch.Tensor
    elapsed_clock: torch.Tensor
    last_write_clock: torch.Tensor
    write_seen: torch.Tensor
    count: int


def _source_inputs(obs, action, dt, writable, *, batched):
    time_axis = 3 if batched else 2
    if not torch.is_tensor(obs) or obs.ndim != time_axis:
        raise ValueError("observations must be [B,T,D] or [B,D]")
    if not torch.is_floating_point(obs):
        raise ValueError("observations require floating dtype")
    prefix = obs.shape[:-1]
    if (not torch.is_tensor(action) or action.shape[:-1] != prefix
        or action.device != obs.device or action.dtype != obs.dtype):
        raise ValueError("observed actions require matching batch/time/device/dtype")
    if (not torch.is_tensor(dt) or dt.shape != prefix
        or dt.device != obs.device or dt.dtype != obs.dtype):
        raise ValueError("elapsed dt must match source batch/time and dtype/device")
    if (not torch.is_tensor(writable) or writable.shape != prefix
        or writable.device != obs.device or writable.dtype != torch.bool):
        raise ValueError("write mask must be source-provided boolean batch/time")
    if batched and (obs.shape[0] < 1 or obs.shape[1] < 1):
        raise ValueError("nonempty chronological sequences required")
    if obs.shape[0] < 1 or not all(torch.isfinite(v).all().item() for v in (obs,action,dt)):
        raise ValueError("all observed inputs and elapsed deltas must be finite")
    if (dt < 0).any().item():
        raise ValueError("backward physical clock not permitted")


def _time_features(index, dt, now, last_clock, had_prior):
    age = (now - last_clock).clamp_min(0)
    return torch.stack((
        torch.log1p(index), torch.sin(index / 16.0),
        torch.cos(index / 16.0), torch.log1p(dt),
        torch.log1p(now), torch.log1p(age),
        had_prior.to(dtype=dt.dtype),
    ), dim=-1)


class TimeAwareParallelCore(ParallelTACWorld):
    """Reuse original learned scan operators; add input-only temporal projection."""

    def __init__(self, *, state_dim=14, action_dim=14, width=88, decoder=132):
        super().__init__(state_dim=state_dim, action_dim=action_dim,
                         width=width, decoder=decoder)
        self.time_projection = nn.Linear(7, 48, bias=False)
        # This creates an EXACT initial no-effect temporal path for the parent,
        # with live gradients to the projection on backward.
        nn.init.zeros_(self.time_projection.weight)

    def _embed(self, obs, act, time_features):
        return (self.encoder(torch.cat((obs,act), -1))
                + self.time_projection(time_features)).tanh()

    def _decode_batch(self, obs, e, writable):
        B,T,_ = obs.shape
        k = F.normalize(self.address(e), dim=-1, eps=1e-6)
        v = F.normalize(self.content(e), dim=-1, eps=1e-6)
        zeros_fast = e.new_zeros(B,T,self.width)
        zeros_read = e.new_zeros(B,T,8)
        basis,_ = self.fast_map(torch.cat((e,zeros_fast,zeros_read),-1)).chunk(2,-1)
        basis=basis.tanh()
        old_sum=F.pad(basis.cumsum(dim=1)[:,:-1], (0,0,1,0))
        ctx=old_sum/torch.arange(T,device=e.device,dtype=e.dtype).clamp_min(1).view(1,T,1)
        p,fg=self.fast_map(torch.cat((e,ctx,zeros_read),-1)).chunk(2,-1)
        pf=torch.lerp(ctx,p.tanh(),fg.sigmoid())
        delta,sg=self.slow_map(torch.cat((pf,ctx,zeros_read),-1)).chunk(2,-1)
        ps=torch.lerp(ctx,(ctx+delta.tanh()).tanh(),.25*sg.sigmoid())
        alpha,beta=self.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1)
        alpha=alpha*(1-beta/4.0)
        P,Q=associative_affine_schedule(k,v,alpha,beta,writable.unsqueeze(-1))
        memory=preserve_nonwrite_prefix(work_efficient_affine_prefix(P,Q),writable)
        prev=torch.cat((memory.new_zeros(B,1,8,8),memory[:,:-1]),1)
        r=(prev@k.unsqueeze(-1)).squeeze(-1)
        f,fg=self.fast_map(torch.cat((e,pf,r),-1)).chunk(2,-1)
        f=torch.lerp(pf,f.tanh(),fg.sigmoid())
        d,g=self.slow_map(torch.cat((f,ps,r),-1)).chunk(2,-1)
        s=torch.lerp(ps,(ps+d.tanh()).tanh(),.25*g.sigmoid())
        correction=self.readout(torch.cat((f,s,r,e),-1)).tanh()
        prediction=obs+self.delta_scale*correction
        return prediction, dict(memory=memory,old_read=r,fast=f,slow=s,
                                embedding=e,temporal_gate_schedule=alpha,
                                sum_basis=basis.cumsum(dim=1))

    def forward(self, obs, observed_action, *, elapsed_dt, write_mask, return_trace=False):
        _source_inputs(obs, observed_action, elapsed_dt, write_mask, batched=True)
        if obs.shape[-1] != self.state_dim or observed_action.shape[-1] != self.action_dim:
            raise ValueError("typed source dimensions incorrect")
        B,T=elapsed_dt.shape
        now=elapsed_dt.cumsum(dim=1)
        if not torch.isfinite(now).all().item():
            raise ValueError("elapsed clock overflow")
        written_time=torch.where(write_mask,now,torch.zeros_like(now)).cummax(dim=1).values
        old_written=F.pad(written_time[:,:-1],(1,0))
        prior_count=F.pad(write_mask.to(torch.int64).cumsum(dim=1)[:,:-1],(1,0))
        position=torch.arange(T,device=obs.device,dtype=obs.dtype).view(1,T).expand(B,T)
        phi=_time_features(position,elapsed_dt,now,old_written,prior_count>0)
        e=self._embed(obs,observed_action,phi)
        base,trace=self._decode_batch(obs,e,write_mask)
        if return_trace:
            return base,{**trace,"time_features":phi,
                         "elapsed_clock":now,"last_prior_write_clock":old_written}
        return base

    def init_stream(self, batch:int) -> TimeAwareStreamState:
        if type(batch) is not int or batch<1: raise ValueError("batch must be positive")
        p=next(self.parameters())
        return TimeAwareStreamState(
            p.new_zeros(batch,self.width),
            p.new_zeros(batch,8,8),
            p.new_zeros(batch),p.new_zeros(batch),
            torch.zeros(batch,device=p.device,dtype=torch.bool),0,
        )

    def step(self,state:TimeAwareStreamState,obs,observed_action,*,elapsed_dt,
             write_mask,return_trace=False):
        _source_inputs(obs,observed_action,elapsed_dt,write_mask,batched=False)
        B=obs.shape[0]
        if (obs.shape[-1]!=self.state_dim or observed_action.shape[-1]!=self.action_dim
            or state.count<0 or state.fast_sum.shape!=(B,self.width)
            or state.memory.shape!=(B,8,8)
            or any(v.shape!=(B,) for v in (
                state.elapsed_clock,state.last_write_clock,state.write_seen))
            or state.write_seen.dtype!=torch.bool):
            raise ValueError("invalid typed stream state or observation")
        if not all(v.device==obs.device for v in
                   (state.fast_sum,state.memory,state.elapsed_clock,
                    state.last_write_clock,state.write_seen)):
            raise ValueError("stream state device mismatch")
        if not all(v.dtype==obs.dtype for v in
                   (state.fast_sum,state.memory,state.elapsed_clock,state.last_write_clock)):
            raise ValueError("stream state dtype mismatch")
        now=state.elapsed_clock+elapsed_dt
        if not torch.isfinite(now).all().item():
            raise ValueError("elapsed clock overflow")
        index=torch.full_like(elapsed_dt,float(state.count))
        phi=_time_features(index,elapsed_dt,now,
                           state.last_write_clock,state.write_seen)
        e=self._embed(obs,observed_action,phi)
        k=F.normalize(self.address(e),dim=-1,eps=1e-6)
        v=F.normalize(self.content(e),dim=-1,eps=1e-6)
        zero_fast=torch.zeros_like(state.fast_sum)
        zero_read=e.new_zeros(B,8)
        basis,_=self.fast_map(torch.cat((e,zero_fast,zero_read),-1)).chunk(2,-1)
        basis=basis.tanh()
        ctx=state.fast_sum/max(state.count,1)
        p,fg=self.fast_map(torch.cat((e,ctx,zero_read),-1)).chunk(2,-1)
        pf=torch.lerp(ctx,p.tanh(),fg.sigmoid())
        d,g=self.slow_map(torch.cat((pf,ctx,zero_read),-1)).chunk(2,-1)
        ps=torch.lerp(ctx,(ctx+d.tanh()).tanh(),.25*g.sigmoid())
        alpha,beta=self.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1)
        alpha=alpha*(1-beta/4.)
        old_read=(state.memory@k.unsqueeze(-1)).squeeze(-1)
        candidate=alpha.unsqueeze(-1)*state.memory + (
            beta*(v-alpha*old_read)).unsqueeze(-1)*k.unsqueeze(-2)
        new_memory=torch.where(write_mask[:,None,None],candidate,state.memory)
        f,fg=self.fast_map(torch.cat((e,pf,old_read),-1)).chunk(2,-1)
        f=torch.lerp(pf,f.tanh(),fg.sigmoid())
        d,g=self.slow_map(torch.cat((f,ps,old_read),-1)).chunk(2,-1)
        s=torch.lerp(ps,(ps+d.tanh()).tanh(),.25*g.sigmoid())
        correction=self.readout(torch.cat((f,s,old_read,e),-1)).tanh()
        prediction=obs+self.delta_scale*correction
        next_state=TimeAwareStreamState(
            state.fast_sum+basis,new_memory,now,
            torch.where(write_mask,now,state.last_write_clock),
            state.write_seen | write_mask,state.count+1,
        )
        if return_trace:
            return prediction,next_state,{
                "memory":new_memory,"old_read":old_read,
                "fast":f,"slow":s,"embedding":e,
                "time_features":phi,"elapsed_clock":now,
                "last_prior_write_clock":state.last_write_clock,
            }
        return prediction,next_state


class TimeAwareParallelResidualWorld(ParallelResidualWorld):
    """Optional temporal encoding; unchanged parent FFN and loss contract."""

    def __init__(self, state_dim=14,action_dim=14,width=88,decoder=132,ffn_hidden=12):
        super().__init__(state_dim=state_dim,action_dim=action_dim,
                         width=width,decoder=decoder,ffn_hidden=ffn_hidden)
        self.core=TimeAwareParallelCore(
            state_dim=state_dim,action_dim=action_dim,width=width,decoder=decoder)

    def _post_read(self,trace):
        z=torch.cat((trace["fast"],trace["slow"],
                     trace["old_read"],trace["embedding"]),dim=-1)
        return self.residual_logit.sigmoid()*self.residual_ffn(z).tanh()

    def forward(self, observation,observed_action,*,elapsed_dt,write_mask,
                return_trace=False):
        base,trace=self.core(observation,observed_action,elapsed_dt=elapsed_dt,
                             write_mask=write_mask,return_trace=True)
        c=self._post_read(trace)
        pred=base+self.core.delta_scale*c
        if return_trace:
            return pred,{**trace,"post_scan_correction":c,
                         "post_scan_gate":self.residual_logit.sigmoid()}
        return pred

    def init_stream(self,batch:int)->TimeAwareStreamState:
        return self.core.init_stream(batch)

    def stream_step(self,state,observation,observed_action,*,elapsed_dt,write_mask,
                    return_trace=False):
        base,after,trace=self.core.step(
            state,observation,observed_action,elapsed_dt=elapsed_dt,
            write_mask=write_mask,return_trace=True)
        c=self._post_read(trace)
        pred=base+self.core.delta_scale*c
        if return_trace:
            return pred,after,{**trace,"post_scan_correction":c,
                               "post_scan_gate":self.residual_logit.sigmoid()}
        return pred,after
