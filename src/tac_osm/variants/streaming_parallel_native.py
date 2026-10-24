"""Constant-size streaming execution for the separate fully-parallel TAC candidate.

This is EXACTLY the candidate's mathematical graph, not the original
DIAGKEY004 fast/slow recurrence. Each stream stores:
  sum_fast[B,88], memory[B,8,8], step_count:int.
No history/cache grows with T. The last output is produced on every event.

The batch candidate computes its causal fast context with torch.cumsum and
its memory with balanced matrix-scan; due to nonassociative floating-point
arithmetic, exact bitwise equality is NOT promised. Numerical output,
intermediate trace and parameter-gradient closeness must be tested.

Causal input at time t: kind/key/value IDs already observed. Role=2 keeps
memory bit-for-bit. Write strength and retention are precomputed from
independent fast/slow proposals, not previous memory. After the update,
the readout uses M_t at current key. No future inputs are inspected.

A streaming step uses the rank-one right-affine formula directly:
  M_new = a*M_old + (beta*v-a*beta*M_old@k) outer k.
This is equivalent in exact real arithmetic to M@P+Q. Cost O(B*8²)
per event, rather than a generic O(8³) matrix-matrix multiply.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TACStreamState:
    sum_fast: torch.Tensor  # [B,88] prefix statistic (not all embeddings)
    memory: torch.Tensor    # [B,8,8] bounded value-by-key store
    step_count: int         # scalar chronological event count


def init_tac_stream(model, batch: int, *, device=None, dtype=None) -> TACStreamState:
    if batch < 1:
        raise ValueError("batch must be positive")
    p=next(model.parameters())
    device=p.device if device is None else device
    dtype=p.dtype if dtype is None else dtype
    return TACStreamState(torch.zeros(batch,model.width,device=device,dtype=dtype),
                          torch.zeros(batch,8,8,device=device,dtype=dtype),0)


def stream_step(model, state: TACStreamState,
                kind: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                *, radius: float=4.0, return_trace: bool=False):
    """Single chronological event. Returns (logits[B,32], next_state[,trace])."""
    if (kind.ndim!=1 or keys.shape!=kind.shape or values.shape!=kind.shape or
        kind.dtype!=torch.long or keys.dtype!=torch.long or values.dtype!=torch.long):
        raise ValueError("kind/key/value must be matching int64 [B]")
    B=kind.shape[0]
    if (state.sum_fast.shape!=(B,model.width) or state.memory.shape!=(B,8,8)
        or state.step_count<0 or state.memory.device!=state.sum_fast.device
        or state.memory.dtype!=state.sum_fast.dtype):
        raise ValueError("invalid fixed-size streaming state")
    if not isinstance(radius,(int,float)) or not math.isfinite(radius) or radius<1.:
        raise ValueError("radius must be finite and >=1")
    if not (kind.device==keys.device==values.device==state.memory.device):
        raise ValueError("inputs and state must be on same device")
    if not (torch.isfinite(state.sum_fast).all().item()
            and torch.isfinite(state.memory).all().item()):
        raise ValueError("nonfinite state")

    # Use 3D tensor inputs, identical to the batch model's learned mappings.
    kinds,ids,vals=kind[:,None],keys[:,None],values[:,None]
    event=torch.tanh(model.encoder(
        model.kind_embed(kinds)+model.key(ids)+model.value(vals)))
    k=F.normalize(model.address(ids),dim=-1,eps=1e-6)
    v=F.normalize(model.content(vals),dim=-1,eps=1e-6)
    zeros_fast=torch.zeros_like(state.sum_fast)[:,None,:]
    zeros_read=event.new_zeros(B,1,8)
    basis,_=model.fast_map(torch.cat((event,zeros_fast,zeros_read),-1)).chunk(2,-1)
    basis=basis.tanh()

    # Constant-size sufficient causal statistic for exclusive prefix mean.
    ctx=(state.sum_fast/max(state.step_count,1))[:,None,:]
    fast_prop,fast_gate=model.fast_map(
        torch.cat((event,ctx,zeros_read),-1)).chunk(2,-1)
    provisional_f=torch.lerp(ctx,fast_prop.tanh(),fast_gate.sigmoid())
    delta,slow_gate=model.slow_map(
        torch.cat((provisional_f,ctx,zeros_read),-1)).chunk(2,-1)
    provisional_s=torch.lerp(ctx,(ctx+delta.tanh()).tanh(),
                             .25*slow_gate.sigmoid())

    raw_alpha,beta=model.memory_gates(
        torch.cat((event,provisional_f,provisional_s),-1)).sigmoid().chunk(2,-1)
    alpha=raw_alpha*(1-beta/radius)
    writable=kinds.ne(2)
    old_m=state.memory
    # Read from previous memory before the current write.
    old_read=(old_m.unsqueeze(1)@k.unsqueeze(-1)).squeeze(-1)
    old_key_read=old_read
    # Low-rank representation of the same scheduled right-affine update.
    effective_alpha=torch.where(writable,alpha,torch.ones_like(alpha))
    effective_beta=torch.where(writable,beta,torch.zeros_like(beta))
    decayed=old_m.unsqueeze(1)*effective_alpha.unsqueeze(-1)
    innovation=v-effective_alpha*old_key_read
    updated=decayed+(
        (effective_beta*innovation).unsqueeze(-1)*k.unsqueeze(-2))
    new_m=torch.where(writable.unsqueeze(-1),updated,old_m.unsqueeze(1))[:,0]

    new_basis_sum=state.sum_fast+basis[:,0]
    next_state=TACStreamState(new_basis_sum,new_m,state.step_count+1)
    proposal,gate=model.fast_map(
        torch.cat((event,provisional_f,old_read),-1)).chunk(2,-1)
    refined_f=torch.lerp(provisional_f,proposal.tanh(),gate.sigmoid())
    d,g=model.slow_map(torch.cat((refined_f,provisional_s,old_read),-1)).chunk(2,-1)
    refined_s=torch.lerp(provisional_s,
                         (provisional_s+d.tanh()).tanh(),
                         .25*g.sigmoid())

    final_read=(new_m.unsqueeze(1)@k.unsqueeze(-1)).squeeze(-1)[:,0]
    codebook=F.normalize(model.content.weight[1:],dim=-1,eps=1e-6)
    logits=model.readout(torch.cat(
        (refined_f[:,0],refined_s[:,0],final_read,event[:,0]),-1))
    logits=logits+model.read_gain.clamp(-2,4).exp()*(final_read@codebook.T)

    if return_trace:
        trace={"memory":new_m,"old_read":old_read[:,0],
               "fast":refined_f[:,0],"slow":refined_s[:,0],
               "gates":torch.cat((alpha,beta),-1)[:,0],
               "past_context":ctx[:,0]}
        return logits,next_state,trace
    return logits,next_state


def streaming_unroll(model,kind,keys,values, *,
                     state: TACStreamState|None=None, radius: float=4.0,
                     return_trace=False):
    """Reference chronological fold for streaming; fixed state O(B·(88+64)).

    This wrapper materializes all output logits/traces only for verification.
    Production deployment calls stream_step and retains one state, not outputs.
    """
    if kind.ndim!=2 or keys.shape!=kind.shape or values.shape!=kind.shape or kind.shape[1]==0:
        raise ValueError("nonempty B,T input shapes required")
    state=init_tac_stream(model,kind.shape[0]) if state is None else state
    outputs=[]
    traces=[]
    for t in range(kind.shape[1]):
        logits,state,trace=stream_step(
            model,state,kind[:,t],keys[:,t],values[:,t],
            radius=radius,return_trace=True)
        outputs.append(logits)
        traces.append(trace)
    if not return_trace:
        return torch.stack(outputs,1),state
    trace={k:torch.stack([tr[k] for tr in traces],1) for k in traces[0]}
    return torch.stack(outputs,1),state,trace
