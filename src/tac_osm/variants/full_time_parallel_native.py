"""TAC-OSM native-operator *candidate* with a mathematically time-parallel DAG.

Entire input→output path is time-parallel up to O(log T) prefix stages:
1. [B,T] causal event embeddings and normalized key/value encodings;
2. time-parallel source fast-map feature and causal prefix-mean context;
3. time-parallel fast/slow provisional features and gate schedules;
4. exact affine-associative batched 8x8 chronological prefix scan;
5. all-time read-before-write, parallel fast/slow read refinement, final read;
6. same original native codebook/readout into final logits.

All 101,527 parameters are the original AlignedAssociativeTAC modules, reused
without additional trainable parameters. NO weights changed by this class.
This is *NOT numerically equivalent* to the original nonlinear recurrent TAC:
its fast/slow state is a causal parallel summary, not old-state recurrence;
retention gates are computed before any memory read (necessary for scan), and
its global Frobenius projection is replaced by a provable conservative
per-write norm budget. Therefore this is a distinct, untrained candidate for
a genuine real-data matched architecture experiment, NOT an upgrade to the
existing checkpoint or a demonstrated intelligence gain.

For each normalized 8D k,v and strength beta in [0,1], define
 alpha_eff = alpha_raw * (1 - beta/R), R=4, alpha_raw in [0,1].
 P=alpha_eff (I - beta*k*kᵀ), Q=beta*v*kᵀ.
Because ||I-beta*k*kᵀ||_2<=1 and ||Q||_F<=beta, we have
 ||M_next||_F <= alpha_eff ||M_prev||_F+beta <= R
whenever ||M_prev||_F<=R (assuming k/v norms <=1). This allows exact
parallel affine scan without a nonlinear global projection. Read-only events
remain P=I,Q=0. This is a sufficient boundedness condition, not a proof of
long-memory quality. Heavy writes necessarily spend finite retention budget.

Uses torch.cumsum (parallelizable reduction) and parallel_affine_prefix.
There is NO loop over sequence steps. Stage count O(log T) small matrix
multiplications, but memory/work and actual CPU/GPU time require measurement.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from .parallel_affine_memory import associative_affine_schedule, work_efficient_affine_prefix


def full_time_parallel_native_candidate(model, kind, keys, values,
                                        *, return_trace=False, radius=4.0):
    if (kind.ndim != 2 or keys.shape!=kind.shape or values.shape!=kind.shape
        or kind.shape[1] < 1):
        raise ValueError("kind,keys,values require nonempty matching [B,T]")
    if radius < 1 or not isinstance(radius,(int,float)):
        raise ValueError("radius>=1 required for beta∈[0,1] budget proof")

    B,T=kind.shape
    encoded=torch.tanh(model.encoder(
        model.kind_embed(kind)+model.key(keys)+model.value(values)))
    addresses=F.normalize(model.address(keys),dim=-1,eps=1e-6)
    contents=F.normalize(model.content(values),dim=-1,eps=1e-6)
    zeros_fast=encoded.new_zeros(B,T,model.width)
    zeros_read=encoded.new_zeros(B,T,8)

    # Construct a causal time-parallel fast information basis using original
    # fast-map weights. Prefix *excludes current event* at every position.
    fast_basis,_=model.fast_map(torch.cat(
        (encoded,zeros_fast,zeros_read),-1)).chunk(2,-1)
    fast_basis=fast_basis.tanh()
    prefix=F.pad(fast_basis.cumsum(dim=1)[:,:-1],(0,0,1,0))
    count=torch.arange(T,device=encoded.device,dtype=encoded.dtype).clamp_min(1)
    past_context=prefix/count.reshape(1,T,1)

    fast_prop,fast_gate=model.fast_map(torch.cat(
        (encoded,past_context,zeros_read),-1)).chunk(2,-1)
    provisional_fast=torch.lerp(past_context,fast_prop.tanh(),fast_gate.sigmoid())
    slow_delta,slow_gate=model.slow_map(torch.cat(
        (provisional_fast,past_context,zeros_read),-1)).chunk(2,-1)
    provisional_slow=torch.lerp(
        past_context,(past_context+slow_delta.tanh()).tanh(),
        .25*slow_gate.sigmoid())

    raw_retention,strength=model.memory_gates(torch.cat(
        (encoded,provisional_fast,provisional_slow),-1)).sigmoid().chunk(2,-1)
    # Genuine *source-independent* write gate schedule. Global radius bound
    # without projection, provided normalized k and v.
    retention=raw_retention*(1-strength/radius)
    writable=kind.ne(2).unsqueeze(-1)
    pair_P,pair_Q=associative_affine_schedule(
        addresses,contents,retention,strength,writable)
    state_at_each_step=work_efficient_affine_prefix(pair_P,pair_Q)

    # Exact bitwise *read-only* trace identity in floating point:
    # even mathematically neutral identity-matrix matmul can round the
    # associative prefix by a ULP if its grouping changes. Gather each
    # read-only timestep from the last actual write (or initial zero)
    # using a parallel prefix-maximum over write indices. This does not
    # change writable memory states or introduce any time-step loop.
    write_indices=torch.arange(T,device=kind.device).view(1,T).expand(B,T)
    last_write=torch.cummax(torch.where(
        writable.squeeze(-1),write_indices,
        torch.full_like(write_indices,-1)),dim=1).values
    pick=last_write.clamp_min(0).unsqueeze(-1).unsqueeze(-1).expand(B,T,8,8)
    state_at_each_step=state_at_each_step.gather(1,pick)
    state_at_each_step=torch.where(
        (last_write>=0).view(B,T,1,1),state_at_each_step,
        state_at_each_step.new_zeros(B,T,8,8))

    old_states=torch.cat(
        (state_at_each_step.new_zeros(B,1,8,8),state_at_each_step[:,:-1]),dim=1)
    old_read=(old_states@addresses.unsqueeze(-1)).squeeze(-1)

    # All-time learned observation+recall refinement is independent across
    # timesteps; it cannot revise the already computed memory gate schedule.
    proposal,gate=model.fast_map(torch.cat(
        (encoded,provisional_fast,old_read),-1)).chunk(2,-1)
    f=torch.lerp(provisional_fast,proposal.tanh(),gate.sigmoid())

    delta,g=model.slow_map(torch.cat((f,provisional_slow,old_read),-1)).chunk(2,-1)
    s=torch.lerp(provisional_slow,(provisional_slow+delta.tanh()).tanh(),
                 .25*g.sigmoid())

    last_read=(state_at_each_step[:,-1]@
               addresses[:,-1].unsqueeze(-1)).squeeze(-1)
    codebook=F.normalize(model.content.weight[1:],dim=-1,eps=1e-6)
    logits=model.readout(torch.cat((f[:,-1],s[:,-1],last_read,encoded[:,-1]),-1))
    logits=logits+model.read_gain.clamp(-2,4).exp()*(last_read@codebook.T)

    if return_trace:
        return logits,{"memory":state_at_each_step,"addresses":addresses,
                       "gates":torch.cat((retention,strength),-1),
                       "old_read":old_read,"fast":f,"slow":s,
                       "past_context":past_context}
    return logits
