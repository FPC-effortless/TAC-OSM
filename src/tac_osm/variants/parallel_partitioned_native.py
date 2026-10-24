"""Source-preserving parallel input/output path for the original 101,527
parameter DIAGKEY004 TAC model, with a strictly sequential nonlinear state
core. This is a *mathematical computation-graph partition*, not a claim of
parallel native-state recursion.

All time-indexed embeddings and input-only affine contributions to fast
and memory gates are computed by batched PyTorch GEMMs [B,T,*] before the
state loop. Final codebook and output routing are batched. The nonlinear
read→fast→slow→memory write transition remains causal and ordered. Exact
real linearity justifies splitting affine matrices; floating-point reduction
order can change, so equivalence is tested numerically in FP32/FP64 including
source traces and all parameter gradients.

The source architecture, weights, state width, masks, read-before-write,
Frobenius projection and public output are unchanged. This code is an
optional exact-semantics execution mode for a source-frozen research model.
It has NO newly learned parameters or reported performance improvement.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F


def parallel_partitioned_forward(model, kind, keys, values, return_trace=False):
    if (kind.ndim!=2 or keys.shape != kind.shape or values.shape != kind.shape or
            kind.shape[1] == 0):
        raise ValueError("input ids must have identical nonempty [B,T] shapes")

    # Batched time-parallel observation encoding and memory key/value
    # preparation, as already legitimate in the original source.
    encoded=torch.tanh(model.encoder(
        model.kind_embed(kind)+model.key(keys)+model.value(values)))
    addresses=F.normalize(model.address(keys),dim=-1,eps=1e-6)
    contents=F.normalize(model.content(values),dim=-1,eps=1e-6)

    width=model.width
    # Exact-real affine partition:
    # W_fast[event,old_fast,old_read] + b =
    #      (W_event[event]+b) + W_fast[old_fast,old_read].
    # The first term is batch/time parallel and reusable for the whole sequence.
    fast_event=F.linear(encoded,model.fast_map.weight[:,:48],model.fast_map.bias)
    fast_state_weight=model.fast_map.weight[:,48:]

    # Learned memory gates likewise split into an all-time event projection
    # and the state-dependent (f_new,s_new) contribution.
    memory_event=F.linear(encoded,model.memory_gates.weight[:,:48],model.memory_gates.bias)
    memory_state_weight=model.memory_gates.weight[:,48:]

    b,t=kind.shape
    f=encoded.new_zeros(b,width)
    s=f.clone()
    m=encoded.new_zeros(b,8,8)
    states=[]
    gates=[]
    for j in range(t):
        recall=torch.bmm(m,addresses[:,j].unsqueeze(-1)).squeeze(-1)
        proposal,fast_gate=(
            fast_event[:,j]+F.linear(torch.cat((f,recall),-1),
                                     fast_state_weight)
        ).chunk(2,-1)
        f=torch.lerp(f,proposal.tanh(),fast_gate.sigmoid())

        delta,slow_gate=model.slow_map(torch.cat((f,s,recall),-1)).chunk(2,-1)
        s=torch.lerp(s,(s+delta.tanh()).tanh(),.25*slow_gate.sigmoid())

        alpha,beta=(
            memory_event[:,j]+F.linear(torch.cat((f,s),-1),
                                       memory_state_weight)
        ).sigmoid().chunk(2,-1)
        alpha=alpha.clamp(max=1.0-1e-4)

        # Use original gated writer exactly, including read-only identity
        # and **global** radius-four norm projection.
        # Identical source-native writer formula below avoids dependence on
        # the source model's import path; each operation is checked against
        # the pinned original numerical reference.
        writable=kind[:,j:j+1]!=2
        keep=torch.where(writable,alpha,torch.ones_like(alpha))
        strength=torch.where(writable,beta,torch.zeros_like(beta))
        decayed=m*keep.unsqueeze(-1)
        innovation=contents[:,j]-torch.bmm(
            decayed,addresses[:,j].unsqueeze(-1)).squeeze(-1)
        updated=decayed+(strength*innovation).unsqueeze(-1)*addresses[:,j].unsqueeze(1)
        norm=updated.square().sum((1,2),keepdim=True).clamp_min(1e-12).sqrt()
        projected=updated/(norm/4.0).clamp_min(1.0)
        m=torch.where(writable.unsqueeze(-1),projected,m)

        if return_trace:
            states.append(m)
            gates.append(torch.cat((alpha,beta),-1))

    read=torch.bmm(m,addresses[:,-1].unsqueeze(-1)).squeeze(-1)
    codebook=F.normalize(model.content.weight[1:],dim=-1,eps=1e-6)
    logits=model.readout(torch.cat((f,s,read,encoded[:,-1]),-1))
    logits=logits+model.read_gain.clamp(-2.,4.).exp()*(read@codebook.T)
    if return_trace:
        return logits,{"addresses":addresses,"memory":torch.stack(states,1),
                       "gates":torch.stack(gates,1),"read":read}
    return logits
