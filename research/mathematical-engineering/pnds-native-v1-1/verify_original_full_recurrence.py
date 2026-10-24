#!/usr/bin/env python3
"""Original pinned entire DIAGKEY004 forward/backward source graph parity.

No retraining, no synthetic test accuracy claim, no real dataset or checkpoint.
The same original 101k parameter instance executes both the native forward and
an independently ordered explicit transition, checking ALL per-step memory
outputs, final logits, and ALL model parameter gradients.

A second local state JVP/finite difference check covers nonlinear f/s/M
couplings and role-dependent writes. This is a numerical original-source
refinement diagnostic; it is NOT a Lean whole-model proof nor a contraction,
Lyapunov, long-memory or trained predictive capability certificate.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from verify_pinned_source import modules


def native_step(model,kind,event,address,content,fast,slow,memory,source):
    old_read=torch.bmm(memory,address.unsqueeze(-1)).squeeze(-1)
    proposal,fg=model.fast_map(torch.cat((event,fast,old_read),-1)).chunk(2,-1)
    new_fast=torch.lerp(fast,proposal.tanh(),fg.sigmoid())
    delta,sg=model.slow_map(torch.cat((new_fast,slow,old_read),-1)).chunk(2,-1)
    new_slow=torch.lerp(slow,(slow+delta.tanh()).tanh(),.25*sg.sigmoid())
    alpha,beta=model.memory_gates(torch.cat((event,new_fast,new_slow),-1)).sigmoid().chunk(2,-1)
    alpha=alpha.clamp(max=1.0-1e-4)
    new_memory=source.gated_write(
        memory,address,content,alpha,beta,kind[:,None]!=2)
    return new_fast,new_slow,new_memory,torch.cat((alpha,beta),-1)


def explicit_whole_graph(model,source,kind,keys,values):
    encoded=torch.tanh(model.encoder(
        model.kind_embed(kind)+model.key(keys)+model.value(values)))
    addresses=F.normalize(model.address(keys),dim=-1,eps=1e-6)
    contents=F.normalize(model.content(values),dim=-1,eps=1e-6)
    b,t=kind.shape
    f=encoded.new_zeros(b,model.width)
    s=encoded.new_zeros(b,model.width)
    m=encoded.new_zeros(b,8,8)
    memory_trace=[]
    gate_trace=[]
    for step in range(t):
        f,s,m,g=native_step(model,kind[:,step],encoded[:,step],addresses[:,step],
                            contents[:,step],f,s,m,source)
        memory_trace.append(m)
        gate_trace.append(g)
    read=torch.bmm(m,addresses[:,-1].unsqueeze(-1)).squeeze(-1)
    codebook=F.normalize(model.content.weight[1:],dim=-1,eps=1e-6)
    logits=model.readout(torch.cat((f,s,read,encoded[:,-1]),-1))
    logits=logits+model.read_gain.clamp(-2.,4.).exp()*(read@codebook.T)
    return logits,{"addresses":addresses,"memory":torch.stack(memory_trace,1),
                   "gates":torch.stack(gate_trace,1),"read":read}


def verify():
    torch.set_num_threads(1)
    torch.manual_seed(20261024)
    source,_=modules()
    model=source.AlignedAssociativeTAC().double().eval()
    total=sum(p.numel() for p in model.parameters())
    if total != 101527:
        raise RuntimeError(f"parameter count drift: expected 101527, got {total}")
    checks=0
    for length in (1,3,8):
        for b in (1,2):
            kind=torch.randint(0,3,(b,length),dtype=torch.long)
            kind[:,-1]=2
            keys=torch.randint(0,65,(b,length),dtype=torch.long)
            values=torch.randint(0,33,(b,length),dtype=torch.long)
            expected,trace=model(kind,keys,values,return_trace=True)
            observed,ref=explicit_whole_graph(model,source,kind,keys,values)
            torch.testing.assert_close(expected,observed,atol=1e-11,rtol=1e-11)
            for name in ("addresses","memory","gates","read"):
                torch.testing.assert_close(trace[name],ref[name],atol=1e-11,rtol=1e-11)
                checks+=1
            checks+=1
            params=tuple(model.parameters())
            g1=torch.autograd.grad(expected.square().sum(),params,allow_unused=True)
            g2=torch.autograd.grad(observed.square().sum(),params,allow_unused=True)
            if any(x is None or y is None for x,y in zip(g1,g2)):
                raise RuntimeError("one or more original native gradients missing")
            for x,y in zip(g1,g2):
                torch.testing.assert_close(x,y,atol=5e-9,rtol=5e-8)
            checks+=1

    # Nonlinear local state propagation and source role: full tuple (f,s,M).
    event=torch.randn(1,48,dtype=torch.float64)*.2
    address=F.normalize(torch.randn(1,8,dtype=torch.float64),dim=-1)
    content=F.normalize(torch.randn(1,8,dtype=torch.float64),dim=-1)
    flat=torch.randn(240,dtype=torch.float64)*.03
    direction=F.normalize(torch.randn(240,dtype=torch.float64),dim=0)
    for event_kind in (0,2):
        kind=torch.tensor([event_kind])
        def fstate(z):
            f=z[:88].reshape(1,88)
            s=z[88:176].reshape(1,88)
            m=z[176:].reshape(1,8,8)
            f1,s1,m1,_=native_step(model,kind,event,address,content,f,s,m,source)
            return torch.cat((f1.flatten(),s1.flatten(),m1.flatten()))
        y, jvp=torch.autograd.functional.jvp(fstate,flat,direction,create_graph=False)
        eps=1e-5
        numerical=(fstate(flat+eps*direction)-fstate(flat-eps*direction))/(2*eps)
        torch.testing.assert_close(jvp,numerical,atol=2e-7,rtol=2e-5)
        if event_kind==2:
            torch.testing.assert_close(y[176:],flat[176:],rtol=0.,atol=0.)
        if not torch.isfinite(jvp).all().item():
            raise RuntimeError("nonfinite nonlinear state directional Jacobian")
        checks+=2

    print({"status":"PASS_ORIGINAL_WHOLE_DIAGKEY004_NUMERICAL_GRAPH_ONLY",
           "checked_outputs_per_step_gradients_and_JVP":checks,
           "original_source_commit":"081e534a074001c6a3b2251b3ff31ad20ec4fca8",
           "trainable_parameters":total,"new_training_steps":0,
           "trained_checkpoint_replayed":False,"real_world_evaluation":False,
           "full_Lean_proof":False,"global_stability_proven":False})


if __name__=="__main__":
    verify()
