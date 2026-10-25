#!/usr/bin/env python3
"""Source-anchored mathematical equivalence tests for bounded four-timescale TAC.

Generated vectors are mechanism tests ONLY. No real-world skill inferred.
"""
import json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/"src"))
from tac_osm.variants.stable_timescale_world import (
  StableTimescaleTACWorld,init_stable_stream,stable_stream_step,
  decay_bands,balanced_elementwise_affine_prefix)
from tac_osm.variants.parallel_continuous_world import ParallelTACWorld


def close(a,b,dtype,label):
    tol=8e-5 if dtype==torch.float32 else 3e-9
    try:
        torch.testing.assert_close(a,b,atol=tol,rtol=tol)
    except AssertionError as e:
        raise AssertionError(f"{label}: {e}") from e


def verify():
    torch.manual_seed(20261025)
    torch.set_num_threads(1)
    checks=0
    maxdiff=0.
    for dtype in (torch.float32,torch.float64):
        model=StableTimescaleTACWorld().to(dtype)
        base=ParallelTACWorld().to(dtype)
        assert sum(p.numel() for p in model.parameters())==sum(
            p.numel() for p in base.parameters())==93324
        assert not any("timescale" in n or "gamma" in n for n,_ in model.named_parameters())
        checks+=2
        for B,T in ((1,1),(2,2),(3,7),(2,33),(2,129),(1,257)):
            obs=torch.randn(B,T,14,dtype=dtype)
            act=torch.randn(B,T,14,dtype=dtype)
            y,trace=model(obs,act,return_trace=True)
            state=init_stable_stream(model,B)
            predictions=[]
            logs={key:[] for key in ("memory","context","summary","fast","slow","old_read")}
            for t in range(T):
                yp,state,tr=stable_stream_step(model,state,obs[:,t],act[:,t],return_trace=True)
                predictions.append(yp)
                for name in logs:logs[name].append(tr[name])
            close(y,torch.stack(predictions,1),dtype,"batch-vs-stream outcome")
            checks+=1
            for key in logs:
                stream=torch.stack(logs[key],1)
                close(trace[key],stream,dtype,f"B{B}T{T} {key}")
                maxdiff=max(maxdiff,(trace[key]-stream).abs().max().item())
                checks+=1
            assert state.summary.shape==(B,88) and state.memory.shape==(B,8,8)
            assert torch.max(torch.abs(trace["summary"])).item()<=1.00001
            assert torch.linalg.vector_norm(trace["memory"].flatten(-2),dim=-1).max().item()<4.0001
            checks+=3
            # Same per-parameter gradients across batch and streaming modes.
            if T in (2,7,33):
                gs1=torch.autograd.grad(y.square().sum(),tuple(model.parameters()),retain_graph=True)
                gs2=torch.autograd.grad(torch.stack(predictions,1).square().sum(),
                                        tuple(model.parameters()))
                for g1,g2 in zip(gs1,gs2):
                    close(g1,g2,dtype,"all model parameter gradients")
                    checks+=1
            if T>2:
                truncated,short=model(obs[:,:T//2],act[:,:T//2],return_trace=True)
                for k in logs:
                    close(short[k],trace[k][:,:T//2],dtype,"causal future exclusion")
                    checks+=1
        for T in (1,2,3,8,31,128,257):
            B,D=2,88
            g=decay_bands(D,"cpu",dtype)[None,None]
            basis=torch.randn(B,T,D,dtype=dtype).tanh().detach().requires_grad_(True)
            scan=balanced_elementwise_affine_prefix(
                (1-g).expand(B,T,D),g*basis)
            s=torch.zeros(B,D,dtype=dtype)
            hist=[]
            for t in range(T):
                s=(1-g[:,0])*s+g[:,0]*basis[:,t]
                hist.append(s)
            seq=torch.stack(hist,1)
            close(scan,seq,dtype,"scalar affine scan time order")
            checks+=1
            if T in (3,31):
                first=torch.autograd.grad(scan.square().sum(),basis,retain_graph=True)[0]
                second=torch.autograd.grad(seq.square().sum(),basis)[0]
                close(first,second,dtype,"affine scalar scan autograd")
                checks+=1
        state=init_stable_stream(model,2)
        with torch.no_grad():
            obs=torch.randn(2,14,dtype=dtype)
            action=torch.randn(2,14,dtype=dtype)
            for i in range(4096):
                _,state=stable_stream_step(model,state,obs,action)
            assert state.summary.abs().amax().item()<=1.00001
            assert torch.linalg.vector_norm(state.memory.flatten(-2),dim=-1).max().item()<=4.0001
            assert state.summary.numel()+state.memory.numel()==2*152
            assert state.count==4096
            checks+=4
    print(json.dumps({"status":"PASS_BOUNDED_MULTISCALE_DUAL_MODE",
      "checks":checks,"maximum_intermediate_difference":maxdiff,
      "per_sample_persistent_fp32_bytes":608,
      "learned_parameters_Tac":93324,"trained_real_gain_established":False,
      "history_tested":4096,"fixed_decay_bands":[.125,.0625,.015625,.00390625],
      "same_original_native_recurrent_graph":False},sort_keys=True))


if __name__=="__main__":verify()
