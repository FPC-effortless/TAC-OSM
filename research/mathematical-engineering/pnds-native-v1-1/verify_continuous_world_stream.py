#!/usr/bin/env python3
"""Continuous TAC world batch/stream equivalence, memory and gradient tests.

Numerical generated inputs exclusively for implementation verification.
Separate real-source testing and world-model capability require authentic
recorded observations/actions/outcomes and strict training-compute parity.
"""
import sys,json,torch
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/"src"))
from tac_osm.variants.parallel_continuous_world import ParallelTACWorld,RealGRUWorld
from tac_osm.variants.streaming_continuous_world import init_world_stream,world_stream_step


def check():
    torch.set_num_threads(1)
    torch.manual_seed(20261025)
    checks=0
    max_error=0.
    for dtype in (torch.float32,torch.float64):
        model=ParallelTACWorld().to(dtype)
        gru=RealGRUWorld().to(dtype)
        pcount=sum(p.numel() for p in model.parameters())
        gcount=sum(p.numel() for p in gru.parameters())
        assert pcount==93324 and gcount==93454
        assert abs(pcount-gcount)/gcount<.01
        checks+=3
        for B,T in ((1,1),(2,5),(2,17),(3,33),(2,65)):
            obs=torch.randn(B,T,14,dtype=dtype)
            act=torch.randn(B,T,14,dtype=dtype)
            yp,ref=model(obs,act,return_trace=True)
            state=init_world_stream(model,B)
            predictions=[];traces=[]
            for t in range(T):
                y,state,tr=world_stream_step(model,state,obs[:,t],act[:,t],
                                             return_trace=True)
                predictions.append(y)
                traces.append(tr)
            actual=torch.stack(predictions,1)
            tol=4e-5 if dtype==torch.float32 else 5e-9
            torch.testing.assert_close(actual,yp,atol=tol,rtol=tol)
            checks+=1
            for name in ("memory","old_read","fast","slow"):
                a=torch.stack([tr[name] for tr in traces],1)
                torch.testing.assert_close(a,ref[name],atol=tol,rtol=tol)
                max_error=max(max_error,(a-ref[name]).abs().max().item())
                checks+=1
            assert state.memory.numel()==B*64 and state.fast_sum.numel()==B*88
            checks+=2
            if T in (5,17):
                weights=tuple(model.parameters())
                # Gradient across every parameter including causal memory
                # addressing and all original-style fast/slow maps.
                d1=torch.autograd.grad(yp.square().sum(),weights,retain_graph=True)
                d2=torch.autograd.grad(actual.square().sum(),weights)
                for a,b in zip(d1,d2):
                    torch.testing.assert_close(a,b,atol=tol,rtol=tol)
                    checks+=1
        state=init_world_stream(model,2)
        with torch.no_grad():
            for t in range(2048):
                y,state=world_stream_step(
                    model,state,torch.randn(2,14,dtype=dtype),
                    torch.randn(2,14,dtype=dtype))
            assert state.memory.numel()+state.fast_sum.numel()==2*152
            assert state.count==2048
            assert torch.linalg.vector_norm(state.memory.flatten(-2),dim=-1).max()<4.0001
            checks+=3
    print(json.dumps({"status":"PASS_CONTINUOUS_WORLD_PARALLEL_STREAM_EQUATION",
       "tests":checks,"max_abs_trace_diff":max_error,
       "parameters_TAC":93324,"parameters_GRU":93454,
       "source_style_memory":True,"stream_float_memory_per_sample":152,
       "trainable_parity_gap_percent":round(abs(93324-93454)/93454*100,4),
       "trained_real_skill_claim":"NONE"},sort_keys=True))

if __name__=="__main__":check()
