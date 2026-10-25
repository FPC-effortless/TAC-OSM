#!/usr/bin/env python3
"""Structural qualification for a NEW native-weight, full-time-parallel TAC variant.

Do not confuse passing these mechanism tests with source-native equivalence,
retrained skill, real-world task results, or a physical safety guarantee.
All 101527 parameters are reused, NONE newly allocated or fitted.
"""
import sys
from pathlib import Path
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/"src"))
from tac_osm.variants.full_time_parallel_native import (
    full_time_parallel_native_candidate)
from verify_pinned_source import modules


def close(x,y,dtype):
    tol=5e-5 if dtype==torch.float32 else 5e-10
    torch.testing.assert_close(x,y,rtol=tol,atol=tol)


def verify():
    source,_=modules()
    torch.set_num_threads(1)
    torch.manual_seed(20261024)
    tested=0
    for dtype in (torch.float32,torch.float64):
        model=source.AlignedAssociativeTAC().to(dtype=dtype)
        assert sum(p.numel() for p in model.parameters())==101527
        for batch,steps in ((1,1),(2,3),(2,8),(1,33),(2,128)):
            kind=torch.randint(0,3,(batch,steps))
            keys=torch.randint(0,65,(batch,steps))
            values=torch.randint(0,33,(batch,steps))
            # Make all runs include a known read-only timestep.
            kind[:,-1]=2
            logits,trace=full_time_parallel_native_candidate(
                model,kind,keys,values,return_trace=True)
            assert logits.shape==(batch,32)
            assert all(torch.isfinite(x).all().item() for x in
                       (logits,trace["fast"],trace["slow"],trace["memory"]))
            bound=torch.linalg.vector_norm(trace["memory"].flatten(-2),dim=-1)
            assert (bound <=4.0+(2e-5 if dtype==torch.float32 else 1e-10)).all().item()
            assert trace["fast"].abs().max().item() <= 1.00001
            assert trace["slow"].abs().max().item() <= 1.00001
            tested+=4
            prev=torch.cat((trace["memory"].new_zeros(batch,1,8,8),
                            trace["memory"][:,:-1]),1)
            read_only=(kind==2).unsqueeze(-1).unsqueeze(-1)
            torch.testing.assert_close(
                torch.where(read_only,trace["memory"],prev),
                prev,rtol=0,atol=0)
            tested+=1
            # Source-independent gates have an explicit reserve:
            # α_eff R + β <= R. No nonlinear global projection.
            a,b=trace["gates"][...,:1],trace["gates"][...,1:]
            assert (4.0*a+b <=4.0+2e-5).all().item()
            tested+=1

            # Check truncation prefix identity of *every* computed time
            # tensor: no leakage from future keys, values or role markers.
            if steps>=3:
                count=steps//2
                _,short=full_time_parallel_native_candidate(
                    model,kind[:,:count],keys[:,:count],values[:,:count],
                    return_trace=True)
                for k in ("memory","addresses","gates","old_read","fast",
                          "slow","past_context"):
                    close(short[k],trace[k][:,:count],dtype)
                    tested+=1

            # Backprop must reach the entire trainable neural graph,
            # including first key, write strengths and decoder.
            if steps in (3,8,33):
                g=torch.autograd.grad(logits.square().sum(),
                                      tuple(model.parameters()),allow_unused=False)
                assert len(g)==len(list(model.parameters()))
                for x in g:
                    assert torch.isfinite(x).all().item()
                tested+=len(g)

        # Source-native and fully parallel candidate are NOT equivalent even
        # with identical 101527 parameters: their recurrence contracts differ.
        kind=torch.tensor([[0,1,2,0,2]])
        keys=torch.tensor([[1,2,1,3,1]])
        values=torch.tensor([[8,17,0,23,0]])
        old=source.AlignedAssociativeTAC.forward(model,kind,keys,values)
        new=full_time_parallel_native_candidate(model,kind,keys,values)
        if torch.allclose(old,new,rtol=1e-9,atol=1e-9):
            raise AssertionError("semantic-model change not detected")
        tested+=1

        # Deterministic independently-masked sequence tests, source inputs only.
        with torch.no_grad():
            kind=torch.randint(0,3,(2,256))
            keys=torch.randint(0,65,(2,256))
            vals=torch.randint(0,33,(2,256))
            l,tr=full_time_parallel_native_candidate(model,kind,keys,vals,return_trace=True)
            assert torch.isfinite(l).all().item()
            assert (torch.linalg.vector_norm(tr["memory"].flatten(-2),dim=-1)<=4.00003).all().item()
            tested+=2

    print({"status":"PASS_FULL_TIME_PARALLEL_CANDIDATE_STRUCTURAL_TESTS",
           "checks":tested,"trainable_parameters_equal_original":101527,
           "input_to_output_time_parallel_algorithm":True,
           "learned_fast_slow_native_recurrence_preserved":False,
           "gated_affine_memory_prefix_scan_no_global_projection":True,
           "causal_temporal_prefix_checked":True,
           "real_dataset_training_or_skill":False,
           "new_optimizer_steps":0,"original_trained_checkpoint_upgraded":False})


if __name__=="__main__":
    verify()
