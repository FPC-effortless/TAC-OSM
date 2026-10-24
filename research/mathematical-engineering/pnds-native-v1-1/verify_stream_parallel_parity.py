#!/usr/bin/env python3
"""Deterministic parity/gradient tests for a *single* dual-mode native TAC candidate.

Parallel full sequence vs fixed-register stream: same parameters, identical
mathematical causal update, toleranced finite-precision equivalence. Neither
path is the original learned nonlinear DIAGKEY004 recurrence.

Tests generated tensors as mechanism diagnostics only: they cannot qualify
real-world long-memory, entity separation or checkpoint skill.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/"src"))
from tac_osm.variants.full_time_parallel_native import full_time_parallel_native_candidate
from tac_osm.variants.streaming_parallel_native import (
    TACStreamState,init_tac_stream,stream_step,streaming_unroll)
from verify_pinned_source import modules


def compare(x,y,dtype,context):
    tol=6e-5 if dtype==torch.float32 else 2e-9
    try: torch.testing.assert_close(x,y,atol=tol,rtol=tol)
    except AssertionError as e: raise AssertionError(f"{context}: {e}") from e


def verify():
    torch.set_num_threads(1)
    torch.manual_seed(20261025)
    source,_=modules()
    checks=0
    max_abs=0.
    for dtype in (torch.float32,torch.float64):
        model=source.AlignedAssociativeTAC().to(dtype=dtype).eval()
        assert sum(p.numel() for p in model.parameters())==101527
        for B,T in ((1,1),(2,2),(2,7),(3,13),(2,32),(2,129),(1,257)):
            kind=torch.randint(0,3,(B,T),dtype=torch.long)
            keys=torch.randint(0,65,(B,T),dtype=torch.long)
            values=torch.randint(0,33,(B,T),dtype=torch.long)
            # Read-only interleaving (some all-read-only, some writable).
            if T>2: kind[:,1::4]=2
            result,trace=full_time_parallel_native_candidate(
                model,kind,keys,values,return_trace=True)
            output,final_state,ref=streaming_unroll(
                model,kind,keys,values,return_trace=True)
            compare(output[:,-1],result,dtype,"final full-parallel vs stream logits")
            checks+=1
            for name in ("memory","old_read","fast","slow","gates","past_context"):
                compare(ref[name],trace[name],dtype,f"B{B}T{T} {name}")
                checks+=1
                max_abs=max(max_abs,(ref[name]-trace[name]).abs().max().item())
            assert final_state.step_count==T
            assert final_state.sum_fast.shape==(B,88)
            assert final_state.memory.shape==(B,8,8)
            checks+=3

            # Chunks and resumed events produce same outputs and bounded
            # persistent state shape, no hidden history or per-timestep cache.
            cut=max(T//2,1)
            if cut<T:
                left,s0=streaming_unroll(model,kind[:,:cut],keys[:,:cut],values[:,:cut])
                right,s1=streaming_unroll(
                    model,kind[:,cut:],keys[:,cut:],values[:,cut:],state=s0)
                compare(torch.cat((left,right),1),output,dtype,"chunked stream equality")
                compare(s1.memory,final_state.memory,dtype,"chunked state")
                checks+=2

            read_mask=(kind==2).unsqueeze(-1).unsqueeze(-1)
            before=torch.cat((ref["memory"].new_zeros(B,1,8,8),
                              ref["memory"][:,:-1]),1)
            assert torch.equal(torch.where(read_mask,ref["memory"],before),before)
            assert torch.linalg.vector_norm(ref["memory"].flatten(-2),dim=-1).max() <=4.00003
            checks+=2

            # New streaming graph must backpropagate to same learned weights as
            # time-parallel graph. Gradient closeness is not task fitness.
            if T in (2,7,32):
                weights=tuple(model.parameters())
                gl=torch.autograd.grad(result.square().sum(),weights,retain_graph=True)
                gs=torch.autograd.grad(output[:,-1].square().sum(),weights)
                assert len(gl)==len(gs)
                for a,b in zip(gl,gs):
                    compare(a,b,dtype,"parameter gradient full-vs-stream")
                    checks+=1

        # Fixed memory footprint independent of entire observed history.
        B=2
        state=init_tac_stream(model,B)
        count0=state.sum_fast.numel()+state.memory.numel()
        kinds=torch.tensor([0,2]); ids=torch.tensor([1,2]); vals=torch.tensor([3,0])
        with torch.no_grad():
            for i in range(2048):
                _,state=stream_step(model,state,kinds,ids,vals)
                if i in (0,7,127,1023,2047):
                    assert state.sum_fast.numel()+state.memory.numel()==count0
                    assert state.step_count==i+1
                    assert state.memory.shape==(B,8,8)
                    checks+=3
        assert torch.isfinite(state.memory).all().item()
        assert torch.linalg.vector_norm(state.memory.flatten(-2),dim=-1).max()<=4.00003
        checks+=2

    print(json.dumps({
      "status":"PASS_DUAL_MODE_PARALLEL_SEQUENCE_AND_FIXED_STATE_STREAM",
      "checks":checks,"max_absolute_intermediate_difference":max_abs,
      "persistent_scalar_count_per_sample":88+8*8,
      "integer_event_count":1,
      "streaming_state_bytes_per_sample_float32":(88+8*8)*4,
      "streaming_memory_grows_with_T":False,
      "associative_per_step_complexity":"O(8^2) via rank-1 update",
      "offline_associative_scan_work":"O(T*8^3) using balanced tree",
      "source_model_semantics_changed":True,
      "trainable_parameters":101527,"new_training_steps":0,
      "real_world_skill":"NOT_EVALUATED"},sort_keys=True))


if __name__=="__main__":
    verify()
