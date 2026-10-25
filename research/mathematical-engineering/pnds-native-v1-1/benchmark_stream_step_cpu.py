#!/usr/bin/env python3
"""Measured fixed-state streaming TAC inference: per-step latency independent
of context length (up to hardware/cache effects), same 101527 parameters.

No streaming GPU latency, real robot deployment, power or trained memory
performance claimed. CPU forward-only with single thread, no gradient.
"""
import json,statistics,sys,time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/"src"))
from tac_osm.variants.streaming_parallel_native import init_tac_stream,stream_step
from verify_pinned_source import modules


def run():
    torch.set_num_threads(1)
    torch.manual_seed(20261025)
    source,_=modules()
    model=source.AlignedAssociativeTAC().float().eval()
    assert sum(p.numel() for p in model.parameters())==101527
    kind=torch.tensor([0,1,2,0],dtype=torch.long)
    keys=torch.tensor([1,2,3,4],dtype=torch.long)
    values=torch.tensor([11,12,13,14],dtype=torch.long)
    rows=[]
    with torch.inference_mode():
        for warm_history in (0,128,2048):
            state=init_tac_stream(model,4)
            for _ in range(warm_history):
                _,state=stream_step(model,state,kind,keys,values)
            for _ in range(20):
                _,state=stream_step(model,state,kind,keys,values)
            measurements=[]
            for _ in range(80):
                t0=time.perf_counter_ns()
                logits,state=stream_step(model,state,kind,keys,values)
                measurements.append((time.perf_counter_ns()-t0)/1e6)
            assert logits.shape==(4,32) and torch.isfinite(logits).all().item()
            assert state.sum_fast.numel()==4*88 and state.memory.numel()==4*64
            rows.append({"initial_history":warm_history,"state_elements_per_sample":152,
                         "state_bytes_per_sample_float32":608,
                         "median_step_ms":round(statistics.median(measurements),6),
                         "p95_step_ms":round(sorted(measurements)[75],6),
                         "min_step_ms":round(min(measurements),6),
                         "measurements":80})
    print(json.dumps({"status":"MEASURED_FIXED_STATE_CPU_STEP_NOT_REAL_SKILL",
        "torch_version":torch.__version__,"cpu_threads":1,"batch_size":4,
        "training_steps":0,"persistent_state_grows_with_history":False,
        "same_model_weights_as_batch_candidate":True,
        "real_world_control_or_skill_claim":False,"rows":rows},sort_keys=True))

if __name__=="__main__":run()
