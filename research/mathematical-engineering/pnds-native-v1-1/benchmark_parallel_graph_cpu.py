#!/usr/bin/env python3
"""Illustrative measured CPU forward resource comparison; never a skill gate.

Same original 101527 parameters and identical observed integer inputs for
all three execution paths. The two architectures are NOT semantically
equivalent; runtime ranking does not establish predictive skill, training
fairness or physical viability. Torch is single-threaded CPU and costs
depend on software/hardware. No performance threshold is used to claim
success; publish the absolute results, even if parallel is slower.
"""
from __future__ import annotations
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/"src"))
from tac_osm.variants.parallel_partitioned_native import parallel_partitioned_forward
from tac_osm.variants.full_time_parallel_native import full_time_parallel_native_candidate
from verify_pinned_source import modules


def bench_one(f,reps=5):
    timings=[]
    with torch.inference_mode():
        for _ in range(2):
            out=f()
            if not torch.isfinite(out).all().item():
                raise RuntimeError("nonfinite benchmark output")
        for _ in range(reps):
            t0=time.perf_counter()
            out=f()
            if not torch.isfinite(out).all().item():
                raise RuntimeError("nonfinite benchmark output")
            timings.append(1000*(time.perf_counter()-t0))
    return {"median_ms":round(statistics.median(timings),5),
            "min_ms":round(min(timings),5),
            "max_ms":round(max(timings),5),"repeats":reps}


def run():
    torch.set_num_threads(1)
    torch.manual_seed(20261024)
    source,_=modules()
    model=source.AlignedAssociativeTAC().float().eval()
    params=sum(p.numel() for p in model.parameters())
    assert params==101527
    rows=[]
    for T in (32,128,256):
        B=4
        kind=torch.randint(0,3,(B,T))
        keys=torch.randint(0,65,(B,T))
        vals=torch.randint(0,33,(B,T))
        methods={
          "original_serial_native":lambda:model(kind,keys,vals),
          "parallel_input_same_output_native":lambda:parallel_partitioned_forward(
              model,kind,keys,vals),
          "new_full_time_parallel_candidate":lambda:full_time_parallel_native_candidate(
              model,kind,keys,vals)
        }
        measures={}
        # Alternate the runner order by T to reduce ordering bias. This
        # remains illustrative, not a controlled hardware benchmark.
        names=list(methods)
        if T==128:names=list(reversed(names))
        for name in names:
            measures[name]=bench_one(methods[name])
        rows.append({"batch":B,"history":T,"device":"cpu",
            "dtype":"float32","parameter_count":params,"timings":measures,
            "parallel_candidate_output_equivalent_to_original":False,
            "training_compute_matched":False,
            "real_world_skill_established":False})
    result={"status":"MEASURED_CPU_FORWARD_ONLY_NOT_QUALIFICATION",
            "torch_version":torch.__version__,"threads":torch.get_num_threads(),
            "new_training_steps":0,"rows":rows}
    print(json.dumps(result,sort_keys=True))
    return result


if __name__=="__main__":
    run()
