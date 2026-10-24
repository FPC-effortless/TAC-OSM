"""Compare exact original vs batched complete model forward latency, no training.

Hardware, torch version, seed and raw per-repeat timings must accompany any
claim. Wall time is descriptive, NOT a compulsory matched-training gate.
"""
from __future__ import annotations
import argparse
import json
import platform
import statistics
import time
import torch
from tac_osm.variants.selective_exchange_world import SelectiveExchangeWorldModel
from tac_osm.variants.vectorized_exchange_world import VectorizedSelectiveExchangeWorldModel


def trial(batch: int, repeats: int):
    d=14
    torch.manual_seed(11011+batch)
    old=SelectiveExchangeWorldModel()
    new=VectorizedSelectiveExchangeWorldModel()
    new.load_state_dict(old.state_dict(),strict=True)
    original_inputs = [
        (torch.randn(batch,d),torch.randn(batch,d),torch.full((batch,),.1),
         torch.arange(batch,dtype=torch.long),torch.ones(batch,d,dtype=torch.bool))
        for _ in range(2)
    ]
    def run(model):
        state=model.initial_selective(batch)
        for x,a,dt,ident,mask in original_inputs:
            out=model(x,a,dt,ident,mask,state=state)
            state=out.state
        return out
    result={}
    with torch.inference_mode():
        o=run(old);v=run(new)
        max_error=max((o.mean_next-v.mean_next).abs().max().item(),
                      (o.scale_next-v.scale_next).abs().max().item(),
                      (o.proposed_action-v.proposed_action).abs().max().item(),
                      (o.movement_logit-v.movement_logit).abs().max().item(),
                      (o.state.world.memory.values-v.state.world.memory.values).abs().max().item())
        assert max_error <= 1e-5, (batch,max_error)
        for name,model in (("original",old),("batched",new)):
            for _ in range(3):run(model)
            times=[]
            for _ in range(repeats):
                tic=time.perf_counter();run(model);times.append(time.perf_counter()-tic)
            result[name]={"median_s":statistics.median(times),"seconds":times}
    result["median_speedup_original_divided_by_batched"]=result["original"]["median_s"]/result["batched"]["median_s"]
    result["max_abs_original_candidate_output_difference"]=max_error
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repeats",type=int,default=20)
    args=parser.parse_args()
    torch.set_num_threads(1)
    report={"study":"ESWAM011_EQUIVALENT_MEMORY_CPU_TIMING_DIAGNOSTIC",
            "real_world_generalization_proof":False,"trained_model":False,
            "torch_version":torch.__version__,
            "python_platform":platform.platform(),
            "threads":torch.get_num_threads(),
            "matmul_precision":torch.get_float32_matmul_precision(),
            "batches":{str(b):trial(b,args.repeats) for b in (1,8,32,128)}}
    print(json.dumps(report,sort_keys=True,indent=2))
