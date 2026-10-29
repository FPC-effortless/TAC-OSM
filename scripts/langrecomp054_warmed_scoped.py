#!/usr/bin/env python3
"""LANGRECOMP054: warmed source-authentic training-graph checkpointing Pareto.

Original trained 499623p TAC checkpoint only. CPU FP32 original next-byte
FIT windows, no optimizer step; GPU peak/GRAPH011 claim NOT measured.
"""
from __future__ import annotations
import argparse,contextlib,json,math,platform,statistics,sys,time
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.langrecomp053_training_parity import (
    GEOMETRY,RNG_SEED,one_pass,compare_outputs,source_batch)
from scripts.langrecomp052_replay import load_original
from scripts.langlayer035_sampling import fit_sources

POLICIES=("native","scoped_upper","checkpoint_entire_graph")
WARMUPS=2
TIMED=5

@contextlib.contextmanager
def checkpoint_original_upper_only(model):
    """Pure functional interception: no new parameters / state dictionary entries.

    The exact original .layernorm registered sequence is executed under a
    non-reentrant checkpoint only. The forward method is restored afterward.
    Backward must happen before exiting; all gradient paths remain live.
    """
    module=model.layernorm
    orig=module.forward
    def wrapped(z):
        return checkpoint(orig,z,use_reentrant=False,preserve_rng_state=True,
                          determinism_check="default")
    module.forward=wrapped
    try:
        yield
    finally:
        module.forward=orig

def execute(model,x,y,arm):
    if arm=="scoped_upper":
        with checkpoint_original_upper_only(model):
            return one_pass(model,x,y,"native")
    if arm=="checkpoint_entire_graph":
        return one_pass(model,x,y,"checkpoint_entire_graph")
    if arm=="native":
        return one_pass(model,x,y,"native")
    raise ValueError("unsupported checkpoint scope")

def dist(values):
    if len(values)!=TIMED:raise ValueError("not enough matched warmed repeats")
    x=sorted(values)
    return {"median":statistics.median(x),
            "p95":x[max(0,math.ceil(.95*len(x))-1)],
            "raw":values}

def run(original_dir):
    if torch.cuda.is_initialized():
        raise ValueError("CUDA not authorized; CPU-only experiment")
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    fit=fit_sources()
    model,receipt=load_original(original_dir,fit)
    model.train()
    params={n:p.detach().clone() for n,p in model.named_parameters()}
    if sum(x.numel() for x in params.values())!=499623:
        raise ValueError("original checkpoint parameter count drift")
    rows=[]
    for batch,length in GEOMETRY:
        x,y,owners,digest=source_batch(fit,batch,length)
        rng=torch.manual_seed(RNG_SEED).get_state().clone()
        torch.set_rng_state(rng.clone())
        anchor=execute(model,x,y,"native")
        anchor_rng=torch.get_rng_state().clone()
        measurements={arm:[] for arm in POLICIES}
        largest_grad_abs=0.;largest_logit_abs=0.
        for iteration in range(WARMUPS+TIMED):
            order=POLICIES[iteration%len(POLICIES):]+POLICIES[:iteration%len(POLICIES)]
            for arm in order:
                torch.set_rng_state(rng.clone())
                result=execute(model,x,y,arm)
                compare=compare_outputs(anchor,result)
                if not torch.equal(torch.get_rng_state(),anchor_rng):
                    raise ValueError("RNG changed across checkpoint mode")
                largest_grad_abs=max(largest_grad_abs,compare["max_gradient_abs"])
                largest_logit_abs=max(largest_logit_abs,compare["max_logits_abs"])
                for name,p in model.named_parameters():
                    if not torch.equal(p.detach(),params[name]):
                        raise ValueError("checkpoint mode changed original trained parameter")
                if iteration>=WARMUPS:
                    measurements[arm].append({k:result[k] for k in (
                        "forward_ms","backward_ms","time_wall_ms",
                        "saved_logical_tensor_requests","saved_logical_tensor_bytes",
                        "saved_logical_floating_bytes")})
        stats={}
        for arm,data in measurements.items():
            if len(data)!=TIMED:raise ValueError("incomplete warm original source runs")
            counts={k:list({int(q[k]) for q in data}) for k in
                    ("saved_logical_tensor_requests","saved_logical_tensor_bytes",
                     "saved_logical_floating_bytes")}
            if any(len(vals)!=1 for vals in counts.values()):
                raise ValueError("autograd saved tensor request graph changed across same source repeats")
            stats[arm]={
                "forward_ms":dist([q["forward_ms"] for q in data]),
                "backward_ms":dist([q["backward_ms"] for q in data]),
                "total_ms":dist([q["time_wall_ms"] for q in data]),
                "logical_saved_requests":counts["saved_logical_tensor_requests"][0],
                "logical_saved_bytes":counts["saved_logical_tensor_bytes"][0],
                "logical_saved_floating_bytes":counts["saved_logical_floating_bytes"][0]}
        rows.append({"batch":batch,"window":length,"original_target_count":batch*length,
            "fit_owners":owners,"actual_fit_window_sha256":digest,
            "max_forward_abs_diff":largest_logit_abs,
            "max_parameter_gradient_abs_diff":largest_grad_abs,
            "active_original_gradient_parameters":499623,
            "original_nll":float(anchor["loss"]),
            "settings":{"warmups":WARMUPS,"timed_repeats":TIMED,
                        "timed_order":"rotating 3 arms after 2 warmups"},
            "arms":stats})
    return {"study":"LANGRECOMP054",
        "status":"INTERNAL_ORIGINAL_TRAINED_CPU_WARM_CHECKPOINT_PARITY_ACTUALLY_MEASURED",
        "original_checkpoint":receipt,
        "source":"authentic original publisher CPython FIT first seed1701 sampled windows; no DEV/TEST",
        "trained_model_parameters":499623,
        "original_training_updates":320,"new_optimizer_updates":0,
        "source_batch_windows":[[b,t] for b,t in GEOMETRY],
        "cpu_threads":2,"torch":torch.__version__,"platform":platform.platform(),
        "graph011_original_source":"NOT_VERIFIED_OR_IMPORTED",
        "cuda_peak_allocated":"NOT_MEASURED_CPU_ONLY",
        "cuda_peak_reserved":"NOT_MEASURED_CPU_ONLY",
        "training_checkpoint_implementation":"original whole graph nonreentrant or original upper-block-only nonreentrant, no parameter changes",
        "rows":rows,
        "limits":["Logical autograd saved-tensor requests count alias views and do not equal peak allocated or reserved memory",
            "CPU source batches are short and source owners previously exposed, no unseen language ability",
            "The original GRAPH011 implementation and 5.51MB→0.0014MB GPU claim are NOT verified",
            "No CUDA GPU, device memory, optimizer updates, real training throughput or phone energy measured",
            "No Lean/Python end-to-end whole-graph proof or IEEE finite-precision guarantee"],
        "protected_TEST":"NOT_ACCESSED","gpu_minutes":0,
        "independently_qualified":False,"promotion":False}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("refuse original evidence overwrite")
    r=run(a.original_dir)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"study":r["study"],"status":r["status"],"rows":[{
        "batch":z["batch"],"window":z["window"],
        "max_forward_abs":z["max_forward_abs_diff"],
        "max_gradient_abs":z["max_parameter_gradient_abs_diff"],
        "arms":{k:{"median_ms":v["total_ms"]["median"],
                  "forward_median_ms":v["forward_ms"]["median"],
                  "backward_median_ms":v["backward_ms"]["median"],
                  "logical_saved_bytes":v["logical_saved_bytes"]}
                for k,v in z["arms"].items()}}
        for z in r["rows"]]},sort_keys=True))
if __name__=="__main__":main()
