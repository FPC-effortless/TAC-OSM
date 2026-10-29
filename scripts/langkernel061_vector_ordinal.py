#!/usr/bin/env python3
"""LANGKERNEL061: no-fit positional ordinal microkernel falsifier.

Compare original ContextTransition026 ordinal_code to a vectorized equivalent
on CPU. This does NOT modify trained TAC, its original _advance graph, state,
checkpoints, output logits, or source-split behavior. Bitwise parity is a
PRECONDITION for considering a model-level candidate, not proof of it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src"),
             str(ROOT/"research/mathematical-engineering/context026")]
from context_transition import PERIODS, ordinal_code

PROMPTS=(128,512,8192)
TRIALS=7
CALLS=64
WARMUP=2


def vectorized_ordinal(indices:torch.Tensor,dtype:torch.dtype)->torch.Tensor:
    if indices.dtype!=torch.long or indices.ndim<1 or indices.numel()==0:
        raise ValueError("only original causal long clock inputs")
    # No new retained state or cached precomputed tensors.
    periods=torch.tensor(PERIODS,device=indices.device,dtype=torch.long)
    factors=torch.tensor(tuple(2*math.pi/p for p in PERIODS),device=indices.device,dtype=dtype)
    angles=indices.unsqueeze(-1).remainder(periods).to(dtype)*factors
    return torch.stack((angles.cos(),angles.sin()),dim=-1).flatten(-2)


def original_pair(indices,dtype):
    return torch.stack((ordinal_code(indices,dtype),
                        ordinal_code((indices-1).clamp_min(0),dtype)),dim=0)


def vectorized_two_calls(indices,dtype):
    return torch.stack((vectorized_ordinal(indices,dtype),
                        vectorized_ordinal((indices-1).clamp_min(0),dtype)),dim=0)


def vectorized_pair(indices,dtype):
    both=torch.stack((indices,(indices-1).clamp_min(0)),dim=0)
    return vectorized_ordinal(both,dtype)


ARMS={"original_two_calls":original_pair,
      "vectorized_two_calls":vectorized_two_calls,
      "vectorized_pair":vectorized_pair}


def statistics_cpu(fn)->dict:
    for _ in range(WARMUP):fn()
    vals=[]
    for _ in range(TRIALS):
        for _ in range(CALLS):
            t=time.perf_counter_ns()
            fn()
            vals.append((time.perf_counter_ns()-t)/1000)
    vals.sort()
    return {"median_us":statistics.median(vals),
            "p95_us":vals[math.ceil(.95*len(vals))-1],
            "samples":len(vals),"raw_us":vals}


def evaluate()->dict:
    if torch.cuda.is_initialized():raise ValueError("CPU-only original operator study")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    probes=torch.arange(0,8193,dtype=torch.long)
    more=torch.tensor([0,1,6,7,12,13,28,29,58,59,127,128,511,512,8191,8192,65535,262144],dtype=torch.long)
    cases=torch.unique(torch.cat((probes,more)))
    accuracy={}
    with torch.inference_mode():
        for dtype in (torch.float32,torch.float64):
            ref=ordinal_code(cases,dtype)
            fast=vectorized_ordinal(cases,dtype)
            diff=(ref-fast).abs()
            accuracy[str(dtype)]={
                "cases":cases.numel(),
                "bitwise_equal":torch.equal(ref,fast),
                "max_abs_error":float(diff.max()),
                "reference_sha256":hashlib.sha256(ref.numpy().tobytes()).hexdigest(),
                "candidate_sha256":hashlib.sha256(fast.numpy().tobytes()).hexdigest(),
            }
        speed=[]
        for clock in PROMPTS:
            idx=torch.tensor([clock],dtype=torch.long)
            ref=original_pair(idx,torch.float32)
            rows={}
            for name,fn in ARMS.items():
                output=fn(idx,torch.float32)
                rows[name]={
                    "bitwise_original_pair":torch.equal(output,ref),
                    "max_abs_pair_error":float((output-ref).abs().max()),
                    "cpu_unprofiled":statistics_cpu(lambda:fn(idx,torch.float32)),
                }
            speed.append({"clock":clock,"arms":rows})
    all_bitwise=all(v["bitwise_equal"] for v in accuracy.values()) and all(
        a["bitwise_original_pair"] for row in speed for a in row["arms"].values())
    return {"study":"LANGKERNEL061",
            "status":"ORIGINAL_ORDINAL_KERNEL_ONLY_NO_FIT_DIAGNOSTIC",
            "original_source":"context026/context_transition.py:ordinal_code; original exact PERIODS",
            "periods":list(PERIODS),
            "reference_checkpoint":"NOT_LOADED_KERNEL_ONLY",
            "original_decoder_modified":False,
            "persistent_decoder_state_bytes":"UNMEASURED_UNCHANGED_SOURCE",
            "bitwise_kernel_accepted":all_bitwise,
            "value_checks":accuracy,
            "cpu":speed,
            "training_updates":0,"gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
            "limits":["kernel-only timings do not predict end-to-end full trained decoder speed",
                      "source kernel tensor operations reordered but original decoder untouched",
                      "kernel bitwise parity cannot certify decoder state/logit parity",
                      "one GitHub CPU, same-input repeats; hosted CPU scheduling uncontrolled",
                      "previous LANGDECODE057 tail latency failure remains open",
                      "new learned model, held-out language quality and PNDS full source equivalence not measured"],
            "independently_qualified":False,"checkpoint_promotion":False}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError("no overwrite of measured evidence")
    data=evaluate()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as fp:fp.write(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"study":data["study"],"bitwise_kernel_accepted":data["bitwise_kernel_accepted"],
                      "value_checks":data["value_checks"],
                      "clock": [{"clock":r["clock"],"arms":{name:{
                          "bitwise":a["bitwise_original_pair"],
                          "median_us":a["cpu_unprofiled"]["median_us"],
                          "p95_us":a["cpu_unprofiled"]["p95_us"]}
                          for name,a in r["arms"].items()}}
                          for r in data["cpu"]]},sort_keys=True),flush=True)


if __name__=="__main__":main()
