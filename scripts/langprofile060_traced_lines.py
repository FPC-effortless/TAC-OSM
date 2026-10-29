#!/usr/bin/env python3
"""LANGPROFILE060: source-line execution duration tracing, original TAC _advance.

CPython sys.settrace labels elapsed intervals between Python line events.
Includes nested callees and PyTorch C operations at the invoking line;
has heavy tracing overhead, no isolation of operators and cannot support
a production latency/speedup estimate. Original neural graph is untouched.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src"),
              str(ROOT/"research/mathematical-engineering/langstream043")]
from incremental import IncrementalTAC043, SHAPES
from scripts.langdecode057_validation import BatchedValidationTAC057
from scripts.langdecode057_benchmark import parity, exact_state
from scripts.langrecomp052_replay import load_original
from scripts.langlayer035_sampling import fit_sources

WARMUPS=2
SAMPLES=24


def trace_lines(target, fn, count: int = SAMPLES, clock=time.perf_counter_ns) -> dict:
    """Only trace supplied Python code object, no accidental future-source reads.

    Line cost attributed to the PREVIOUS line, until next line or return.
    Other Python function calls are not traced within their frames, so
    time is inclusive of nested Python/C callees and dispatch effects.
    """
    if count < 1 or not hasattr(target,"__code__"):
        raise ValueError("invalid trace target")
    if sys.gettrace() is not None:
        raise ValueError("already tracing: refuse contaminated measurement")
    target_code=target.__code__
    byline=defaultdict(list)
    call_count=0
    current={}
    def local(frame,event,arg):
        now=clock()
        marker=id(frame)
        previous=current.get(marker)
        if previous is not None:
            line,start=previous
            elapsed=now-start
            if elapsed < 0:
                raise ValueError("clock moved backward")
            byline[line].append(elapsed)
        if event=="line":
            current[marker]=(frame.f_lineno,clock())
        elif event in ("return","exception"):
            current.pop(marker,None)
        return local
    def global_tracer(frame,event,arg):
        nonlocal call_count
        if event=="call" and frame.f_code is target_code:
            call_count+=1
            current[id(frame)]=None
            return local
        return None
    start=clock()
    try:
        sys.settrace(global_tracer)
        for _ in range(count):
            fn()
    finally:
        sys.settrace(None)
    wall=clock()-start
    if call_count!=count or current or not byline:
        raise ValueError("trace did not cover exactly requested original calls")
    src,first=inspect.getsourcelines(target)
    allowed=set(range(first,first+len(src)))
    if set(byline)-allowed:
        raise ValueError("traced unexpected source lines")
    rows=[]
    for line,values in byline.items():
        if not values or any(v < 0 for v in values):
            raise ValueError("invalid per-line observations")
        total=sum(values)
        rows.append({"line":line,
                     "source":src[line-first].strip()[:240],
                     "samples":len(values),
                     "median_us":statistics.median(values)/1000,
                     "total_us":total/1000})
    rows.sort(key=lambda x:(-x["total_us"],x["line"]))
    line_total=sum(x["total_us"] for x in rows)
    if line_total <= 0:
        raise ValueError("zero line durations")
    return {"status":"TRACED_PYTHON_LINES_NOT_PRODUCTION_LATENCY",
            "calls":call_count,"line_events":sum(len(v) for v in byline.values()),
            "total_traced_wall_us":wall/1000,
            "sum_line_intervals_us":line_total,
            "top_lines":rows[:25],
            "all_lines":sorted(rows,key=lambda x:x["line"]),
            "clock":"perf_counter_ns includes nested calls, C kernels and tracer overhead; no ATen attribution",
            "no_speedup_claim":True}


def run(folder:Path) -> dict:
    if torch.cuda.is_initialized():
        raise ValueError("CPU only")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    fit=fit_sources()
    model,receipt=load_original(folder,fit)
    engines={"original":IncrementalTAC043(model,"cap_all"),
             "batched":BatchedValidationTAC057(model,"cap_all")}
    orig_hash=engines["original"]._weight_hash()
    owner=next(x for x in sorted(fit) if len(fit[x])>=544)
    rows=[]
    with torch.inference_mode():
        for length in (128,512):
            raw=fit[owner][:length]
            tokens=torch.tensor([list(raw)],dtype=torch.long)
            generated,checks=parity(engines,tokens,(owner,))
            _,initial=engines["batched"].prefill(tokens,owners=(owner,))
            token=generated[0].clone()
            mask=torch.ones_like(token,dtype=torch.bool)
            a,sa=engines["batched"].step(token,initial,owners=(owner,))
            b,sb=engines["batched"]._advance(token,initial,mask)
            if not torch.equal(a,b):
                raise ValueError("step/transition logits diverged")
            exact_state(sa,sb)
            before={name:x.clone() for name,x in initial.tensors.items()}
            fn=lambda: engines["batched"]._advance(token,initial,mask)
            for _ in range(WARMUPS):fn()
            result=trace_lines(IncrementalTAC043._advance,fn)
            if any(not torch.equal(initial.tensors[name],before[name]) for name in SHAPES):
                raise ValueError("tracing mutated persistent state")
            if engines["original"]._weight_hash()!=orig_hash:
                raise ValueError("tracing mutated frozen original checkpoint")
            rows.append({"prompt_bytes":length,
                         "original_source_sha256":hashlib.sha256(raw).hexdigest(),
                         "parity":checks,"transition_bitwise_equal":True,
                         "persistent_state_bytes":initial.tensor_bytes(),
                         "line_trace":result})
    return {"study":"LANGPROFILE060",
            "status":"ORIGINAL_TRAINED_SOURCE_LINE_TRACE_INTERNAL_DIAGNOSTIC",
            "original_checkpoint_receipt":receipt,
            "source_split":"FIT_ONLY_INFERENCE","source_owner":owner,
            "model_parameters":499623,
            "new_training_updates":0,"gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
            "results":rows,
            "limits":["CPython sys.settrace greatly perturbs wall time; never production latency",
                      "per-line source timing is inclusive of nested callees and C kernels",
                      "line statistics cannot isolate single ATen operations or numerical costs",
                      "frozen state same clock and byte repeated, not growing-sequence latency",
                      "one original trained checkpoint and already-exposed publisher FIT owner",
                      "no model quality, learned architecture, mobile/GPU or TEST result",
                      "LANGDECODE057 p95 repeatability remains unresolved"],
            "independently_qualified":False,"checkpoint_promotion":False}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tac-dir",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError("do not overwrite source evidence")
    report=run(args.tac_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as fp:
        fp.write(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"study":report["study"],"results":[
        {"prompt":x["prompt_bytes"],"top_lines":x["line_trace"]["top_lines"][:12],
         "calls":x["line_trace"]["calls"],"status":x["line_trace"]["status"]}
        for x in report["results"]]},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
