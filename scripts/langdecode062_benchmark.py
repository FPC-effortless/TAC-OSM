#!/usr/bin/env python3
"""LANGDECODE062: frozen checkpoint identical-graph candidate & fair CPU decode.

The proposed step changes two original ordinal_code computations to one
bitwise-verified vectorized tensor computation, otherwise byte-for-byte
original mathematical operations. Never train, promote or alter checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src"),
             str(ROOT/"research/mathematical-engineering/langstream043")]
from scripts.langdecode057_benchmark import parity, trial
from scripts.langdecode057_validation import BatchedValidationTAC057
from scripts.langdecode062_vectorized import VectorOrdinalTAC062
from scripts.langrecomp052_replay import load_original,dist
from scripts.langlayer035_sampling import fit_sources


def run(folder:Path)->dict:
    if torch.cuda.is_initialized():raise ValueError("registered CPU-only benchmark")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    fit=fit_sources()
    model,receipt=load_original(folder,fit)
    engines={"original":BatchedValidationTAC057(model,"cap_all"),
             "vector_ordinal":VectorOrdinalTAC062(model,"cap_all")}
    original_weight_hash=engines["original"]._weight_hash()
    for meth in ("step","prefill","save_state","load_state","validate"):
        if getattr(BatchedValidationTAC057,meth) is not getattr(VectorOrdinalTAC062,meth):
            raise ValueError("candidate unexpectedly changed non-transition method: "+meth)
    owner=(next(x for x in sorted(fit) if len(fit[x])>=544),)
    results=[]
    with torch.inference_mode():
        for length in (128,512):
            raw=fit[owner[0]][:length]
            tok=torch.tensor([list(raw)],dtype=torch.long)
            generated,checks=parity(engines,tok,owner)
            measured={k:[] for k in engines}
            for repetition in range(9):
                order=("original","vector_ordinal") if repetition%2==0 else ("vector_ordinal","original")
                for name in order:
                    row=trial(engines[name],tok,owner,generated)
                    if repetition>=2:measured[name].append(row)
            arms={}
            for name,rows in measured.items():
                if any(row["state_bytes"]!=3904 for row in rows):
                    raise ValueError("persistent state grew")
                arms[name]={
                    "decode":dist([x for row in rows for x in row["step_ms"]]),
                    "prefill":dist([row["prefill_ms"] for row in rows]),
                    "validation":dist([x for row in rows for x in row["validate_ms"]]),
                    "whole_repeat":dist([row["prefill_ms"]+sum(row["step_ms"]) for row in rows]),
                }
            ratio=arms["vector_ordinal"]["decode"]["median_ms"]/arms["original"]["decode"]["median_ms"]
            tail=arms["vector_ordinal"]["decode"]["p95_ms"]/arms["original"]["decode"]["p95_ms"]
            results.append({
                "prompt_bytes":length,
                "source_sha256":hashlib.sha256(raw).hexdigest(),
                "generated_sha256":hashlib.sha256(bytes(int(x[0]) for x in generated)).hexdigest(),
                "original_equivalence":checks,
                "arms":arms,
                "median_reduction_fraction":1-ratio,
                "p95_ratio":tail,
                "latency_gate":(ratio<=.95 and tail<=1.10),
            })
    if engines["original"]._weight_hash()!=original_weight_hash:
        raise ValueError("checkpoint mutated")
    return {
        "study":"LANGDECODE062",
        "status":"ORIGINAL_TRAINED_VECTOR_ORDINAL_INTERNAL_CPU_CANDIDATE",
        "original_checkpoint_receipt":receipt,
        "candidate_source_sha256":hashlib.sha256(
            (ROOT/"scripts/langdecode062_vectorized.py").read_bytes()).hexdigest(),
        "original_transition_sha256":hashlib.sha256(
            (ROOT/"research/mathematical-engineering/langstream043/incremental.py").read_bytes()).hexdigest(),
        "original_kernel_sha256":hashlib.sha256(
            (ROOT/"research/mathematical-engineering/context026/context_transition.py").read_bytes()).hexdigest(),
        "all_latency_gates_pass":all(r["latency_gate"] for r in results),
        "trained_parameters":499623,
        "persistent_state_bytes_per_owner":3904,
        "original_source_owner":owner[0],"source_partition":"FIT_ONLY_INFERENCE",
        "source_model_unchanged":True,"new_training_updates":0,
        "gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
        "torch":torch.__version__,"host":platform.platform(),
        "timing":"CPU FP32 eager batch1 two threads; 2 warmups, 7 measured repeats with alternating arm order; 16 original greedy bytes; original validations included",
        "results":results,
        "limits":["one hosted CPU and historical original checkpoint/source FIT owner",
                  "original full prefix parity within frozen numerical tolerances, not a universal proof",
                  "candidate is a source-copy two-position math substitution requiring PNDS source bridge",
                  "no mobile or GPU, RAM/energy or new quality claims",
                  "third LANGDECODE057 128-byte p95 regression remains unresolved",
                  "CPU tail behavior requires independently repeated matched host validation"],
        "independently_qualified":False,"checkpoint_promotion":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tac-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("original candidate evidence immutable")
    result=run(a.tac_dir)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as fp:fp.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"study":result["study"],"all_latency_gates_pass":result["all_latency_gates_pass"],
                      "results":[{"prompt":x["prompt_bytes"],"median_reduction":x["median_reduction_fraction"],
                                  "p95_ratio":x["p95_ratio"],"latency_gate":x["latency_gate"],
                                  "exact_incremental":x["original_equivalence"]["bitwise_original_incremental_logits_and_all_states"],
                                  "max_abs_full_prefix":x["original_equivalence"]["full_prefix_max_abs_error"]}
                                 for x in result["results"]]},sort_keys=True),flush=True)


if __name__=="__main__":main()
