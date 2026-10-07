#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT_ID="TACOSM-ADDRESS-BASELINE-001"
CONTRACT_PATH=ROOT/"contracts"/f"{EXPERIMENT_ID}.json"
DIMS=(4,8,16,32)
MEMORY_SIZES=(3,8,16,32,64,128,256)
SIGMAS=(0.0,0.05,0.1,0.2,0.4,0.8)
SEEDS=(0,1,2,3,4)
TRIALS=100000

def contract_hash():
    return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()

def run_condition(d:int,m:int,sigma:float,seed:int):
    rng=np.random.default_rng(seed)
    correct=0
    total=0
    batch=5000
    for start in range(0,TRIALS,batch):
        b=min(batch,TRIALS-start)
        keys=rng.normal(size=(b,m,d))
        keys/=np.linalg.norm(keys,axis=2,keepdims=True)
        target=rng.integers(m,size=b)
        query=keys[np.arange(b),target]
        if sigma:
            query=query+sigma*rng.normal(size=(b,d))
        scores=np.einsum("bd,bmd->bm",query,keys)
        correct += int((scores.argmax(axis=1)==target).sum())
        total += b
    return correct/total

def main():
    contract=json.loads(CONTRACT_PATH.read_text())
    assert contract["status"]=="pre-registered"
    assert contract["protocol"]["primary_dimension"]==16
    rows=[]
    for d in DIMS:
        for m in MEMORY_SIZES:
            for sigma in SIGMAS:
                accs=[run_condition(d,m,sigma,seed+d*1000+m*17+int(sigma*10000))
                      for seed in SEEDS]
                rows.append({
                    "d":d,"M":m,"sigma":sigma,
                    "mean_accuracy":float(np.mean(accs)),
                    "min_seed_accuracy":float(np.min(accs)),
                    "seed_accuracies":[float(x) for x in accs],
                })
    output={
        "experiment_id":EXPERIMENT_ID,
        "status":"measured",
        "provenance":{
            "git_commit":os.environ.get("GITHUB_SHA","unknown"),
            "workflow_run_id":os.environ.get("GITHUB_RUN_ID","unknown"),
            "numpy_version":np.__version__,
            "python_version":platform.python_version(),
            "platform":platform.platform(),
            "contract_sha256":contract_hash(),
        },
        "protocol":{
            "address_dimensions":list(DIMS),
            "memory_sizes":list(MEMORY_SIZES),
            "gaussian_query_noise_sigmas":list(SIGMAS),
            "trials_per_condition":TRIALS,
            "seeds":list(SEEDS),
            "scorer":"argmax raw dot product",
        },
        "rows":rows,
        "primary_d16_capacity":[r for r in rows if r["d"]==16],
        "claim_boundary":[
            "Bayes/raw-dot reference for random unit-key addressing",
            "no learned-addressing capability claim",
            "no content or semantic addressing claim",
        ],
    }
    out=ROOT/"artifacts"/f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"conditions":len(rows),"d16_conditions":len(output["primary_d16_capacity"])},indent=2))

if __name__=="__main__":
    main()
