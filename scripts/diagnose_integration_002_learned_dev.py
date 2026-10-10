#!/usr/bin/env python3
"""Train and evaluate INTEGRATION-002 learned-address development controls.

Not confirmatory research. Produces raw cells plus provenance. The 64-slot
cache is a nonlearned write-overwrite stress baseline, not a GRU/CASM/PLM.
Dense learned scoring still traverses every candidate and is NOT sublinear.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import torch  # noqa: E402
from tac_osm.integration_002_learned_dev import (  # noqa: E402
    CACHE_BUCKETS,EVAL_NAMESPACE,EVAL_SEEDS,TRAIN_NAMESPACE,TRAIN_SEEDS,
    TRAIN_STEPS,evaluate_seed,train_seed,
)

ID="INTEGRATION-002-DEV-TRAINED-ADDRESS-001"
OUT=ROOT/"artifacts"/(ID+".json")


def sha256(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_by(rows:list[dict],key:str)->float:
    return statistics.fmean(float(r[key]) for r in rows)


def main()->None:
    torch.set_num_threads(2)
    started=time.perf_counter()
    measurements=[]
    train_records=[]
    source_files=[
        "src/tac_osm/integration_002_benchmark.py",
        "src/tac_osm/integration_002_learned_dev.py",
        "scripts/diagnose_integration_002_learned_dev.py",
    ]
    for tr_seed,ev_seed in zip(TRAIN_SEEDS,EVAL_SEEDS):
        model,train=train_seed(tr_seed)
        if not (train["parameters_changed"] and train["nonzero_finite_gradient"]):
            raise AssertionError("training integrity gate violated")
        cells=evaluate_seed(model,ev_seed)
        if len(cells)!=54 or any(c["reset_correct"]!=0.5 for c in cells):
            raise AssertionError("evaluation grid or reset causal gate invalid")
        measurements.extend(cells)
        train_records.append(train)
    if len(measurements)!=162:
        raise AssertionError("evaluation grid incomplete")
    metrics=(
        "learned_correct","euclidean_correct","learned_recall_at_1",
        "euclidean_recall_at_1","bounded_correct","bounded_nowrite_correct",
        "bounded_swapped_correct","bounded_retained","bounded_nowrite_retained",
        "reset_correct","mean_cache_overwrites","mean_bounded_query_candidates",
        "dense_query_candidates",
    )
    summary={name:mean_by(measurements,name) for name in metrics}
    summary.update({
        "condition_rows":len(measurements),
        "episodes":sum(x["episode_count"] for x in measurements),
        "paired_histories":sum(x["pair_count"] for x in measurements),
        "trained_seeds":len(TRAIN_SEEDS),
        "training_steps_per_seed":TRAIN_STEPS,
        "train_seed_namespaces":list(TRAIN_SEEDS),
        "evaluation_seeds":list(EVAL_SEEDS),
        "fixed_memory_capacity":CACHE_BUCKETS,
        "cpu_wall_seconds":round(time.perf_counter()-started,3),
    })
    if summary["reset_correct"]!=0.5:
        raise AssertionError("reset pairing invalid")
    by_interference={}
    for frac in (0.0,0.5,1.0):
        z=[row for row in measurements if row["hard_fraction"]==frac]
        by_interference[str(frac)]={
            "learned_accuracy":mean_by(z,"learned_correct"),
            "euclidean_accuracy":mean_by(z,"euclidean_correct"),
            "learned_recall_at_1":mean_by(z,"learned_recall_at_1"),
            "cache_accuracy":mean_by(z,"bounded_correct"),
            "cache_target_retention":mean_by(z,"bounded_retained"),
        }
    summary["by_interference_fraction"]=by_interference
    report={
        "experiment_id":ID,
        "status":"MEASURED_DEVELOPMENT_ONLY",
        "confirmatory_status":"NOT_REGISTERED_OR_EXECUTED",
        "not_a_confirmatory_pass":True,
        "provenance":{
            "commit":os.environ.get("GITHUB_SHA","local-unverified"),
            "workflow_run_id":os.environ.get("GITHUB_RUN_ID","local-unverified"),
            "source_sha256":{p:sha256(ROOT/p) for p in source_files},
            "train_rng_namespace":TRAIN_NAMESPACE,
            "evaluation_rng_namespace":EVAL_NAMESPACE,
            "distinct_seed_sets":set(TRAIN_SEEDS).isdisjoint(EVAL_SEEDS),
        },
        "limitations":[
            "Training uses oracle target-slot labels exclusively for cross-entropy supervision; model inference never uses oracle metadata",
            "Model is only a learned address metric, not a learned recurrent memory",
            "Fixed 64-slot overwrite cache is nonlearned with sign-hash assignment; a stress control, not TAC-OSM architecture validation",
            "Address scoring over full history and cache is dense O(H+D) and O(64) respectively; cache construction includes H+D writes",
            "The development pool influenced design and is not an untouched confirmatory holdout",
            "Binary answer accuracy can exceed correct-target retrieval recall due to guessing an equal bit",
            "No seed-level CIs, checkpoint-selection audit or full operation-normalized cost parity claim is included",
        ],
        "training":train_records,
        "summary":summary,
        "raw_cells":measurements,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True))
    print("RESULT: DEVELOPMENT-ONLY TRAINED ADDRESS / NONLEARNED CACHE CONTROLS")
    print("NO INTEGRATION-002 CONFIRMATORY RESULT")
    print("artifact:",OUT)


if __name__=="__main__":
    main()
