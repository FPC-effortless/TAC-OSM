#!/usr/bin/env python3
"""Aggregate 15 original Github artifact receipts; never read protected TEST.

A missing/failed checkpoint never silently drops from the mean. This script is
not a model fit or a trained-FLOP-matched architecture comparison.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ARMS=("tac_context026","tac_layer2","tac_layer3")
SEEDS=(1701,1702,1703,1704,1705)
COUNTS={"tac_context026":500122,"tac_layer2":499725,"tac_layer3":499934}
P3_HISTORICAL_EXPOSED_DEV_NLL=1.877306112
STUDY="LANGLAYER035"


def aggregate(root: Path):
    receipts=list(root.rglob("FIT_AND_EXPOSED_DEV_RECEIPT.json"))
    rows={}
    for path in receipts:
        r=json.loads(path.read_text())
        if r["study"]!=STUDY or r["arm"] not in ARMS or r["seed"] not in SEEDS:
            raise ValueError(f"unregistered artifact: {path}")
        key=(r["arm"],r["seed"])
        if key in rows:
            raise ValueError(f"duplicate receipt: {key}")
        if r["parameters"] != {"trainable":COUNTS[r["arm"]],"deployed":COUNTS[r["arm"]]}:
            raise ValueError(f"wrong trained parameter count: {key}")
        if r["updates"]!=320 or r["target_bytes_seen"]!=327680:
            raise ValueError(f"FIT exposure mismatch: {key}")
        if r["protected_TEST"]!="NOT_ACCESSED" or r["GPU_minutes"]!=0:
            raise ValueError(f"unexpected resource/access receipt: {key}")
        if r["matched_training_FLOPs"]!="NOT_VERIFIED":
            raise ValueError(f"unverified compute match falsely claimed: {key}")
        if not r["checkpoint_reloaded_identical_scored_DEV"]:
            raise ValueError(f"cold replay failure: {key}")
        score=r["DEV_EXPOSED_4096_ONLY"]
        if score["n"]!=4096:
            raise ValueError(f"wrong exposed DEV target count: {key}")
        cp=path.parent/r["checkpoint_filename"]
        if not cp.is_file() or hashlib.sha256(cp.read_bytes()).hexdigest()!=r["checkpoint_sha256"]:
            raise ValueError(f"checkpoint missing or SHA mismatch: {key}")
        for name in ("nll","brier","top1","top5"):
            if not math.isfinite(float(score[name])):
                raise ValueError(f"nonfinite DEV score: {key}/{name}")
        rows[key]=r
    expected={(a,s) for a in ARMS for s in SEEDS}
    missing=sorted(expected-set(rows))
    if missing:
        return {"status":"INCOMPLETE_NO_AGGREGATE","found":len(rows),
                "missing":[f"{a}/{s}" for a,s in missing],
                "score":"NOT_QUALIFIED"}
    for seed in SEEDS:
        sample={rows[a,seed]["source_sample_order_sha256"] for a in ARMS}
        source={rows[a,seed]["source_bytes_sha256"] for a in ARMS}
        if len(sample)!=1 or len(source)!=1:
            raise ValueError(f"per-seed original source samples differ: {seed}")
    result={"status":"COMPLETE_INTERNAL_EXPOSED_DEV_EXPLORATORY_NOT_FLOP_MATCHED",
            "study":STUDY,"cohort":15,"seeds":list(SEEDS),
            "native_cores":["CONTEXT026","LANGLAYER034_DEPTH2","LANGLAYER034_DEPTH3"],
            "developer_data":"original PSF CPython 3.13.0 exposed DEV; NOT independent TEST",
            "training_FLOP_parity":"NOT_VERIFIED",
            "historical_p3_exposed_dev_nll":P3_HISTORICAL_EXPOSED_DEV_NLL,
            "models":{},"per_seed":{},"paired_differences":{},
            "protected_TEST":"NOT_ACCESSED","neural_architecture_superiority_certified":False}
    for a in ARMS:
        fields={}
        for key in ("nll","brier","top1","top5"):
            v=[float(rows[a,s]["DEV_EXPOSED_4096_ONLY"][key]) for s in SEEDS]
            fields[key+"_mean"]=statistics.mean(v)
            fields[key+"_sd"]=statistics.stdev(v)
        fields["perplexity_mean_of_seeds"]=statistics.mean(
            math.exp(float(rows[a,s]["DEV_EXPOSED_4096_ONLY"]["nll"])) for s in SEEDS)
        fields["cpu_process_seconds_mean"]=statistics.mean(
            float(rows[a,s]["measured_cpu_process_seconds"]) for s in SEEDS)
        fields["parameters"]=COUNTS[a]
        result["models"][a]=fields
        result["per_seed"][a]={
            str(s):{k:rows[a,s]["DEV_EXPOSED_4096_ONLY"][k] for k in ("nll","brier","top1","top5")}
            for s in SEEDS}
    for a in ("tac_layer2","tac_layer3"):
        delta=[rows[a,s]["DEV_EXPOSED_4096_ONLY"]["nll"]
               -rows["tac_context026",s]["DEV_EXPOSED_4096_ONLY"]["nll"] for s in SEEDS]
        result["paired_differences"][a]={
            "mean_nll_layer_minus_parent":statistics.mean(delta),
            "seeds_better_nll_than_parent":sum(d<0 for d in delta),
            "nll_differences_by_seed":dict(zip(map(str,SEEDS),delta)),
        }
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    r=aggregate(a.results)
    if a.output.exists():
        raise FileExistsError("refuse overwrite of any aggregate receipt")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(r,sort_keys=True,allow_nan=False))
    if r["status"]!="COMPLETE_INTERNAL_EXPOSED_DEV_EXPLORATORY_NOT_FLOP_MATCHED":
        raise SystemExit(2)


if __name__=="__main__":
    main()
