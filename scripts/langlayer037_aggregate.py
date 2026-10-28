#!/usr/bin/env python3
"""Independent prospective 10-checkpoint FFN control versus FROZEN LANGLAYER035.

Reads authentic original source and publisher-owned FIT schedule, validates
binary checkpoints and original training receipts; cannot access TEST or fit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.langlayer035_sampling import fit_sources,schedule_receipt

ARMS={"tac_ffn2":499661,"tac_ffn3":499684}
REF={"tac_ffn2":"tac_layer2","tac_ffn3":"tac_layer3"}
SEEDS=(1701,1702,1703,1704,1705)
FIELDS=("nll","top1","top5","brier")

def aggregate(root:Path,old:Path):
    reference=json.loads(old.read_text())
    if reference.get("study")!="LANGLAYER035" or reference.get("cohort")!=15 or reference.get("status")!="COMPLETE_INTERNAL_EXPOSED_DEV_EXPLORATORY_NOT_FLOP_MATCHED":
        raise ValueError("original 15-model benchmark not frozen verified original")
    fit=fit_sources()
    schedules={seed:schedule_receipt(fit,seed,steps=320) for seed in SEEDS}
    rows={}
    for path in root.rglob("FIT_AND_EXPOSED_DEV_RECEIPT.json"):
        r=json.loads(path.read_text())
        key=(r.get("arm"),r.get("seed"))
        if key in rows or key[0] not in ARMS or key[1] not in SEEDS:
            raise ValueError("unexpected/duplicated arm or seed")
        if r["study"]!="LANGLAYER037" or r["updates"]!=320:
            raise ValueError("wrong study or updates")
        if r["parameters"]!={"trainable":ARMS[key[0]],"deployed":ARMS[key[0]]}:
            raise ValueError("actual parameters mismatch")
        if r["examples_seen"]!=2560 or r["target_bytes_seen"]!=327680:
            raise ValueError("FIT target exposure mismatch")
        schedule=schedules[key[1]]
        if (r["source_sample_order_sha256"]!=schedule["sample_order_sha256"] or
            r["source_bytes_sha256"]!=schedule["sample_bytes_sha256"]):
            raise ValueError("unmatched original publisher FIT seed/owner/offset sample")
        if not r["checkpoint_reloaded_identical_scored_DEV"]:
            raise ValueError("original model+optimizer+RNG cold checkpoint replay failed")
        cp=path.parent/r["checkpoint_filename"]
        if not cp.is_file() or hashlib.sha256(cp.read_bytes()).hexdigest()!=r["checkpoint_sha256"]:
            raise ValueError("missing or tampered original checkpoint")
        if r["protected_TEST"]!="NOT_ACCESSED" or r["GPU_minutes"]!=0:
            raise ValueError("protected or GPU scope breached")
        if r["matched_training_FLOPs"]!="NOT_VERIFIED":
            raise ValueError("unsupported FLOP-matched comparison claim")
        scores=r["DEV_EXPOSED_4096_ONLY"]
        if scores["n"]!=4096 or any(not math.isfinite(scores[k]) for k in FIELDS):
            raise ValueError("wrong or nonfinite DEV measurement")
        for ref_arm in ("tac_context026",REF[key[0]]):
            old_nll=reference["per_seed"][ref_arm][str(key[1])]["nll"]
            if not math.isfinite(old_nll):raise ValueError("old score invalid")
        rows[key]=r
    expected={(a,s) for a in ARMS for s in SEEDS}
    if set(rows)!=expected:
        raise ValueError(f"missing complete 10-checkpoint matrix: {sorted(expected-set(rows))}")
    report={
       "study":"LANGLAYER037","status":"COMPLETE_10_CHECKPOINTS_INTERNAL_EXPOSED_DEV_ONLY",
       "old_study":"LANGLAYER035","old_reference_original_verified_run":40786118546,
       "new_optimizer_steps_per_model":320,"real_fit_targets_per_model":327680,
       "protected_TEST":"NOT_ACCESSED","GPU_minutes":0,
       "training_FLOP_parity":"NOT_VERIFIED",
       "eligible_for_architectural_superiority":False,
       "mean_per_model":{},"paired_reference_differences":{},"per_seed":{}}
    for arm in ARMS:
        values=[rows[arm,seed] for seed in SEEDS]
        score={}
        for k in FIELDS:
            x=[v["DEV_EXPOSED_4096_ONLY"][k] for v in values]
            score[k+"_mean"]=statistics.mean(x)
            score[k+"_sd"]=statistics.stdev(x)
        score["perplexity_mean"]=statistics.mean(math.exp(v["DEV_EXPOSED_4096_ONLY"]["nll"]) for v in values)
        score["cpu_process_seconds_mean"]=statistics.mean(v["measured_cpu_process_seconds"] for v in values)
        score["parameter_count"]=ARMS[arm]
        report["mean_per_model"][arm]=score
        report["per_seed"][arm]={str(seed):rows[arm,seed]["DEV_EXPOSED_4096_ONLY"] for seed in SEEDS}
        report["paired_reference_differences"][arm]={}
        for ref_arm in ("tac_context026",REF[arm]):
            diffs=[rows[arm,seed]["DEV_EXPOSED_4096_ONLY"]["nll"]-reference["per_seed"][ref_arm][str(seed)]["nll"]
                   for seed in SEEDS]
            report["paired_reference_differences"][arm][ref_arm]={
                "mean_ffn_minus_reference_nll":statistics.mean(diffs),
                "sd":statistics.stdev(diffs),
                "ffn_wins":sum(x<0 for x in diffs),
                "paired_by_seed":dict(zip(map(str,SEEDS),diffs))}
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",required=True,type=Path)
    p.add_argument("--original",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    report=aggregate(a.results,a.original)
    if a.output.exists():raise FileExistsError("refuse overwrite")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
if __name__=="__main__":main()
