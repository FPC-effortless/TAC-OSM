#!/usr/bin/env python3
"""Fail-closed real-source 039 5-checkpoint audit vs immutable 035/037 refs.

Only complete SHA-checked original trained checkpoint+receipt matrix can
produce a report. No training occurs. All numbers are exposed-DEV only.
"""
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

ARM="tac_recuronly038"
PARAMS=499901
SEEDS=(1701,1702,1703,1704,1705)
FIELDS=("nll","brier","top1","top5")
P3_EXPOSED_DEV_NLL=1.877306112

def summarise(results:Path,old035:Path,old037:Path):
    hist=json.loads(old035.read_text())
    ffn=json.loads(old037.read_text())
    if hist.get("study")!="LANGLAYER035" or hist.get("cohort")!=15 or hist.get("status")!="COMPLETE_INTERNAL_EXPOSED_DEV_EXPLORATORY_NOT_FLOP_MATCHED":
        raise ValueError("frozen 15-arm source-verified reference not verified")
    if ffn.get("study")!="LANGLAYER037" or ffn.get("status")!="COMPLETE_10_CHECKPOINTS_INTERNAL_EXPOSED_DEV_ONLY" or ffn.get("old_reference_original_verified_run")!=40786118546:
        raise ValueError("frozen ten-arm FFN control reference not verified")
    fit=fit_sources()
    schedules={seed:schedule_receipt(fit,seed,steps=320) for seed in SEEDS}
    rows={}
    for path in results.rglob("FIT_AND_EXPOSED_DEV_RECEIPT.json"):
        r=json.loads(path.read_text())
        if r.get("study")!="LANGLAYER039" or r.get("arm")!=ARM:
            raise ValueError("unexpected model artifact")
        seed=r.get("seed")
        if seed not in SEEDS or seed in rows:raise ValueError("missing identity or duplicate seed")
        if r["updates"]!=320 or r["examples_seen"]!=2560 or r["target_bytes_seen"]!=327680:
            raise ValueError("missing real FIT exposure")
        if r["parameters"]!={"trainable":PARAMS,"deployed":PARAMS}:
            raise ValueError("trainable/deployed parameter mismatch")
        schedule=schedules[seed]
        if r["source_sample_order_sha256"]!=schedule["sample_order_sha256"] or r["source_bytes_sha256"]!=schedule["sample_bytes_sha256"]:
            raise ValueError("source-owner / byte sample schedule mismatch")
        if r["source_commit"]!="60403a5409ff2c3f3b07dd2ca91a7a3e096839c7":
            raise ValueError("original publisher source version drifted")
        if r["protected_TEST"]!="NOT_ACCESSED" or r["GPU_minutes"]!=0 or r["matched_training_FLOPs"]!="NOT_VERIFIED":
            raise ValueError("undeclared resource/compute certification")
        if not r["checkpoint_reloaded_identical_scored_DEV"]:
            raise ValueError("original checkpoint cold replay failed")
        checkpoint=path.parent/r["checkpoint_filename"]
        if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=r["checkpoint_sha256"]:
            raise ValueError("missing/tampered original training checkpoint")
        dev=r["DEV_EXPOSED_4096_ONLY"]
        if dev["n"]!=4096 or any(not math.isfinite(float(dev[k])) for k in FIELDS):
            raise ValueError("original source DEV target/metric mismatch")
        for refarm in ("tac_context026","tac_layer2","tac_layer3"):
            if str(seed) not in hist["per_seed"][refarm]:
                raise ValueError("old original control incomplete")
        for refarm in ("tac_ffn2","tac_ffn3"):
            if str(seed) not in ffn["per_seed"][refarm]:
                raise ValueError("old FFN control incomplete")
        rows[seed]=r
    if set(rows)!=set(SEEDS):raise ValueError("full prospective five-seed matrix not complete")
    report={
      "study":"LANGLAYER039",
      "status":"COMPLETE_5_ORIGINAL_CPU_FITS_INTERNAL_EXPOSED_DEV_ONLY",
      "new_fit_seeds":list(SEEDS),
      "parameter_count":PARAMS,
      "new_training_updates_per_seed":320,
      "real_next_byte_targets_per_seed":327680,
      "source":"original PSF CPython3.13.0 FIT; 4 previously exposed DEV docs",
      "old_035_run":40786118546,
      "old_037_run":40788869823,
      "training_FLOP_parity":"NOT_VERIFIED",
      "protected_TEST":"NOT_ACCESSED",
      "independently_qualified":False,
      "architectural_superiority_proven":False,
      "p3_historical_unmatched_exposed_dev_nll":P3_EXPOSED_DEV_NLL,
      "model_summary":{},
      "matched_seed_differences":{},
      "per_seed":{},
      "checkpoint_sha256_by_seed":{},
      "original_controls":{}}
    for k in FIELDS:
        v=[float(rows[s]["DEV_EXPOSED_4096_ONLY"][k]) for s in SEEDS]
        report["model_summary"][k+"_mean"]=statistics.mean(v)
        report["model_summary"][k+"_sd"]=statistics.stdev(v)
    report["model_summary"]["perplexity_mean"]=statistics.mean(math.exp(rows[s]["DEV_EXPOSED_4096_ONLY"]["nll"]) for s in SEEDS)
    report["model_summary"]["cpu_process_seconds_mean"]=statistics.mean(float(rows[s]["measured_cpu_process_seconds"]) for s in SEEDS)
    report["model_summary"]["cpu_wall_seconds_mean"]=statistics.mean(float(rows[s]["measured_wall_seconds"]) for s in SEEDS)
    report["per_seed"]={str(s):{k:rows[s]["DEV_EXPOSED_4096_ONLY"][k] for k in FIELDS} for s in SEEDS}
    report["checkpoint_sha256_by_seed"]={str(s):rows[s]["checkpoint_sha256"] for s in SEEDS}
    for arm in ("tac_context026","tac_layer2","tac_layer3","tac_ffn2","tac_ffn3"):
        source=hist if arm.startswith("tac_") and arm not in ("tac_ffn2","tac_ffn3") else ffn
        if source is hist:
            s=source["models"][arm]
            ref={k:float(s[k+"_mean"]) for k in FIELDS}
            ref["cpu_seconds"]=float(s["cpu_process_seconds_mean"])
            ref["params"]=s["parameters"]
        else:
            s=source["mean_per_model"][arm]
            ref={k:float(s[k+"_mean"]) for k in FIELDS}
            ref["cpu_seconds"]=float(s["cpu_process_seconds_mean"])
            ref["params"]=s["parameter_count"]
        report["original_controls"][arm]=ref
        differences=[float(rows[seed]["DEV_EXPOSED_4096_ONLY"]["nll"])-
                     float(source["per_seed"][arm][str(seed)]["nll"])
                     for seed in SEEDS]
        mean=statistics.mean(differences)
        sd=statistics.stdev(differences)
        # Purely descriptive 95% paired t interval, df=4. NOT an independent
        # test or post-hoc multiplicity corrected significance claim.
        margin=2.7764451051977987*sd/math.sqrt(5)
        report["matched_seed_differences"][arm]={
           "mean_new_minus_old_nll":mean,
           "sd":sd,
           "paired_95pct_t_interval_descriptive":[mean-margin,mean+margin],
           "new_wins":sum(d<0 for d in differences),
           "differences_by_seed":dict(zip(map(str,SEEDS),differences))}
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--original035",type=Path,required=True)
    p.add_argument("--original037",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=summarise(a.results,a.original035,a.original037)
    if a.output.exists():raise FileExistsError("refuse overwriting scientific artifact")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(result,sort_keys=True,allow_nan=False))
if __name__=="__main__":main()
