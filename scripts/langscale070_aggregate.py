#!/usr/bin/env python3
"""LANGSCALE070 honest 9 original-checkpoint FIT/exposed DEV receipts aggregator.

Three FFN capacities, three matched publisher original-source schedules.
No winner/superiority/compute-optimal or sealed TEST claims.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

TIERS={"p250k":250133,"p500k":499623,"p1000k":1000239}
SEEDS=(1701,1702,1703)
UPDATES=160
TARGETS=163840


def collect(root:Path)->dict:
    files=list(root.rglob("FIT_AND_EXPOSED_DEV_RECEIPT.json"))
    expected={(a,s) for a in TIERS for s in SEEDS}
    rows={}
    for f in files:
        r=json.loads(f.read_text())
        k=r.get("arm"),r.get("seed")
        if k not in expected or k in rows:
            raise ValueError("unknown/duplicated original capacity checkpoint: "+str(k))
        if r.get("study")!="LANGSCALE070" or r.get("updates")!=UPDATES:
            raise ValueError("study or optimizer updates mismatch")
        if r.get("target_bytes_seen")!=TARGETS or r.get("examples_seen")!=UPDATES*8:
            raise ValueError("publisher source exposure mismatch")
        if r.get("parameters")!={"trainable":TIERS[k[0]],"deployed":TIERS[k[0]]}:
            raise ValueError("wrong active trainable/deployed live count")
        if r.get("ffn_width")!={"p250k":150,"p500k":455,"p1000k":1067}[k[0]]:
            raise ValueError("unregistered FFN width")
        if r.get("upper_state_width")!=48 or r.get("scale_axis")!="FFN_WIDTH_ONLY":
            raise ValueError("memory architecture mutated across capacity")
        if r.get("checkpoint_reloaded_identical_scored_DEV") is not True:
            raise ValueError("checkpoint cold replay failed")
        if r.get("GPU_minutes")!=0 or r.get("protected_TEST")!="NOT_ACCESSED":
            raise ValueError("resource or partition policy violated")
        if r.get("matched_training_FLOPs")!="NOT_MATCHED_BY_DESIGN_PARAMETER_SCALING":
            raise ValueError("different compute must be disclosed")
        if not math.isfinite(float(r.get("measured_cpu_process_seconds",float("nan")))):
            raise ValueError("bad measured CPU cost")
        score=r.get("DEV_EXPOSED_4096_ONLY",{})
        if score.get("n")!=4096:
            raise ValueError("wrong original DEV scoring")
        for met in ("nll","top1","brier","top5"):
            if not math.isfinite(float(score.get(met,float("nan")))):
                raise ValueError("non-finite authentic DEV metric")
        checkpoint=f.parent/r["checkpoint_filename"]
        if not checkpoint.exists():
            raise ValueError("missing original checkpoint artifact for "+str(k))
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=r["checkpoint_sha256"]:
            raise ValueError("checkpoint digest failed for "+str(k))
        rows[k]=r
    if set(rows)!=expected:
        raise ValueError("need all 9 original completed checkpoint receipts, found "+str(sorted(rows)))
    for seed in SEEDS:
        group=[rows[tier,seed] for tier in TIERS]
        if len({row["source_sample_order_sha256"] for row in group})!=1:
            raise ValueError("different original FIT batches for paired size arms")
        if len({row["source_bytes_sha256"] for row in group})!=1:
            raise ValueError("different original original-bytes samples")
        if len({row["source_commit"] for row in group})!=1:
            raise ValueError("publisher source different across arms")
    metrics={}
    for tier in TIERS:
        subset=[rows[tier,seed] for seed in SEEDS]
        metrics[tier]={
            "trainable_parameters":TIERS[tier],
            "ffn_width":{"p250k":150,"p500k":455,"p1000k":1067}[tier],
            "nll_mean":statistics.mean(r["DEV_EXPOSED_4096_ONLY"]["nll"] for r in subset),
            "nll_sample_sd":statistics.stdev(r["DEV_EXPOSED_4096_ONLY"]["nll"] for r in subset),
            "top1_mean":statistics.mean(r["DEV_EXPOSED_4096_ONLY"]["top1"] for r in subset),
            "brier_mean":statistics.mean(r["DEV_EXPOSED_4096_ONLY"]["brier"] for r in subset),
            "cpu_process_seconds_mean":statistics.mean(r["measured_cpu_process_seconds"] for r in subset),
            "raw_by_seed":{str(seed):{
                "nll":rows[tier,seed]["DEV_EXPOSED_4096_ONLY"]["nll"],
                "top1":rows[tier,seed]["DEV_EXPOSED_4096_ONLY"]["top1"],
                "cpu_process_seconds":rows[tier,seed]["measured_cpu_process_seconds"],
                "checkpoint_sha256":rows[tier,seed]["checkpoint_sha256"],
                "sample_order_sha256":rows[tier,seed]["source_sample_order_sha256"],
            } for seed in SEEDS}
        }
    paired={}
    for a,b in (("p250k","p500k"),("p500k","p1000k"),("p250k","p1000k")):
        diffs={str(seed):rows[b,seed]["DEV_EXPOSED_4096_ONLY"]["nll"]-
                         rows[a,seed]["DEV_EXPOSED_4096_ONLY"]["nll"] for seed in SEEDS}
        paired[a+"_to_"+b]={"delta_nll_by_seed":diffs,
                            "delta_nll_mean":statistics.mean(diffs.values()),
                            "seed_count":len(diffs),
                            "all_seeds_larger_lower_nll":all(x<0 for x in diffs.values())}
    return {"study":"LANGSCALE070","status":"SAME_SOURCE_SAME_UPDATES_EXPOSED_DEV_INTERNAL_ONLY",
            "fitted_checkpoints":len(rows),"models_per_scale":len(SEEDS),
            "architecture":"TAC fixed base and graph-free upper48, FFN width only",
            "fitness_axis":"next-byte natural NLL per exposed publisher original DEV byte",
            "training_updates_per_model":UPDATES,"source_target_bytes_seen_per_fit":TARGETS,
            "all_checkpoint_cold_replays":True,"all_source_schedules_paired":True,
            "training_FLOP_parity":"NOT_MATCHED_BY_DESIGN",
            "architecture_superiority":"NOT_CLAIMED",
            "statistical_significance":"NOT_QUALIFIED_THREE_SEED_EXPLORATORY",
            "independently_qualified":False,"protected_TEST":"NOT_ACCESSED",
            "gpu_minutes":0,"metrics":metrics,"paired":paired,
            "limits":["same 160 updates and bytes, but parameter count and training FLOPs differ",
                      "only FFN head capacity changes; recurrent state memory stays fixed",
                      "source DEV previously exposed, not untouched independent TEST",
                      "no streaming API qualification for newly scaled models",
                      "small three-seed pilot not a general scaling law",
                      "whole PyTorch/PNDS source and backward/IEEE proof incomplete"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("immutable aggregate must not overwrite")
    result=collect(args.results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"study":result["study"],"status":result["status"],
                      "metrics":{k:{x:v[x] for x in ("trainable_parameters","nll_mean",
                                                     "top1_mean","cpu_process_seconds_mean")}
                                 for k,v in result["metrics"].items()},
                      "paired":{k:v["delta_nll_mean"] for k,v in result["paired"].items()}},sort_keys=True))


if __name__=="__main__":
    main()
