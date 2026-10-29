#!/usr/bin/env python3
"""Aggregate all nine source-verified no-fit context length reset-window tests."""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path

TIERS={"p250k":250133,"p500k":499623,"p1000k":1000239}
SEEDS=(1701,1702,1703)
CONTEXTS=(16,32,64,128,256)


def aggregate(folder:Path):
    items=list(folder.rglob("CONTEXT_RESULTS.json"))
    expected={(a,s) for a in TIERS for s in SEEDS}
    rows={}
    for path in items:
        row=json.loads(path.read_text())
        key=(row.get("tier"),row.get("seed"))
        if key not in expected or key in rows:raise ValueError("wrong or duplicate source checkpoint arm")
        if (row.get("study")!="LANGSCALE071" or row.get("training_updates")!=0
                or row.get("gpu_minutes")!=0 or row.get("protected_TEST")!="NOT_ACCESSED"):
            raise ValueError("new fitting/test leakage")
        if row.get("parameters")!=TIERS[key[0]] or row.get("context_target_count")!=256:
            raise ValueError("original model count/target change")
        if not isinstance(row.get("target_cohort_sha256"),str) or len(row["target_cohort_sha256"])!=64:
            raise ValueError("missing original source target fingerprint")
        if len(row.get("lengths",[]))!=len(CONTEXTS) or tuple(x["context"] for x in row["lengths"])!=CONTEXTS:
            raise ValueError("wrong original source context set")
        for length_row in row["lengths"]:
            if length_row["n"]!=256 or not all(
                    math.isfinite(length_row[k]) for k in ("nll","top1","brier","cpu_process_seconds")):
                raise ValueError("wrong original sample count / nonfinite quality")
        if abs(row["original_full_DEV_replay_nll"]-row["trained_checkpoint"]["original_DEV_NLL"])>1e-6:
            raise ValueError("trained original model checkpoint replay diverged")
        rows[key]=row
    if set(rows)!=expected:
        raise ValueError("missing original trained checkpoint context result "+str(sorted(expected-set(rows))))
    cohort_ids={r["target_cohort_sha256"] for r in rows.values()}
    if len(cohort_ids)!=1:raise ValueError("context cohorts use different original DEV target labels")
    for i,length in enumerate(CONTEXTS):
        if len({row["lengths"][i]["inputs_and_targets_sha256"] for row in rows.values()})!=1:
            raise ValueError("different original source inputs at context "+str(length))
    scores={}
    for tier in TIERS:
        scores[tier]={}
        for i,length in enumerate(CONTEXTS):
            series=[rows[tier,s]["lengths"][i] for s in SEEDS]
            scores[tier][str(length)]={
                "nll_mean":statistics.mean(r["nll"] for r in series),
                "nll_sd":statistics.stdev(r["nll"] for r in series),
                "top1_mean":statistics.mean(r["top1"] for r in series),
                "brier_mean":statistics.mean(r["brier"] for r in series),
                "cpu_process_seconds_mean":statistics.mean(r["cpu_process_seconds"] for r in series),
                "n_targets_per_seed":256,
                "nll_by_seed":{str(s):rows[tier,s]["lengths"][i]["nll"] for s in SEEDS}}
    paired={}
    for tier in TIERS:
        for a,b in ((16,32),(32,64),(64,128),(128,256),(16,128)):
            deltas={str(seed):scores[tier][str(b)]["nll_by_seed"][str(seed)]-
                               scores[tier][str(a)]["nll_by_seed"][str(seed)] for seed in SEEDS}
            paired[tier+":"+str(a)+"->"+str(b)]={"delta_nll_mean":statistics.mean(deltas.values()),
                "delta_nll_by_seed":deltas,"all_seeds_more_context_lower_nll":all(z<0 for z in deltas.values())}
    return {"study":"LANGSCALE071",
            "status":"ORIGINAL_FROZEN_CHECKPOINTS_EXPOSED_DEV_RESET_CONTEXT_ONLY",
            "models_evaluated":9,"new_neural_fits":0,"original_fitted_checkpoint_run":40871491842,
            "target_cohort_sha256":next(iter(cohort_ids)),
            "all_original_source_cohorts_and_context_inputs_identical":True,
            "same_targets_per_model_per_context":256,
            "context_lengths":list(CONTEXTS),"trained_window_bytes":128,
            "original_source_context_mode":"RESET_AT_EACH_TARGET_NOT_CARRIED",
            "quality_by_tier_context":scores,"paired_context_deltas":paired,
            "independently_qualified":False,"protected_TEST":"NOT_ACCESSED","gpu_minutes":0,
            "limits":["previously exposed DEV original 4 documents, target positions fixed and re-used",
                      "all context inputs reset, not persistent online streaming and no long-memory claim",
                      "L256 outside trained 128B history and may be distribution shift",
                      "no new training, different fitted model sizes with unequal historical training FLOPs",
                      "small 3-seed, 256 target no-fit exploration; no architecture superiority or general scaling law",
                      "full PNDS actual PyTorch source/IEEE/AD and mobile execution remain OPEN"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("refuse to overwrite original evidence")
    result=aggregate(a.results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("CONTEXT_AGGREGATE "+json.dumps({
        "study":result["study"],"status":result["status"],
        "nll_by_tier":{t:{k:round(z["nll_mean"],6) for k,z in d.items()}
                       for t,d in result["quality_by_tier_context"].items()},
        "paired":{key:round(x["delta_nll_mean"],6) for key,x in result["paired_context_deltas"].items()}},
        sort_keys=True),flush=True)


if __name__=="__main__":
    main()
