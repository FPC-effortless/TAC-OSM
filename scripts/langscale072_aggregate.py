#!/usr/bin/env python3
"""All-nine source-matched cap256-vs-native256 diagnostic, post-exposed-DEV."""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path

TIERS={"p250k":250133,"p500k":499623,"p1000k":1000239}
SEEDS=(1701,1702,1703)
TARGET_SHA="5c267b3f6cf43ebbfd8e46a6e1e1376f2f730ec35a5db19293d4f0af6659f803"


def aggregate(folder:Path):
    names=list(folder.rglob("POSITION_RESULTS.json"))
    required={(tier,s) for tier in TIERS for s in SEEDS}
    rows={}
    for path in names:
        row=json.loads(path.read_text())
        key=(row.get("tier"),row.get("seed"))
        if key not in required or key in rows:raise ValueError("unexpected/duplicate model")
        if row.get("study")!="LANGSCALE072" or row.get("training_updates")!=0:
            raise ValueError("not genuine no-fit 072")
        if row.get("gpu_minutes")!=0 or row.get("protected_TEST")!="NOT_ACCESSED":
            raise ValueError("GPU/TEST attempted")
        if row.get("target_cohort_sha256")!=TARGET_SHA or row.get("same_targets")!=256:
            raise ValueError("same 071 source cohort not reused")
        if not row.get("hook_removed_and_model_weights_original"):
            raise ValueError("position hook/weights changed")
        a=row["original_position_128"];b=row["cap_position_128_negative_control"]
        for field in ("nll","top1","brier","n","inputs_and_targets_sha256"):
            if a[field]!=b[field]:raise ValueError("cap changed the in-distribution L128 control")
        x=row["original_position_256"];y=row["cap_position_256"]
        if x["n"]!=256 or y["n"]!=256 or x["inputs_and_targets_sha256"]!=y["inputs_and_targets_sha256"]:
            raise ValueError("position comparison changed actual source inputs")
        if abs(row["cap256_minus_native256_nll"]-(y["nll"]-x["nll"]))>1e-12:
            raise ValueError("incorrect source NLL delta")
        if any(not math.isfinite(v) for v in (x["nll"],y["nll"],a["nll"])):
            raise ValueError("nonfinite source score")
        rows[key]=row
    if set(rows)!=required:raise ValueError("need exactly 9 fully audited source checkpoints")
    if len({row["original_position_256"]["inputs_and_targets_sha256"] for row in rows.values()})!=1:
        raise ValueError("original source inputs differ across models")
    stats={}
    for tier in TIERS:
        pairs=[rows[tier,seed] for seed in SEEDS]
        diffs={str(seed):rows[tier,seed]["cap256_minus_native256_nll"] for seed in SEEDS}
        stats[tier]={
            "native128_mean":statistics.mean(r["original_position_128"]["nll"] for r in pairs),
            "native256_mean":statistics.mean(r["original_position_256"]["nll"] for r in pairs),
            "cap256_mean":statistics.mean(r["cap_position_256"]["nll"] for r in pairs),
            "cap_minus_native256_mean":statistics.mean(diffs.values()),
            "cap_improvement_count":sum(x<0 for x in diffs.values()),
            "cap_delta_by_seed":diffs,
            "n_targets_per_model":256,
        }
    return {"study":"LANGSCALE072",
            "status":"POST_EXPOSED_DEV_NO_FIT_POSITION_EXTRAPOLATION_DIAGNOSTIC_ONLY",
            "model_source_checkpoints":9,"new_training_updates":0,
            "original_target_sha256":TARGET_SHA,
            "cap128_noop_controls_pass":True,
            "model_weights_unchanged":True,
            "original_same_labels":256,
            "metrics":stats,"gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
            "independently_qualified":False,
            "limits":["posthoc inference transform, selected after 071 original exposed DEV worsening",
                      "positive scores not primary independent generalization or trusted TEST",
                      "256 positions exceed original 128-trained positional context",
                      "source cap_all affects position feature, not persistent recurrent scan",
                      "no training, memory causality proof, Android/mobile or SSM/GRU comparison",
                      "PNDS cap law partial, exact PyTorch full-source IEEE/AD OPEN"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("immutable output exists")
    out=aggregate(a.results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("POSITION_AGGREGATE "+json.dumps({"study":out["study"],
        "models":out["model_source_checkpoints"],
        "scores":out["metrics"]},sort_keys=True))


if __name__=="__main__":main()
