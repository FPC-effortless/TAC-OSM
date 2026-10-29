#!/usr/bin/env python3
"""Fail-closed original 070/075 nine-checkpoint true carried source summary."""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path

TIERS={"m8":(499623,8,3904),"m12":(500175,12,4752),"m16":(500663,16,5728)}
SEEDS=(1701,1702,1703)
ARMS=("full","early_native_clear_at_128","one_step_native_zero")
CLOCKS=tuple(range(128,320,3))


def aggregate(folder:Path):
    files=list(folder.rglob("STREAM_RESULTS.json"))
    expected={(tier,seed) for tier in TIERS for seed in SEEDS}
    rows={}
    for path in files:
        x=json.loads(path.read_text())
        key=(x.get("tier"),x.get("seed"))
        if key not in expected or key in rows:
            raise ValueError("missing/duplicate original trained streaming source identity")
        param,k,size=TIERS[key[0]]
        if (x.get("study")!="LANGSTREAM076" or x.get("parameters")!=param
            or x.get("native_matrix_dim")!=k or x.get("state_bytes_FP32_per_stream")!=size
            or x.get("source_target_count")!=256 or x.get("warm_prefix_original_bytes")!=128
            or x.get("carried_history_last_clock")!=320
            or x.get("position_mode")!="cap_all_AFTER127_UNTRAINED_INFERENCE_POLICY"
            or tuple(x.get("target_clock_positions",[]))!=CLOCKS
            or x.get("no_new_training") is not True or x.get("gpu_minutes")!=0
            or x.get("protected_TEST")!="NOT_ACCESSED"):
            raise ValueError("original streaming source geometry, target or resource changed")
        for field in ("trained_checkpoint_sha256","trained_source_schedule_sha256","source_target_fingerprint"):
            if not isinstance(x.get(field),str) or len(x[field])!=64:
                raise ValueError("missing original source provenance: "+field)
        if len(x.get("original_source_owners",[]))!=4 or len(x.get("source_rows",[]))!=256:
            raise ValueError("original source outcomes missing")
        if set(x.get("mean_metrics",{}))!=set(ARMS):
            raise ValueError("causal memory lesion control missing")
        for arm in ARMS:
            metrics=x["mean_metrics"][arm]
            for field in ("nll","brier","top1"):
                if not math.isfinite(metrics[field]):
                    raise ValueError("nonfinite real source score")
            if not 0<=metrics["top1"]<=1:raise ValueError("invalid original top1")
        rows[key]=x
    if set(rows)!=expected:
        raise ValueError("requires all nine original source model+seed stream results")
    fingerprints={x["source_target_fingerprint"] for x in rows.values()}
    if len(fingerprints)!=1:
        raise ValueError("different original publisher target bytes across compared arms")
    signature=None
    for row in rows.values():
        current=tuple((x["owner"],x["t"],x["source_target_byte"]) for x in row["source_rows"])
        if signature is None:signature=current
        if current!=signature:raise ValueError("different per-owner source event or target order")
        if len(set((o,t) for o,t,y in current))!=256:
            raise ValueError("source next-byte target duplicate")
    out={}
    for tier in TIERS:
        group=[rows[tier,seed] for seed in SEEDS]
        measures={}
        for arm in ARMS:
            measures[arm]={m:statistics.mean(x["mean_metrics"][arm][m] for x in group)
                           for m in ("nll","brier","top1")}
        per_seed={}
        for x in group:
            nll=x["mean_metrics"]
            per_seed[str(x["seed"])]={"full_nll":nll["full"]["nll"],
                "early_native_clear_minus_full":nll["early_native_clear_at_128"]["nll"]-nll["full"]["nll"],
                "one_step_native_zero_minus_full":nll["one_step_native_zero"]["nll"]-nll["full"]["nll"],
                "early_native_max_logit_diff":x["mean_abs_logit_early_diff"],
                "lesion_max_logit_diff":x["mean_abs_logit_lesion_diff"],
                "source_checkpoint_sha256":x["trained_checkpoint_sha256"]}
        out[tier]={"parameters":TIERS[tier][0],"native_dim":TIERS[tier][1],
                   "persistent_FP32_bytes_per_owner":TIERS[tier][2],
                   "metrics":measures,"by_seed":per_seed,
                   "cpu_process_seconds_mean":statistics.mean(x["cpu_process_seconds"] for x in group)}
    paired={}
    for tier in ("m12","m16"):
        vals={str(seed):rows[tier,seed]["mean_metrics"]["full"]["nll"]-
                        rows["m8",seed]["mean_metrics"]["full"]["nll"] for seed in SEEDS}
        paired[tier+"_minus_m8"]={"nll_delta_by_seed":vals,"nll_delta_mean":statistics.mean(vals.values()),
                                   "new_lower_nll_count":sum(z<0 for z in vals.values())}
    return {"study":"LANGSTREAM076",
            "status":"NINE_ORIGINAL_TRAINED_CARRIED_STREAMS_EXPOSED_DEV_CAUSAL_LESIONS",
            "model_seed_results":9,"no_new_training":True,"gpu_minutes":0,
            "protected_TEST":"NOT_ACCESSED",
            "same_original_observed_next_byte_targets":256,
            "original_targets_sha256":next(iter(fingerprints)),
            "max_state_clock":320,"trained_prefix_window":128,
            "position_mode":"cap_all_POST_EXPOSED_DEV_SELECTION",
            "native_erasure_only_does_not_erase_other_memories":True,
            "metrics":out,"paired":paired,"independently_qualified":False,
            "limits":["previously exposed publisher original DEV, 256 source targets/model",
                      "true carried state through byte319, but no direct long-range fact retrieval diagnosis",
                      "native lesioned before current write; other aligned/upper/wave/concepts still contain earlier data",
                      "inference time source cap_all is post-selected after 073 EXPOSED DEV results",
                      "different k training FLOPs unequal and aligned address/readout change with native k",
                      "not independent model quality/safety/Android/real autonomous intelligence claim",
                      "whole PNDS actual source IEEE/AD mathematical and long-horizon proof remain OPEN"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("preserve evidence")
    r=aggregate(a.results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("CARRIED_NATIVE_AGGREGATE "+json.dumps({
        "status":r["status"],"models":r["model_seed_results"],
        "metrics":{k:{"full_nll":v["metrics"]["full"]["nll"],
                      "early_clear_nll":v["metrics"]["early_native_clear_at_128"]["nll"],
                      "one_step_zero_nll":v["metrics"]["one_step_native_zero"]["nll"],
                      "state_bytes":v["persistent_FP32_bytes_per_owner"]}
                   for k,v in r["metrics"].items()},
        "paired":r["paired"]},sort_keys=True))


if __name__=="__main__":main()
