#!/usr/bin/env python3
"""LANGMEM077: original nine-checkpoint source-dependent retention audit."""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path

TIERS={"m8":(499623,8,3904),"m12":(500175,12,4752),"m16":(500663,16,5728)}
SEEDS=(1701,1702,1703)
CLOCKS=(128,129,131,135,143,159,191,255,287)
AGES=tuple(t-127 for t in CLOCKS)
SOURCE_ROWS=4*(320-128)


def aggregate(root:Path):
    files=list(root.rglob("RETENTION_RESULTS.json"))
    expected={(tier,seed) for tier in TIERS for seed in SEEDS}
    seen={}
    for f in files:
        r=json.loads(f.read_text())
        tier,seed=r.get("tier"),r.get("seed")
        key=(tier,seed)
        if key not in expected or key in seen:
            raise ValueError("original arm/seed unexpected or duplicate")
        params,k,size=TIERS[tier]
        if (r.get("study")!="LANGMEM077" or r.get("model_parameters")!=params
            or r.get("registered_native_dim")!=k
            or r.get("fp32_state_bytes_per_owner")!=size
            or r.get("original_target_count")!=SOURCE_ROWS
            or r.get("clock_first")!=128 or r.get("clock_end_exclusive")!=320
            or r.get("native_erased_at_clock")!=128
            or tuple(r.get("sample_ages",[]))!=AGES
            or r.get("no_fits") is not True or r.get("no_optimizer_steps") is not True
            or r.get("gpu_minutes")!=0 or r.get("protected_TEST")!="NOT_ACCESSED"):
            raise ValueError("registered original source & nofit receipt differs")
        if len(r.get("original_source_owners",[]))!=4:
            raise ValueError("original source owner count changed")
        for field in ("original_model_checkpoint_sha256","original_fit_sample_order_sha256",
                      "original_source_label_sha256"):
            x=r.get(field)
            if type(x) is not str or len(x)!=64 or not all(c in "0123456789abcdef" for c in x):
                raise ValueError("source/checkpoint fingerprint missing or malformed")
        if len(r.get("source_rows",[]))!=SOURCE_ROWS:
            raise ValueError("original source row target missing")
        if set(r.get("selected_age_summary",{}))!=set(map(str,AGES)):
            raise ValueError("predeclared original retention delays missing")
        if not math.isfinite(r["metrics"]["max_source_native_delta_update_abs_error"]) or r["metrics"]["max_source_native_delta_update_abs_error"]>1e-4:
            raise ValueError("native difference identity not verified")
        if not 0<r["metrics"]["native_retention_alpha_mean"]<.995:
            raise ValueError("trained gate not original source range")
        if not 0<r["metrics"]["native_forget_half_life_source_steps_from_mean_log_alpha"]<100000:
            raise ValueError("invalid source-native retention bound")
        for age,v in r["selected_age_summary"].items():
            if not 0<=v["mean_retention_upper_bound"]<=1:
                raise ValueError("retention bound outside [0,1]")
            if v["mean_actual_native_difference_ratio"]>v["mean_retention_upper_bound"]+1e-4:
                raise ValueError("measured native difference violates source upper bound")
        seen[key]=r
    if set(seen)!=expected:raise ValueError("some original trained checkpoint diagnostics missing")
    target_shas={r["original_source_label_sha256"] for r in seen.values()}
    if len(target_shas)!=1:raise ValueError("original source next-byte labels differ across arms")
    row_signature=None
    for r in seen.values():
        sig=tuple((x["owner"],x["clock"],x["source_target_byte"]) for x in r["source_rows"])
        if len(set((name,t) for name,t,y in sig))!=SOURCE_ROWS:
            raise ValueError("duplicate original source target owner/clock")
        if row_signature is None:row_signature=sig
        if row_signature!=sig:
            raise ValueError("original source byte targets changed across models")
    metrics={}
    for tier in TIERS:
        group=[seen[tier,seed] for seed in SEEDS]
        summary={}
        for age in AGES:
            rows=[r["selected_age_summary"][str(age)] for r in group]
            summary[str(age)]={field:statistics.mean(x[field] for x in rows)
                for field in ("mean_alpha","mean_erase_gate",
                              "mean_retention_upper_bound",
                              "mean_actual_native_difference_ratio",
                              "mean_native_read_delta_l2","mean_full_nll",
                              "mean_early_erase_minus_full_nll")}
        metrics[tier]={
            "parameters":TIERS[tier][0],
            "native_matrix_dim":TIERS[tier][1],
            "state_bytes_per_owner":TIERS[tier][2],
            "num_trained_seeds":len(group),
            "mean_source_native_alpha":statistics.mean(r["metrics"]["native_retention_alpha_mean"] for r in group),
            "mean_source_retention_half_life_upper_bound_tokens":statistics.mean(r["metrics"]["native_forget_half_life_source_steps_from_mean_log_alpha"] for r in group),
            "mean_source_erase_gate":statistics.mean(r["metrics"]["native_erase_beta_mean"] for r in group),
            "mean_abs_query_write_cosine":statistics.mean(r["metrics"]["mean_abs_query_write_cosine"] for r in group),
            "mean_fraction_abs_cosine_at_least_0_9":statistics.mean(r["metrics"]["fraction_abs_query_write_cosine_ge_0_9"] for r in group),
            "mean_original_full_nll":statistics.mean(r["metrics"]["mean_full_nll"] for r in group),
            "mean_early_native_erase_minus_full_nll":statistics.mean(r["metrics"]["mean_early_erase_minus_full_nll"] for r in group),
            "max_native_delta_equation_abs_error":max(r["metrics"]["max_source_native_delta_update_abs_error"] for r in group),
            "retention_by_age":summary,
            "original_source_checkpoints_sha256":{str(r["seed"]):r["original_model_checkpoint_sha256"] for r in group}
        }
    return {"study":"LANGMEM077",
        "status":"NINE_ORIGINAL_SOURCE_NATIVE_RETENTION_AFFINE_DIAGNOSTICS_REPLAYED",
        "original_source_target_sha256":next(iter(target_shas)),
        "source_targets_per_checkpoint":SOURCE_ROWS,
        "original_trained_checkpoint_results":9,
        "no_fits":True,"gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
        "all_source_delta_equations_pass":True,"all_source_retention_bounds_pass":True,
        "position_mode":"cap_all_POST_EXPOSED_DEV",
        "independently_qualified":False,"metrics":metrics,
        "limits":[
          "authentic licensed original source but previously exposed DEV and post-exposed positional policy",
          "only old native memory state intervention; other recurrences are not erased",
          "alpha source upper bound shows rapid decay of a particular old native state, not total memory capacity",
          "query/write geometric similarity does not certify correct historical content retrieval",
          "float32 cancellation floor means observed tiny matrix norm is not exact zero",
          "original source models 075 had unequal FLOPs, no comparative architecture superiority claim"]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError("source results must not be overwritten")
    r=aggregate(args.results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LANGMEM077_AGGREGATE "+json.dumps({
        "models":9,"metrics":{tier:{
             "mean_alpha":v["mean_source_native_alpha"],
             "half_life_upper":v["mean_source_retention_half_life_upper_bound_tokens"],
             "full_nll":v["mean_original_full_nll"],
             "early_clear_nll_delta":v["mean_early_native_erase_minus_full_nll"],
             "ages":{age:{"upper":z["mean_retention_upper_bound"],
                          "actual":z["mean_actual_native_difference_ratio"]}
                     for age,z in v["retention_by_age"].items()}}
             for tier,v in r["metrics"].items()}},sort_keys=True))


if __name__=="__main__":
    main()
