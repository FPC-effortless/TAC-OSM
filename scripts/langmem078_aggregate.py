#!/usr/bin/env python3
"""LANGMEM078 original source-verified 3 dual native fits vs original 070 m8."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
SEEDS=(1701,1702,1703)
COUNT=499627
ORIGINAL=499623

def aggregate(root:Path,reference_file:Path):
    old=json.loads(reference_file.read_text())
    if (old.get("study")!="LANGSCALE070" or old.get("fitted_checkpoints")!=9
        or old.get("all_source_schedules_paired") is not True
        or old.get("protected_TEST")!="NOT_ACCESSED"):
        raise ValueError("unverified original baseline source training")
    original=old["metrics"]["p500k"]
    if original["trainable_parameters"]!=ORIGINAL or original["ffn_width"]!=455:
        raise ValueError("original baseline learned geometry mismatch")
    receipts={}
    for p in root.rglob("FIT_AND_EXPOSED_DEV_RECEIPT.json"):
        d=json.loads(p.read_text())
        seed=d.get("seed")
        if seed not in SEEDS or seed in receipts or d.get("study")!="LANGMEM078" or d.get("arm")!="dual":
            raise ValueError("unregistered or duplicate trained dual source")
        if (d.get("updates")!=160 or d.get("parameters")!={"trainable":COUNT,"deployed":COUNT}
            or d.get("examples_seen")!=1280 or d.get("target_bytes_seen")!=163840
            or d.get("scale_axis")!="NATIVE_FAST_PLUS_SLOW_BOUNDED_AFFINE_LANE"
            or d.get("original_fast_native_state_unchanged") is not True
            or d.get("new_active_learned_scalars")!=4
            or d.get("native_matrix_dim")!=8 or d.get("slow_native_matrix_dim")!=8
            or d.get("slow_extra_FP32_persistent_bytes_per_owner")!=256
            or d.get("upper_state_width")!=48 or d.get("ffn_width")!=455
            or d.get("matched_training_FLOPs")!="NOT_MATCHED_EXTRA_SLOW_NATIVE_AFFINE_SCAN"
            or d.get("checkpoint_reloaded_identical_scored_DEV") is not True
            or d.get("GPU_minutes")!=0 or d.get("protected_TEST")!="NOT_ACCESSED"):
            raise ValueError("preregistered dual source/fit/checkpoint mismatch")
        target=d.get("DEV_EXPOSED_4096_ONLY")
        if not isinstance(target,dict) or target.get("n")!=4096 or not all(
              math.isfinite(target[k]) for k in ("nll","top1","brier","top5")):
            raise ValueError("genuine source exposed DEV metric missing")
        artifact=p.parent/d["checkpoint_filename"]
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest()!=d["checkpoint_sha256"]:
            raise ValueError("original checkpoint SHA mismatch")
        ref=original["raw_by_seed"][str(seed)]
        if d["source_sample_order_sha256"]!=ref["sample_order_sha256"]:
            raise ValueError("original 070 FIT owner/offset schedule differed")
        if d["source_commit"]!="60403a5409ff2c3f3b07dd2ca91a7a3e096839c7":
            raise ValueError("source publisher original commit mismatch")
        receipts[seed]=d
    if set(receipts)!=set(SEEDS):
        raise ValueError("missing one or more of three original dual trained source checkpoints")
    metrics={}
    for arm in ("original_fast","dual_fast_slow"):
        entries=[]
        for seed in SEEDS:
            if arm=="original_fast":
                ref=original["raw_by_seed"][str(seed)]
                entries.append({"seed":seed,"nll":ref["nll"],"top1":ref["top1"],
                      "cpu_process_seconds":ref["cpu_process_seconds"],
                      "checkpoint_sha256":ref["checkpoint_sha256"],
                      "source_sample_order_sha256":ref["sample_order_sha256"]})
            else:
                r=receipts[seed]
                entries.append({"seed":seed,"nll":r["DEV_EXPOSED_4096_ONLY"]["nll"],
                      "top1":r["DEV_EXPOSED_4096_ONLY"]["top1"],
                      "cpu_process_seconds":r["measured_cpu_process_seconds"],
                      "checkpoint_sha256":r["checkpoint_sha256"],
                      "source_sample_order_sha256":r["source_sample_order_sha256"]})
        metrics[arm]={"mean_nll":statistics.mean(x["nll"] for x in entries),
          "sd_nll":statistics.stdev(x["nll"] for x in entries),
          "mean_top1":statistics.mean(x["top1"] for x in entries),
          "mean_cpu_process_seconds":statistics.mean(x["cpu_process_seconds"] for x in entries),
          "learned_parameters":ORIGINAL if arm=="original_fast" else COUNT,
          "original_seed_records":entries}
    diffs={str(seed):receipts[seed]["DEV_EXPOSED_4096_ONLY"]["nll"]-
          original["raw_by_seed"][str(seed)]["nll"] for seed in SEEDS}
    return {"study":"LANGMEM078","status":"THREE_NEW_DUAL_NATIVE_TRAINED_CHECKPOINTS_VS_ORIGINAL_070",
      "new_original_checkpoint_count":3,"source_reference_checkpoint_count":3,
      "original_training_run_id":40871491842,
      "source_paired":True,"parameter_parity_within_one_percent":True,
      "source_tested":"LICENSED_ORIGINAL_PUBLISHER_DEV_PREVIOUSLY_EXPOSED",
      "training_FLOP_parity":"NOT_MATCHED",
      "original_fast_native_8x8":"UNCHANGED",
      "slow_native_8x8":"ADDED",
      "extra_fp32_persistent_matrix_bytes":256,
      "whole_model_carried_streaming":"NOT_QUALIFIED",
      "metrics":metrics,
      "paired_dual_minus_original_nll":diffs,
      "mean_delta_nll":statistics.mean(diffs.values()),
      "new_better_seed_count":sum(v<0 for v in diffs.values()),
      "gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
      "independently_qualified":False,
      "limits":["four DEV source owners previously exposed; no independent generalization",
                "reset128 windows, no carried learned slow lane quality assessment",
                "different training FLOPs due extra slow scan; original 070 baseline reused",
                "no protected TEST or trained model promotion, partial PNDS mathematical correspondence"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("source aggregate overwrite prohibited")
    r=aggregate(a.results,a.reference)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("DUAL_NATIVE078_AGGREGATE "+json.dumps({
      "metrics":{k:{"nll":v["mean_nll"],"top1":v["mean_top1"],
                    "cpu_sec":v["mean_cpu_process_seconds"]}
                 for k,v in r["metrics"].items()},
      "diffs":r["paired_dual_minus_original_nll"],
      "new_better_seed_count":r["new_better_seed_count"]},sort_keys=True))
if __name__=="__main__":main()
