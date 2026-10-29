#!/usr/bin/env python3
"""LANGSCALE075 source, checkpoint, seed-paired m8/m12/m16 evidence.

Six NEW native width12/16 original-source 075 fits; original m8 reused
from original verified LANGSCALE070 result artifact and never retrained.
Training FLOPs across different upper state/FFN geometry are NOT matched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

COUNTS={"m8":499623,"m12":500175,"m16":500663}
UPPER={"m8":8,"m12":12,"m16":16}
FFN={"m8":455,"m12":455,"m16":455}
NEW=("m12","m16")
SEEDS=(1701,1702,1703)


def collect(root:Path,original_070_summary:Path):
    previous=json.loads(original_070_summary.read_text())
    if (previous.get("study")!="LANGSCALE070" or previous.get("fitted_checkpoints")!=9
            or previous.get("all_source_schedules_paired") is not True
            or previous.get("protected_TEST")!="NOT_ACCESSED"):
        raise ValueError("not original SHA-validated nine 070 checkpoints aggregate")
    old=previous["metrics"]["p500k"]
    if old["trainable_parameters"]!=COUNTS["m8"] or old["ffn_width"]!=FFN["m8"]:
        raise ValueError("original 070 m8 trained source geometry mismatch")
    rows={}
    originals=list(root.rglob("FIT_AND_EXPOSED_DEV_RECEIPT.json"))
    for path in originals:
        r=json.loads(path.read_text())
        key=(r.get("arm"),r.get("seed"))
        if key[0] not in NEW or key[1] not in SEEDS or key in rows:
            raise ValueError("duplicate/new/unregistered arm "+str(key))
        if r.get("study")!="LANGSCALE075" or r.get("updates")!=160:
            raise ValueError("study/step mismatch")
        if (r.get("target_bytes_seen")!=163840 or r.get("examples_seen")!=1280
                or r.get("scale_axis")!="NATIVE_ASSOCIATIVE_KEY_VALUE_AND_ALIGNED_ADDRESS"):
            raise ValueError("source exposure/reallocation mismatch")
        if (r.get("parameters")!={"trainable":COUNTS[key[0]],
                                    "deployed":COUNTS[key[0]]}
                or r.get("native_matrix_dim")!=UPPER[key[0]]
                or r.get("ffn_width")!=FFN[key[0]]):
            raise ValueError("actual original source geometry mismatch")
        if (r.get("native_and_aligned_address_scaled") is not True
                or r.get("native_matrix_FP32_active_bytes_per_owner")!=4*UPPER[key[0]]*UPPER[key[0]]
                or r.get("aligned_matrix_FP32_bytes_per_owner")!=2*16*UPPER[key[0]]*4
                or r.get("matched_training_FLOPs")!="NOT_MATCHED_DIFFERENT_NATIVE_MATRIX_WIDTH"):
            raise ValueError("wrong native state or undeclared FLOPs parity")
        if (r.get("checkpoint_reloaded_identical_scored_DEV") is not True
                or r.get("protected_TEST")!="NOT_ACCESSED" or r.get("GPU_minutes")!=0):
            raise ValueError("trained checkpoint or partition evidence fail")
        if r.get("DEV_EXPOSED_4096_ONLY",{}).get("n")!=4096:
            raise ValueError("DEV score not original 4096 genuine next-byte targets")
        if not all(math.isfinite(r["DEV_EXPOSED_4096_ONLY"][k]) for k in
                   ("nll","top1","brier","top5")):
            raise ValueError("nonfinite trained original source DEV score")
        path_to_cp=path.parent/r["checkpoint_filename"]
        if (not path_to_cp.exists() or
                hashlib.sha256(path_to_cp.read_bytes()).hexdigest()!=r["checkpoint_sha256"]):
            raise ValueError("original checkpoint SHA bytes do not match")
        ref=old["raw_by_seed"][str(key[1])]
        if r["source_sample_order_sha256"]!=ref["sample_order_sha256"]:
            raise ValueError("new training source sample order not original 070")
        if r["source_commit"]!="60403a5409ff2c3f3b07dd2ca91a7a3e096839c7":
            raise ValueError("publisher original source commit changed")
        rows[key]=r
    if set(rows)!={(tier,seed) for tier in NEW for seed in SEEDS}:
        raise ValueError("missing original nine new trained full checkpoint receipts")
    if max(COUNTS.values())-min(COUNTS.values())>0.01*max(COUNTS.values()):
        raise ValueError("parameter-parity condition violated")
    per_tier={}
    for tier in COUNTS:
        rec=[]
        for seed in SEEDS:
            if tier=="m8":
                d=old["raw_by_seed"][str(seed)]
                rec.append({"seed":seed,"nll":d["nll"],"top1":d["top1"],
                    "cpu_process_seconds":d["cpu_process_seconds"],
                    "checkpoint_sha256":d["checkpoint_sha256"],
                    "source_sample_order_sha256":d["sample_order_sha256"],
                    "source":"ORIGINAL_LANGSCALE070_UNCHANGED"})
            else:
                d=rows[tier,seed]
                rec.append({"seed":seed,"nll":d["DEV_EXPOSED_4096_ONLY"]["nll"],
                    "top1":d["DEV_EXPOSED_4096_ONLY"]["top1"],
                    "cpu_process_seconds":d["measured_cpu_process_seconds"],
                    "checkpoint_sha256":d["checkpoint_sha256"],
                    "source_sample_order_sha256":d["source_sample_order_sha256"],
                    "source":"NEW_LANGSCALE075_CPU_FIT"})
        per_tier[tier]={
            "native_associative_dimension":UPPER[tier],"ffn_width":FFN[tier],
            "trainable_parameters":COUNTS[tier],
            "native_active_matrix_FP32_bytes_per_owner":4*UPPER[tier]*UPPER[tier],
            "aligned_FP32_bytes_per_owner":2*16*UPPER[tier]*4,
            "mean_DEV_nll":statistics.mean(x["nll"] for x in rec),
            "sd_DEV_nll":statistics.stdev(x["nll"] for x in rec),
            "mean_DEV_top1":statistics.mean(x["top1"] for x in rec),
            "mean_cpu_process_seconds":statistics.mean(x["cpu_process_seconds"] for x in rec),
            "original_seed_receipts":rec}
    paired={}
    for tier in NEW:
        delta={str(seed):per_tier[tier]["original_seed_receipts"][i]["nll"]-
                         per_tier["m8"]["original_seed_receipts"][i]["nll"]
               for i,seed in enumerate(SEEDS)}
        paired[tier+"_minus_m8"]={"per_seed_delta_nll":delta,
            "mean_delta_nll":statistics.mean(delta.values()),
            "n_seeds_new_lower_nll":sum(x<0 for x in delta.values())}
    return {"study":"LANGSCALE075",
            "status":"SIX_NEW_NATIVE_MATRIX_FITS_PLUS_THREE_ORIGINAL_M8_SOURCE_PAIRED",
            "new_checkpoints_verified":6,"original_070_reference_checkpoints":3,
            "original_070_training_run":40871491842,
            "source_paired":True,"parameter_parity_within_one_percent":True,
            "native_fast_slow_projection_widths":"88_88_UNCHANGED",
            "native_associative_and_aligned_address":"SCALED_8_TO_12_TO_16",
            "upper_recurrent_state":"48_UNCHANGED",
            "training_FLOP_parity":"NOT_MATCHED",
            "trained_update_count_each":160,
            "dev_target_count_per_seed":4096,
            "metrics":per_tier,"paired_against_m8":paired,
            "protected_TEST":"NOT_ACCESSED","gpu_minutes":0,
            "independently_qualified":False,
            "limits":["historical original 070 m8 reference same original source samples and 160 steps, replay SHA from prior run",
                      "native key/value/readout and aligned address geometry change; not pure native-matrix-only lesion",
                      "training FLOPs not measured/matched across new geometries",
                      "native and aligned address capacity varies; upper48, fast/slow88/88 and FFN455 unchanged",
                      "development source previously exposed; no independent generalization/TEST",
                      "full PNDS original source IEEE/AD graph and carried-streaming math OPEN"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("do not overwrite evidence")
    result=collect(a.results,a.reference)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("UPPER_REALLOCATION_AGGREGATE "+json.dumps({
        "source_paired":result["source_paired"],
        "metrics":{k:{n:v[n] for n in ("mean_DEV_nll","mean_DEV_top1",
              "mean_cpu_process_seconds","trainable_parameters","upper_hidden_width","ffn_width")}
              for k,v in result["metrics"].items()},
        "paired":result["paired_against_m8"],
    },sort_keys=True))


if __name__=="__main__":
    main()
