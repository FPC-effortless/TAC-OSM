"""ESWAM014 independent paired original-source real-case audit, no model fitting.

Reads frozen source-measured true future labels and model original predictions
from genuine guarded ESWAM013 and ESWAM014 checkpoints. Actor-ID resampling
preserves within-actor dependency; not a new physical-site test. No tuning.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

ROOT=Path("research/replay-bundles")
HORIZONS=(10,30)
SEEDS=(87001,87002)
REPEATS=5000
EXPECTED=532


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(study,seed):
    path=ROOT/study/f"seed-{seed}-ORIGINAL-REAL-DEV-CASES.json"
    result=json.loads(path.read_text())
    if len(result)!=EXPECTED:
        raise RuntimeError(f"original real case count shifted for {study}/{seed}")
    return result,sha(path)


def audit():
    output={"study":"ESWAM014_PAIRED_GUARDED_GENUINE_ACTOR_DEV_AUDIT",
            "original_source_physically_recorded_not_synthetic":True,
            "original_exposed_dev_site":"TC_BGR_Intersection_VA",
            "new_independent_site":False,
            "original_publisher_archive_sha256":
                "9aa78102006dc8d8a64675cf64eba5264085cfb1c006b1f4d0e856d94e657aec",
            "competing_native_architectures_both_guarded":True,
            "training_flops_identical_certified":False,
            "original_actor_bootstrap_repeats":REPEATS,
            "new_optimizer_updates":0,
            "paired_real_rows":[]}
    for seed in SEEDS:
        old, sha_old = load("ESWAM013",seed)
        new, sha_new = load("ESWAM014",seed)
        for horizon in HORIZONS:
            paired=[]
            for x,y in zip(old,new):
                if x["horizon_frames"]!=y["horizon_frames"]:
                    raise RuntimeError("original source forecast horizon order changed")
                for key in ("source_original_recording","source_original_frame",
                            "source_original_timestamp_ms","source_original_actor_id",
                            "source_actor_cluster","true_original_future_xy_m",
                            "observed_current_xy_m","original_fixed_physical_predictions"):
                    if x[key]!=y[key]:
                        raise RuntimeError(f"source original actor/future/physical prior mismatch {key}")
                if x["horizon_frames"]==horizon:
                    paired.append((x,y))
            if len(paired)!=266:
                raise RuntimeError("source original actor-case count changed")
            groups=defaultdict(list)
            stationary=[]
            moving=[]
            for x,y in paired:
                delta=y["l2_meters"]["model"]-x["l2_meters"]["model"]
                groups[x["source_actor_cluster"]].append(delta)
                if x["current_measured_source_velocity_norm_m_per_s"]<.2:
                    stationary.append((x,y))
                else:
                    moving.append((x,y))
            ids=sorted(groups)
            rng=random.Random(140000+seed*37+horizon)
            simulated=[]
            for k in range(REPEATS):
                sampled=[rng.choice(ids) for _ in ids]
                num=sum(len(groups[a]) for a in sampled)
                total=sum(sum(groups[a]) for a in sampled)
                simulated.append(total/num)
            simulated.sort()
            means=lambda ar,name:sum(v["l2_meters"][name] for v in ar)/len(ar) if ar else None
            result={
                "seed":seed,"horizon_frames":horizon,
                "real_actor_cases":len(paired),"actor_clusters":len(ids),
                "old_original_checkpoint_real_cases_sha256":sha_old,
                "new_original_checkpoint_real_cases_sha256":sha_new,
                "candidate_old_mean_L2_m":means([x for x,_ in paired],"model"),
                "candidate_new_mean_L2_m":means([y for _,y in paired],"model"),
                "original_real_physical_prior_mean_L2_m":means([x for x,_ in paired],"prior"),
                "new_minus_old_original_L2_m":
                    sum(y["l2_meters"]["model"]-x["l2_meters"]["model"] for x,y in paired)/len(paired),
                "original_actor_cluster_bootstrap_new_minus_old_95pct":
                    [simulated[int(.025*REPEATS)],simulated[int(.975*REPEATS)-1]],
                "stationary_actor_cases_speed_below_0_2_mps":len(stationary),
                "stationary_new_mean_L2_m":means([y for _,y in stationary],"model"),
                "stationary_original_physical_prior_L2_m":means([x for x,_ in stationary],"prior"),
                "stationary_no_regression_gate_passed":
                    means([y for _,y in stationary],"model")<=
                    means([x for x,_ in stationary],"prior"),
                "stationary_old_minus_new_mean_L2_m":
                    means([x for x,_ in stationary],"model")-
                    means([y for _,y in stationary],"model"),
                "moving_actor_cases_speed_at_least_0_2_mps":len(moving),
            }
            output["paired_real_rows"].append(result)
            print("ESWAM014_PAIRED_REAL_ACTOR_AUDIT="+json.dumps(result,sort_keys=True),flush=True)
    output["both_seeds_stationary_no_regression_gate_passed"]=all(
        x["stationary_no_regression_gate_passed"] for x in output["paired_real_rows"])
    return output


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    o=audit()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(o,sort_keys=True,indent=2)+"\n")
    print("ESWAM014_PAIRED_AUDIT_COMPLETE",flush=True)
