"""ME076 no-fit independent reading of original Git-owned real predictions.

Episode-cluster DEV uncertainty and historical task strata, NOT preregistered
independent confirmation, architecture superiority, or a new model checkpoint.
"""
from __future__ import annotations
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import statistics

ROOT=Path("research/replay-bundles/ME076")
OUT=Path("research/mathematical-engineering/me076/POSTHOC_REAL_SCORE_DIAGNOSTIC.json")
SEEDS=(76001,76002,76003,76004,76005)
ITERATIONS=5000
BOOTSTRAP_SEED=76076

def mean(values):
    return sum(values)/len(values) if values else None

def summarize(rows):
    by=[]
    for r in rows:by.append(r["average_model"]-r["persist"])
    return {"observed_query_count":len(rows),
            "original_episodes":len(set(r["episode"] for r in rows)),
            "native_minus_persistence_MAE":mean(by),
            "native_model_MAE":mean([r["average_model"] for r in rows]),
            "persistence_MAE":mean([r["persist"] for r in rows]),
            "fit_only_servo_MAE":mean([r["servo"] for r in rows]),
            "number_of_episode_motion_rich_queries":sum(r["moving"] for r in rows)}

def bootstrap_episodes(rows, *, seed=BOOTSTRAP_SEED):
    grouped=defaultdict(list)
    for r in rows:grouped[r["episode"]].append(r)
    keys=sorted(grouped)
    rng=random.Random(seed)
    values=[]
    for i in range(ITERATIONS):
        sampled=[keys[rng.randrange(len(keys))] for _ in keys]
        all_rows=[r for k in sampled for r in grouped[k]]
        values.append(mean([r["average_model"]-r["persist"] for r in all_rows]))
    values.sort()
    return {"method":"paired independent REAL EPISODE cluster bootstrap, 40 units, 5 seeds averaged per observed query",
            "bootstrap_iterations":ITERATIONS,"original_episodes":len(keys),
            "ci95_native_minus_persistence": [values[124],values[4874]],
            "seed":seed,
            "limits":"POSTHOC, exposed-FIT-derived development only; does not establish independent gain; model-seed uncertainty reported separately"}

def main():
    records=[]
    scores={}
    for seed in SEEDS:
        folder=ROOT/f"seed-{seed}"
        study=json.loads((folder/"RESULTS.json").read_text())
        rawfile=folder/"ORIGINAL_REAL_PREDICTIONS.json"
        if sha256(rawfile.read_bytes()).hexdigest()!=study["original_prediction_sha256"]:
            raise RuntimeError("original raw real predictions changed since original fit")
        r=json.loads(rawfile.read_text())
        if len(r)!=960:raise RuntimeError("per-seed real original score count changed")
        for row in r:
            key=(row["task"],row["episode_index"],row["frame_index"])
            values=scores.setdefault(key,[])
            if values and (row["physical_mae_persistence"]!=values[0]["physical_mae_persistence"] or
                          row["physical_mae_fit_servo_lag1"]!=values[0]["physical_mae_fit_servo_lag1"] or
                          row["motion_rich_threshold_0_025"]!=values[0]["motion_rich_threshold_0_025"]):
                raise RuntimeError("original factual label or baseline changed between model seeds")
            values.append(row)
        records.append({"seed":seed,"native_all_MAE":study["metrics"]["all"]["physical_mae_model"],
                        "native_motion_MAE":study["metrics"]["movement_rich"]["physical_mae_model"],
                        "original_checkpoint_sha256":study["original_checkpoint_sha256"],
                        "original_forecast_sha256":study["original_prediction_sha256"]})
    if len(scores)!=960 or any(len(v)!=5 for v in scores.values()):
        raise RuntimeError("original cross-seed cohort mismatch, no diagnosis")
    merged=[]
    for (task,ep,frame),values in sorted(scores.items()):
        r=values[0]
        merged.append({"task":task,"episode":f"{task}:{ep}","frame":frame,
                       "average_model":mean([v["physical_mae_model"] for v in values]),
                       "persist":r["physical_mae_persistence"],
                       "servo":r["physical_mae_fit_servo_lag1"],
                       "moving":bool(r["motion_rich_threshold_0_025"])})
    allresult=summarize(merged)
    motion=[r for r in merged if r["moving"]]
    rest=[r for r in merged if not r["moving"]]
    tasks={}
    for task in sorted(set(r["task"] for r in merged)):
        rows=[r for r in merged if r["task"]==task]
        tasks[task]={"all":summarize(rows),
                     "moving":summarize([r for r in rows if r["moving"]]),
                     "stationary":summarize([r for r in rows if not r["moving"]])}
    result={"study":"ME076_POSTHOC_REAL_CASE_SCORE_INSPECTION",
            "scientific_tier":"REAL_PREVIOUSLY_EXPOSED_DEV_DIAGNOSTIC_ONLY",
            "new_model_fits":0,"original_data_source_overridden":False,
            "future_labels_used_as_inference_features":False,
            "seeds":records,"all":allresult,
            "motion_rich":summarize(motion),
            "near_stationary":summarize(rest),
            "task_strata":tasks,
            "case_cluster_bootstrap_all":bootstrap_episodes(merged),
            "case_cluster_bootstrap_motion":bootstrap_episodes(motion,seed=BOOTSTRAP_SEED+1),
            "original_raw_case_count":960,
            "independent_real_case_units":40,
            "per_seed_reused_exact_same_cases":True,
            "decision":"The registered >=5 percent physical error gain over persistence FAILED on overall and motion-rich cohorts for every seed. Even a favorable CI would not rescue an already exposed-development generalization claim.",
            "limitations":["posthoc interval, no preregistered scientific promotion", "selection requires source verified future autonomous continuation","old publisher original FIT history already exposed","no previously untouched external task opened","does not prove a specific model-module attribution","no causal action evaluation, no permission or sub-ms runtime qualification"] }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("ME076_ORIGINAL_REAL_POSTHOC_AUDIT="+json.dumps({
      "all":allresult,"motion":result["motion_rich"],
      "bootstrap_ci":result["case_cluster_bootstrap_all"]["ci95_native_minus_persistence"],
      "task_n":len(tasks),"regression_over_persistence":allresult["native_minus_persistence_MAE"]>0}),flush=True)

if __name__=="__main__":
    main()
