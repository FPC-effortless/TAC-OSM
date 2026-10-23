"""ESWAM004 original paired real-episode audit: no fitting, no new source data.

Use only complete original Git-owned ESWAM003/004 real original scores,
same original seeds, exact same task/episode/frame identities. Episode
clusters, not individual frame-targets or five seed replicas, are resampled.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random

SEEDS=(76001,76002,76003,76004,76005)
PRIOR=Path("research/replay-bundles/ESWAM003")
CURRENT=Path("research/replay-bundles/ESWAM004")
OUT=Path("research/mathematical-engineering/eswam004/PAIRED_EPISODE_AUDIT.json")
FIELDS=("physical_mae_model", "next_recorded_command_imitation_l1", "movement_brier")


def load_paired():
    cases=defaultdict(list)
    for seed in SEEDS:
        previous=json.loads((PRIOR/f"seed-{seed}"/"ORIGINAL_REAL_PREDICTIONS.json").read_text())
        current=json.loads((CURRENT/f"seed-{seed}"/"ORIGINAL_REAL_PREDICTIONS.json").read_text())
        if len(previous)!=len(current) or len(current)!=960:
            raise RuntimeError("original real-score case count changed")
        for earlier,later in zip(previous,current):
            key=("task","episode_index","frame_index")
            if tuple(earlier[k] for k in key)!=tuple(later[k] for k in key):
                raise RuntimeError("different original source case compared")
            if earlier["motion_rich_threshold_0_025"]!=later["motion_rich_threshold_0_025"]:
                raise RuntimeError("future original actual movement label mismatch")
            for phys in ("physical_mae_persistence","physical_mae_fit_servo_lag1"):
                if abs(earlier[phys]-later[phys])>1e-6:
                    raise RuntimeError("source/factual physical prediction normalization changed")
            case=(earlier["task"],earlier["episode_index"])
            cases[case].append({
                "seed":seed,
                "moving":bool(earlier["motion_rich_threshold_0_025"]),
                **{field:later[field]-earlier[field] for field in FIELDS},
            })
    if len(cases)!=40 or set(len(v) for v in cases.values())!={120}:
        raise RuntimeError("not exactly 40 original 24-case episodes *5 seed replicas")
    return cases


def subset_mean(cases,selected,field,subcohort):
    values=[]
    for name in selected:
        for r in cases[name]:
            if subcohort=="all" or (subcohort=="moving")==r["moving"]:
                values.append(r[field])
    if not values:
        return None
    return sum(values)/len(values)


def audit():
    cases=load_paired()
    episodes=sorted(cases)
    tasks=defaultdict(list)
    for name in episodes:
        tasks[name[0]].append(name)
    if len(tasks)!=5 or set(len(x) for x in tasks.values())!={8}:
        raise RuntimeError("expected exactly 8 DEV source episodes per original task")
    rng=random.Random(760404)
    overall={}
    for field in FIELDS:
        overall[field]={}
        for group in ("all","moving","stationary"):
            score=subset_mean(cases,episodes,field,group)
            distribution=[]
            skipped=0
            for it in range(5000):
                selected=[rng.choice(task_cases) for task_cases in tasks.values()
                          for _ in task_cases]
                diff=subset_mean(cases,selected,field,group)
                if diff is None:
                    skipped+=1
                else:
                    distribution.append(diff)
            distribution.sort()
            if not distribution:
                raise RuntimeError("no valid original real bootstrap samples")
            overall[field][group]={
                "new_minus_prior_original_mean":score,
                "paired_task_stratified_episode_bootstrap_95":(
                    distribution[int(0.025*len(distribution))],
                    distribution[min(len(distribution)-1,int(0.975*len(distribution)))]),
                "sampled_original_episodes":40,
                "source_task_strata":5,
                "resamples":5000,
                "empty_case_draws":skipped,
                "lower_difference_means_ESWAM004_better":True,
            }
    return {
        "study":"ESWAM004_ORIGINAL_SOURCE_EPISODE_PAIRED_DEVELOPMENT_AUDIT",
        "original_case_source":"same previously exposed BodenAI real 40 episode DEV",
        "original_scoring_models_each":5,
        "source_distinct_episodes":len(episodes),
        "real_query_cases_per_model":960,
        "source_train_fits_per_original_model":640,
        "original_weights_unchanged":True,
        "optimizer_updates_during_audit":0,
        "full_training_compute_parity_established":False,
        "independent_source_test_established":False,
        "paired_differences":overall,
    }


if __name__=="__main__":
    result=audit()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("ESWAM004_ORIGINAL_PAIRED_EPISODE_AUDIT="+json.dumps(
        result["paired_differences"]["physical_mae_model"],sort_keys=True),flush=True)
