"""ME066 once-frozen scores for four real-trained models on ONE unseen robot task.

NEVER fit, tune, normalize, calibrate, select, or change weights on new task.
Must verify 20 original Git checkpoint bytes and CLEAN_CHECKOUT_MODEL_REPLAY
before reading any original source CAL or new publisher target trajectory.
ME064 spray-nozzle old exposed test is prohibited. Real factual action prefixes
are not evidence for unexecuted counterfactual outcomes or policy utility.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch

from scripts.me064_mixed_real_fit import load_members,get_meta
from scripts.me064_score_frozen import evaluate
from scripts.me065_prefit_original_fit import CONFIG,SEEDS,model_for
from scripts.me057_train_fused_anchor_robot import normalize
from scripts.me060_train_trajectory import digest

ROOT=Path("research/mathematical-engineering/me066")
OUT=Path("research/replay-bundles/ME066")
ME064_TASKS=Path("research/mathematical-engineering/me064/TASK_LEVEL_SPLIT.json")
ARMS=tuple(CONFIG)

def write_json(path, content):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(content,indent=2,sort_keys=True)+"\n")

def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()

def verify_inventory():
    task=json.loads((ROOT/"TASK_LEVEL_SPLIT.json").read_text())
    manifest=json.loads((ROOT/"FAIR_COMPARISON.json").read_text())
    final=json.loads((OUT/"TRAINING_RESULT.json").read_text())
    original=json.loads((OUT/"CLEAN_CHECKOUT_MODEL_REPLAY.json").read_text())
    if (not task["heldout_selected_from_metadata_only"] or
            task["heldout_original_trajectory_files_opened"]!=0 or
            task["training_optimizer_updates"]!=0 or
            task["heldout_task_name"]=="install_spray_nozzle_onto_bottle" or
            task["heldout_task_name"]!=manifest["prospective_untouched_target"]["task"]):
        raise RuntimeError("new untouched genuine robot task registration violated")
    if (original["model_replay"]!="PASS" or
            original["original_git_trained_model_bytes_hashed_and_restored"]!=20 or
            original["new_heldout_trajectories_opened"]!=0):
        raise RuntimeError("independent original Git model restore not yet proven")
    if (final["real_task_transfer_result"]!="NOT_RUN" or
            final["target_trajectory_files_opened"]!=0 or
            final["source_cal_trajectory_files_opened"]!=0 or
            final["model_git_replay"]!="PENDING_SEPARATE_GIT_CHECKOUT"):
        raise RuntimeError("original FIT models or heldout restriction invalid")
    records=final["original_checkpoints"]
    if len(records)!=20 or {(x["arm"],x["seed"]) for x in records}!={(a,s) for a in ARMS for s in SEEDS}:
        raise RuntimeError("not all four-arm original FIT model bytes exist")
    for row in records:
        if sha(row["path"])!=row["checkpoint_sha256"]:
            raise RuntimeError("original model checkpoint SHA changed before target read")
    if (OUT/"ORIGINAL_UNTOUCHED_TASK_SCORES.json").exists():
        raise RuntimeError("original unseen robot task already opened and scored; refuse redo")
    return task,manifest,records

def frozen_model(row, task):
    raw=torch.load(row["path"],weights_only=False,map_location="cpu")
    if (not raw["training_complete"] or raw["training_updates"]!=600 or
            raw["scoring_status"]!="NO_SOURCE_CAL_OR_FUTURE_TARGET_OPENED" or
            raw["heldout_task_metadata_hash"]!=task["heldout_member_digest"]):
        raise RuntimeError("original checkpoint not frozen prior to heldout read")
    identified={k:np.asarray(v,np.float32) for k,v in raw["identified"].items()}
    model=model_for(row["arm"],identified)
    model.load_state_dict(raw["model"],strict=True)
    model.eval()
    return model,identified

def read_original_source_cal(cache,records,task):
    source=json.loads(ME064_TASKS.read_text())
    if source["holdout_task"]=="plug_charger_into_power_strip":
        raise RuntimeError("new CAL loader attempted revealed future task")
    cal_by_task={}
    provenance={}
    for name in task["training_task_names"]:
        member=source["task_evidence"][name]
        fetched=load_members(cache,name,member["CAL"],get_meta(cache,name,member))
        cal_by_task[name]={ep:v for (taskname,ep),v in fetched.items()}
        provenance[name]=digest({str(ep):v["sha256"] for ep,v in cal_by_task[name].items()})
    all_scores={}
    for row in records:
        model,identified=frozen_model(row,task)
        scores={}
        for name in task["training_task_names"]:
            sample=normalize(cal_by_task[name],identified)
            scores.update({f"{name}:{ep}":val for ep,val in evaluate(model,sample,identified).items()})
        all_scores[f"{row['arm']}-s{row['seed']}"]=scores
    write_json(OUT/"ORIGINAL_SOURCE_CAL_SCORES.json",{
        "study":task["study"],
        "original_source_CAL_episode_counts":{name:len(rows) for name,rows in cal_by_task.items()},
        "CAL_origin_source_sha256":provenance,
        "CAL_not_a_new_independent_task":True,
        "no_training_or_model_selection":True,
        "scores":all_scores,
    })
    return all_scores

def read_once_unseen_original(cache,records,task):
    name=task["heldout_task_name"]
    receipt={
        "info_sha256":task["heldout_meta_info_sha256"],
        "episodes_metadata_sha256":task["heldout_meta_episode_sha256"]
    }
    original=load_members(cache,name,task["heldout_original_episode_members"],
                          get_meta(cache,name,receipt))
    hash_original=digest({str(ep):row["sha256"] for (t,ep),row in original.items()})
    by_ep={ep:row for (t,ep),row in original.items()}
    if len(by_ep)!=80:raise RuntimeError("must read exactly frozen 80 real untouched episodes")
    scored={}
    for row in records:
        model,identified=frozen_model(row,task)
        scored[f"{row['arm']}-s{row['seed']}"]=evaluate(
            model,normalize(by_ep,identified),identified)
    write_json(OUT/"ORIGINAL_UNTOUCHED_TASK_SCORES.json",{
        "study":task["study"],"task":name,
        "original_episode_count":len(by_ep),
        "heldout_publisher_parquet_sha256":{str(ep):row["sha256"] for ep,row in by_ep.items()},
        "original_new_task_source_digest":hash_original,
        "target_model_updates":0,"target_normalization":False,
        "target_calibration":False,"old_exposed_me064_test_opened":False,
        "scores":scored,
    })
    return scored,hash_original

def describe(cal,target,task):
    ids=sorted(next(iter(target.values())),key=int)
    if len(ids)!=80:raise RuntimeError("original 80 task episodes absent")
    arrays={}
    for arm in ARMS:
        arrays[arm]=np.asarray([[target[f"{arm}-s{seed}"][ep]["MAE"] for ep in ids]
                               for seed in SEEDS],dtype=np.float64)
    episode_mean={arm:matrix.mean(0) for arm,matrix in arrays.items()}
    means={arm:float(np.mean(data)) for arm,data in episode_mean.items()}
    base=target[f"{ARMS[0]}-s{SEEDS[0]}"]
    baselines={n:np.asarray([base[ep][f"{n}_MAE"] for ep in ids],dtype=np.float64)
               for n in ("persistence","servo")}
    for n,v in baselines.items():means[n]=float(np.mean(v))
    # Frozen original recording block IDs, not nominally 80 independent units.
    blocks=sorted({int(ep)//4 for ep in ids})
    if len(blocks)!=20:raise RuntimeError("exact original 20 physical recording blocks needed")
    rng=np.random.default_rng(66066)
    comparison={}
    for name in (*ARMS[1:],"persistence","servo"):
        reference=episode_mean[name] if name in episode_mean else baselines[name]
        differences=episode_mean["native_tac"]-reference
        block_means=np.asarray([differences[np.asarray([int(ep)//4==b for ep in ids])].mean()
                                for b in blocks],dtype=np.float64)
        picks=rng.integers(len(blocks),size=(20000,len(blocks)))
        confidence=np.quantile(block_means[picks].mean(axis=1),[.025,.975]).tolist()
        comparison[name]={
            "tac_minus_reference_block_bootstrap_95":confidence,
            "tac_minus_reference_mean":float(differences.mean()),
            "tac_seed_wins":(int(sum(a<b for a,b in zip(
                arrays["native_tac"].mean(1),arrays[name].mean(1))))
                if name in arrays else None),
            "tac_minus_each_seed":(
                (arrays["native_tac"].mean(1)-arrays[name].mean(1)).tolist()
                if name in arrays else None),
        }
    coverage={}
    routing={}
    for arm in ARMS:
        entries=[]
        for seed in SEEDS:
            key=f"{arm}-s{seed}"
            source=list(cal[key].values())
            if len(source)<40:raise RuntimeError("original 40 source CAL episodes required")
            thresholds=np.sort([max(x["scaled_errors"]) for x in source])
            rank=min(len(thresholds),math.ceil((len(thresholds)+1)*.90))
            quantile=float(thresholds[rank-1])
            episodes=[target[key][ep] for ep in ids]
            count=float(np.mean([max(x["scaled_errors"])<=quantile for x in episodes]))
            width=float(np.mean([2*quantile*x["scale_mean"] for x in episodes]))
            weights=np.mean([x["regime_weights_mean"] for x in episodes],axis=0).tolist()
            entries.append({"seed":seed,"source_CAL_episode_count":len(source),
                            "whole_episode_coverage":count,
                            "mean_physical_interval_width":width,
                            "source_CAL_quantile":quantile,
                            "weights_persistence_servo_neural":weights})
        coverage[arm]={
            "nominal":.90,"mean_whole_episode_coverage":float(np.mean([v["whole_episode_coverage"] for v in entries])),
            "mean_physical_interval_width":float(np.mean([v["mean_physical_interval_width"] for v in entries])),
            "width_to_mean_MAE_ratio":float(np.mean([v["mean_physical_interval_width"] for v in entries])/max(means[arm],1e-9)),
            "exchangeability_qualified":False,"per_seed":entries}
        routing[arm]=np.mean([e["weights_persistence_servo_neural"] for e in entries],axis=0).tolist()
    # Robust task value must beat all learned + practical controls with >=2% margin
    # plus block uncertainty and at least 4/5 paired independent seed wins vs neural controls.
    gate=all(
        means["native_tac"]<=.98*means[name] and
        comparison[name]["tac_minus_reference_block_bootstrap_95"][1]<0
        for name in (*ARMS[1:],"persistence","servo")
    ) and all(comparison[name]["tac_seed_wins"]>=4 for name in ARMS[1:])
    p=baselines["persistence"]
    motion={
        "me064_retrospective_cutoffs_not_used_for_model_selection":True,
        "near_stationary_persistence_mae_lt_0_025":int(np.sum(p<.025)),
        "limited_motion_0_025_to_0_100":int(np.sum((p>=.025)&(p<.1))),
        "moving_persistence_mae_gte_0_100":int(np.sum(p>=.1)),
        "warning":"This uses future outcome-derived persistence errors for posthoc diagnostics only; never used as input to the trained forecast."
    }
    return {
        "task":task["heldout_task_name"],
        "original_episode_count":len(ids),"original_recording_block_count":len(blocks),
        "physical_MAE":means,"native_tac_against_each_reference":comparison,
        "source_ONLY_calibrated_uncertainty":coverage,
        "mean_router_weights":routing,
        "motion_diagnostics_NOT_realtime":motion,
        "registered_superiority_gate_pass":bool(gate),
        "target_model_updates":0,
        "memory_mechanism_necessary_proven":False,
        "action_counterfactual_policy_value_proven":False,
        "independent_robot_or_site_proven":False,
        "real_safe_autonomous_execution_proven":False,
    }

def run(cache):
    torch.set_num_threads(1)
    task,manifest,records=verify_inventory()
    cal=read_original_source_cal(cache,records,task)
    target,sha_new=read_once_unseen_original(cache,records,task)
    result=describe(cal,target,task)
    write_json(OUT/"REAL_WORLD_EVALUATION_SUMMARY.json",{
        "study":manifest["study_id"],
        "original_FIT_source_sha256":task["source_FIT_hash"],
        "original_new_task_holdout_sha256":sha_new,
        "new_task_registered_before_all_FIT_models":True,
        "trained_original_checkpoints_replayed_from_clean_Git":20,
        "source_CAL_only_calibration":True,
        "heldout_normalization_fit":False,
        "heldout_training_updates":0,
        "old_ME064_test_reused":False,
        "full_score_replay":"PENDING_INDEPENDENT_SOURCE_RECALCULATION",
        "results":result,
    })
    print("ME066_REAL_FOUR_ARM_ZERO_SHOT_RESULT="+json.dumps({
        "task":result["task"],"MAE":result["physical_MAE"],
        "native_vs_control":result["native_tac_against_each_reference"],
        "gate":result["registered_superiority_gate_pass"]},sort_keys=True),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",required=True,type=Path)
    args=parser.parse_args()
    run(args.cache)
