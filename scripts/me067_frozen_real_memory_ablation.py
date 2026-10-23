"""ME067: frozen-checkpoint native memory lesions on ALREADY EXPOSED original robot data.

Exploratory mechanistic diagnosis ONLY, not a new independent model evaluation.
No fitting, gradients, checkpoint selection, new target, extra labels, or policy.
Original ME066 source CAL / plug charger episodes were revealed already.
A lesion of a trained latent state is out-of-distribution and cannot by itself
prove either memory necessity or the value of a new architecture.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch

from scripts.me060_train_trajectory import HISTORY,HORIZON
from scripts.me064_mixed_real_fit import load_members,get_meta
from scripts.me057_train_fused_anchor_robot import normalize
from scripts.me066_score_frozen import frozen_model

REPLAY = Path("research/replay-bundles/ME066")
OUT = Path("research/replay-bundles/ME067")
ORIGINAL_SOURCE = Path("research/mathematical-engineering/me064/TASK_LEVEL_SPLIT.json")
SEEDS = (65001,65002,65003,65004,65005)
NATIVE_LESIONS = (
    "full", "association_zero", "slow_zero", "fast_zero",
    "fast_slow_zero", "kinematics_zero", "all_hidden_zero")
GRU_LESIONS = ("full","fast_zero","kinematics_zero","all_hidden_zero")

def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def original_frozen_inventory():
    task=json.loads((Path("research/mathematical-engineering/me066")/"TASK_LEVEL_SPLIT.json").read_text())
    manifest=json.loads((Path("research/mathematical-engineering/me066")/"FAIR_COMPARISON.json").read_text())
    training=json.loads((REPLAY/"TRAINING_RESULT.json").read_text())
    audited=json.loads((REPLAY/"CLEAN_CHECKOUT_MODEL_REPLAY.json").read_text())
    if (audited.get("model_replay")!="PASS" or
            audited.get("original_git_trained_model_bytes_hashed_and_restored")!=20 or
            training.get("real_task_transfer_result")!="NOT_RUN" or
            training.get("target_trajectory_files_opened")!=0):
        raise RuntimeError("cannot reopen immutable original ME066 model evidence")
    rows=training["original_checkpoints"]
    if len(rows)!=20 or len({(r["arm"],r["seed"]) for r in rows})!=20:
        raise RuntimeError("not all original 20 trained model identities preserved")
    for row in rows:
        if digest_file(row["path"])!=row["checkpoint_sha256"]:
            raise RuntimeError("original trained model binary SHA mismatch")
    if task["heldout_task_name"]!=manifest["prospective_untouched_target"]["task"]:
        raise RuntimeError("frozen ME066 metadata target altered")
    return task,manifest,rows

def require_original():
    if not (REPLAY/"EVALUATION_REPLAY_RECEIPT.json").is_file():
        raise RuntimeError("independent ME066 source/target score replay absent")
    replay=json.loads((REPLAY/"EVALUATION_REPLAY_RECEIPT.json").read_text())
    if replay["numeric_score_replay"]!="PASS_WITH_FLOAT_TOLERANCE":
        raise RuntimeError("do not diagnose nonreproducible underlying real score")
    scored=json.loads((REPLAY/"ORIGINAL_UNTOUCHED_TASK_SCORES.json").read_text())
    cal=json.loads((REPLAY/"ORIGINAL_SOURCE_CAL_SCORES.json").read_text())
    if scored["task"]!="plug_charger_into_power_strip" or scored["original_episode_count"]!=80:
        raise RuntimeError("reusing anything but already opened ME066 real charger task")
    if scored["target_model_updates"]!=0 or scored["target_calibration"]:
        raise RuntimeError("previous scoring provenance is not clean")
    return scored,cal

def apply_lesion(belief,variant):
    if variant=="full":
        return belief
    if variant not in NATIVE_LESIONS:
        raise RuntimeError("unregistered memory intervention: "+str(variant))
    zeros=lambda name:torch.zeros_like(getattr(belief,name))
    replacements={}
    if variant in ("association_zero","all_hidden_zero"):
        replacements["memory"]=zeros("memory")
    if variant in ("slow_zero","fast_slow_zero","all_hidden_zero"):
        replacements["slow"]=zeros("slow")
    if variant in ("fast_zero","fast_slow_zero","all_hidden_zero"):
        replacements["fast"]=zeros("fast")
    if variant in ("kinematics_zero","all_hidden_zero"):
        replacements["velocity"]=zeros("velocity")
        replacements["innovation"]=zeros("innovation")
    return belief._replace(**replacements)

def physical_errors(model, identified, episodes, lesions):
    result={}
    model.eval()
    scale=torch.from_numpy(identified["state_scale"])
    with torch.no_grad():
        for ep,row in sorted(episodes.items(),key=lambda item:str(item[0])):
            candidates=[t for t in range(HISTORY,row["n"]-HORIZON)
                        if not bool(row["human"][t])]
            if not candidates:
                continue
            t=candidates[len(candidates)//2]
            belief=model.initial(1)
            for j in range(t-HISTORY,t):
                belief=model.observe(
                    belief,row["state"][j:j+1],row["action"][j:j+1],
                    row["state"][j+1:j+2],row["human"][j:j+1,None])
            current=row["state"][t:t+1]
            action=row["action"][None,t:t+HORIZON]
            actual=row["state"][None,t+1:t+HORIZON+1]
            alive=(~row["human"][t:t+HORIZON]).cumprod(0).bool()
            if not bool(alive.any()):
                continue
            scores={}
            for lesion in lesions:
                output=model.forecast(apply_lesion(belief,lesion),current,action)
                err=((output.mean-actual).abs()[0,alive]*scale).mean()
                scores[lesion]=float(err)
            result[str(ep)]={
                "origin":int(t),
                "scored_steps":int(alive.sum()),
                "physical_mae":scores,
            }
    return result

def summarize(scores,lesions,site):
    keys=sorted(scores[f"native_tac-s{SEEDS[0]}"],key=lambda s: int(s.split(":")[-1]))
    if site=="previously_opened_target" and len(keys)!=80:
        raise RuntimeError("all 80 original exposed real robot task episodes required")
    if site=="exposed_source_CAL" and len(keys)!=40:
        raise RuntimeError("all 40 exposed original source CAL episodes required")
    def mean(vals):
        return float(np.mean(vals))
    table={}
    for arm in ("native_tac","gru_world"):
        names=lesions[arm]
        for lesion in names:
            by_seed=np.asarray([[scores[f"{arm}-s{seed}"][ep]["physical_mae"][lesion]
                                for ep in keys] for seed in SEEDS])
            table[f"{arm}.{lesion}"]={
                "mean_physical_MAE":mean(by_seed),
                "seed_mean_MAE":by_seed.mean(axis=1).tolist(),
            }
        baseline=np.asarray([[scores[f"{arm}-s{seed}"][ep]["physical_mae"]["full"]
                            for ep in keys] for seed in SEEDS])
        for lesion in names:
            if lesion=="full":
                continue
            compared=np.asarray([[scores[f"{arm}-s{seed}"][ep]["physical_mae"][lesion]
                                 for ep in keys] for seed in SEEDS])
            differences=compared-baseline
            table[f"{arm}.{lesion}"]["lesion_minus_full_seed_delta"]=differences.mean(1).tolist()
            table[f"{arm}.{lesion}"]["lesion_worse_seed_count"]=int((differences.mean(1)>0).sum())
    bootstrap={}
    if site=="previously_opened_target":
        # Preserved original 4-episode recording blocks, not 80 iid samples.
        blocks=sorted({int(ep)//4 for ep in keys})
        if len(blocks)!=20:
            raise RuntimeError("must preserve original 20 real robot recording blocks")
        rng=np.random.default_rng(67067)
        draws=rng.integers(len(blocks),size=(20000,len(blocks)))
        for lesion in NATIVE_LESIONS[1:]:
            episode_delta=np.asarray([
                np.mean([scores[f"native_tac-s{seed}"][ep]["physical_mae"][lesion]-
                         scores[f"native_tac-s{seed}"][ep]["physical_mae"]["full"]
                         for seed in SEEDS])
                for ep in keys])
            block_delta=np.asarray([episode_delta[np.asarray([int(ep)//4==b for ep in keys])].mean()
                                    for b in blocks])
            bootstrap[lesion]={
                "mean_lesion_minus_full":mean(episode_delta),
                "20_recording_block_bootstrap_95":np.quantile(
                    block_delta[draws].mean(axis=1),[.025,.975]).tolist(),
                "important_limitation":"CI resamples 20 original blocks averaged over five FIXED fitted seeds; not a CI across independent training seeds",
            }
    return {"source_group":site,"episode_count":len(keys),
            "preselected_observation_time":"middle eligible episode origin; same original ME066 methodology",
            "descriptive_scores":table,
            "native_lesion_cluster_differences":bootstrap,
            "original_test_previously_exposed":site=="previously_opened_target",
            "lesions_are_out_of_training_distribution":True,
            "scientific_superiority_or_memory_necessity_qualified":False}

def main(cache):
    torch.set_num_threads(1)
    task,manifest,rows=original_frozen_inventory()
    # ME066 verifies no repeat score-file write and cannot run here because
    # target is already exposed. Reuse only its original reader + checkpoints.
    original_target,original_cal=require_original()
    if task["heldout_task_name"]!="plug_charger_into_power_strip":
        raise RuntimeError("new untouched target is NOT an eligible diagnosis source")
    target_receipt={
        "info_sha256":task["heldout_meta_info_sha256"],
        "episodes_metadata_sha256":task["heldout_meta_episode_sha256"],
    }
    target=load_members(cache,task["heldout_task_name"],
                        task["heldout_original_episode_members"],
                        get_meta(cache,task["heldout_task_name"],target_receipt))
    actual_target={str(ep):row for (name,ep),row in target.items()}
    for ep,record in actual_target.items():
        if record["sha256"]!=original_target["heldout_publisher_parquet_sha256"][ep]:
            raise RuntimeError("real original exposed robot episode bytes altered")
    source_receipts=json.loads(ORIGINAL_SOURCE.read_text())["task_evidence"]
    source={}
    for name in task["training_task_names"]:
        receipt=source_receipts[name]
        orig=load_members(cache,name,receipt["CAL"],get_meta(cache,name,receipt))
        for (taskname,ep),row in orig.items():
            source[f"{taskname}:{ep}"]=row
    if len(source)!=40:
        raise RuntimeError("original source CAL episode count changed")
    output={"exposed_source_CAL":{},"previously_opened_target":{}}
    for row in rows:
        model,identified=frozen_model(row,task)
        key=f"{row['arm']}-s{row['seed']}"
        lesions=NATIVE_LESIONS if row["arm"]=="native_tac" else GRU_LESIONS
        for title,raw in (("exposed_source_CAL",source),
                          ("previously_opened_target",actual_target)):
            results=physical_errors(model,identified,normalize(raw,identified),lesions)
            # Confirm same original observation origins/uncorrupted model score.
            original=(original_cal if title=="exposed_source_CAL" else original_target)
            if len(results)!=len(original["scores"][key]):
                raise RuntimeError("original scored real episode set changed")
            for ep,entry in results.items():
                ref=original["scores"][key][ep]
                if entry["origin"]!=ref["origin"] or entry["scored_steps"]!=ref["scored_steps"]:
                    raise RuntimeError("original physical scoring origins changed")
                if abs(entry["physical_mae"]["full"]-ref["MAE"])>1e-4:
                    raise RuntimeError("full-state frozen original model score no longer replays")
            output[title][key]=results
        print("ME067_EXPOSED_SCORE_ONLY_FROZEN_CHECKPOINT="+json.dumps({
            "arm":row["arm"],"seed":row["seed"],"zero_model_updates":True}),flush=True)
    final={
        "study":"ME067_POSTHOC_FROZEN_REAL_ROBOT_MEMORY_ABLATION",
        "status":"EXPLORATORY_POSTHOC_NOT_INDEPENDENT",
        "original_trained_model_count":len(rows),
        "original_ME066_score_replay_sha256":digest_file(REPLAY/"EVALUATION_REPLAY_RECEIPT.json"),
        "original_ME066_target_sha256":digest_file(REPLAY/"ORIGINAL_UNTOUCHED_TASK_SCORES.json"),
        "source_CAL_already_exposed":True,
        "new_independent_task_opened":False,
        "remaining_reserved_tissue_box_task_unopened":True,
        "model_training_updates":0,
        "checkpoint_or_optimizer_mutated":False,
        "exact_model_weights_original_sha256_verified":True,
        "native_lesion_names":list(NATIVE_LESIONS),
        "gru_control_lesion_names":list(GRU_LESIONS),
        "results":{k:summarize(v,
                  {"native_tac":NATIVE_LESIONS,"gru_world":GRU_LESIONS},k)
                   for k,v in output.items()},
        "limits":[
            "Latent-state interventions are out-of-distribution and diagnose sensitivity, not strict causal memory necessity.",
            "Exposed outcomes cannot qualify independent model generalization or be used for tuning thresholds.",
            "No true new action intervention, no autonomous policy cost, no publisher-independent robot/site.",
            "Block-resampled intervals for fixed averaged trained seeds are not independent model-fit confidence.",
        ],
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"ME067_ABLATION_SUMMARY.json").write_text(json.dumps(final,sort_keys=True,indent=2)+"\n")
    print("ME067_EXPLORATORY_RESULTS="+json.dumps({
        "original_replayed_models":len(rows),
        "original_target_native_full_MAE":final["results"]["previously_opened_target"][
            "descriptive_scores"]["native_tac.full"]["mean_physical_MAE"],
        "mean_native_lesion_delta":{
            name:final["results"]["previously_opened_target"]["native_lesion_cluster_differences"][name][
                "mean_lesion_minus_full"] for name in NATIVE_LESIONS[1:]}},sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--cache",required=True,type=Path)
    args=p.parse_args()
    main(args.cache)
