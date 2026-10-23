"""ME064 frozen 5-task source-trained model scoring on ONE untouched real task.

Must run only AFTER ten Git-stored checkpoint bytes are clean-checkout verified.
No target retraining, fitted normalization, target calibration, checkpoint
selection, or old exposed TEST. Original publisher task data is real.
"""
from __future__ import annotations
import json, math, argparse
from pathlib import Path
import numpy as np
import torch

from scripts.me064_mixed_real_fit import task_registration,get_meta,load_members,SEEDS
from scripts.me063_prefit_original_fit import model_for, CONFIG
from scripts.me060_train_trajectory import HISTORY,HORIZON,digest
from scripts.me057_train_fused_anchor_robot import normalize,sha_file,write_json

ROOT=Path("research/mathematical-engineering/me064")
REPLAY=Path("research/replay-bundles/ME064")


def checkpoint_inventory():
    report=json.loads((REPLAY/"TRAINING_RESULT.json").read_text())
    verified=json.loads((REPLAY/"CLEAN_CHECKOUT_MODEL_REPLAY.json").read_text())
    if verified["model_replay"]!="PASS" or verified["checkpoints_restored"]!=10:
        raise RuntimeError("required independent original Git checkpoint restoration absent")
    items=report["checkpoints"]
    keys={(r["arm"],r["seed"]) for r in items}
    expected={(a,s) for a in CONFIG for s in SEEDS}
    if len(items)!=10 or keys!=expected:
        raise RuntimeError("not all original paired-seed trained checkpoints present")
    for row in items:
        if sha_file(Path(row["path"]))!=row["checkpoint_sha256"]:
            raise RuntimeError("frozen real checkpoint bytes have changed")
    return items


def load_model(row):
    data=torch.load(row["path"],weights_only=False,map_location="cpu")
    if not data["training_complete"] or data["update_count"]!=600 or (
            data["scoring_status"]!="NO_CAL_OR_HELDOUT_TRAJECTORY_OPENED"):
        raise RuntimeError("checkpoint training/selection freeze violated")
    identified={k:np.asarray(v,np.float32) for k,v in data["identified"].items()}
    model=model_for(row["arm"],identified)
    model.load_state_dict(data["model"],strict=True)
    model.eval()
    return model,identified


def evaluate(model, source, identified):
    scale=torch.from_numpy(identified["state_scale"])
    outcome={}
    model.eval()
    with torch.no_grad():
        for ep,row in sorted(source.items()):
            candidates=[t for t in range(HISTORY,row["n"]-HORIZON)
                if not bool(row["human"][t])]
            if not candidates: continue
            t=candidates[len(candidates)//2]
            belief=model.initial(1)
            for j in range(t-HISTORY,t):
                belief=model.observe(belief,row["state"][j:j+1],row["action"][j:j+1],
                     row["state"][j+1:j+2],row["human"][j:j+1,None])
            current=row["state"][t:t+1]
            forecast=model.forecast(belief,current,row["action"][None,t:t+HORIZON])
            actual=row["state"][None,t+1:t+HORIZON+1]
            live=(~row["human"][t:t+HORIZON]).cumprod(0).bool()
            if live.sum()==0: continue
            error=(forecast.mean-actual).abs()[0]
            servo=current
            states=[]
            for action in row["action"][t:t+HORIZON]:
                servo=servo+model.servo_step(servo,action[None])
                states.append(servo[0])
            prior=torch.stack(states)
            outcome[str(ep)]={
                "origin":int(t),"scored_steps":int(live.sum()),
                "MAE":float((error[live]*scale).mean()),
                "persistence_MAE":float(((current-actual).abs()[0,live]*scale).mean()),
                "servo_MAE":float(((prior-actual[0]).abs()[live]*scale).mean()),
                "scaled_errors":(error[live]/forecast.scale[0,live]).flatten().tolist(),
                "scale_mean":float((forecast.scale[0,live]*scale).mean()),
                "regime_weights_mean":forecast.weights[0].mean(0).tolist(),
                "physical_error_by_horizon":(error[live]*scale).mean(-1).tolist(),
            }
    return outcome


def read_calibration(cache,registration,records):
    sources={}
    origin_hashes={}
    for task in registration["training_tasks"]:
        membership=registration["task_evidence"][task]
        original=load_members(cache,task,membership["CAL"],
                              get_meta(cache,task,membership))
        receipt=digest({str(ep):v["sha256"] for (name,ep),v in original.items()})
        origin_hashes[task]=receipt
        sources[task]={ep:row for (name,ep),row in original.items()}
    all_scores={}
    for row in records:
        model,identified=load_model(row)
        named={}
        for task in registration["training_tasks"]:
            source=normalize(sources[task],identified)
            outputs=evaluate(model,source,identified)
            named.update({f"{task}:{e}":v for e,v in outputs.items()})
        all_scores[f"{row['arm']}-s{row['seed']}"]=named
    write_json(REPLAY/"ORIGINAL_SOURCE_CAL_SCORES.json",{
        "source":"5 previously exposed training task source-CAL episodes",
        "calibration_is_not_independent":True,
        "fit_updated":False,
        "original_cal_source_hashes":origin_hashes,
        "scores":all_scores
    })
    return all_scores


def read_unseen(cache,registration,records):
    task=registration["holdout_task"]
    receipt=registration["task_evidence"][task]
    raw=load_members(cache,task,receipt["FROZEN_HOLDOUT"],
                     get_meta(cache,task,receipt))
    provenance=digest({str(ep):row["sha256"] for (name,ep),row in raw.items()})
    original={ep:row for (name,ep),row in raw.items()}
    per_model={}
    for row in records:
        model,identified=load_model(row)
        per_model[f"{row['arm']}-s{row['seed']}"]=evaluate(
             model,normalize(original,identified),identified)
    write_json(REPLAY/"ORIGINAL_UNTOUCHED_TASK_SCORES.json",{
        "study":"ME064_GENUINELY_UNSEEN_SIXTH_TASK",
        "task":task,"original_episode_count":len(original),
        "per_episode_parquet_sha256":{str(ep):row["sha256"] for ep,row in original.items()},
        "holdout_original_source_sha256":provenance,
        "no_target_training":True,"no_target_normalization":True,
        "no_target_calibration":True,"original_source_TEST_opened":False,
        "scores":per_model,
    })
    return per_model,provenance


def summarize(cal, target, task):
    ids=sorted(next(iter(target.values())),key=int)
    errors={arch:np.asarray([
        [target[f"{arch}-s{seed}"][e]["MAE"] for e in ids] for seed in SEEDS])
            for arch in CONFIG}
    per_episode={arch:rows.mean(axis=0) for arch,rows in errors.items()}
    means={arch:float(vals.mean()) for arch,vals in per_episode.items()}
    base=next(iter(target.values()))
    for name in ("persistence","servo"):
        means[name]=float(np.mean([base[e][f"{name}_MAE"] for e in ids]))
    base_values={
        "gru_world":per_episode["gru_world"],
        "persistence":np.asarray([base[e]["persistence_MAE"] for e in ids]),
        "servo":np.asarray([base[e]["servo_MAE"] for e in ids])
    }
    blocks=sorted({int(ep)//4 for ep in ids})
    if len(blocks)<10:raise RuntimeError("too few real target recording blocks for honest CI")
    rng=np.random.default_rng(64064)
    intervals={}
    for name,reference in base_values.items():
        differences=per_episode["native_tac"]-reference
        aggregates=np.asarray([
            differences[np.asarray([int(ep)//4==b for ep in ids])].mean()
            for b in blocks])
        sample=rng.integers(len(blocks),size=(20000,len(blocks)))
        intervals[name]=np.quantile(aggregates[sample].mean(axis=1),[.025,.975]).tolist()
    calibrated={}
    routing={}
    for arch in CONFIG:
        coverage=[]
        widths=[]
        seeds=[]
        routes=[]
        for seed in SEEDS:
            key=f"{arch}-s{seed}"
            calibration=list(cal[key].values())
            if len(calibration)<20: raise RuntimeError("source CAL episode count insufficient")
            eps=np.sort([max(x["scaled_errors"]) for x in calibration])
            k=min(len(eps),math.ceil((len(eps)+1)*.90))
            quantile=float(eps[k-1])
            observations=[target[key][e] for e in ids]
            covered=float(np.mean([max(x["scaled_errors"])<=quantile for x in observations]))
            width=float(np.mean([2*quantile*x["scale_mean"] for x in observations]))
            weights=np.mean(np.asarray([x["regime_weights_mean"] for x in observations]),axis=0).tolist()
            routes.append(weights)
            coverage.append(covered); widths.append(width)
            seeds.append({"seed":seed,"CAL_episodes":len(calibration),
                          "whole_episode_coverage":covered,"mean_physical_width":width,
                          "quantile":quantile,"router_weights_persistence_servo_neural":weights})
        calibrated[arch]={
            "nominal":.90,"mean_coverage":float(np.mean(coverage)),
            "mean_physical_interval_width":float(np.mean(widths)),
            "per_seed":seeds,"exchangeability_certified":False,
            "transfer_calibration_qualified":False}
        routing[arch]={"mean_weights_persistence_servo_neural":np.mean(routes,axis=0).tolist()}
    seed_wins=int(sum(a<b for a,b in zip(
        errors["native_tac"].mean(axis=1),errors["gru_world"].mean(axis=1))))
    gate=bool(all(
        means["native_tac"]<=.98*means[name] and intervals[name][1]<0
        for name in base_values))
    return {
        "task":task,"episode_count":len(ids),"recording_blocks":len(blocks),
        "physical_MAE":means,
        "TAC_paired_seed_wins":seed_wins,
        "TAC_minus_each_comparator_block_bootstrap_95_intervals":intervals,
        "registered_superiority_gate_pass":gate,
        "source_only_calibration":calibrated,
        "learned_router_weights":routing,
        "new_real_task_not_independent_robot_or_operator_site":True,
        "zero_target_retraining":True,
        "memory_necessity_proven":False,
        "actual_action_counterfactual_policy_value_proven":False,
    }


def run(cache):
    torch.set_num_threads(1)
    metadata=task_registration()
    original=checkpoint_inventory()
    cal=read_calibration(cache,metadata,original)
    target,source_sha=read_unseen(cache,metadata,original)
    summary=summarize(cal,target,metadata["holdout_task"])
    document={
        "study":"ME064_MULTI_TASK_REAL_ROBOT_LEAVE_ONE_TASK_OUT",
        "training_tasks":metadata["training_tasks"],
        "training_fit_episode_count":metadata["training_episode_count"],
        "untouched_evaluation_task":metadata["holdout_task"],
        "untouched_target_sha256":source_sha,
        "trained_original_git_checkpoint_count":10,
        "model_checkpoints_independently_reloaded":True,
        "source_CAL_is_historically_exposed":True,
        "original_source_TEST_opened":False,
        "model_retrained_using_target":False,
        "results":summary,
        "generalization_independent_site_operator_proven":False,
        "score_replay":"PENDING_SEPARATE_CLEAN_CHECKOUT",
    }
    write_json(REPLAY/"REAL_WORLD_EVALUATION_SUMMARY.json",document)
    print("ME064_REAL_MULTITASK_ZERO_SHOT_SCORE="+json.dumps({
        "task":summary["task"],"MAE":summary["physical_MAE"],
        "TAC_seed_wins":summary["TAC_paired_seed_wins"],
        "block_CI":summary["TAC_minus_each_comparator_block_bootstrap_95_intervals"],
        "gate":summary["registered_superiority_gate_pass"],
        "calibration":{k:{"coverage":v["mean_coverage"],"width":v["mean_physical_interval_width"]}
              for k,v in summary["source_only_calibration"].items()},
    },sort_keys=True),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--cache",type=Path,required=True)
    a=cli.parse_args()
    run(a.cache)
