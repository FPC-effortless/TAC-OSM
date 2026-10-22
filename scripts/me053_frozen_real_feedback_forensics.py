#!/usr/bin/env python3
"""ME053: diagnose ME052 REAL observed physical-state prediction failure.

This is NO-TRAINING forensics on the exact 10 original genuine trained models.
Original real DEV episodes only; TEST never opened. Same original publisher
revision, original labels and authenticated frozen training checkpoints.
Document no-operator-feedback and physical-only feedback diagnostics, using
the same prior real state/command information for every model and mask.
No hypothetical fake rewards, simulated world, unobserved "do" outcomes,
new model superiority claims or new training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.me052_train_real_robot import (
    SEEDS,load_episode,normalize,network,source_index,sha_file,write_json)
from scripts.qualify_real_robot_cohort import REVISION

torch.set_num_threads(1)
SEED_SUBSET=(52001,52003)
MODES=("full_original_feedback","physical_only","no_feedback")
DEV_EPISODES=12

def selected_dev_episodes(members):
    ids=[i for i,_ in members]
    ids.sort(key=lambda i:hashlib.sha256(
        f"ME053:2026-10-22:{REVISION}:DEV:{i}".encode()).hexdigest())
    return sorted(ids[:DEV_EPISODES])

def norm_of(z):
    if isinstance(z,torch.Tensor):
        return float(torch.linalg.vector_norm(z).item())
    return float(sum(float(torch.linalg.vector_norm(t).item()) for t in z))

def forensic_one(model,ep,mode,group,scaling):
    n=ep["n"]
    z=model.initial(1)
    src=ep["state"];acts=ep["action"];human=ep["human"]
    joint_std=torch.as_tensor(scaling["state_std"],dtype=torch.float32)
    buckets={
        "initial_256_to_511":{"mae_sum":0.,"coords":0,"state_norm_sum":0.,"prediction_norm_sum":0.,"observations":0},
        "later_512_to_1023":{"mae_sum":0.,"coords":0,"state_norm_sum":0.,"prediction_norm_sum":0.,"observations":0},
        "long_1024_plus":{"mae_sum":0.,"coords":0,"state_norm_sum":0.,"prediction_norm_sum":0.,"observations":0}
    }
    total=0;correct=0.;persistence=0.;max_state=0.
    with torch.inference_mode():
        for t in range(n-1):
            observed=src[t:t+1]
            cmd=acts[t:t+1]
            authority=human[t:t+1].unsqueeze(-1)
            next_observed=src[t+1:t+2]
            next_human=human[t+1:t+2].unsqueeze(-1)
            z=model.observe(z,observed)
            forecast=model.predict(z,observed,cmd[:,None],authority)
            current_norm=norm_of(z)
            max_state=max(max_state,current_norm)
            if not np.isfinite(current_norm):
                raise ValueError("nonfinite recurrent state from actually observed robot trajectory")
            actual_onset=((~authority)&next_human).to(dtype=next_observed.dtype)
            if t>=255 and (t-255)%64==0:
                if t<512:bucket=buckets["initial_256_to_511"]
                elif t<1024:bucket=buckets["later_512_to_1023"]
                else:bucket=buckets["long_1024_plus"]
                p=forecast.next_joint_state[:,0]
                err=float(torch.abs((p-next_observed)*joint_std).sum().item())
                baseline=float(torch.abs((observed-next_observed)*joint_std).sum().item())
                bucket["mae_sum"]+=err
                bucket["coords"]+=14
                bucket["state_norm_sum"]+=current_norm
                bucket["prediction_norm_sum"]+=float(torch.linalg.vector_norm(p).item())
                bucket["observations"]+=1
                total+=14
                correct+=err
                persistence+=baseline
            is_physical=mode!="no_feedback"
            is_onset=mode=="full_original_feedback"
            next_phys_mask=torch.full_like(authority,is_physical,dtype=torch.bool)
            onset_mask=(~authority) if is_onset else torch.zeros_like(authority)
            z=model.verify_real_feedback(
                z,cmd,authority,next_observed,forecast.next_joint_state[:,0],
                next_phys_mask,actual_onset,forecast.takeover_logits[:,0],
                onset_mask)
    return {"source_original_physical_frames":n,"original_next_state_coordinates":total,
            "physical_absolute_error_sum":correct,
            "last_state_persistence_error_sum":persistence,
            "max_recurrent_state_norm":max_state,"by_original_horizon":buckets}

def summarize(rows):
    agg={}
    for mode in MODES:
        selected=[r for r in rows if r["mode"]==mode]
        denominator=sum(r["original_next_state_coordinates"] for r in selected)
        if denominator==0:raise RuntimeError("no actual measured DEV targets")
        horizons={}
        for key in selected[0]["by_original_horizon"]:
            parts=[r["by_original_horizon"][key] for r in selected]
            ncoords=sum(p["coords"] for p in parts)
            nobs=sum(p["observations"] for p in parts)
            horizons[key]={
                "scored_real_physical_frames":nobs,
                "MAE_original_robot_joint_units":
                  sum(p["mae_sum"] for p in parts)/ncoords if ncoords else None,
                "mean_recurrent_state_norm":
                  sum(p["state_norm_sum"] for p in parts)/nobs if nobs else None,
                "mean_normalized_next_prediction_norm":
                  sum(p["prediction_norm_sum"] for p in parts)/nobs if nobs else None}
        agg[mode]={
            "original_DEV_physical_MAE":
                sum(r["physical_absolute_error_sum"] for r in selected)/denominator,
            "original_DEV_last_state_persistence_MAE":
                sum(r["last_state_persistence_error_sum"] for r in selected)/denominator,
            "max_real_episode_recurrent_state_norm":
                max(r["max_recurrent_state_norm"] for r in selected),
            "real_sequential_horizon_segments":horizons}
    return agg

def run(root,receipt):
    root=Path(root)
    origin=json.loads((root/"results.json").read_text())
    if len(origin["actual_completed_training_models"])!=10 or not origin["all_frozen_original_models_before_heldout_opening"]:
        raise RuntimeError("original 10 model fitting evidence incomplete")
    records={(r["architecture"],r["seed"]):r
             for r in origin["actual_completed_training_models"]}
    cache=root/"private-DEV-forensic-NOT-AN-ARTIFACT"
    cache.mkdir(parents=True,exist_ok=True)
    cohorts,original_metadata,source_hashes=source_index(cache)
    if source_hashes!=origin["original_source_metadata_sha256"]:
        raise RuntimeError("publisher source metadata differs from original actual model fit")
    dev_ids=selected_dev_episodes(cohorts["DEV"])
    original={}
    for i in dev_ids:
        for attempt in range(3):
            parquet_file=cache/f"scoop_cat_litter_into_trash_can/data/chunk-{i//1000:03d}/episode_{i:06d}.parquet"
            if parquet_file.exists() and parquet_file.stat().st_size==0:
                parquet_file.unlink()
            try:
                original[i]=load_episode(cache,i,original_metadata)
                break
            except Exception:
                if attempt==2:
                    raise RuntimeError(f"original authentic DEV episode {i} could not be read from immutable source")
                if parquet_file.exists():
                    parquet_file.unlink()
    cp=torch.load(root/records[("native_tac",SEED_SUBSET[0])]["original_checkpoint_path"],
                  map_location="cpu",weights_only=False)
    scaling=cp["fixed_fit_only_scaling"]
    data=normalize(original,scaling)
    result={"study":"ME053_DIAGNOSTIC_REAL_DEV_FROZEN_ME052_MODELS",
        "actual_model_fits":0,"model_optimizer_updates":0,
        "original_model_training_run":"39714369920",
        "frozen_source_revision":REVISION,
        "selected_original_DEV_IDs":dev_ids,
        "actual_original_DEV_parquet_SHA256":{str(k):v["sha256"] for k,v in original.items()},
        "source_episodes_chosen_without_reading_labels":True,
        "original_TEST_trajectories_not_opened":True,
        "original_REAL_factual_postprediction_physical_feedback":True,
        "no_fabricated_reward_success_or_counterfactual_action":True,
        "run_diagnostic_only_no_architecture_superiority_claim":True,
        "frozen_actual_seeds":list(SEED_SUBSET),
        "comparison_modes":list(MODES),"scores":{}}
    write_json(receipt,result)
    for arm in ("native_tac","gru_world"):
        result["scores"][arm]={}
        for seed in SEED_SUBSET:
            rec=records[(arm,seed)]
            if sha_file(root/rec["original_checkpoint_path"])!=rec["original_checkpoint_sha256"]:
                raise RuntimeError("original actual trained model checksum disagrees")
            model=network(arm)
            model.load_state_dict(torch.load(
                root/rec["original_checkpoint_path"],map_location="cpu",
                weights_only=False)["state_dict"],strict=True)
            model.eval()
            rows=[]
            for mode in MODES:
                for ep in dev_ids:
                    rows.append({"original_episode_id":ep,"mode":mode,
                                 **forensic_one(model,data[ep],mode,arm,scaling)})
            summary=summarize(rows)
            result["scores"][arm][str(seed)]=summary
            result["completed_frozen_checkpoint_and_real_DEV_count"]={
                "original_trained_checkpoints_in_archive":len(records),
                "original_DEV_episodes_per_comparison":len(dev_ids)}
            print(json.dumps({"event":"ME053_AUTHENTIC_DEV_FROZEN_MODEL_DIAGNOSTIC",
                "arm":arm,"seed":seed,
                "full_feedback_MAE":summary["full_original_feedback"]["original_DEV_physical_MAE"],
                "no_feedback_MAE":summary["no_feedback"]["original_DEV_physical_MAE"],
                "physical_only_MAE":summary["physical_only"]["original_DEV_physical_MAE"]}),
                flush=True)
            write_json(receipt,result)
    result["status"]="ORIGINAL_REAL_DEV_DIAGNOSTIC_COMPLETE__ZERO_TRAINING"
    write_json(receipt,result)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="results/me052")
    p.add_argument("--receipt",default="results/me053/original-real-dev-feedback-forensics.json")
    args=p.parse_args()
    run(args.root,args.receipt)
