#!/usr/bin/env python3
"""ME054: substantial ACTUAL real-robot recurrent world-model training.

Reuses original ME052 source plumbing, not its unstable architectures,
training run, hyperparameters or evaluated TEST for independent claims.
Native TAC has mathematically bounded zero-innovation physical correction;
both native and strong GRU get identity prior, FIT-only genuine joint delta
scale and identical actual original training windows, 600 x batch2 x 16
supervised genuine transitions per seed. CPU time is *not* matched.
All 10 trained checkpoint hashes freeze before original DEV/TEST acquisition.
The original ME052 TEST was already inspected; new TEST scores are explicitly
developmental corroboration, NEVER fresh independent architecture superiority.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from scripts import me052_train_real_robot as base
from scripts.me052_score_frozen_real_robot import load_original_cohort,group_bootstrap
from scripts.validate_research_parity import check_manifest
from tac_osm.real_robot_me054_stable import (
    StableRealTAC,StableRealGRU,actual_real_physical_and_takeover_loss)

base.UPDATES=600
base.BATCH=2
base.HISTORY=256
base.BACKPROP=16
base.NEXT_FRAMES=base.HISTORY+base.BACKPROP
torch.set_num_threads(1)
ARMS=("native_tac","gru_world")
SEEDS=(52001,52002,52003,52004,52005)
REGISTRATION=Path("research/mathematical-engineering/me054/FAIR_COMPARISON.json")

def network(name,scale):
    if name=="native_tac":return StableRealTAC(scale)
    if name=="gru_world":return StableRealGRU(scale)
    raise ValueError("unknown original actual robot model")

def active(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def FIT_only_original_delta_scales(actual_fit,statistics):
    # Within-episode true measured qpos transitions only; NEVER concatenate
    # ep boundary into a fake physical transition.
    increments=np.concatenate(
        [np.abs(row["state"][1:]-row["state"][:-1])
         for _,row in sorted(actual_fit.items())],axis=0)
    actual_mean_delta=increments.mean(axis=0,dtype=np.float64)
    source_fit_state_scale=np.asarray(statistics["state_std"],dtype=np.float64)
    scale=np.maximum(actual_mean_delta/source_fit_state_scale,0.0005)
    if scale.shape!=(14,) or not np.isfinite(scale).all():
        raise RuntimeError("genuine FIT-only scale not established")
    return scale.astype(np.float32).tolist(),{
        "FIT_only_mean_original_one_step_absolute_delta":actual_mean_delta.tolist(),
        "FIT_normalized_one_step_delta_scale":scale.tolist(),
        "original_physical_transitions_used":len(increments)}

def verify_real_source_before_any_optimizer(original_fit,groups,plan):
    ids=sorted(original_fit)
    frozen_receipts=[{
        "original_episode_index":idx,
        "source_parquet_SHA256":original_fit[idx]["sha256"],
        "authentic_frame_count":original_fit[idx]["length"]} for idx in ids]
    measured_hash=base.sha_bytes(json.dumps(
        frozen_receipts,sort_keys=True,separators=(",",":")).encode())
    if measured_hash!=plan["dataset_fit_receipt_sha256"]:
        raise RuntimeError("original genuine 325 FIT Parquet source SHA altered")
    for split in ("FIT","DEV","TEST"):
        observed=[idx for idx,_ in groups[split]]
        observed_hash=base.sha_bytes(json.dumps(
            observed,separators=(",",":")).encode())
        if observed_hash!=plan["frozen_episode_split_sha256"][split]:
            raise RuntimeError("original episode memberships differ from source evidence")
    checked=check_manifest(plan,"preflight")
    if not checked["status"].startswith("PASS_OWNER"):
        raise RuntimeError("matched genuine parameters/data/update budget failed")
    return checked

def fit_one(name,seed,source,real_schedule,scale,model_folder):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m=network(name,scale)
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    start_cpu=time.process_time()
    records=[]
    for step,actual_pair in enumerate(real_schedule,1):
        states,actions,human=base.get_batch(source,actual_pair)
        z=m.initial(2)
        with torch.no_grad():
            for t in range(256):
                z,_,_,_,_=base.single_step(m,z,states,actions,human,t)
        opt.zero_grad(set_to_none=True)
        objective=None
        for t in range(256,272):
            # Genuine measured x_(t+1) exists only after physical command a_t.
            observed_now=states[:,t]
            z,forecast,real_future,real_onset,eligible=base.single_step(
                m,z,states,actions,human,t)
            term,_=actual_real_physical_and_takeover_loss(
                m,forecast,observed_now,real_future,real_onset,eligible)
            objective=term if objective is None else objective+term
        objective=objective/16
        if not torch.isfinite(objective):
            raise RuntimeError("nonfinite genuine normalized physical-delta loss")
        objective.backward()
        gradients=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        if not torch.isfinite(gradients):
            raise RuntimeError("nonfinite optimizer gradients on genuine physical data")
        opt.step()
        if step%100==0 or step==1:
            record={"genuine_real_fit_updates":step,
                "FIT_actual_physical_joint_delta_and_takeover_loss":float(objective.detach()),
                "CPU_seconds_for_efficiency_ONLY":time.process_time()-start_cpu}
            records.append(record)
        if step%100==0:
            print(json.dumps({"event":"ME054_FIT_REAL_PHYSICAL_WORLD_MODEL",
                "arm":name,"seed":seed,"optimizer_updates":step,
                "true_joint_delta_loss":round(float(objective.detach()),6)}),flush=True)
    original_path=model_folder/f"{name}-s{seed}"
    original_path.mkdir(parents=True,exist_ok=False)
    weights=original_path/"original-trained-real-robot-model.pt"
    torch.save({"architecture":name,"seed":seed,"model":m.state_dict(),
        "optimizer":opt.state_dict(),"original_FIT_delta_scale":scale,
        "original_real_source_revision":base.REVISION,
        "actual_optimizer_updates":600,
        "frozen_training_order_SHA256":
            base.sha_bytes(json.dumps(real_schedule,separators=(",",":")).encode()),
        "rng_torch_state":torch.get_rng_state(),
        "code_training_commit_sha":os.getenv("GITHUB_SHA","LOCAL")},
       weights)
    base.write_json(original_path/"training_loss_and_efficiency.json",records)
    result={"architecture":name,"seed":seed,
        "active_params":active(m),
        "original_actual_fit_updates":600,
        "original_real_history_frames":256,
        "real_backward_steps_per_update":16,
        "original_real_training_frames_presented":600*2*272,
        "original_checkpoint_sha256":base.sha_file(weights),
        "original_checkpoint_path":str(weights.relative_to(model_folder)),
        "training_schedule_SHA256":
            base.sha_bytes(json.dumps(real_schedule,separators=(",",":")).encode()),
        "actual_training_cpu_seconds":time.process_time()-start_cpu}
    print(json.dumps({"event":"ME054_ORIGINAL_TRAINED_PHYSICAL_MODEL_FROZEN",
        "arm":name,"seed":seed,"original_optimizer_updates":600,
        "checkpoint_SHA256":result["original_checkpoint_sha256"]}),flush=True)
    return result

def run(folder):
    root=Path(folder)
    root.mkdir(parents=True,exist_ok=False)
    original_cache=root/"private-original-robot-FIT-not-upload"
    original_cache.mkdir(parents=True)
    plan=json.loads(REGISTRATION.read_text())
    original_groups,original_metadata,source_meta_hashes=base.source_index(original_cache)
    actual_fit=base.load_part(original_cache,original_groups["FIT"],original_metadata,"FIT")
    prior_fit_stats=base.fit_scaling(actual_fit)
    delta_scale,delta_evidence=FIT_only_original_delta_scales(actual_fit,prior_fit_stats)
    normalized_fit=base.normalize(actual_fit,prior_fit_stats)
    prepared={seed:base.sample_schedule(normalized_fit,seed) for seed in SEEDS}
    primary_models={name:active(network(name,delta_scale)) for name in ARMS}
    if primary_models!=plan["parameter_counts"]:
        raise RuntimeError("original matched active model params changed")
    audit=verify_real_source_before_any_optimizer(actual_fit,original_groups,plan)
    report={"study":"ME054_REAL_JOINT_DELTA_PERSISTENT_STABLE_WORLD_MODEL",
        "status":"original_real_FIT_training",
        "owner_mandatory_parameter_data_budget_preflight":audit,
        "registered_before_first_fit":True,
        "original_source_revision":base.REVISION,
        "original_FIT_episode_count":len(actual_fit),
        "original_real_FIT_frames":sum(row["length"] for row in actual_fit.values()),
        "original_FIT_delta_scale_evidence":delta_evidence,
        "original_FIT_scaling":prior_fit_stats,
        "model_parameters":primary_models,"actual_trained_checkpoints":[],
        "frozen_source_metadata_SHA256":source_meta_hashes,
        "same_training_data_order_by_seed":{str(s):v[1]["training_window_order_SHA256"]
                                         for s,v in prepared.items()},
        "CPU_execution_time_required_to_match":False,
        "all_original_checkpoints_frozen_before_heldout":False,
        "ME052_TEST_previously_seen_fresh_confirmation":False,
        "no_verified_reward_success_or_causal_action_claim":True,
        "no_synthetic_model_training_or_evaluation":True}
    base.write_json(root/"results.json",report)
    for seed in SEEDS:
        schedule,_=prepared[seed]
        for name in ARMS:
            record=fit_one(name,seed,normalized_fit,schedule,delta_scale,root)
            report["actual_trained_checkpoints"].append(record)
            base.write_json(root/"results.json",report)
    if len(report["actual_trained_checkpoints"])!=10:
        raise RuntimeError("all original ten trained world model checkpoints required")
    for row in report["actual_trained_checkpoints"]:
        if base.sha_file(root/row["original_checkpoint_path"])!=row["original_checkpoint_sha256"]:
            raise RuntimeError("trained original state SHA changed")
    final_plan=dict(plan)
    final_plan["completed_model_records"]=report["actual_trained_checkpoints"]
    final_plan["all_frozen_original_models_before_heldout_opening"]=True
    checked=check_manifest(final_plan,"final")
    if not checked["status"].startswith("PASS_OWNER"):
        raise RuntimeError("actual original source parameter/data/update budget mismatch")
    report["postfit_actual_source_parameters_and_training_budget_check"]=checked
    report["all_original_checkpoints_frozen_before_heldout"]=True
    report["status"]="all_ten_original_real_fits_frozen_no_DEV_TEST_opened"
    base.write_json(root/"FAIR_COMPARISON_FINAL.json",final_plan)
    base.write_json(root/"results.json",report)
    print("ME054_ALL_TEN_TRUE_TRAINED_ROBOT_WORLD_MODEL_CHECKPOINTS_FROZEN",flush=True)

    # Both cohorts have been exposed by ME052: diagnostic/research evaluation,
    # NOT a newly untouched confirmatory dataset claim.
    del normalized_fit,actual_fit
    for part in ("DEV","TEST"):
        source,miss=load_original_cohort(original_cache,original_groups[part],
                                          original_metadata,part)
        report.setdefault("original_source_omissions",{})[part]=miss
        reports={}
        frames=base.normalize(source,prior_fit_stats)
        for row in report["actual_trained_checkpoints"]:
            m=network(row["architecture"],delta_scale)
            cp=torch.load(root/row["original_checkpoint_path"],
                          map_location="cpu",weights_only=False)
            m.load_state_dict(cp["model"],strict=True)
            metrics=base.score(m,frames,original_metadata,prior_fit_stats,part)
            model_key=f'{row["architecture"]}-s{row["seed"]}'
            reports[model_key]=metrics
            print(json.dumps({"event":"ME054_AUTHENTIC_PHYSICAL_WORLD_MODEL_EVAL",
                "cohort":part,"arm":row["architecture"],"seed":row["seed"],
                "original_joint_state_MAE":metrics["physical_MAE_real_units"],
                "real_persistence_MAE":metrics["persistence_MAE_real_units"]}),flush=True)
            base.write_json(root/"results.json",report)
        report[part+"_original_real_physical_predictions"]=reports
        report[part+"_original_real_episodes_available"]=len(source)
        base.write_json(root/"results.json",report)
    scored=report["TEST_original_real_physical_predictions"]
    paired={};wins=0
    for seed in SEEDS:
        a=scored[f"native_tac-s{seed}"]["physical_MAE_real_units"]
        b=scored[f"gru_world-s{seed}"]["physical_MAE_real_units"]
        wins+=int(a<b)
        paired[str(seed)]={"native_TAC_original_MAE":a,
             "learned_GRU_original_MAE":b,"native_TAC_improved":a<b}
    report["paired_original_five_seed_scores"]=paired
    report["original_TAC_seed_wins"]=wins
    report["original_TEST_already_exposed_so_independent_generalization_claim"]=False
    report["status"]="TEN_TRAINED_REAL_MODELS_AND_DEVELOPMENTAL_REAL_WORLD_SCORES_COMPLETE"
    base.write_json(root/"results.json",report)
    print(json.dumps({"event":"ME054_REAL_WORLD_PHYSICS_TRAINING_AND_DIAGNOSTIC_EVAL_DONE",
             "scores":paired,
             "fresh_independent_superiority_qualification":False}),flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/me054")
    args=p.parse_args()
    run(args.output)
