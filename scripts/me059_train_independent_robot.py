#!/usr/bin/env python3
"""ME-059: genuine independent-task FIT model training, guarded by preregistration.

This runner MUST NOT fit until a committed architecture-comparison manifest
explicitly authorizes it and passes the fail-closed parity validator. All ten
checkpoints are frozen before any genuine DEV or TEST trajectory is read.
Never infer autonomous policy or counterfactual benefit from a human log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import resource
import time
from pathlib import Path

import numpy as np
import torch

from scripts import me058_profile_original_fit as prior
from scripts.me056_profile_paired_execution import boundary_objective
from scripts.me055_profile_prefit import objective, update
from scripts.validate_research_parity import check_manifest

PLAN_PATH=Path("research/mathematical-engineering/me059/FAIR_COMPARISON.json")
ARMS=("native_tac","gru_world")
HORIZON=16
ORIGINS=(256,512,768)

def digest_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(2**20),b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")

def allow_fit(plan):
    """Refuse diagnostics, missing fixed receipts, or incomplete registration."""
    if plan.get("study_type")!="architecture_comparison" or plan.get("training_authorized") is not True:
        raise RuntimeError("ME059_NO_TRAINING: committed full scientific architecture comparison registration required")
    if plan.get("schema_version")!="tac-osm.parity.v3":
        raise RuntimeError("ME059_NO_TRAINING: prospective CPU-independent parity-v3 required")
    if plan.get("fit_source_manifest_sha256")!=prior.SOURCE_MANIFEST_SHA256:
        raise RuntimeError("ME059_NO_TRAINING: original FIT source manifest not frozen")
    if plan.get("expected_physical_prior_sha256") is None:
        raise RuntimeError("ME059_NO_TRAINING: frozen genuine FIT physical prior required")
    orders=plan.get("frozen_sample_orders",{})
    if {str(s) for s in prior.SEEDS}!=set(orders):
        raise RuntimeError("ME059_NO_TRAINING: missing five paired original source schedules")
    if (plan.get("optimizer_updates")!=prior.UPDATES or
            plan.get("batch_size")!=prior.BATCH_SIZE or
            plan.get("history_frames")!=prior.HISTORY or
            plan.get("forecast_frames")!=prior.HORIZON):
        raise RuntimeError("ME059_NO_TRAINING: source exposure changed")
    if not plan.get("me058_source_backed_FLOP_preflight_pass",False):
        raise RuntimeError("ME059_NO_TRAINING: corrected original FIT FLOP parity not verified")
    # The native/GRU model graphs and original profile must remain immutable.
    for path,expected in plan.get("frozen_code_blob_sha",{}).items():
        observed=subprocess.check_output(["git","hash-object",path],text=True).strip()
        if observed!=expected:
            raise RuntimeError("ME059_NO_TRAINING: graph changed after true FIT profiler: "+path)
    checked=check_manifest(plan,"preflight")
    if checked["status"]!="PASS":
        raise RuntimeError("ME059_NO_TRAINING: machine preregistration failed")
    return checked

def fit_one(name,seed,source,plan,identified,directory):
    schedule,evidence=prior.schedule_fit_only(source,seed)
    if evidence["schedule_sha256"]!=plan["frozen_sample_orders"][str(seed)]:
        raise RuntimeError("ME059 source sample order not immutable")
    torch.manual_seed(seed)
    model=prior.construct(name,identified)
    counts=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if counts!=prior.FROZEN_CAPACITY[name]:
        raise RuntimeError("ME059 architecture parameter geometry changed")
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    t0=time.process_time()
    journal=[]
    for step,pair in enumerate(schedule,1):
        batch=prior.original_batch(source,pair,identified)
        if name=="native_tac":
            kernels=(model._burnin_training,model._burnin_boundary_training,
                     model._observe_training,model._predict_training,
                     model._dense_transition_auxiliary_training)
            loss=lambda:boundary_objective(model,batch,kernels)
        else:
            loss=lambda:objective(model,batch)
        value=update(model,batch,optimizer,loss)
        if not np.isfinite(value):
            raise RuntimeError("ME059 genuine source objective became nonfinite")
        if step==1 or step%100==0:
            row={"step":step,"measured_FIT_training_loss":value,
                 "CPU_seconds_efficiency_only":time.process_time()-t0}
            journal.append(row)
            print(json.dumps({"event":"ME059_GENUINE_FIT_MODEL_UPDATE",
                 "architecture":name,"seed":seed,**row}),flush=True)
    directory.mkdir(parents=True,exist_ok=False)
    destination=directory/"genuine-independent-robot-model.pt"
    torch.save({
        "model":model.state_dict(),"optimizer":optimizer.state_dict(),
        "architecture":name,"seed":seed,"source_manifest_sha256":prior.SOURCE_MANIFEST_SHA256,
        "training_schedule_sha256":evidence["schedule_sha256"],
        "fit_only_identification":identified,"training_steps":prior.UPDATES,
        "code_commit":os.getenv("GITHUB_SHA","LOCAL"),
        "torch_rng":torch.get_rng_state(),
    },destination)
    write_json(directory/"training_loss_trace.json",journal)
    return {"architecture":name,"seed":seed,"original_optimizer_updates":prior.UPDATES,
            "active_params":counts,"training_schedule_sha256":evidence["schedule_sha256"],
            "checkpoint_sha256":digest_file(destination),
            "checkpoint_path":str(destination),
            "train_CPU_seconds_efficiency_only":time.process_time()-t0}

@torch.no_grad()
def score_episode(name,model,original,identified):
    """Only genuine observed physical state under logged causal actions.

    Each prediction starts from a *past-only* belief; future action sequence
    is supplied as logged action conditions, not treated as an autonomous plan.
    Human takeover censors all subsequent steps. Return additive scores.
    """
    ss=np.asarray(identified["state_scale"],np.float32)
    sm=np.asarray(identified["state_mean"],np.float32)
    am=np.asarray(identified["action_mean"],np.float32)
    ast=np.asarray(identified["action_scale"],np.float32)
    x=torch.from_numpy(((original["state"]-sm)/ss).astype(np.float32))
    a=torch.from_numpy(((original["action"]-am)/ast).astype(np.float32))
    human=torch.from_numpy(original["human"].astype(np.bool_))
    score={"errors":{m:[0.]*HORIZON for m in
                     (name,"persistence","servo_prior")},
           "counts":[0]*HORIZON,"origins":0,"onset_positives":0}
    for origin in ORIGINS:
        if origin+HORIZON>=len(x) or bool(human[origin]):
            continue
        belief=(model.initial_from_observation(x[:1]) if name=="native_tac"
                else model.initial(1))
        for step in range(origin):
            fn=(model._burnin_boundary_training
                if name=="native_tac" and (step+1)%128==0
                else model._burnin_training)
            belief=fn(belief,x[step:step+1],a[step:step+1],
                human[step:step+1].view(1,1),x[step+1:step+2],
                human[step+1:step+2].view(1,1))
        current=x[origin:origin+1]
        first=current.clone()
        servo=current.clone()
        known_human=torch.zeros(1,1,dtype=torch.bool)
        active=True
        score["origins"]+=1
        for h in range(HORIZON):
            if bool(human[origin+h]):
                active=False
            if not active:
                break
            belief=model._observe_training(belief,current)
            fore=model._predict_training(
                belief,current,a[origin+h:origin+h+1].unsqueeze(1),known_human)
            current=fore.next_joint_state[:,0]
            dest=original["state"][origin+h+1]
            prediction=current[0].numpy()*ss+sm
            persistence=first[0].numpy()*ss+sm
            target_normalized=((a[origin+h:origin+h+1] *
                torch.as_tensor(identified["action_scale"])) +
                torch.as_tensor(identified["action_mean"]) -
                torch.as_tensor(identified["state_mean"])) / (
                torch.as_tensor(identified["state_scale"]))
            inc=torch.as_tensor(identified["servo_gain"])*(
                target_normalized-servo)
            limit=torch.as_tensor(identified["servo_step_limit"])
            servo=servo+inc.clamp(-limit,limit)
            servo_value=servo[0].numpy()*ss+sm
            for label,raw in ((name,prediction),("persistence",persistence),
                              ("servo_prior",servo_value)):
                score["errors"][label][h]+=float(np.abs(raw-dest).sum())
            score["counts"][h]+=14
            if not bool(human[origin+h]) and bool(human[origin+h+1]):
                score["onset_positives"]+=1
    return score

def score_original_cohort(name,model,group,identified):
    accumulated=None
    for episode,row in group.items():
        result=score_episode(name,model,row,identified)
        if accumulated is None:
            accumulated=result
        else:
            for label,items in result["errors"].items():
                accumulated["errors"][label]=[
                    x+y for x,y in zip(accumulated["errors"][label],items)]
            accumulated["counts"]=[x+y for x,y in zip(
                accumulated["counts"],result["counts"])]
            accumulated["origins"]+=result["origins"]
            accumulated["onset_positives"]+=result["onset_positives"]
    if accumulated is None:
        raise RuntimeError("no original cohort episodes scored")
    return {"origins":accumulated["origins"],
            "recorded_takeover_onsets":accumulated["onset_positives"],
            "evaluable_horizon_steps":accumulated["counts"],
            "physical_MAE_by_horizon":{
                label:[err/n if n else None for err,n in zip(
                    items,accumulated["counts"])]
                for label,items in accumulated["errors"].items()},
            "physical_MAE_pooled":{
                label:(sum(items)/sum(accumulated["counts"])
                       if sum(accumulated["counts"]) else None)
                for label,items in accumulated["errors"].items()}}

def load_true_cohort(group,cache):
    import io
    import pyarrow.parquet as pq
    from scripts import me057_audit_original_fit as audit
    original={}
    for index,expected in group:
        raw=audit.get_episode(index,Path(cache))
        audit.verify_parquet(raw,expected)
        table=pq.read_table(io.BytesIO(raw),columns=[
            "observations.state.qpos","action.qpos","intervention"])
        original[index]={
            "state":np.asarray(table["observations.state.qpos"].to_pylist(),np.float32),
            "action":np.asarray(table["action.qpos"].to_pylist(),np.float32),
            "human":np.asarray(table["intervention"].to_pylist(),np.bool_)}
    return original

def run(output):
    # A *separate* complete genuine architecture comparison registration is
    # mandatory before any optimizer update or opening any DEV/TEST trajectory.
    plan=json.loads(PLAN_PATH.read_text())
    allow_fit(plan)
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    fit,groups,source_sha=prior.read_verified_fit(
        root/"private-FIT-data",plan["source_registry"])
    identified=prior.identify_fit_only(fit)
    if prior.hashed(identified)!=plan["expected_physical_prior_sha256"]:
        raise RuntimeError("ME059 source-derived physical prior differs from frozen prefit")
    all_schedules={str(seed):prior.schedule_fit_only(fit,seed)[1]["schedule_sha256"]
                   for seed in prior.SEEDS}
    if all_schedules!=plan["frozen_sample_orders"]:
        raise RuntimeError("ME059 frozen original sample order mismatch")
    receipt={"status":"source_verified_before_fit",
        "experiment":"ME059_INDEPENDENT_PHYSICAL_ROBOT",
        "original_fit_sha256":source_sha,
        "original_frozen_order_sha256":all_schedules,
        "active_parameter_counts":prior.FROZEN_CAPACITY,
        "actual_checkpoints":[],"all_checkpoints_frozen_before_heldout":False,
        "policy_value_claim":False,
        "full_true_training_FLOP_parity_certified":False}
    write_json(root/"results.json",receipt)
    for seed in prior.SEEDS:
        for name in ARMS:
            checkpoint=fit_one(name,seed,fit,plan,identified,root/f"{name}-s{seed}")
            receipt["actual_checkpoints"].append(checkpoint)
            write_json(root/"results.json",receipt)
    if len(receipt["actual_checkpoints"])!=10:
        raise RuntimeError("ME059 missing complete ten original model checkpoints")
    for row in receipt["actual_checkpoints"]:
        if digest_file(row["checkpoint_path"])!=row["checkpoint_sha256"]:
            raise RuntimeError("ME059 persisted checkpoint hash corruption")
    receipt["all_checkpoints_frozen_before_heldout"]=True
    # Reconcile exact optimizer step count, original per-seed sample order and
    # actual checkpoint hashes BEFORE accessing independent heldouts.
    # FLOPs here are partial operator-profile estimates, never certified
    # exact full original training FLOP accounting.
    final_plan=json.loads(json.dumps(plan))
    records={(x["architecture"],x["seed"]):x for x in receipt["actual_checkpoints"]}
    for arm in final_plan["arms"]:
        original=records[(arm["architecture"],arm["seed"])]
        budget=arm["budget"]
        arm["measured"]={
            "actual_steps":original["original_optimizer_updates"],
            "actual_examples":budget["examples_seen"],
            "actual_tokens":budget["tokens_seen"],
            "training_flops":budget["prefit_profile_training_flops"]*original["original_optimizer_updates"],
            "training_flops_estimate_only":True,
            "optimizer_cpu_seconds":max(1.e-6,original["train_CPU_seconds_efficiency_only"]),
            "peak_memory_bytes":max(1,int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024),
            "flops_profiler":"Extrapolated torch.profiler CPU reported operators only; no true full FLOP certification",
            "checkpoint_sha256":original["checkpoint_sha256"],
            "actual_sample_order_sha256":original["training_schedule_sha256"],
            "evaluation_split_sha256":arm["test_split_sha256"],
        }
    final_plan["all_frozen_original_models_before_heldout_opening"]=True
    original_budget_review=check_manifest(final_plan,"final")
    if original_budget_review["eligible_for_architecture_superiority"] is True:
        raise RuntimeError("ME059 partial profiled FLOPs must NOT qualify superiority")
    receipt["postfit_resource_accounting"]=original_budget_review
    write_json(root/"FAIR_COMPARISON_FINAL.json",final_plan)
    write_json(root/"results.json",receipt)
    # Never open independent heldout trajectories until all original models
    # and the final machine budget accounting have been verified.
    del fit
    for cohort in ("DEV","TEST"):
        original=load_true_cohort(groups[cohort],root/"private-"+cohort+"-data")
        cohort_results={}
        for row in receipt["actual_checkpoints"]:
            model=prior.construct(row["architecture"],identified)
            weights=torch.load(row["checkpoint_path"],map_location="cpu",weights_only=False)
            model.load_state_dict(weights["model"],strict=True)
            model.eval()
            outcome=score_original_cohort(row["architecture"],model,original,identified)
            cohort_results[f'{row["architecture"]}-s{row["seed"]}']=outcome
        receipt[cohort+"_original_physical_forecasts"]=cohort_results
        receipt["status"]="developmental_physical_results_no_matched_FLOP_certificate"
        write_json(root/"results.json",receipt)
    return receipt

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--output",default="results/me059-original-real-robot")
    args=cli.parse_args()
    run(args.output)
