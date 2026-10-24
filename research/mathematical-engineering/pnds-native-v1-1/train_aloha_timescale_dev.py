#!/usr/bin/env python3
"""Authentic LeRobot ALOHA DEV-ONLY timescale experiment, no test re-use.

Two 93,324-param architectures initialized IDENTICALLY per seed, same 64
updates and same exact 14D original recorded (obs,action,next_obs) windows.
Baseline is v1.5 all-history mean TAC; candidate is bounded fixed four-band
decay TAC. All state/action scaling comes from TRAIN recordings alone.

Uses 59 real train episodes and 13 previously exposed development episodes.
Original test episodes 72-84 are NOT evaluated. Compute/FLOP matching is
NOT ESTABLISHED by identical params/data/step count; perf comparisons are
descriptive and cannot qualify architecture superiority.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import random
import sys
import torch
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/"src"))
from tac_osm.variants.parallel_continuous_world import ParallelTACWorld
from tac_osm.variants.stable_timescale_world import StableTimescaleTACWorld
from train_aloha_real_pilot import (
    load_authentic_records, splits, windows, normalization, pack
)

DATA_REV="7a752b39f7e69de7e38aee485a6bab07528a061a"
DATA_SHA256="a3409d3282b2cf9a94d32b301756fbd15dca5862cc1c32973c7316dd8853c57a"


def metrics(model,tensors,*,ablate=None,early=32,threshold=None):
    obs,act,truth=tensors
    model.eval()
    kwargs={}
    if ablate=="memory": kwargs={"ablate_memory":True}
    elif ablate=="context": kwargs={"ablate_context":True}
    elif ablate is not None:raise ValueError("unknown ablation")
    with torch.no_grad():
        pred=model(obs,act,**kwargs)
        se=(pred-truth).square()
        delta=(truth-obs).square().sum(-1).sqrt()
        # Threshold is computed only from authentic training trajectories.
        fast=delta>=threshold if threshold is not None else torch.ones_like(delta,dtype=torch.bool)
        return {
          "full_mse":round(se.mean().item(),9),
          "late_half_mse":round(se[:,early:].mean().item(),9),
          "high_change_mse":round(se[fast].mean().item(),9),
          "high_change_source_points":int(fast.sum().item()),
        }


def train_one(model,init_state,source_train,source_val,indices,scale,threshold):
    model.load_state_dict(copy.deepcopy(init_state),strict=True)
    model.set_train_delta_scale(scale)
    opt=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.001)
    last_loss=None
    for row in indices:
        o,a,y=(part[row] for part in source_train)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=(model(o,a)-y).square().mean()
        if not torch.isfinite(loss).item():
            raise RuntimeError("nonfinite factual observed-world training")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        last_loss=float(loss.detach().item())
    score=metrics(model,source_val,threshold=threshold,
                  early=source_val[0].shape[1]//2)
    return score,last_loss


def main(steps=64,seeds=(20261025,20261026),length=64):
    torch.set_num_threads(1)
    records,prov=load_authentic_records()
    if prov["dataset_revision_sha"]!=DATA_REV or prov["parquet_sha256"]!=DATA_SHA256:
        raise RuntimeError("authentic source changed; no benchmark substitution")
    ids=splits(records)
    # Mandatory: no heldout records or targets from TEST episodes enter
    # train, train-only scaler, candidate tuning or DEV evaluation.
    train_records=windows(records,ids["train"],length=length,max_windows_per_episode=8)
    val_records=windows(records,ids["validation"],length=length,max_windows_per_episode=8)
    norm=normalization(train_records)
    train_data=pack(train_records,norm)
    val_data=pack(val_records,norm)
    train_change=(train_data[2]-train_data[0]).square().sum(-1).sqrt()
    threshold=train_change.flatten().quantile(.75)
    scale=(train_data[2]-train_data[0]).square().mean(dim=(0,1)).sqrt().clamp_min(1e-5)
    output=[]
    for seed in seeds:
        torch.manual_seed(seed);random.seed(seed)
        baseline=ParallelTACWorld()
        candidate=StableTimescaleTACWorld()
        assert sum(p.numel() for p in baseline.parameters())==sum(
            p.numel() for p in candidate.parameters())==93324
        init=copy.deepcopy(baseline.state_dict())
        batch_indices=torch.randint(0,len(train_records),(steps,8),
             generator=torch.Generator().manual_seed(seed))
        digest=hashlib.sha256(json.dumps(batch_indices.tolist(),separators=(',',':')).encode()).hexdigest()
        base_scores,loss_b=train_one(
            baseline,init,train_data,val_data,batch_indices,scale,threshold)
        candidate_scores,loss_c=train_one(
            candidate,init,train_data,val_data,batch_indices,scale,threshold)
        ablate_mem=metrics(candidate,val_data,ablate="memory",
            threshold=threshold,early=length//2)
        ablate_ctx=metrics(candidate,val_data,ablate="context",
            threshold=threshold,early=length//2)
        with torch.no_grad():
            obs,action,real_next=val_data
            persist=((obs-real_next)**2)
            real_change=(real_next-obs).square().sum(-1).sqrt()
            fast=real_change>=threshold
            pers={
              "full_mse":round(persist.mean().item(),9),
              "late_half_mse":round(persist[:,length//2:].mean().item(),9),
              "high_change_mse":round(persist[fast].mean().item(),9),
              "high_change_source_points":int(fast.sum().item())}
        output.append({
          "seed":seed,"batch_schedule_sha256":digest,"train_steps_each":steps,
          "source_train_windows_used":len(train_records),
          "mean_tac_validation":base_scores,
          "timescale_tac_validation":candidate_scores,
          "timescale_memory_disabled_validation":ablate_mem,
          "timescale_context_disabled_validation":ablate_ctx,
          "observed_persistence_validation":pers,
          "train_terminal_loss_mean":loss_b,"train_terminal_loss_timescale":loss_c
        })
    print(json.dumps({
       "status":"GENUINE_ROBOT_DEV_ONLY_TIMESCALE_ABLATION_NO_TEST_ACCESS",
       "original_data":prov,"data_revision_pinned":True,
       "training_episode_count":len(ids["train"]),
       "development_episode_count":len(ids["validation"]),
       "test_episode_count_not_read":len(ids["test"]),
       "train_windows":len(train_records),"dev_windows":len(val_records),
       "history_frames":length,"recorded_states_actions_targets_only":True,
       "train_derived_high_change_threshold":float(threshold),
       "same_active_parameters":93324,
       "identical_train_source_schedules":True,
       "training_steps_and_data_equal":True,
       "training_FLOPs_equal_verified":False,
       "full_fair_architecture_superiority_claim":False,
       "development_is_previously_exposed":True,
       "independent_real_capability_qualified":False,
       "entity_admission_jointly_trained":False,
       "seed_results":output},sort_keys=True))


if __name__=="__main__":
    main(steps=int(os.environ.get("TAC_TIMESCALE_STEPS","64")),
         length=int(os.environ.get("TAC_TIMESCALE_LENGTH","64")))
