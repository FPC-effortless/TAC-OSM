#!/usr/bin/env python3
"""ME-058: genuine FIT-only source-backed, corrected two-arm prefit parity.

This source-driven *diagnostic* performs one discarded optimiser update on a
fully original recorded FIT batch per arm. No scored models are trained.
Test/DEV physical data are not fetched. No synthetic performance data.
The prior frozen ME-057 receipt is checked on *all* original FIT episodes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from scripts import me055_reserve_independent_robot_task as reservation
from scripts import me057_audit_original_fit as audit
from scripts.me055_profile_prefit import HISTORY, HORIZON, update, objective
from scripts.me056_profile_paired_execution import boundary_objective
from tac_osm.real_robot_me055_rollout import ServoRolloutGRU
from tac_osm.real_robot_me056_boundary import BoundaryServoRolloutTAC

SEEDS=(58001,58002,58003,58004,58005)
UPDATES=600
BATCH_SIZE=2
SOURCE_MANIFEST_SHA256="c4064651e91cced7aafc11337c32a27266f470ea10388bb0dd2562ae955f88fd"
FROZEN_CAPACITY={"native_tac":38266,"gru_world":38295}
FLOP_MAX_GAP=.01
PARAM_MAX_GAP=.01

def hashed(x):
    return hashlib.sha256(json.dumps(x,separators=(",",":"),sort_keys=True).encode()).hexdigest()

def read_verified_fit(cache,registration):
    frozen=json.loads(Path(registration).read_text())
    name=reservation.chosen_task()
    info=reservation.download(name,"info.json",1_000_000)
    episodes=reservation.download(name,"episodes.jsonl",2_000_000)
    membership=audit.verify_registration(info,episodes,frozen)
    originals={}
    receipts=[]
    for n,(episode,expected_frames) in enumerate(membership["FIT"],1):
        raw=audit.get_episode(episode,Path(cache))
        stats=audit.verify_parquet(raw,expected_frames)
        import pyarrow.parquet as pq
        table=pq.read_table(io.BytesIO(raw),
             columns=["observations.state.qpos","action.qpos","intervention"])
        original={
            "state":np.asarray(table["observations.state.qpos"].to_pylist(),dtype=np.float32),
            "action":np.asarray(table["action.qpos"].to_pylist(),dtype=np.float32),
            "human":np.asarray(table["intervention"].to_pylist(),dtype=np.bool_),
        }
        originals[episode]=original
        receipts.append({"episode_index":episode,"sha256":reservation.sha(raw),**stats})
        if n%50==0:
            print(json.dumps({"event":"ME058_ORIGINAL_FIT_CHECKED",
                              "episodes":n,"total":len(membership["FIT"])}),flush=True)
    assert len(originals)==228
    actual_hash=audit.canonical_hash(receipts)
    if actual_hash!=SOURCE_MANIFEST_SHA256:
        raise RuntimeError("ME058 original 228 FIT source hashes differ from frozen ME057 receipt")
    return originals,membership,actual_hash

def identify_fit_only(originals):
    """Within-episode FIT moments and action->state prior; never join episodes."""
    episodes=list(originals.values())
    states=np.concatenate([x["state"] for x in episodes]).astype(np.float64)
    actions=np.concatenate([x["action"] for x in episodes]).astype(np.float64)
    sm=states.mean(0); ss=np.maximum(states.std(0),.05)
    am=actions.mean(0); ast=np.maximum(actions.std(0),.05)
    differences=np.concatenate([(x["state"][1:]-x["state"][:-1])/ss
                                for x in episodes]).astype(np.float64)
    errors=np.concatenate([(x["action"][:-1]-x["state"][:-1])/ss
                           for x in episodes]).astype(np.float64)
    gain=np.clip((errors*differences).sum(0)/
                 (np.square(errors).sum(0)+1.e-8),0.,.25)
    dscale=np.maximum(np.abs(differences).mean(0),.0005)
    step=np.maximum(np.quantile(np.abs(differences),.99,axis=0),dscale)
    values={
        "state_mean":sm.astype(np.float32).tolist(),
        "state_scale":ss.astype(np.float32).tolist(),
        "action_mean":am.astype(np.float32).tolist(),
        "action_scale":ast.astype(np.float32).tolist(),
        "servo_gain":gain.astype(np.float32).tolist(),
        "delta_scale":dscale.astype(np.float32).tolist(),
        "servo_step_limit":step.astype(np.float32).tolist(),
    }
    for key,value in values.items():
        if len(value)!=14 or not all(math.isfinite(x) for x in value):
            raise RuntimeError("invalid FIT-only physical prior "+key)
    return values

def schedule_fit_only(originals,seed,steps=UPDATES):
    """Outcome-blind fixed-length sample origins, excluding human-controlled start.

    Only the *current* FIT control authority determines eligibility. The
    preceding and future human labels do not determine selection priority.
    Identical immutable sampling order is given to both architecture arms.
    """
    eligible=[]
    for episode,record in sorted(originals.items()):
        n=len(record["state"])
        for start in range(max(0,n-(HISTORY+HORIZON+1))):
            if not record["human"][start+HISTORY]:
                eligible.append((episode,start))
    if len(eligible)<100:
        raise RuntimeError("insufficient real robot-controlled FIT sample origins")
    rng=np.random.default_rng(seed)
    picks=rng.integers(len(eligible),size=(steps,BATCH_SIZE))
    plan=[[list(eligible[int(i)]) for i in pair] for pair in picks]
    return plan,{"eligible_source_windows":len(eligible),
                 "schedule_sha256":hashed(plan)}

def original_batch(originals,selected,identified):
    state=[];action=[];human=[]
    sm=np.asarray(identified["state_mean"],np.float32)
    ss=np.asarray(identified["state_scale"],np.float32)
    am=np.asarray(identified["action_mean"],np.float32)
    ast=np.asarray(identified["action_scale"],np.float32)
    for episode,first in selected:
        e=originals[episode]
        sl=slice(first,first+HISTORY+HORIZON+1)
        sa=slice(first,first+HISTORY+HORIZON)
        state.append((e["state"][sl]-sm)/ss)
        action.append((e["action"][sa]-am)/ast)
        human.append(e["human"][sl])
    b=torch.from_numpy(np.stack(state).astype(np.float32))
    a=torch.from_numpy(np.stack(action).astype(np.float32))
    h=torch.from_numpy(np.stack(human).astype(np.bool_)).unsqueeze(-1)
    if b.shape!=(BATCH_SIZE,HISTORY+HORIZON+1,14):
        raise RuntimeError("genuine source batch state dimensions wrong")
    if h[:,HISTORY].any():
        raise RuntimeError("genuine rollout origin was under human control")
    return b,a,h,tuple((int(ep),int(start)) for ep,start in selected)

def construct(name,identified):
    return (BoundaryServoRolloutTAC(**identified) if name=="native_tac"
            else ServoRolloutGRU(**identified))

def profile_discarded_update(name,batch,identified):
    torch.manual_seed(SEEDS[0])
    model=construct(name,identified)
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params!=FROZEN_CAPACITY[name]:
        raise RuntimeError(f"{name} parameter count changed: {params}")
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    if name=="native_tac":
        kernels=(model._burnin_training,model._burnin_boundary_training,
                 model._observe_training,model._predict_training,
                 model._dense_transition_auxiliary_training)
        loss=lambda:boundary_objective(model,batch,kernels)
    else:
        loss=lambda:objective(model,batch)
    start=time.process_time()
    with profile(activities=[ProfilerActivity.CPU],with_flops=True) as recording:
        actual_loss=update(model,batch,optimizer,loss)
    cpu=time.process_time()-start
    if not np.isfinite(actual_loss):
        raise RuntimeError("real FIT prefit nonfinite loss")
    flops=sum(int(x.flops) for x in recording.key_averages())
    if flops<=0:
        raise RuntimeError("actual original FIT profiler returned no operator FLOPs")
    return {
        "architecture":name,
        "parameters":params,
        "profiled_operator_flops_per_update":flops,
        "cpu_process_seconds_efficiency_only":cpu,
        "discarded_fit_real_objective":actual_loss,
        "objective":"causal anchored + 128 real-history dense auxiliary + 16 rollout"
                    if name=="native_tac" else "GRU 16 rollout; same observed real history/targets",
        "full_true_training_flops_certified":False,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache",default="results/me058/private-original-FIT")
    p.add_argument("--registration",default="research/me055-independent-robot-task-source.json")
    p.add_argument("--output",default="results/me058/ME058_ORIGINAL_FIT_PREFIT.json")
    args=p.parse_args()
    torch.set_num_threads(1)
    originals,membership,source_sha=read_verified_fit(args.cache,args.registration)
    values=identify_fit_only(originals)
    schedules={}
    for seed in SEEDS:
        schedule,stat=schedule_fit_only(originals,seed)
        schedules[str(seed)]={"schedule_sha256":stat["schedule_sha256"],
            "eligible_source_windows":stat["eligible_source_windows"],
            "first_selected_pair":schedule[0]}
    first_seed_schedule,_=schedule_fit_only(originals,SEEDS[0])
    batch=original_batch(originals,first_seed_schedule[0],values)
    print(json.dumps({"event":"ME058_REAL_FIT_PREFIT_BEGIN",
         "fit_episodes":len(originals),
         "original_source_receipt_hash":source_sha,
         "paired_seeds":list(SEEDS),
         "fit_physical_prior_sha256":hashed(values)}),flush=True)
    arms=[profile_discarded_update(name,batch,values)
          for name in ("native_tac","gru_world")]
    def relative(key):
        v=[row[key] for row in arms]
        return (max(v)-min(v))/max(v)
    pg=relative("parameters")
    fg=relative("profiled_operator_flops_per_update")
    result={
        "study":"ME058_PROSPECTIVE_AUTHENTIC_FIT_CAUSAL_PROFILE",
        "schema_version":"me058.factual-FIT-prefit.v1",
        "fit_source_manifest_sha256":source_sha,
        "original_source_episodes":len(originals),
        "source_task":reservation.chosen_task(),
        "publisher_revision":reservation.REVISION,
        "frozen_cohort_id_sha256":{
            k:reservation.sha(json.dumps([i for i,_ in membership[k]],
                separators=(",",":")).encode()) for k in ("FIT","DEV","TEST")},
        "frozen_sample_orders":schedules,
        "fit_physical_prior":values,
        "fit_physical_prior_sha256":hashed(values),
        "parameter_relative_gap":pg,
        "profiled_operator_FLOP_relative_gap":fg,
        "parameter_parity_pass":pg<=PARAM_MAX_GAP,
        "profiled_operator_FLOP_parity_pass":fg<=FLOP_MAX_GAP,
        "CPU_runtime_matching_required":False,
        "full_actual_training_FLOP_parity_certified":False,
        "feature_access_same":True,
        "observed_transition_fields_same":True,
        "architecture_objective_differs_due_to_dense_TAC_auxiliary":True,
        "training_budget_next_stage":{"seeds":list(SEEDS),"updates":UPDATES,
            "batch_size":BATCH_SIZE,"history_frames":HISTORY,"rollout_frames":HORIZON,
            "training_tuning_trials":0},
        "original_DEV_TEST_trajectories_opened":0,
        "comparative_scored_models_fitted":0,
        "only_discarded_FIT_profiler_updates_per_arm":1,
        "arms":arms,
        "status":"PROFILED_ONLY_NO_TRAINING",
    }
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"event":"ME058_AUTHENTIC_FIT_PREFIT_RECEIPT",
       "parameter_relative_gap":pg,"profiled_operator_FLOP_relative_gap":fg,
       "parameter_pass":pg<=PARAM_MAX_GAP,
       "profiled_FLOP_pass":fg<=FLOP_MAX_GAP,
       "trained_models":0}),flush=True)
    if pg>PARAM_MAX_GAP or fg>FLOP_MAX_GAP:
        raise SystemExit("ME058_PRETRAINING_RESOURCE_PARITY_BLOCKED_ORIGINAL_GENUINE_FIT")
    print("ME058_ORIGINAL_FIT_OPERATOR_PROFILE_PASSED_NO_TRAINING_AUTHORIZED",flush=True)

if __name__=="__main__":
    main()
