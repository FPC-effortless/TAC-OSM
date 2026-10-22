#!/usr/bin/env python3
"""Paired ME-056 execution profile on the frozen authentic ME-055 FIT frame.

This is an engineering profiler, not a model fit or capability experiment. It
times both arms in alternating order on the same pinned process, records every
sample, and bootstraps the paired median ratio. TEST data is never opened.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from scripts.me055_profile_prefit import (
    HORIZON, HISTORY, authentic_batch, build, load_authentic, objective,
    physical_identification, sha_file, update)
from tac_osm.real_robot_me055_rollout import TrajectoryForecast, factual_multistep_loss
from tac_osm.real_robot_me056_boundary import BoundaryServoRolloutTAC


def percentile_interval(values, *, seed=56001, samples=20_000):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=1 or len(values)<5 or not np.isfinite(values).all():
        raise ValueError("at least five finite paired measurements required")
    rng=np.random.default_rng(seed)
    medians=np.median(values[rng.integers(0,len(values),(samples,len(values)))],axis=1)
    return [float(np.quantile(medians,.025)),float(np.quantile(medians,.975))]


def burnin_with_causal_snapshots(model, states, actions, human, burnin, boundary):
    """Snapshots for the transition t -> t+1 must precede seeing state[t+1].

    The former post-feedback snapshots could encode the supervised target
    through verification, turning the dense prediction loss into hindsight
    reconstruction. Only time-t evidence may appear in the auxiliary state.
    """
    belief=model.initial_from_observation(states[:,0])
    histories=[]
    with torch.no_grad():
        for step in range(HISTORY):
            histories.append(type(belief)(*(value.detach() for value in belief)))
            transition=boundary if (step+1)%128==0 else burnin
            belief=transition(belief,states[:,step],actions[:,step],
                human[:,step],states[:,step+1],human[:,step+1])
    return belief,histories


def boundary_objective(model,batch,kernels):
    states,actions,human,_=batch
    burnin,boundary,observe,predict,dense_auxiliary=kernels
    belief,histories=burnin_with_causal_snapshots(
        model,states,actions,human,burnin,boundary)
    # Half of the transitions exactly fills the profiled compute released by
    # anchored consolidation; the fixed stride is outcome-independent.
    selected=histories[::2]
    fast=torch.stack([row.fast for row in selected],1).flatten(0,1)
    slow=torch.stack([row.slow for row in selected],1).flatten(0,1)
    memory=torch.stack([row.memory for row in selected],1).flatten(0,1)
    auxiliary=dense_auxiliary(
        fast,slow,memory,states[:,:HISTORY:2].flatten(0,1),
        states[:,1:HISTORY+1:2].flatten(0,1),
        actions[:,:HISTORY:2].flatten(0,1),
        human[:,:HISTORY:2].flatten(0,1),
        human[:,1:HISTORY+1:2].flatten(0,1))
    current=states[:,HISTORY]
    control=human[:,HISTORY]
    future_human=human[:,HISTORY:HISTORY+HORIZON+1]
    alive=(~future_human[:,:-1]).to(torch.int64).cumprod(1).bool()
    onset=(~future_human[:,:-1]&future_human[:,1:]).to(current.dtype)
    predicted_states=[];predicted_logits=[]
    for step in range(HORIZON):
        belief=observe(belief,current)
        prediction=predict(belief,current,
            actions[:,HISTORY+step:HISTORY+step+1],control)
        current=prediction.next_joint_state[:,0]
        predicted_states.append(current)
        predicted_logits.append(prediction.takeover_logits[:,0])
    forecast=TrajectoryForecast(torch.stack(predicted_logits,1),
                                torch.stack(predicted_states,1))
    primary=factual_multistep_loss(model,forecast,
        states[:,HISTORY+1:HISTORY+HORIZON+1],alive,onset,alive)[0]
    return primary+.05*auxiliary


def compiled_runner(model,batch,architecture):
    methods=[model._burnin_training,model._observe_training,
             model._predict_training]
    if architecture=="native_tac":
        methods.insert(1,model._burnin_boundary_training)
        methods.append(model._dense_transition_auxiliary_training)
    kernels=tuple(torch.compile(method,dynamic=False,fullgraph=True,
                                mode="reduce-overhead")
                  for method in methods)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    loss=(lambda: boundary_objective(model,batch,kernels)) if architecture=="native_tac" else (
        lambda: objective(model,batch,kernels,use_scan=False))
    return optimizer,loss


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fit-parquet",required=True)
    cli.add_argument("--expected-sha256",required=True)
    cli.add_argument("--output",required=True)
    cli.add_argument("--pairs",type=int,default=41)
    cli.add_argument("--warmup",type=int,default=5)
    cli.add_argument("--batch-size",type=int,default=2)
    args=cli.parse_args()
    if args.pairs<11 or args.warmup<2:
        raise SystemExit("at least 11 pairs and two warmups required")
    if sha_file(args.fit_parquet)!=args.expected_sha256:
        raise SystemExit("authentic FIT profiler episode SHA mismatch")
    affinity=[]
    if hasattr(os,"sched_getaffinity"):
        allowed=sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0,{allowed[0]})
        affinity=[allowed[0]]
    torch.set_num_threads(1)
    state,action,human=load_authentic(args.fit_parquet)
    identified=physical_identification(state,action)
    batch=authentic_batch(state,action,human,identified,args.batch_size)
    runners={}
    compile_cpu={}
    for architecture in ("native_tac","gru_world"):
        torch.manual_seed(56001)
        if architecture=="native_tac":
            model=BoundaryServoRolloutTAC(**{
                name:value.tolist() for name,value in identified.items()})
        else:
            model=build(architecture,identified)
        optimizer,loss=compiled_runner(model,batch,architecture)
        started=time.process_time_ns()
        update(model,batch,optimizer,loss)
        compile_cpu[architecture]=(time.process_time_ns()-started)/1e9
        for _ in range(args.warmup-1):
            update(model,batch,optimizer,loss)
        runners[architecture]=(model,optimizer,loss)
    durations={name:[] for name in runners}
    paired=[]
    for pair in range(args.pairs):
        order=("native_tac","gru_world") if pair%2==0 else ("gru_world","native_tac")
        measured={}
        for architecture in order:
            model,optimizer,loss=runners[architecture]
            started=time.process_time_ns()
            update(model,batch,optimizer,loss)
            elapsed=(time.process_time_ns()-started)/1e9
            durations[architecture].append(elapsed)
            measured[architecture]=elapsed
        ratio=measured["native_tac"]/measured["gru_world"]-1.
        paired.append(ratio)
    median_gap=float(statistics.median(paired))
    interval=percentile_interval(paired)
    receipt={
        "study":"ME056_PAIRED_EXECUTION_PROFILE",
        "capability_claim":False,
        "model_fits_started":0,
        "source_split":"FIT",
        "source_episode":12,
        "source_sha256":args.expected_sha256,
        "cpu_affinity":affinity,
        "torch_threads":torch.get_num_threads(),
        "warmup_updates_per_arm":args.warmup,
        "paired_updates":args.pairs,
        "batch_size":args.batch_size,
        "alternating_arm_order":True,
        "timer":"time.process_time_ns",
        "compiler":"torch.compile fullgraph reduce-overhead per transition kernel",
        "compile_cpu_seconds":compile_cpu,
        "cpu_seconds_samples":durations,
        "paired_TAC_over_GRU_relative_samples":paired,
        "paired_median_TAC_over_GRU_relative_gap":median_gap,
        "paired_median_bootstrap_95_interval":interval,
        "strict_5pct_gate_pass":interval[1]<=.05,
        "strict_5pct_gate_definitive_fail":interval[0]>.05,
    }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps({key:receipt[key] for key in (
        "paired_median_TAC_over_GRU_relative_gap",
        "paired_median_bootstrap_95_interval",
        "strict_5pct_gate_pass","strict_5pct_gate_definitive_fail")}))


if __name__=="__main__":
    raise SystemExit(main())

