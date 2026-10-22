#!/usr/bin/env python3
"""Honest PRE-FIT real-data resource probe for native TAC versus recurrent GRU.

No optimizer, no parameter updates, no DEV/TEST file, no heldout labels.
Uses 256 original temporally linked FIT frames from same source revision and
the exact same 14D states/actions/authority for both untrained architectures.

Crucial: torch.profiler(with_flops=True) only estimates some matrix/convolution
operations. It is NOT full training FLOP certification, especially for GRU
and recurrent pointwise/backward operations. Preserve partial measurements,
explicitly FAIL comparative training eligibility, and don't invent counts.
A future complete operator-level hardware trace or validated analytical
instrument is compulsory before FAIR_COMPARISON.json can pass.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import profile,ProfilerActivity
from scripts.audit_real_hil_trajectory import TASK,download_file,audit_frame_table
from scripts.qualify_real_robot_cohort import (
    REVISION,METADATA_PATHS,metadata_only_rows,partition_metadata,
    deterministic_fit_sample)
from tac_osm.real_robot_predictive_state import RealRobotPredictiveState
from tac_osm.real_robot_gru_world_control import GRURobotWorldModel

FRAMES=256
GRAD_STEPS=16
CANDIDATE_HIDDEN=tuple(range(32,65))  # meaningful GRU capacity, no unused pads
SEED=52001
torch.set_num_threads(1)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_one_genuine_fit_episode(private):
    import pyarrow.parquet as pq
    root=Path(private);root.mkdir(parents=True,exist_ok=True)
    metadata={}
    for f in METADATA_PATHS:
        metadata[f]=download_file(REVISION,f,root/f,5_000_000)
    groups=partition_metadata(metadata_only_rows(metadata[METADATA_PATHS[1]]))
    original_id=deterministic_fit_sample(groups)[0]
    path=f"{TASK}/data/chunk-{original_id//1000:03d}/episode_{original_id:06d}.parquet"
    raw=download_file(REVISION,path,root/path,15_000_000)
    table=pq.read_table(io.BytesIO(raw))
    episodes={r["episode_index"]:r for r in (
        json.loads(line) for line in metadata[METADATA_PATHS[1]].splitlines())}
    evidence=audit_frame_table(table,episodes[original_id])
    if table.num_rows<FRAMES+1:
        raise ValueError("actual original FIT episode cannot support full 256-event pilot")
    # No raw data in receipt and no access to DEV/TEST parquets.
    state=torch.tensor(table["observations.state.qpos"].to_pylist()[:FRAMES+1],
                       dtype=torch.float32)
    action=torch.tensor(table["action.qpos"].to_pylist()[:FRAMES],
                        dtype=torch.float32)
    human=torch.tensor(table["intervention"].to_pylist()[:FRAMES],
                       dtype=torch.bool).unsqueeze(-1)
    return (state,action,human,
            {"episode_index":original_id,"original_publisher_parquet_SHA256":sha(raw),
             "real_state_action_frames":FRAMES,"original_metadata_frame_count":evidence["frame_count"],
             "sample_source_is_FIT":True,"source_revision":REVISION})


def nparameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def construct_models():
    torch.manual_seed(SEED)
    native=RealRobotPredictiveState()
    count=nparameters(native)
    measurements=[]
    for width in CANDIDATE_HIDDEN:
        torch.manual_seed(SEED)
        control=GRURobotWorldModel(width)
        qty=nparameters(control)
        measurements.append({"width":width,"active_parameters":qty,
                             "relative_gap":abs(qty-count)/max(qty,count)})
    closest=min(measurements,key=lambda d:(d["relative_gap"],d["width"]))
    torch.manual_seed(SEED)
    control=GRURobotWorldModel(closest["width"])
    return native,control,measurements,closest


def exercise_pre_fit(model,obs,act,human):
    """Single recurrent 256-frame burn-in with 16-gradient-step continuation.

    Native and GRU see exact original source frames and correction timing.
    No labeled outcome loss is computed or used to update any parameters.
    Data-dependent scalar on predictions solely drives FLOP/backward profiling.
    """
    model.zero_grad(set_to_none=True)
    state=model.initial(1)
    with torch.no_grad():
        for t in range(FRAMES-GRAD_STEPS):
            measured=obs[t:t+1]
            authority=human[t:t+1]
            state=model.observe(state,measured)
            forecast=model.predict(state,measured,act[t:t+1].unsqueeze(1),authority)
            # Actual x(t+1) arrives AFTER prediction; mask onset as absent here,
            # so we use only real physical state for the pilot (no reward labels).
            state=model.verify_real_feedback(
                state,act[t:t+1],authority,obs[t+1:t+2],
                forecast.next_joint_state[:,0],torch.ones(1,1,dtype=torch.bool),
                torch.full((1,1),float("nan")),
                torch.full((1,1),float("nan")),
                torch.zeros(1,1,dtype=torch.bool))
    objective=None
    for t in range(FRAMES-GRAD_STEPS,FRAMES):
        measured=obs[t:t+1]
        authority=human[t:t+1]
        state=model.observe(state,measured)
        forecast=model.predict(state,measured,act[t:t+1].unsqueeze(1),authority)
        # No fake real outcomes; squared model output provides a scalar to
        # trace backward operators without running an optimization step.
        local=forecast.next_joint_state.square().mean()+forecast.takeover_logits.square().mean()
        objective=local if objective is None else objective+local
        state=model.verify_real_feedback(
            state,act[t:t+1],authority,obs[t+1:t+2],
            forecast.next_joint_state[:,0],torch.ones(1,1,dtype=torch.bool),
            torch.full((1,1),float("nan")),
            torch.full((1,1),float("nan")),
            torch.zeros(1,1,dtype=torch.bool))
    objective.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad):
        raise ValueError("original prospective model not differentiable through entire factual path")
    return float(objective.detach())


def profile_one(model,obs,act,human):
    # Burn-in first, then trace one sample including full 256 genuine physical
    # frames; no model weight updated. One unprofiled warmup and three profiled
    # independent runs, each with same prefit seed and source batch.
    exercise_pre_fit(model,obs,act,human)
    observations=[]
    operator_receipts={}
    for trial in range(3):
        started=time.process_time()
        with profile(activities=[ProfilerActivity.CPU],record_shapes=True,with_flops=True) as p:
            exercise_pre_fit(model,obs,act,human)
        cpu=time.process_time()-started
        events=p.key_averages()
        partial=sum(int(evt.flops or 0) for evt in events)
        listed=sorted(
            ({"operator":evt.key,"profiled_flops":int(evt.flops or 0),
              "calls":evt.count} for evt in events if evt.flops),
            key=lambda row:-row["profiled_flops"])
        observations.append({"trial":trial,"profiler_cpu_process_seconds":cpu,
                             "instrumented_matrix_conv_flops_lower_bound":partial,
                             "nonzero_counted_operators":listed[:10]})
    return {"median_profiler_CPU_seconds":statistics.median(
            x["profiler_cpu_process_seconds"] for x in observations),
            "median_partial_matmul_FLOPs":statistics.median(
            x["instrumented_matrix_conv_flops_lower_bound"] for x in observations),
            "runs":observations,
            "complete_training_FLOPs_certified":False,
            "flop_instrumentation":"torch.profiler(with_flops) estimates matmul/conv only; other recurrent and backward operators missing"}


def run(private,output):
    state,action,human,source=load_one_genuine_fit_episode(private)
    native,gru,counts,closest=construct_models()
    result={
        "study":"INTEGRATED_WORLD_MODEL_PREFIT_RESOURCE_SURVEY",
        "original_source":source,
        "verified_original_fit_trajectory_used":True,
        "trainable_parameter_census":{"native":nparameters(native),
            "gru_candidates":counts,"selected_nearest_meaningful_gru":closest},
        "source_trace":{"real_frames_per_dry_run":FRAMES,
                         "real_observed_frames_for_backward":GRAD_STEPS,
                         "physical_feedback_from_actual_NEXT_frame_after_prediction":True,
                         "onset_outcome_masked_both_models":True},
        "model_weight_updates":0,"actual_training_data_sampled_for_fitting":False,
        "original_dev_test_files_opened":False,
        "no_synthetic_model_skill_evidence":True,
        "all_real_outcome_prediction_accuracy_scores_computed":False,
        "operator_complete_training_FLOPs_available":False,
        "full_resource_parity_qualified":False,
        "model_training_authorized":False,
        "result_may_not_claim_architecture_superiority":True,
        "pending":"complete validated FLOP accounting plus actual CPU parity, full equal-data training protocol and source SHA FAIR_COMPARISON manifest"
    }
    # Both pilots are always run and retained; failed parity is informative.
    result["native"]=profile_one(native,state,action,human)
    result["established_gru"]=profile_one(gru,state,action,human)
    param_ok=closest["relative_gap"]<=.01
    n_cpu=result["native"]["median_profiler_CPU_seconds"]
    g_cpu=result["established_gru"]["median_profiler_CPU_seconds"]
    cpu_gap=abs(n_cpu-g_cpu)/max(n_cpu,g_cpu)
    # Execution speed is NOT a matched-training-budget requirement.
    result["resource_fairness_diagnostics"]={
        "trainable_parameters_within_one_percent":param_ok,
        "measured_CPU_seconds_relative_difference":cpu_gap,
        "CPU_runtime_within_five_percent_DESCRIPTIVE_ONLY":cpu_gap<=.05,
        "CPU_runtime_is_not_a_fair_comparison_requirement":True,
        "matmul_partial_FLOP_parity_NOT_full_compute_certification":abs(
            result["native"]["median_partial_matmul_FLOPs"]-
            result["established_gru"]["median_partial_matmul_FLOPs"])/
            max(1,result["native"]["median_partial_matmul_FLOPs"],
                result["established_gru"]["median_partial_matmul_FLOPs"]),
        "strict_preflight_PASS":False,
        "reason":"complete compute-budget accounting and formal matched data/optimization registration not yet established; CPU runtime is diagnostic only"}
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"event":"UNTRAINED_INTEGRATED_WORLD_MODEL_REAL_FIT_PREFIT_SURVEY",
          "native_parameters":nparameters(native),"nearest_gru_parameters":nparameters(gru),
          "parameter_gap":closest["relative_gap"],"measured_CPU_gap":cpu_gap,
          "all_operator_FLOPs_covered":False,
          "architecture_training_allowed":False,"optimizer_updates":0}),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--private-cache",required=True)
    p.add_argument("--receipt",required=True)
    a=p.parse_args()
    run(a.private_cache,a.receipt)
