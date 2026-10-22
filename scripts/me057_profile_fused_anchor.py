#!/usr/bin/env python3
"""Strict paired preflight for the ME-057 fused predictive-anchor TAC.

No model is fitted.  Both arms consume one checksum-locked authentic FIT
episode, identical history/rollout frames, and the same optimizer exposure.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from scripts.me055_profile_prefit import (
    HISTORY, HORIZON, authentic_batch, build, load_authentic, objective,
    physical_identification, sha_file, update)
from scripts.me056_profile_paired_execution import percentile_interval
from tac_osm.real_robot_me055_rollout import (
    TrajectoryForecast, factual_multistep_loss)
from tac_osm.real_robot_me057_fused_anchor import FusedAnchorServoRolloutTAC


def fused_objective(model, batch, kernels=None):
    states, actions, human, _ = batch
    if kernels is None:
        burnin = model._burnin_training
        boundary = model._burnin_boundary_training
        observe = model._observe_training
        predict = model._predict_training
        dense = model._dense_transition_auxiliary_training
    else:
        burnin, boundary, observe, predict, dense = kernels
    belief = model.initial_from_observation(states[:, 0])
    with torch.no_grad():
        for step in range(HISTORY):
            transition = boundary if (step+1) % 128 == 0 else burnin
            belief = transition(
                belief, states[:, step], actions[:, step], human[:, step],
                states[:, step+1], human[:, step+1])
    # Train the final persistent belief to retain dynamics from the recent
    # authentic segment.  Expanding one terminal belief avoids replay-state
    # materialization and makes this a direct retention objective.
    count = 144
    fast = belief.fast[:, None].expand(-1, count, -1).reshape(-1, 32)
    slow = belief.slow[:, None].expand(-1, count, -1).reshape(-1, 32)
    memory = belief.memory[:, None].expand(-1, count, -1).reshape(-1, 8)
    start = HISTORY-144
    def selected_frames(value, offset=0):
        width = value.shape[-1]
        return value[:, start+offset:HISTORY+offset].reshape(-1, width)
    auxiliary = dense(
        fast, slow, memory, selected_frames(states),
        selected_frames(states, 1), selected_frames(actions),
        selected_frames(human), selected_frames(human, 1))
    current = states[:, HISTORY]
    control = human[:, HISTORY]
    future_human = human[:, HISTORY:HISTORY+HORIZON+1]
    alive = (~future_human[:, :-1]).to(torch.int64).cumprod(1).bool()
    onset = (~future_human[:, :-1] & future_human[:, 1:]).to(current.dtype)
    predicted_states, predicted_logits = [], []
    for step in range(HORIZON):
        belief = observe(belief, current)
        prediction = predict(
            belief, current, actions[:, HISTORY+step:HISTORY+step+1], control)
        current = prediction.next_joint_state[:, 0]
        predicted_states.append(current)
        predicted_logits.append(prediction.takeover_logits[:, 0])
    forecast = TrajectoryForecast(
        torch.stack(predicted_logits, 1), torch.stack(predicted_states, 1))
    primary = factual_multistep_loss(
        model, forecast, states[:, HISTORY+1:HISTORY+HORIZON+1],
        alive, onset, alive)[0]
    return primary+.05*auxiliary


def candidate(identified):
    return FusedAnchorServoRolloutTAC(**{
        name: value.tolist() for name, value in identified.items()})


def compiled_runner(model, batch, architecture, compile_mode):
    methods = [model._burnin_training, model._observe_training,
               model._predict_training]
    if architecture == "native_tac":
        methods.insert(1, model._burnin_boundary_training)
        methods.append(model._dense_transition_auxiliary_training)
    kernels = tuple(torch.compile(
        method, dynamic=False, fullgraph=True, mode=compile_mode)
        for method in methods)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=.001, weight_decay=.0001)
    loss = ((lambda: fused_objective(model, batch, kernels))
            if architecture == "native_tac"
            else (lambda: objective(model, batch, kernels, use_scan=False)))
    return optimizer, loss


def eager_flops(architecture, identified, batch):
    torch.manual_seed(57001)
    model = candidate(identified) if architecture == "native_tac" else build(
        architecture, identified)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=.001, weight_decay=.0001)
    loss = ((lambda: fused_objective(model, batch))
            if architecture == "native_tac" else None)
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as recording:
        update(model, batch, optimizer, loss)
    result = int(sum(event.flops for event in recording.key_averages()))
    if result <= 0:
        raise RuntimeError("PyTorch profiler emitted no operator FLOPs")
    return result


def relative_gap(left, right):
    return abs(left-right)/max(left, right)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fit-parquet", required=True)
    cli.add_argument("--expected-sha256", required=True)
    cli.add_argument("--output", required=True)
    cli.add_argument("--pairs", type=int, default=41)
    cli.add_argument("--warmup", type=int, default=5)
    cli.add_argument("--batch-size", type=int, default=2)
    cli.add_argument("--compile-mode", default="reduce-overhead",
                     choices=("default", "reduce-overhead", "max-autotune"))
    args = cli.parse_args()
    if args.pairs < 11 or args.warmup < 2:
        raise SystemExit("at least 11 pairs and two warmups required")
    if sha_file(args.fit_parquet) != args.expected_sha256:
        raise SystemExit("authentic FIT profiler episode SHA mismatch")
    affinity = []
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {allowed[0]})
        affinity = [allowed[0]]
    torch.set_num_threads(1)
    state, action, human = load_authentic(args.fit_parquet)
    identified = physical_identification(state, action)
    batch = authentic_batch(
        state, action, human, identified, args.batch_size)
    runners, compile_cpu, parameters = {}, {}, {}
    for architecture in ("native_tac", "gru_world"):
        torch.manual_seed(57001)
        model = candidate(identified) if architecture == "native_tac" else build(
            architecture, identified)
        parameters[architecture] = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        optimizer, loss = compiled_runner(
            model, batch, architecture, args.compile_mode)
        started = time.process_time_ns()
        update(model, batch, optimizer, loss)
        compile_cpu[architecture] = (time.process_time_ns()-started)/1e9
        for _ in range(args.warmup-1):
            update(model, batch, optimizer, loss)
        runners[architecture] = (model, optimizer, loss)
    flops = {name: eager_flops(name, identified, batch)
             for name in runners}
    durations = {name: [] for name in runners}
    paired = []
    for pair in range(args.pairs):
        order = (("native_tac", "gru_world") if pair % 2 == 0
                 else ("gru_world", "native_tac"))
        measured = {}
        for architecture in order:
            model, optimizer, loss = runners[architecture]
            started = time.process_time_ns()
            update(model, batch, optimizer, loss)
            elapsed = (time.process_time_ns()-started)/1e9
            durations[architecture].append(elapsed)
            measured[architecture] = elapsed
        paired.append(measured["native_tac"]/measured["gru_world"]-1.)
    interval = percentile_interval(paired, seed=57001)
    receipt = {
        "study": "ME057_FUSED_PREDICTIVE_ANCHOR_PREFLIGHT",
        "capability_claim": False,
        "model_fits_started": 0,
        "source_split": "FIT", "source_episode": 12,
        "source_sha256": args.expected_sha256,
        "cpu_affinity": affinity, "torch_threads": torch.get_num_threads(),
        "warmup_updates_per_arm": args.warmup,
        "paired_updates": args.pairs, "batch_size": args.batch_size,
        "alternating_arm_order": True,
        "timer": "time.process_time_ns",
        "compiler": f"torch.compile fullgraph {args.compile_mode} per transition kernel",
        "compile_cpu_seconds": compile_cpu,
        "parameter_counts": parameters,
        "parameter_relative_gap": relative_gap(*parameters.values()),
        "profiled_update_flops": flops,
        "profiled_flop_relative_gap": relative_gap(*flops.values()),
        "cpu_seconds_samples": durations,
        "paired_TAC_over_GRU_relative_samples": paired,
        "paired_median_TAC_over_GRU_relative_gap": float(
            statistics.median(paired)),
        "paired_median_bootstrap_95_interval": interval,
        "strict_parameter_gate_pass": relative_gap(*parameters.values()) <= .01,
        "strict_flop_gate_pass": relative_gap(*flops.values()) <= .01,
        "strict_5pct_gate_pass": interval[1] <= .05,
        "strict_5pct_gate_definitive_fail": interval[0] > .05,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(receipt, sort_keys=True, indent=2)+"\n")
    print(json.dumps({key: receipt[key] for key in (
        "parameter_relative_gap", "profiled_flop_relative_gap",
        "paired_median_TAC_over_GRU_relative_gap",
        "paired_median_bootstrap_95_interval",
        "strict_parameter_gate_pass", "strict_flop_gate_pass",
        "strict_5pct_gate_pass", "strict_5pct_gate_definitive_fail")}))
    return 0 if all(receipt[key] for key in (
        "strict_parameter_gate_pass", "strict_flop_gate_pass",
        "strict_5pct_gate_pass")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
