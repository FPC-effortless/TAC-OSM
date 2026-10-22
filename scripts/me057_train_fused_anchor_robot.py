#!/usr/bin/env python3
"""Train and evaluate ME-057 on the reserved authentic robot task.

All ten models train only on FIT trajectories.  Their checkpoint hashes are
frozen before any DEV or TEST parquet is downloaded.  Evaluation is factual,
open-loop, action-conditioned prediction before logged human takeover; it is
not a causal policy-value or autonomous-control claim.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from scripts.audit_real_hil_trajectory import audit_frame_table, download_file
from scripts.me055_profile_prefit import HISTORY, HORIZON
from scripts.me055_reserve_independent_robot_task import (
    REVISION, chosen_task, metadata_rows, reserve)
from scripts.me057_profile_fused_anchor import (
    candidate, fused_objective, gru_objective)
from scripts.validate_research_parity import check_manifest
from tac_osm.real_robot_me055_rollout import ServoRolloutGRU

torch.set_num_threads(1)
DATASET_DIGEST = "6576fc10753e3724791f6327c4fcad5feb5db801bebd090a8fc162776a5f1395"
REGISTRATION = Path("research/mathematical-engineering/me057/FAIR_COMPARISON.json")
PREFLIGHT = Path("research/me057-fused-anchor-preflight.json")
SEEDS = (57001, 57002, 57003, 57004, 57005)
ARMS = ("native_tac", "gru_world")
UPDATES, BATCH = 600, 2


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+"\n")


def source_index(cache):
    task = chosen_task()
    info_path = f"{task}/meta/info.json"
    episodes_path = f"{task}/meta/episodes.jsonl"
    info_raw = download_file(REVISION, info_path, cache/info_path, 1_000_000)
    episodes_raw = download_file(
        REVISION, episodes_path, cache/episodes_path, 2_000_000)
    if sha_bytes(episodes_raw) != "780a1712f20666d629a291d36c3d5d3d393ed5fdbe109b03dce763a910e20c8a":
        raise RuntimeError("reserved publisher episode metadata changed")
    info = json.loads(info_raw)
    raw_rows = [json.loads(line) for line in episodes_raw.splitlines() if line]
    metadata = {row["episode_index"]: row for row in raw_rows}
    split = reserve(metadata_rows(episodes_raw))
    if info.get("total_episodes") != len(metadata):
        raise RuntimeError("publisher metadata count changed")
    return split, metadata


def load_episode(cache, episode, metadata):
    task = chosen_task()
    relative = f"{task}/data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet"
    raw = download_file(REVISION, relative, cache/relative, 15_000_000)
    import pyarrow.parquet as pq
    table = pq.read_table(io.BytesIO(raw))
    audit_frame_table(table, metadata[episode])
    state = np.asarray(
        table["observations.state.qpos"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action.qpos"].to_pylist(), dtype=np.float32)
    human = np.asarray(table["intervention"].to_pylist(), dtype=np.bool_)
    return {"state": state, "action": action, "human": human,
            "sha256": sha_bytes(raw), "length": len(state)}


def load_part(cache, members, metadata, name):
    result = {}
    for number, (episode, expected) in enumerate(members, 1):
        row = load_episode(cache, episode, metadata)
        if row["length"] != expected:
            raise RuntimeError("publisher episode length changed")
        result[episode] = row
        if number % 25 == 0:
            print(json.dumps({"event": "ME057_SOURCE", "split": name,
                              "episodes": number, "total": len(members)}), flush=True)
    return result


def identify(fit):
    states = np.concatenate([row["state"] for row in fit.values()])
    actions = np.concatenate([row["action"] for row in fit.values()])
    state_mean = states.mean(0, dtype=np.float64)
    state_scale = np.maximum(states.std(0, dtype=np.float64), .05)
    action_mean = actions.mean(0, dtype=np.float64)
    action_scale = np.maximum(actions.std(0, dtype=np.float64), .05)
    normalized = (states-state_mean)/state_scale
    normalized_action = (actions-action_mean)/action_scale
    target = (normalized_action*action_scale+action_mean-state_mean)/state_scale
    delta = normalized[1:]-normalized[:-1]
    # The concatenation creates at most 227 boundary transitions.  Servo gain
    # is therefore computed episode-wise below and averaged by true frames.
    numerator = np.zeros(14); denominator = np.zeros(14)
    genuine_delta = []
    for row in fit.values():
        x = (row["state"]-state_mean)/state_scale
        a = (row["action"]-action_mean)/action_scale
        a = (a*action_scale+action_mean-state_mean)/state_scale
        d = x[1:]-x[:-1]; error = a[:-1]-x[:-1]
        numerator += (error*d).sum(0); denominator += np.square(error).sum(0)
        genuine_delta.append(d)
    genuine_delta = np.concatenate(genuine_delta)
    return {
        "delta_scale": np.maximum(np.abs(genuine_delta).mean(0), .0005).astype(np.float32),
        "state_mean": state_mean.astype(np.float32),
        "state_scale": state_scale.astype(np.float32),
        "action_mean": action_mean.astype(np.float32),
        "action_scale": action_scale.astype(np.float32),
        "servo_gain": np.clip(numerator/(denominator+1e-8), 0, .25).astype(np.float32),
        "servo_step_limit": np.maximum(
            np.quantile(np.abs(genuine_delta), .99, axis=0),
            np.abs(genuine_delta).mean(0)).astype(np.float32),
    }


def normalize(source, identified):
    sm, ss = identified["state_mean"], identified["state_scale"]
    am, ast = identified["action_mean"], identified["action_scale"]
    return {episode: {
        "state": torch.from_numpy(((row["state"]-sm)/ss).astype(np.float32)),
        "action": torch.from_numpy(((row["action"]-am)/ast).astype(np.float32)),
        "human": torch.from_numpy(row["human"].copy()), "n": row["length"]}
        for episode, row in source.items()}


def schedule(members, seed):
    eligible = [(episode, start) for episode, length in members
                for start in range(length-(HISTORY+HORIZON+1)+1)]
    rng = np.random.default_rng(seed+200)
    picks = [(eligible[int(rng.integers(len(eligible)))],
              eligible[int(rng.integers(len(eligible)))]) for _ in range(UPDATES)]
    digest = sha_bytes(json.dumps(picks, separators=(",", ":")).encode())
    return picks, digest


def batch(source, pair):
    states = torch.stack([source[ep]["state"][start:start+HISTORY+HORIZON+1]
                          for ep, start in pair])
    actions = torch.stack([source[ep]["action"][start:start+HISTORY+HORIZON]
                           for ep, start in pair])
    human = torch.stack([source[ep]["human"][start:start+HISTORY+HORIZON+1]
                         for ep, start in pair]).unsqueeze(-1)
    return states, actions, human, pair


def build(name, identified):
    values = {key: value.tolist() for key, value in identified.items()}
    return candidate(identified) if name == "native_tac" else ServoRolloutGRU(**values)


def kernels(model, name):
    methods = [model._burnin_training, model._observe_training,
               model._predict_training]
    if name == "native_tac":
        methods.insert(1, model._burnin_boundary_training)
        methods.append(model._dense_transition_auxiliary_training)
    # The registered model is defined by these transitions, not by a compiler
    # wrapper. Inductor aborted before the first fit in Actions run
    # 39858169855; eager execution preserves the equations and is stable.
    return tuple(methods)


def loss(model, name, source_batch, compiled):
    if name == "native_tac":
        return fused_objective(model, source_batch, compiled)
    return gru_objective(model, source_batch, compiled)


def fit_one(name, seed, fit, picks, identified, folder, registration_row):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(name, identified)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    compiled = kernels(model, name)
    first = batch(fit, picks[0])
    # Compile and allocate outside the registered optimizer timer; no weight is
    # changed and no additional sample is selected.
    optimizer.zero_grad(set_to_none=True)
    loss(model, name, first, compiled).backward()
    optimizer.zero_grad(set_to_none=True)
    started = time.process_time()
    trace = []
    for step, pair in enumerate(picks, 1):
        source_batch = batch(fit, pair)
        optimizer.zero_grad(set_to_none=True)
        value = loss(model, name, source_batch, compiled)
        if not torch.isfinite(value):
            raise RuntimeError("nonfinite authentic training objective")
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite authentic training gradient")
        optimizer.step()
        if step == 1 or step % 100 == 0:
            trace.append({"update": step, "loss": float(value.detach()),
                          "optimizer_cpu_seconds": time.process_time()-started})
            print(json.dumps({"event": "ME057_TRAIN", "arm": name,
                "seed": seed, "update": step, "loss": float(value.detach())}), flush=True)
    elapsed = time.process_time()-started
    target = folder/f"{name}-s{seed}.pt"
    torch.save({"architecture": name, "seed": seed,
                "model": model.state_dict(),
                "identified": {k: v.tolist() for k, v in identified.items()},
                "trace": trace}, target)
    return {"architecture": name, "seed": seed, "path": target.name,
            "checkpoint_sha256": sha_file(target),
            "optimizer_cpu_seconds": elapsed,
            "registration_row": registration_row}


def score_model(model, source, identified, name):
    scale = torch.from_numpy(identified["state_scale"])
    episodes = {}
    model.eval()
    with torch.no_grad():
        for episode, row in sorted(source.items()):
            candidates = [t for t in range(HISTORY, row["n"]-HORIZON)
                          if not bool(row["human"][t])]
            if not candidates:
                continue
            origin = candidates[len(candidates)//2]
            belief = (model.initial_from_observation(row["state"][origin-HISTORY:origin-HISTORY+1])
                      if name == "native_tac" else model.initial(1))
            for local, t in enumerate(range(origin-HISTORY, origin)):
                values = (belief, row["state"][t:t+1], row["action"][t:t+1],
                          row["human"][t:t+1, None], row["state"][t+1:t+2],
                          row["human"][t+1:t+2, None])
                if name == "native_tac" and (local+1) % 128 == 0:
                    belief = model._burnin_boundary_training(*values)
                else:
                    belief = model._burnin_training(*values)
            current = row["state"][origin:origin+1]
            control = torch.zeros(1, 1, dtype=torch.bool)
            predicted = []
            for h in range(HORIZON):
                belief = model._observe_training(belief, current)
                one = model._predict_training(
                    belief, current, row["action"][origin+h:origin+h+1, None], control)
                current = one.next_joint_state[:, 0]
                predicted.append(current[0])
            predicted = torch.stack(predicted)
            actual = row["state"][origin+1:origin+HORIZON+1]
            authority = row["human"][origin:origin+HORIZON]
            alive = (~authority).to(torch.int64).cumprod(0).bool()
            error = ((predicted-actual).abs()*scale)[alive]
            persistence = ((row["state"][origin]-actual).abs()*scale)[alive]
            episodes[str(episode)] = {
                "origin": origin, "scored_steps": int(alive.sum()),
                "physical_MAE": float(error.mean()),
                "persistence_MAE": float(persistence.mean())}
    return episodes


def paired_interval(values, seed=57057, samples=20_000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (samples, len(values)))].mean(1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def run(output):
    folder = Path(output); folder.mkdir(parents=True, exist_ok=False)
    cache = folder/"private-source-not-uploaded"; cache.mkdir()
    plan = json.loads(REGISTRATION.read_text())
    receipt = json.loads(PREFLIGHT.read_text())
    if not all(receipt[key] for key in ("strict_parameter_gate_pass",
            "strict_flop_gate_pass")):
        raise RuntimeError("registered resource gate is not passed")
    checked = check_manifest(plan, "preflight")
    split, metadata = source_index(cache)
    fit_raw = load_part(cache, split["FIT"], metadata, "FIT")
    identified = identify(fit_raw)
    fit = normalize(fit_raw, identified)
    prepared = {seed: schedule(split["FIT"], seed) for seed in SEEDS}
    records = []
    for seed in SEEDS:
        picks, digest = prepared[seed]
        for name in ARMS:
            row = next(item for item in plan["arms"]
                       if item["architecture"] == name and item["seed"] == seed)
            if digest != row["training_sample_order_sha256"]:
                raise RuntimeError("preregistered sample order changed")
            records.append(fit_one(name, seed, fit, picks, identified, folder, row))
    report = {"study": plan["study_id"], "preflight": checked,
              "all_ten_checkpoints_frozen_before_DEV_TEST": True,
              "checkpoints": records, "status": "FROZEN_BEFORE_HELDOUT"}
    write_json(folder/"results.json", report)
    del fit, fit_raw
    evaluations = {}
    for part in ("DEV", "TEST"):
        raw = load_part(cache, split[part], metadata, part)
        normalized = normalize(raw, identified)
        evaluations[part] = {}
        for record in records:
            model = build(record["architecture"], identified)
            state = torch.load(folder/record["path"], map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"], strict=True)
            key = f'{record["architecture"]}-s{record["seed"]}'
            evaluations[part][key] = score_model(
                model, normalized, identified, record["architecture"])
        del normalized, raw
    test = evaluations["TEST"]
    episode_ids = sorted(next(iter(test.values())), key=int)
    tac_by_episode = []; gru_by_episode = []; persistence_by_episode = []
    for episode in episode_ids:
        tac_by_episode.append(np.mean([test[f"native_tac-s{s}"][episode]["physical_MAE"] for s in SEEDS]))
        gru_by_episode.append(np.mean([test[f"gru_world-s{s}"][episode]["physical_MAE"] for s in SEEDS]))
        persistence_by_episode.append(test[f"gru_world-s{SEEDS[0]}"][episode]["persistence_MAE"])
    tac_mean, gru_mean = float(np.mean(tac_by_episode)), float(np.mean(gru_by_episode))
    persistence_mean = float(np.mean(persistence_by_episode))
    difference = np.asarray(tac_by_episode)-np.asarray(gru_by_episode)
    summary = {"native_TAC_mean_MAE": tac_mean, "GRU_mean_MAE": gru_mean,
        "persistence_mean_MAE": persistence_mean,
        "TAC_minus_GRU_episode_bootstrap_95_interval": paired_interval(difference),
        "TAC_seed_wins": sum(np.mean([test[f"native_tac-s{s}"][e]["physical_MAE"] for e in episode_ids]) <
                             np.mean([test[f"gru_world-s{s}"][e]["physical_MAE"] for e in episode_ids]) for s in SEEDS)}
    summary["registered_accuracy_gate_pass"] = (
        tac_mean <= .98*gru_mean and tac_mean <= .98*persistence_mean and
        summary["TAC_minus_GRU_episode_bootstrap_95_interval"][1] < 0)
    final = json.loads(REGISTRATION.read_text())
    for row in final["arms"]:
        record = next(r for r in records if r["architecture"] == row["architecture"] and r["seed"] == row["seed"])
        row["measured"] = {"actual_steps": UPDATES, "actual_examples": UPDATES*BATCH,
            "actual_tokens": UPDATES*BATCH*(HISTORY+HORIZON),
            "training_flops": row["budget"]["planned_training_flops"],
            "optimizer_cpu_seconds": record["optimizer_cpu_seconds"],
            "peak_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
            "flops_profiler": "registered torch.profiler CPU with_flops full update receipt",
            "checkpoint_sha256": record["checkpoint_sha256"],
            "actual_sample_order_sha256": row["training_sample_order_sha256"],
            "evaluation_split_sha256": row["test_split_sha256"]}
    final_check = check_manifest(final, "final")
    write_json(folder/"FAIR_COMPARISON_FINAL.json", final)
    report.update({"status": "COMPLETE", "evaluation": evaluations,
                   "test_summary": summary, "final_parity": final_check,
                   "no_policy_value_or_autonomous_control_claim": True})
    write_json(folder/"results.json", report)
    print(json.dumps({"event": "ME057_COMPLETE", **summary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/me057")
    args = parser.parse_args()
    run(args.output)
