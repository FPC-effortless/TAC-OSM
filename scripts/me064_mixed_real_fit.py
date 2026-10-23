"""ME064 5-task original robotic FIT-only mixed source loader and schedules.

No CAL, DEV, heldout task trajectories or exposed original source TEST are read.
Task identifiers are part of every episode key; no physical episode mixing.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
import pyarrow.parquet as pq

from scripts.audit_real_hil_trajectory import download_file, audit_frame_table
from scripts.me055_reserve_independent_robot_task import REVISION
from scripts.me057_train_fused_anchor_robot import identify, normalize, sha_bytes
from scripts.me060_train_trajectory import HISTORY, HORIZON, BATCH, UPDATES, digest

ROOT=Path("research/mathematical-engineering/me064")
TASKS=ROOT/"TASK_LEVEL_SPLIT.json"
SEEDS=(64001,64002,64003,64004,64005)


def task_registration():
    registration=json.loads(TASKS.read_text())
    if len(registration["training_tasks"]) != 5:
        raise RuntimeError("five source tasks not registered")
    if registration["holdout_task"] in registration["training_tasks"]:
        raise RuntimeError("holdout task leaks into training")
    if any(registration[key] != 0 for key in (
            "train_trajectory_files_opened","calibration_trajectory_files_opened",
            "holdout_trajectory_files_opened","training_updates")):
        raise RuntimeError("metadata-only source freeze violated")
    return registration


def get_meta(cache, task, receipt):
    info=download_file(REVISION,f"{task}/meta/info.json",
                       cache/task/"meta/info.json",1_000_000)
    episodes=download_file(REVISION,f"{task}/meta/episodes.jsonl",
                       cache/task/"meta/episodes.jsonl",2_000_000)
    if sha_bytes(info)!=receipt["info_sha256"] or (
            sha_bytes(episodes)!=receipt["episodes_metadata_sha256"]):
        raise RuntimeError(f"publisher original metadata changed: {task}")
    return {item["episode_index"]:item for item in (
        json.loads(line) for line in episodes.splitlines())}


def load_members(cache, task, members, meta):
    result={}
    for i,(episode,length) in enumerate(members,1):
        relative=f"{task}/data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet"
        raw=download_file(REVISION,relative,cache/relative,15_000_000)
        table=pq.read_table(io.BytesIO(raw))
        audit_frame_table(table,meta[episode])
        if table.num_rows!=length:
            raise RuntimeError("publisher episode length changed")
        result[(task,episode)]={
            "state":np.asarray(table["observations.state.qpos"].to_pylist(),np.float32),
            "action":np.asarray(table["action.qpos"].to_pylist(),np.float32),
            "human":np.asarray(table["intervention"].to_pylist(),np.bool_),
            "length":length,"sha256":sha_bytes(raw)}
        if i%16==0:
            print(json.dumps({"event":"ME064_AUTHENTIC_REAL_EPISODES","task":task,
                 "read":i,"total":len(members)}),flush=True)
    return result


def prepare_fit(cache):
    plan=task_registration()
    source={}
    task_hashes={}
    for task in plan["training_tasks"]:
        receipt=plan["task_evidence"][task]
        raw=load_members(cache,task,receipt["FIT"],get_meta(cache,task,receipt))
        source.update(raw)
        task_hashes[task]=digest({str(ep):rec["sha256"] for (taskname,ep),rec in raw.items()})
    if len(source)!=160:
        raise RuntimeError("did not retrieve all original multi-task real FIT episodes")
    identified=identify(source)
    norm=normalize(source,identified)
    full_hash=digest({f"{task}:{episode}":row["sha256"]
                       for (task,episode),row in sorted(source.items())})
    return plan,identified,norm,full_hash,task_hashes


def schedules(source, seed, task_names):
    # Each optimizer update has exactly eight factual windows from one task;
    # round-robin makes source domain counts equal 120 updates per task.
    # Every seed uses identical paired TAC/GRU task sequence and samples.
    rng=np.random.default_rng(seed)
    windows={}
    for task in task_names:
        windows[task]=[(key,t) for key,row in sorted(source.items())
            if key[0]==task
            for t in range(HISTORY,row["n"]-HORIZON)
            if not bool(row["human"][t])]
        if not windows[task]:
            raise RuntimeError("task lacks real autonomous state/action history: "+task)
    order=[]
    for update in range(UPDATES):
        task=task_names[update%len(task_names)]
        pool=windows[task]
        picks=[]
        for _ in range(BATCH):
            index=int(rng.integers(len(pool)))
            key,t=pool[index]
            picks.append([key[0],key[1],t])
        order.append(picks)
    return order


def batch(source, picks):
    # Use per-task compound key; return exactly the same tensors to either arm.
    return tuple(torch.stack([
        source[(task,episode)][name][t-HISTORY:t+HORIZON+1]
        for task,episode,t in picks]) for name in ("state","action","human"))


def fit_metadata(cache):
    plan,identified,source,source_hash,per_task_hashes=prepare_fit(cache)
    orders={str(seed):schedules(source,seed,plan["training_tasks"]) for seed in SEEDS}
    receipt={
        "study":"ME064_ORIGINAL_FIVE_TASK_FIT_PREFLIGHT",
        "task_level_split_sha256":digest(plan),
        "fit_original_sha256":source_hash,
        "per_task_fit_sha256":per_task_hashes,
        "per_seed_order_sha256":{seed:digest(order) for seed,order in orders.items()},
        "fit_episodes":len(source),"task_count":len(plan["training_tasks"]),
        "fit_task_episodes":{t:sum(key[0]==t for key in source) for t in plan["training_tasks"]},
        "cal_episodes_opened":0,"holdout_trajectories_opened":0,"optimizer_updates":0,
    }
    return plan,identified,source,receipt,orders
