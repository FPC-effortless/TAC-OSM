#!/usr/bin/env python3
"""Qualify ORIGINAL real-robot intervention trajectory support, never model skill.

Fetch exact official BodenAI HuggingFace dataset revision, metadata and ONE
representative parquet episode. Do not download videos, fit models, inspect
holdout predictions, synthesize missing reward or claim causal effects.
Outputs aggregate source checksums only; never original robot trajectory rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

DATASET = "BodenAI/RW-RL-HIL-Dataset"
TASK = "r1lite_scoop_cat_litter"
EXPECTED_EPISODES = 505
EXPECTED_ROBOT_DIM = 14
FILES = (
    f"{TASK}/meta/info.json",
    f"{TASK}/meta/episodes.jsonl",
    f"{TASK}/data/chunk-000/episode_000000.parquet",
)
FIELDS = ("timestamp", "frame_index", "episode_index", "intervention",
          "control_source", "action.qpos")


class SourceFailure(ValueError):
    pass


def require(value, reason):
    if not value:
        raise SourceFailure(reason)


def checksum_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def load_hub_json(url, max_bytes):
    req = Request(url, headers={"User-Agent": "TAC-OSM-source-audit/1.0"})
    with urlopen(req, timeout=90) as response:
        raw = response.read(max_bytes + 1)
    require(len(raw) <= max_bytes, "publisher response exceeded source audit byte limit")
    return json.loads(raw)


def immutable_revision():
    info = load_hub_json(
        f"https://huggingface.co/api/datasets/{DATASET}",
        2_000_000)
    sha = info.get("sha")
    require(isinstance(sha, str) and len(sha) == 40 and
            all(c in "0123456789abcdef" for c in sha.lower()),
            "publisher immutable revision SHA not available")
    require(info.get("id") == DATASET, "unexpected publisher dataset ID")
    return sha


def download_file(revision, path, dest, max_bytes):
    request = Request(
        f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/" +
        quote(path, safe="/"),
        headers={"User-Agent": "TAC-OSM-source-audit/1.0"},
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest.read_bytes() if dest.stat().st_size <= max_bytes else (
            (_ for _ in ()).throw(SourceFailure("cached publisher file too large")))
    with urlopen(request, timeout=150) as response:
        raw = response.read(max_bytes + 1)
    require(len(raw) <= max_bytes, "published file exceeds bounded metadata/sample audit budget")
    dest.write_bytes(raw)
    return raw


def audit_episodes(raw):
    """Official episode metadata: no fabricated success or recovery labels."""
    records = []
    for i, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        require(isinstance(event, dict), f"episode metadata line {i} not object")
        require(all(k in event for k in ("episode_index", "length",
                    "has_intervention", "intervention_ranges", "intervention_frames")),
                f"episode metadata line {i} required fields missing")
        index = event["episode_index"]
        length = event["length"]
        require(isinstance(index, int) and not isinstance(index,bool) and index >= 0,
                "original episode index must be nonnegative integer")
        require(isinstance(length,int) and length >= 1,
                "original episode length must be positive integer")
        segments = event["intervention_ranges"]
        require(isinstance(segments,list), "missing published intervention segments")
        total = 0
        for s in segments:
            a,b = s.get("start_frame"),s.get("end_frame")
            require(isinstance(a,int) and isinstance(b,int) and 0 <= a <= b < length,
                    "invalid human intervention bounds")
            total += b-a+1
        require(total == event["intervention_frames"],
                "original intervention ranges and frame count disagree")
        require(bool(event["has_intervention"]) == bool(total),
                "original control-authority summary contradictory")
        records.append((index, length, total))
    require(records, "no actual publisher episodes found")
    indexes=[r[0] for r in records]
    require(len(indexes)==len(set(indexes)), "original episode IDs collide")
    require(set(indexes)==set(range(len(indexes))),
            "published episode IDs must be contiguous within this task")
    return {"episode_count":len(records),
            "frames_total":sum(x[1] for x in records),
            "episodes_with_human_intervention":sum(x[2]>0 for x in records),
            "episodes_without_human_intervention":sum(x[2]==0 for x in records),
            "human_intervention_frames":sum(x[2] for x in records),
            "episode_index_contiguous":True}


def audit_frame_table(table, meta):
    columns=set(table.column_names)
    require(set(FIELDS).issubset(columns),
            "publisher per-frame table missing time/action/feedback authority")
    state_name = "observations.state.qpos" if "observations.state.qpos" in columns else (
        "observation.state.qpos" if "observation.state.qpos" in columns else None)
    require(state_name is not None, "publisher measured robot joint state missing")
    require(table.num_rows == meta["length"],
            "original measured episode frames do not match metadata")
    frames=table["frame_index"].to_pylist()
    require(frames == list(range(len(frames))),
            "publisher episode frame index must be contiguous from 0")
    epids=table["episode_index"].to_pylist()
    require(set(epids)=={meta["episode_index"]}, "publisher episode IDs mismatch")
    times=table["timestamp"].to_pylist()
    require(all(isinstance(t,(int,float)) and math.isfinite(t) for t in times)
            and all(t2>t1 for t1,t2 in zip(times,times[1:])),
            "measured trajectory timestamps not strictly chronological")
    mask=table["intervention"].to_pylist()
    authority=table["control_source"].to_pylist()
    require(all(isinstance(m,bool) for m in mask),
            "human-correction mask missing or nonbinary")
    require(all((bool(m) and a=="intervention") or
                ((not m) and a=="policy") for m,a in zip(mask,authority)),
            "policy vs human action ownership incorrectly recorded")
    require(sum(mask)==meta["intervention_frames"],
            "parquet/episode metadata intervention flags disagree")
    for field in (state_name, "action.qpos"):
        vals=table[field].to_pylist()
        require(all(v is not None and len(v)==EXPECTED_ROBOT_DIM and
                    all(isinstance(x,(float,int)) and math.isfinite(x) for x in v)
                    for v in vals),
                f"publisher {field} vector not 14 finite measured DOFs")
    forbidden=("reward","done","success")
    available_labels=[f for f in forbidden if f in columns]
    return {"frame_count":len(frames),"episode_index":meta["episode_index"],
            "measured_state_dimensions":EXPECTED_ROBOT_DIM,
            "executed_action_dimensions":EXPECTED_ROBOT_DIM,
            "time_monotonic":True,"human_control_frames":sum(mask),
            "policy_control_frames":len(mask)-sum(mask),
            "explicit_scalar_reward_done_success_columns":available_labels}


def run(root, output, expected_revision=None):
    import pyarrow.parquet as pq
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    revision=immutable_revision()
    if expected_revision:
        require(revision == expected_revision,
                "dataset HEAD changed from frozen expected revision")
    payloads={}
    for path in FILES:
        limit=15_000_000 if path.endswith(".parquet") else 5_000_000
        payloads[path]=download_file(revision,path,root/path,limit)
    info=json.loads(payloads[FILES[0]])
    summary=audit_episodes(payloads[FILES[1]])
    require(summary["episode_count"]==EXPECTED_EPISODES,
            "new source revision has changed published cat litter episode count")
    with __import__("io").BytesIO(payloads[FILES[2]]) as blob:
        table=pq.read_table(blob)
    first=json.loads(payloads[FILES[1]].splitlines()[0])
    detail=audit_frame_table(table,first)
    receipt={
        "study":"TAC_OSM_HIL_SOURCE_QUALIFICATION_NO_TRAINING",
        "dataset":DATASET,"task":TASK,"revision":revision,
        "license":"CC-BY-4.0 per publisher dataset card; verify intended downstream rights",
        "source_file_sha256":{path:checksum_bytes(payloads[path]) for path in FILES},
        "official_source_metadata_only":summary,
        "one_real_episode_state_action_feedback_audit":detail,
        "training_examples_scored":0,"optimizer_steps":0,
        "retained_trajectory_content_uploaded":False,
        "episode_boundaries_and_time_verified_for_one_real_episode":True,
        "recorded_human_intervention_not_equivalent_to_success":True,
        "no_verified_causal_counterfactual_policy_values":True,
        "no_scalar_real_reward_claim":True,
        "real_state_transition_study_possible":True,
        "integrated_decision_intelligence_qualified":False,
        "matched_parameter_data_compute_parity_qualified":False,
        "source_only_not_permission_to_fit":True
    }
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--private-cache",required=True)
    p.add_argument("--receipt",required=True)
    p.add_argument("--expected-revision")
    args=p.parse_args()
    try:
        result=run(args.private_cache,args.receipt,args.expected_revision)
        print(json.dumps({"event":"HIL_REAL_SOURCE_AUDIT_PASS",
            "revision":result["revision"],"task":TASK,
            "original_episodes":result["official_source_metadata_only"]["episode_count"],
            "first_original_episode_frames":result["one_real_episode_state_action_feedback_audit"]["frame_count"],
            "trained_models":0,"reward_or_success_certified":False}))
    except (SourceFailure,ValueError,KeyError, OSError) as e:
        print("HIL_REAL_SOURCE_QUALIFICATION_FAIL: "+str(e),file=__import__("sys").stderr)
        return 2
    return 0

if __name__=="__main__":
    raise SystemExit(main())
