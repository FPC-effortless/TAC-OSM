#!/usr/bin/env python3
"""ME-057: audit ORIGINAL authentic FIT episode bytes before registration.

This script does NOT create a training manifest, train a model, read DEV/TEST
trajectory contents, or claim predictive capability. All observed hashes are
recomputed from publisher bytes rather than assumed from metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np

from scripts import me055_reserve_independent_robot_task as source

MAX_EPISODE_BYTES = 15_000_000


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_registration(info_raw, episodes_raw, frozen):
    if source.sha(info_raw) != frozen["publisher_info_sha256"]:
        raise RuntimeError("immutable original publisher info SHA does not match")
    if source.sha(episodes_raw) != frozen["publisher_episodes_metadata_sha256"]:
        raise RuntimeError("immutable publisher episode metadata SHA does not match")
    observed = source.build(info_raw, episodes_raw)
    if observed["immutable_revision"] != frozen["immutable_revision"]:
        raise RuntimeError("publisher revision changed")
    if observed["selected_task"] != frozen["selected_task"]:
        raise RuntimeError("source task differs from registered outcome-blind task")
    for cohort in ("FIT", "DEV", "TEST"):
        if observed["cohorts"][cohort] != frozen["cohorts"][cohort]:
            raise RuntimeError("source split " + cohort + " differs from frozen reservation")
    return source.reserve(source.metadata_rows(episodes_raw))


def episode_path(episode):
    return (f"{source.chosen_task()}/data/chunk-{episode // 1000:03d}/"
            f"episode_{episode:06d}.parquet")


def get_episode(episode, local_dir):
    relative=episode_path(episode)
    cache=local_dir / relative
    if cache.exists():
        raw=cache.read_bytes()
    else:
        url=(f"https://huggingface.co/datasets/{source.DATASET}/resolve/"
             f"{source.REVISION}/" + quote(relative,safe="/"))
        with urlopen(Request(url,headers={"User-Agent":"TAC-OSM-ME057-FIT-audit/1.0"}),
                     timeout=90) as response:
            raw=response.read(MAX_EPISODE_BYTES+1)
        if not raw or len(raw)>MAX_EPISODE_BYTES:
            raise RuntimeError("empty or oversized original FIT episode")
        cache.parent.mkdir(parents=True,exist_ok=True)
        cache.write_bytes(raw)
    if not raw or len(raw)>MAX_EPISODE_BYTES:
        raise RuntimeError("invalid cached original FIT episode")
    return raw


def verify_parquet(raw, expected_frames):
    import pyarrow.parquet as pq
    table=pq.read_table(io.BytesIO(raw),
          columns=["observations.state.qpos","action.qpos","intervention"])
    states=np.asarray(table["observations.state.qpos"].to_pylist(),dtype=np.float32)
    actions=np.asarray(table["action.qpos"].to_pylist(),dtype=np.float32)
    human=np.asarray(table["intervention"].to_pylist())
    if states.shape!=(expected_frames,14) or actions.shape!=states.shape:
        raise RuntimeError("authentic source dimension or episode length changed")
    if human.shape!=(expected_frames,) or human.dtype.kind!="b":
        raise RuntimeError("authentic intervention mask missing or malformed")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise RuntimeError("nonfinite publisher joint state/action values")
    if expected_frames < 274:
        raise RuntimeError("episode too short for 256-frame history plus 16 rollout")
    eligible=int(np.count_nonzero(~human[256:expected_frames-17]))
    return {"frames":int(expected_frames),
            "robot_controlled_sample_origins":eligible,
            "actual_takeover_onsets":int(np.count_nonzero((~human[:-1]) & human[1:]))}


def inspect_all_fit(frozen_path,cache_dir,output):
    frozen=json.loads(Path(frozen_path).read_text())
    task=source.chosen_task()
    info_raw=source.download(task,"info.json",1_000_000)
    episodes_raw=source.download(task,"episodes.jsonl",2_000_000)
    memberships=verify_registration(info_raw,episodes_raw,frozen)
    entries=[]
    for index,(episode,length) in enumerate(memberships["FIT"],1):
        raw=get_episode(episode,Path(cache_dir))
        stats=verify_parquet(raw,length)
        entries.append({"episode_index":episode,"sha256":source.sha(raw),**stats})
        if index%25==0:
            print(json.dumps({"event":"ME057_ORIGINAL_FIT_AUDIT_PROGRESS",
                              "audited":index,"total":len(memberships["FIT"])}),flush=True)
    if len(entries)!=frozen["cohorts"]["FIT"]["episode_count"]:
        raise RuntimeError("not all original FIT episodes audited")
    if sum(item["robot_controlled_sample_origins"] for item in entries)<100:
        raise RuntimeError("insufficient authentic controlled FIT training origins")
    manifest_digest=canonical_hash(entries)
    receipt={"schema":"tac-osm.me057.authentic-fit-source-audit.v1",
             "status":"PASS_SOURCE_AUDIT_ONLY_NO_TRAINING",
             "dataset":source.DATASET,"revision":source.REVISION,"task":task,
             "reservation_path":str(frozen_path),
             "metadata_info_sha256":source.sha(info_raw),
             "metadata_episodes_sha256":source.sha(episodes_raw),
             "episode_split_sha256":{
                 split:frozen["cohorts"][split]["episode_id_sha256"]
                 for split in ("FIT","DEV","TEST")},
             "fit_episode_count":len(entries),
             "fit_total_frames":sum(row["frames"] for row in entries),
             "fit_episode_byte_sha256_manifest":entries,
             "fit_manifest_sha256":manifest_digest,
             "fit_robot_controlled_sample_origins":sum(
                 row["robot_controlled_sample_origins"] for row in entries),
             "fit_actual_takeover_onsets":sum(
                 row["actual_takeover_onsets"] for row in entries),
             "DEV_TEST_trajectory_files_opened":0,
             "source_only_no_model_updates":True,
             "training_authorized":False}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"event":"ME057_FIT_SOURCE_AUDIT_COMPLETE",
        "episodes":len(entries),"frames":receipt["fit_total_frames"],
        "fit_manifest_sha256":manifest_digest,"training_authorized":False}),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--frozen",default="research/me055-independent-robot-task-source.json")
    p.add_argument("--cache",default="results/me057/private-original-FIT")
    p.add_argument("--output",default="results/me057/FIT_SOURCE_RECEIPT.json")
    args=p.parse_args()
    inspect_all_fit(args.frozen,args.cache,args.output)


if __name__=="__main__":
    main()
