#!/usr/bin/env python3
"""Source-only, preregistered REAL-ROBOT episodes and causal-order data contract.

DOES NOT TRAIN A MODEL. Does not access DEV/TEST parquet trajectories or
takeover labels. Cohorts are selected from episode ID and length metadata
only, independently of future rewards/control-authority labels.

Reuses published and prior-repository engineering; not a model innovation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

from audit_real_hil_trajectory import (
    DATASET, TASK, EXPECTED_EPISODES, EXPECTED_ROBOT_DIM,
    SourceFailure, require, download_file, audit_frame_table, checksum_bytes)

REVISION="9290f5a96dc91ea58256e81d9ce2dd9621890b51"
BLOCK=5
SEEDS=(52001,52002,52003,52004,52005)
HISTORY_LENGTHS=(16,64,256)
N_FIT_AUDIT=20
METADATA_PATHS=(
    f"{TASK}/meta/info.json",
    f"{TASK}/meta/episodes.jsonl",
)
COHORT_SCHEMA="tac-osm.real-robot-cohort.v1"


def _hash(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def metadata_only_rows(raw):
    """No intervention, reward, control-authority or future targets inspected."""
    rows=[]
    for i,line in enumerate(raw.splitlines()):
        if not line.strip():continue
        original=json.loads(line)
        eid=original.get("episode_index")
        length=original.get("length")
        require(type(eid) is int and eid>=0, "original episode ID must be integer")
        require(type(length) is int and length>0, "original metadata length must be positive")
        rows.append((eid,length))
    rows.sort()
    require(len(rows)==EXPECTED_EPISODES,"publisher episode count changed")
    require([eid for eid,_ in rows]==list(range(EXPECTED_EPISODES)),
            "original episode IDs contain duplicate/gaps")
    return rows


def partition_metadata(rows):
    """Disjoint 5-adjacent-episode blocks; membership independent of outcome."""
    require([i for i,_ in rows]==list(range(EXPECTED_EPISODES)),
            "wrong original episode metadata population")
    blocks=list(range(math.ceil(len(rows)/BLOCK)))
    blocks.sort(key=lambda x:(_hash(f"{REVISION}:{TASK}:v1:block:{x}"),x))
    # Frozen 65 / 16 / 20 blocks = 325 / 80 / 100 original episodes.
    # The index order is not guaranteed to be actual collection time.
    require(len(blocks)==101,"source episode-block count changed")
    indexed={"FIT":blocks[:65],"DEV":blocks[65:81],"TEST":blocks[81:]}
    result={}
    for part,ids in indexed.items():
        members=sorted((i,n) for i,n in rows if (i//BLOCK) in set(ids))
        result[part]=members
    flat=[i for members in result.values() for i,_ in members]
    require(sorted(flat)==list(range(EXPECTED_EPISODES)) and len(set(flat))==len(flat),
            "cohort cross-contamination or missing episode IDs")
    for one in ("FIT","DEV","TEST"):
        for two in ("FIT","DEV","TEST"):
            if one != two:
                require(set(i//BLOCK for i,_ in result[one]).isdisjoint(
                    i//BLOCK for i,_ in result[two]),
                    "nearby episode block crosses cohort")
    return result


def available_window_counts(members):
    return {str(h):sum(max(0,n-h-1) for _,n in members) for h in HISTORY_LENGTHS}


def cohort_receipt(rows,parts,source_hashes):
    report={"schema_version":COHORT_SCHEMA,"dataset":DATASET,
        "task":TASK,"immutable_source_revision":REVISION,
        "source_file_SHA256":source_hashes,
        "membership_does_not_use_reward_or_intervention":True,
        "source_index_not_proven_wall_clock_order":True,
        "adjacent_episode_block_size":BLOCK,
        "shared_physical_session_id_available":False,
        "strictly_sealed_dev_and_test_parquet":True,
        "frozen_seed_matrix":list(SEEDS),
        "candidate_history_lengths_real_events":list(HISTORY_LENGTHS),
        "label_contract":"observed next joint state and next-frame operator takeover only",
        "no_causal_do_action_or_counterfactual_claim":True,
        "no_reward_or_task_success_available":True,
        "equal_resource_comparison_preflight_passed":False,
        "number_of_models_trained":0,"sampled_predicted_outcomes":0}
    splits={}
    for part,members in parts.items():
        eid=[i for i,_ in members]
        members_eligible_256=[i for i,n in members if n>=258]
        splits[part]={
            "episode_ids":eid,
            "episode_count":len(eid),
            "actual_publisher_frames":sum(n for _,n in members),
            "episodes_with_at_least_257_frames":len(members_eligible_256),
            "eligible_real_one_step_pairs_by_prior_history":available_window_counts(members),
            "episode_id_SHA256":_hash(json.dumps(eid,separators=(",",":"))),
        }
    report["splits"]=splits
    require([splits[k]["episode_count"] for k in ("FIT","DEV","TEST")]==[325,80,100],
            "freeze split count integrity failure")
    require(all(splits[k]["eligible_real_one_step_pairs_by_prior_history"]["256"]>0
                for k in ("FIT","DEV","TEST")),"no genuine H256 opportunities in split")
    return report


def deterministic_fit_sample(parts):
    fitted=parts["FIT"]
    eligible=[i for i,n in fitted if n>=258]
    eligible.sort(key=lambda i:(_hash(f"{REVISION}:{TASK}:train-only-audit:{i}"),i))
    require(len(eligible)>=N_FIT_AUDIT,
            "too few authentic FIT episodes for prefit source audit")
    return sorted(eligible[:N_FIT_AUDIT])


def inspect_real_fit_episodes(root, parts, limit=N_FIT_AUDIT):
    import pyarrow.parquet as pq
    sample=deterministic_fit_sample(parts)[:limit]
    meta_file=f"{TASK}/meta/episodes.jsonl"
    episode_metadata={}
    for line in (Path(root)/meta_file).read_text().splitlines():
        event=json.loads(line)
        if event["episode_index"] in sample:
            episode_metadata[event["episode_index"]]=event
    summary={"actual_original_FIT_episodes_inspected":0,
             "original_physical_FIT_frames_validated":0,
             "source_parquet_SHA256":{},
             "human_intervention_frames_in_inspected_FIT_only":0,
             "explicit_reward_columns_found":False}
    for i in sample:
        path=f"{TASK}/data/chunk-{i//1000:03d}/episode_{i:06d}.parquet"
        raw=download_file(REVISION,path,Path(root)/path,15_000_000)
        table=pq.read_table(io.BytesIO(raw))
        quality=audit_frame_table(table,episode_metadata[i])
        require(quality["measured_state_dimensions"]==EXPECTED_ROBOT_DIM,
                "wrong authentic robot state contract")
        require(not quality["explicit_scalar_reward_done_success_columns"],
                "publisher-released reward labels changed; re-register study")
        summary["actual_original_FIT_episodes_inspected"]+=1
        summary["original_physical_FIT_frames_validated"]+=quality["frame_count"]
        summary["human_intervention_frames_in_inspected_FIT_only"]+=quality["human_control_frames"]
        summary["source_parquet_SHA256"][path]=checksum_bytes(raw)
    return summary


def run(cache,receipt):
    cache=Path(cache)
    cache.mkdir(parents=True,exist_ok=True)
    raw={}
    for p in METADATA_PATHS:
        raw[p]=download_file(REVISION,p,cache/p,5_000_000)
    rows=metadata_only_rows(raw[METADATA_PATHS[1]])
    parts=partition_metadata(rows)
    manifest=cohort_receipt(rows,parts,{p:checksum_bytes(raw[p]) for p in METADATA_PATHS})
    manifest["training_only_source_sample"]=inspect_real_fit_episodes(cache,parts)
    require(manifest["training_only_source_sample"]["actual_original_FIT_episodes_inspected"]==N_FIT_AUDIT,
            "original FIT-only real trajectories not fully audited")
    manifest["original_full_episode_parquet_validation_completed"]=False
    manifest["all_dev_test_outcome_files_remain_unopened"]=True
    manifest["source_only_not_permission_to_fit"]=True
    Path(receipt).parent.mkdir(parents=True,exist_ok=True)
    Path(receipt).write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--private-cache",required=True)
    p.add_argument("--receipt",required=True)
    args=p.parse_args()
    try:
        result=run(args.private_cache,args.receipt)
        print(json.dumps({"event":"REAL_ROBOT_COHORT_AND_TRAIN_ONLY_SOURCE_PASS",
                           "counts":{name:part["episode_count"] for name,part in result["splits"].items()},
                           "fitted_models":0,
                           "fit_episodes_with_verified_physical_records":
                               result["training_only_source_sample"]["actual_original_FIT_episodes_inspected"],
                           "all_heldout_outcome_files_unopened":True},sort_keys=True))
    except (SourceFailure, OSError,ValueError,KeyError,ImportError) as e:
        print("REAL_ROBOT_COHORT_HARD_FAIL: "+str(e),file=__import__("sys").stderr)
        return 2
    return 0

if __name__=="__main__":
    raise SystemExit(main())
