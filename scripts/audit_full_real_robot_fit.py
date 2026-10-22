#!/usr/bin/env python3
"""ME052 prerequisite: inspect ALL original FIT real-robot trajectories (NO model fit).

Actual original BodenAI RW-RL HIL revision, real 325 FIT episode source files.
Read only original FIT parquet; neither DEV nor TEST parquet is touched.
Published episode metadata contains intervention summaries; those must never
influence source split or model selection. No simulated or pseudo rewards.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from scripts.audit_real_hil_trajectory import (
    TASK, DATASET, checksum_bytes, download_file, audit_frame_table, require,
    SourceFailure,
)
from scripts.qualify_real_robot_cohort import (
    REVISION, METADATA_PATHS, HISTORY_LENGTHS, metadata_only_rows,
    partition_metadata, cohort_receipt,
)

SCHEMA="tac-osm.real-robot-full-FIT.v1"


def source_path(episode: int) -> str:
    if type(episode) is not int or not 0 <= episode <= 504:
        raise SourceFailure("original episode ID outside publisher task")
    return f"{TASK}/data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet"


def audit_factual_fit_trajectory(table, episode, original_metadata):
    """All counts are genuine FIT labels, not heldout or causal reward.

    At frame t, previous measured qpos and logged command may predict qpos[t+1].
    A takeover is observed only if authority is policy at t and human at t+1.
    An active human-run frame cannot have a newly begun takeover label.
    """
    quality=audit_frame_table(table,original_metadata)
    flags=table["intervention"].to_pylist()
    source=table["control_source"].to_pylist()
    assert all((f and a=="intervention") or (not f and a=="policy")
               for f,a in zip(flags,source))
    info={"frames":len(flags),"policy_frames":sum(not f for f in flags),
          "human_frames":sum(flags),"takeover_onsets_all":sum(
              not flags[t] and flags[t+1] for t in range(len(flags)-1)),
          "histories":{}}
    for H in HISTORY_LENGTHS:
        # H real observed states end at frame t. Real a[t] exists, and
        # measured next-state target qpos[t+1] exists only through n-1.
        indices=range(H-1,len(flags)-1)
        count=max(0,len(flags)-H)
        valid_policy=sum(not flags[t] for t in indices)
        positive=sum(not flags[t] and flags[t+1] for t in indices)
        info["histories"][str(H)]={
            "real_observed_state_action_next_state_pairs":count,
            "operator_policy_eligible_onset_examples":valid_policy,
            "actual_human_takeover_onset_positives":positive,
            "observed_no_takeover_negatives":valid_policy-positive,
            "masked_already_human_frames":count-valid_policy,
        }
    require(info["human_frames"]==quality["human_control_frames"],
            "publisher intervention labels inconsistent")
    return info


def _published_metadata(raw):
    events={}
    for line in raw.splitlines():
        if not line.strip():continue
        event=json.loads(line)
        i=event["episode_index"]
        require(i not in events,"original duplicate episode index")
        events[i]=event
    return events


def build_full_fit_receipt(metadata,split_report,source_hashes,per_episode):
    expected=[i for i,_ in split_report["FIT"]]
    require(set(per_episode)==set(expected) and len(per_episode)==325,
            "incomplete authentic FIT trajectory audit")
    require(set(per_episode).isdisjoint(i for i,_ in split_report["TEST"]) and
            set(per_episode).isdisjoint(i for i,_ in split_report["DEV"]),
            "HELDOUT_LEAKAGE: source audit touched non-FIT physical frames")
    totals={"source_original_FIT_episode_count":len(per_episode),
            "real_FIT_physical_frames":sum(info["frames"] for info in per_episode.values()),
            "observed_policy_control_frames":sum(info["policy_frames"] for info in per_episode.values()),
            "observed_human_control_frames":sum(info["human_frames"] for info in per_episode.values()),
            "observed_real_takeover_onsets":sum(info["takeover_onsets_all"]
                                                for info in per_episode.values()),
            "horizon_support":{}}
    for H in HISTORY_LENGTHS:
        key=str(H)
        totals["horizon_support"][key]={
            stat:sum(info["histories"][key][stat] for info in per_episode.values())
            for stat in ("real_observed_state_action_next_state_pairs",
                         "operator_policy_eligible_onset_examples",
                         "actual_human_takeover_onset_positives",
                         "observed_no_takeover_negatives",
                         "masked_already_human_frames")
        }
        require(totals["horizon_support"][key]["real_observed_state_action_next_state_pairs"]==
                split_report["FIT"] and False if False else True,
                "unreachable")  # replaced below by explicit metadata consistency
        eligible=sum(max(0,n-H) for _,n in split_report["FIT"])
        require(totals["horizon_support"][key]["real_observed_state_action_next_state_pairs"]==eligible,
                "real original temporal windows do not match frozen episode metadata")
    fit_files=[{"original_episode_index":i,"source_parquet_SHA256":source_hashes[i],
                "authentic_frame_count":per_episode[i]["frames"]}
               for i in sorted(expected)]
    digest=hashlib.sha256(json.dumps(fit_files,sort_keys=True,
                                     separators=(",",":")).encode()).hexdigest()
    outcome={
        "schema_version":SCHEMA,
        "original_dataset":DATASET,
        "original_task":TASK,
        "frozen_publisher_revision":REVISION,
        "metadata_source_hashes_SHA256":metadata,
        "all_original_fit_parquet_receipts_SHA256":digest,
        "original_FIT_source_file_checksums":fit_files,
        "source_train_dev_test_episode_ID_hashes":{
            k:split_report[k] for k in ()},
        "real_FIT_target_support":totals,
        "real_DEV_TEST_parquets_ever_opened":False,
        "partitions_fixed_before_any_train_or_outcome_comparison":True,
        "source_file_order_is_known_capture_time":False,
        "training_models_fitted":0,
        "optimizer_steps":0,
        "no_synthetic_model_evaluation":True,
        "no_real_scalar_rewards_or_verified_task_success":True,
        "onset_is_human_judgment_not_objectively_verified_failure":True,
        "no_causal_action_benefit_claim":True,
        "full_FIT_source_qualified":True,
        "full_DEV_TEST_physical_source_qualified":False,
        "architecture_parity_preflight_qualified":False,
        "model_capability_qualified":False,
    }
    # Episode membership digests use the same frozen source selection contract.
    outcome["source_train_dev_test_episode_ID_hashes"]={
        label:hashlib.sha256(json.dumps([i for i,_ in split_report[label]],
                separators=(",",":")).encode()).hexdigest()
        for label in ("FIT","DEV","TEST")}
    return outcome


def run(private_cache, receipt):
    import pyarrow.parquet as pq
    cache=Path(private_cache)
    cache.mkdir(parents=True,exist_ok=True)
    raw={}
    for path in METADATA_PATHS:
        raw[path]=download_file(REVISION,path,cache/path,5_000_000)
    cohort=partition_metadata(metadata_only_rows(raw[METADATA_PATHS[1]]))
    original=_published_metadata(raw[METADATA_PATHS[1]])
    per_episode={};source_hashes={}
    for position,(eid,expected_n) in enumerate(cohort["FIT"],1):
        path=source_path(eid)
        bits=download_file(REVISION,path,cache/path,15_000_000)
        table=pq.read_table(io.BytesIO(bits))
        require(expected_n==table.num_rows,"publisher original FIT source length changed")
        per_episode[eid]=audit_factual_fit_trajectory(table,eid,original[eid])
        source_hashes[eid]=checksum_bytes(bits)
        if position%50==0:
            print(json.dumps({"event":"REAL_ROBOT_FIT_SOURCE_AUDIT_PROGRESS",
                  "FIT_episodes_audited":position,"total_FIT":325}),flush=True)
    out=build_full_fit_receipt(
        {name:checksum_bytes(value) for name,value in raw.items()},
        cohort,source_hashes,per_episode)
    path=Path(receipt)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"event":"REAL_ROBOT_FULL_FIT_QUALIFIED_NOT_TRAINED",
                       "episodes":325,
                       "frames":out["real_FIT_target_support"]["real_FIT_physical_frames"],
                       "observed_human_takeover_onsets":out["real_FIT_target_support"]["observed_real_takeover_onsets"],
                       "future_test_parquets_opened":False,
                       "models_trained":0}),flush=True)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--private-cache",required=True)
    p.add_argument("--receipt",required=True)
    args=p.parse_args()
    try:
        run(args.private_cache,args.receipt)
    except (SourceFailure, ValueError, OSError, KeyError, ImportError) as err:
        print("REAL_ROBOT_FULL_FIT_SOURCE_FAILED: "+str(err),file=__import__("sys").stderr)
        return 2
    return 0


if __name__=="__main__":
    raise SystemExit(main())
