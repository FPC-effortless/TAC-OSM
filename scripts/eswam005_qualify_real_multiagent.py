"""Read-only qualification of original real INTERACTION tracked-agent observations.

NOT training or synthetic key-recall. No task-outcome/action/causal claims.
Archive SHA from public publisher mirror; data cached outside Git. Source
eligibility is determined ONLY from original recorded track CSVs, without
shuffling, relabeling futures, interpolation or generating fake interactions.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile


EXPECTED_ARCHIVE_SHA256 = "9aa78102006dc8d8a64675cf64eba5264085cfb1c006b1f4d0e856d94e657aec"
SOURCE = (
    "https://huggingface.co/datasets/INTERACTION-dataset/"
    "INTERACTION-dataset/resolve/3587152b529f26649ada0838a308eeaf77f5bdc2/"
    "INTERACTION-Dataset-TC-v1_0.zip"
)
FIELDS = ("track_id","frame_id","timestamp_ms","x","y","vx","vy")


def sha256_file(path: Path):
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_archive(path: Path):
    if not path.is_file():
        raise RuntimeError("Publisher original archive not downloaded")
    actual_sha=sha256_file(path)
    if actual_sha!=EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("Original publisher archive hash mismatch; STOP")
    tracking={}
    simultaneous=collections.Counter()
    bad_rows=collections.Counter()
    members=[]
    all_track_runs=[]
    with zipfile.ZipFile(path) as z:
        for entry in sorted(z.infolist(),key=lambda e:e.filename):
            if entry.is_dir():
                continue
            name=entry.filename
            if name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError("Archive path traversal encountered")
            # Maps, challenge scores and scripts are NOT substitute evidence.
            if not (name.lower().endswith(".csv") and
                    ("vehicle_tracks_" in name.lower() or
                     "pedestrian_tracks_" in name.lower())):
                continue
            original=collections.defaultdict(list)
            record_name=(name.replace("vehicle_tracks_","tracks_")
                             .replace("pedestrian_tracks_","tracks_"))
            valid=0
            with z.open(entry) as binary:
                reader=csv.DictReader(io.TextIOWrapper(binary,encoding="utf-8-sig",newline=""))
                if reader.fieldnames is None:
                    raise RuntimeError("Missing real track-file CSV header")
                reader.fieldnames=[v.strip() for v in reader.fieldnames]
                if not set(FIELDS).issubset(reader.fieldnames):
                    raise RuntimeError(
                        "Original trackfile schema missing "+repr(set(FIELDS)-set(reader.fieldnames))
                    )
                for row in reader:
                    try:
                        tid=str(row["track_id"]).strip()
                        frame=int(row["frame_id"])
                        timestamp=int(row["timestamp_ms"])
                        coords=tuple(float(row[k]) for k in ("x","y","vx","vy"))
                        if not tid or frame<0 or timestamp<0 or not all(math.isfinite(v) for v in coords):
                            raise ValueError("invalid source")
                    except (ValueError,TypeError,KeyError):
                        bad_rows[name]+=1
                        continue
                    valid+=1
                    original[tid].append((frame,timestamp))
                    simultaneous[(record_name,timestamp)]+=1
            lengths=[]
            for tid,events in original.items():
                seen=set()
                longest=0
                current_run=0
                previous_frame=None
                previous_timestamp=None
                # Full original chronology, sort only source events by source
                # timestamp/frame; no event shuffling or missing-frame fill.
                for frame,timestamp in sorted(events,key=lambda x:(x[1],x[0])):
                    if (frame,timestamp) in seen:
                        bad_rows[name]+=1
                        continue
                    seen.add((frame,timestamp))
                    if (previous_timestamp is not None and
                            frame==previous_frame+1 and
                            timestamp>previous_timestamp):
                        current_run+=1
                    else:
                        current_run=1
                    longest=max(longest,current_run)
                    previous_frame=frame
                    previous_timestamp=timestamp
                lengths.append(longest)
                all_track_runs.append(longest)
            members.append({"source_member":name,"original_valid_rows":valid,
                            "distinct_local_track_ids":len(original),
                            "max_uninterrupted_track_frames":max(lengths,default=0),
                            "genuine_256_frame_tracks":sum(n>=256 for n in lengths),
                            "genuine_128_frame_tracks":sum(n>=128 for n in lengths)})
            tracking[name]=len(original)
    if not members or not sum(r["original_valid_rows"] for r in members):
        raise RuntimeError("Original camera-source contains no trackfile records; STOP")
    eligible_long=sum(n>=256 for n in all_track_runs)
    output={
        "study":"ESWAM005_INTERACTION_AUTHENTIC_MULTIAGENT_SOURCE_QUALIFICATION",
        "source_url":SOURCE,
        "publisher_archive_sha256":actual_sha,
        "license_as_published_on_hf_dataset_card":"CC-BY-4.0",
        "zip_original":path.name,
        "source_origin":"Recorded real traffic participant trajectories; no synthetic motion",
        "real_observations_present":True,
        "distinct_recorded_csv_members":len(members),
        "total_real_track_rows":sum(r["original_valid_rows"] for r in members),
        "total_distinct_track_ids_scoped_to_member":sum(tracking.values()),
        "real_tracks_with_uninterrupted_128_or_more_frames":sum(n>=128 for n in all_track_runs),
        "real_tracks_with_uninterrupted_256_or_more_frames":eligible_long,
        "max_original_uninterrupted_agent_frames":max(all_track_runs,default=0),
        "real_overlapping_timestamp_samples_at_least_2_agents":sum(v>=2 for v in simultaneous.values()),
        "real_overlapping_timestamp_samples_at_least_4_agents":sum(v>=4 for v in simultaneous.values()),
        "invalid_or_duplicate_source_rows":dict(bad_rows),
        "members":members,
        "source_only_no_fitting":True,
        "retrospective_real_data_not_interventional_policy":True,
        "full_episode_task_success_or_decision_cost_available":False,
        "distinct_train_dev_test_split_registered":False,
        "real_long_256_memory_eligibility":eligible_long>0 and sum(v>=4 for v in simultaneous.values())>0,
        "independent_long_memory_capability_established":False,
        "real_world_decision_intelligence_established":False,
        "next_step_if_eligible":"Register held-out scene/recording splits and a real future-state task; do not train on archive as a benchmark until source split is frozen.",
    }
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=inspect_archive(args.archive)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("ESWAM005_SOURCE_QUALIFICATION="+json.dumps({
        k:report[k] for k in (
        "publisher_archive_sha256","distinct_recorded_csv_members",
        "total_real_track_rows","real_tracks_with_uninterrupted_256_or_more_frames",
        "real_overlapping_timestamp_samples_at_least_4_agents",
        "real_long_256_memory_eligibility")}),flush=True)
