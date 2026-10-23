"""ESWAM006: genuine 256-frame tracked vehicle physical future prediction.

No model fitting or synthetic sequences. Only original source real positions,
real measured *past* velocities and a previously recorded past timestamp step.
Compare fixed nonlearned physical priors before proposing trained long memory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
import json
import math
from pathlib import Path
import random
import statistics
import zipfile

from scripts.eswam005_qualify_real_multiagent import EXPECTED_ARCHIVE_SHA256, sha256_file

ROOT=Path("research/mathematical-engineering/eswam006")
HORIZONS=(10,30)
HISTORY=256
STEP=10
PRIOR_NAMES=("persistence","instantaneous_velocity","eight_frame_velocity","twofiftysix_frame_velocity")


def load_dev_tracks(archive: Path):
    if sha256_file(archive)!=EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("original publisher zip hash mismatch, STOP")
    tracks=defaultdict(list)
    matched=[]
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if (name.startswith("__MACOSX/") or "/__MACOSX/" in name
                    or Path(name).name.startswith("._")):
                continue
            if not name.endswith("vehicle_tracks_003.csv"):
                continue
            if "TC_BGR_Intersection_VA" not in name:
                raise RuntimeError("original registered location changed")
            matched.append(name)
            with z.open(name) as f:
                reader=csv.DictReader(io.TextIOWrapper(f,encoding="utf-8-sig",newline=""))
                if not {"track_id","frame_id","timestamp_ms","x","y","vx","vy"}.issubset(reader.fieldnames or []):
                    raise RuntimeError("required publisher real observation fields missing")
                for record in reader:
                    key=(name,str(record["track_id"]))
                    frame=int(record["frame_id"])
                    ms=int(record["timestamp_ms"])
                    data=tuple(float(record[k]) for k in ("x","y","vx","vy"))
                    if not all(math.isfinite(v) for v in data):
                        raise RuntimeError("nonfinite original real actor state")
                    tracks[key].append((frame,ms,*data))
    if len(matched)!=1:
        raise RuntimeError("exact one prespecified real DEV member required")
    return tracks,matched[0]


def make_contiguous(raw):
    consecutive=[]
    current=[]
    last=None
    for row in sorted(raw,key=lambda a:(a[1],a[0])):
        if last is not None and (row[0]!=last[0]+1 or row[1]<=last[1]):
            if current:consecutive.append(current)
            current=[]
        current.append(row)
        last=row
    if current:consecutive.append(current)
    return consecutive


def forecast_past_only(history, horizon):
    if len(history)!=HISTORY:
        raise RuntimeError("complete genuine history needed")
    last=history[-1]
    delta=(last[1]-history[-2][1])/1000
    if not (delta>0 and math.isfinite(delta)):
        raise RuntimeError("invalid source observed time interval")
    x,y=last[2:4]
    v1=last[4:6]
    v8=tuple(sum(row[4+j] for row in history[-8:])/8 for j in (0,1))
    v256=tuple(sum(row[4+j] for row in history)/HISTORY for j in (0,1))
    return {
        "persistence":(x,y),
        "instantaneous_velocity":(x+horizon*delta*v1[0],y+horizon*delta*v1[1]),
        "eight_frame_velocity":(x+horizon*delta*v8[0],y+horizon*delta*v8[1]),
        "twofiftysix_frame_velocity":(x+horizon*delta*v256[0],y+horizon*delta*v256[1]),
    },delta


def score_real(archive:Path):
    tracks,source_member=load_dev_tracks(archive)
    observations=[]
    for (member,track),raw in sorted(tracks.items()):
        for segment in make_contiguous(raw):
            for horizon in HORIZONS:
                # Training context and future must be genuine original
                # consecutive track frames, not synthesized padding/fill.
                if len(segment)<HISTORY+horizon:
                    continue
                for index in range(HISTORY-1,len(segment)-horizon,STEP):
                    history=segment[index-HISTORY+1:index+1]
                    future=segment[index+horizon]
                    estimates,delta=forecast_past_only(history,horizon)
                    true=(future[2],future[3])
                    errors={}
                    for name,pred in estimates.items():
                        euclidean=math.hypot(pred[0]-true[0],pred[1]-true[1])
                        coordinate_mae=(abs(pred[0]-true[0])+abs(pred[1]-true[1]))/2
                        errors[name]={"l2_meters":euclidean,"xy_mae_meters":coordinate_mae}
                    observations.append({
                        "recorded_track_id":track,
                        "source_member":member,
                        "original_current_frame":segment[index][0],
                        "original_current_timestamp_ms":segment[index][1],
                        "original_future_frame":future[0],
                        "registered_future_horizon_frames":horizon,
                        "known_prior_frame_interval_ms":delta*1000,
                        "original_source_physically_measured_future_xy":true,
                        "original_fixed_baseline_predictions":estimates,
                        "real_errors":errors,
                    })
    if not observations:
        raise RuntimeError("No original >=256-plus-real-future trajectories in DEV recording")
    summaries={}
    rng=random.Random(6006)
    for horizon in HORIZONS:
        cases=[r for r in observations if r["registered_future_horizon_frames"]==horizon]
        if not cases:
            summaries[str(horizon)]={"genuine_queries":0,"status":"NOT_QUALIFIED"}
            continue
        groups=defaultdict(list)
        for row in cases:
            groups[row["recorded_track_id"]].append(row)
        results={}
        for name in PRIOR_NAMES:
            vals=[r["real_errors"][name] for r in cases]
            results[name]={
                "l2_meters":sum(x["l2_meters"] for x in vals)/len(vals),
                "xy_mae_meters":sum(x["xy_mae_meters"] for x in vals)/len(vals),
            }
        # Paired full actor-ID resampling; do not treat overlapping frames as
        # independent source events. Fixed prior comparison, not tuned model.
        ids=sorted(groups)
        paired=[]
        for repeat in range(2000):
            selected=[rng.choice(ids) for _ in ids]
            pooled=[r for tid in selected for r in groups[tid]]
            diff=sum(
                r["real_errors"]["twofiftysix_frame_velocity"]["l2_meters"]
                -r["real_errors"]["eight_frame_velocity"]["l2_meters"]
                for r in pooled
            )/len(pooled)
            paired.append(diff)
        paired.sort()
        summaries[str(horizon)]={
            "genuine_queries":len(cases),
            "distinct_original_actor_tracks":len(groups),
            "fixed_baselines":results,
            "mean_256_minus_8_l2_meters":(
                results["twofiftysix_frame_velocity"]["l2_meters"]
                -results["eight_frame_velocity"]["l2_meters"]),
            "original_actor_cluster_paired_bootstrap_95pct":(
                paired[int(.025*len(paired))],paired[int(.975*len(paired))-1]),
            "bootstraps":2000,
            "bootstrap_source":"original actor track ID, paired and clustered",
            "no_future_timestamp_as_predictor":True,
        }
    result={
        "study":"ESWAM006_GENUINE_TRAFFIC_LONG_HISTORY_FIXED_PRIORS",
        "publisher_archive_sha256":EXPECTED_ARCHIVE_SHA256,
        "source_member_DEV":source_member,
        "original_recording_site":"TC_BGR_Intersection_VA",
        "separate_site_generalization":False,
        "source_original_track_count":len(tracks),
        "original_factual_history_frames":HISTORY,
        "genuine_original_future_frames":list(HORIZONS),
        "stride_original_frames":STEP,
        "real_future_state_targets_only":True,
        "fixed_priors_no_training":True,
        "evidence_of_learned_memory_intelligence":False,
        "causal_action_intelligence":False,
        "reported_real_case_count":len(observations),
        "summaries":summaries,
    }
    return result,observations


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",type=Path,required=True)
    args=parser.parse_args()
    metrics,rows=score_real(args.archive)
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"REAL_FIXED_PRIORS_RESULTS.json").write_text(json.dumps(metrics,sort_keys=True,indent=2)+"\n")
    (ROOT/"ORIGINAL_REAL_CASES.json").write_text(json.dumps(rows,sort_keys=True,indent=2)+"\n")
    print("ESWAM006_REAL_LONG_HISTORY_PRIORS="+json.dumps(metrics["summaries"],sort_keys=True),flush=True)
