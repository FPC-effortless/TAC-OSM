"""ESWAM012: genuine original 256-history multi-actor world-state training.

Frozen protocol: original real INTERACTION TC source 000-002 FIT, 003
previously exposed DEV, five matched 100k candidate/control models, two
seeds, same 320-update/8-batch source sampling schedules, no tuning.
No controlled action or future labels are supplied as input.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import itertools
import json
import math
import random
import time
from pathlib import Path
import zipfile

import torch
from torch.nn import functional as F

from scripts.eswam005_qualify_real_multiagent import (
    EXPECTED_ARCHIVE_SHA256, sha256_file,
)
from tac_osm.variants.eswam012_multiagent import Native, GRU, Transformer

SEEDS = (86001, 86002)
HISTORY = 256
FUTURES = (10, 30)
STRIDE = 20
UPDATES = 320
BATCH = 8
LEARNING_RATE = 0.001
MAX_AGENTS = 4
BUNDLE = Path("research/replay-bundles/ESWAM012")


def models():
    return {"native": Native(68), "gru": GRU(78),
            "attention": Transformer(68, 155),
            "native_short": Native(68),
            "native_unscoped": Native(68, identity_protected=False)}


def preflight():
    counts = {k: sum(p.numel() for p in v.parameters() if p.requires_grad)
              for k, v in models().items()}
    if any(not 95_000 <= n <= 105_000 for n in counts.values()):
        raise RuntimeError(f"100k active parameter gate FAIL {counts}")
    if (max(counts.values()) - min(counts.values())) / max(counts.values()) > 0.01:
        raise RuntimeError(f"Matched parameter gate FAIL {counts}")
    expected = {"native": 101702, "gru": 100788, "attention": 101370,
                "native_short": 101702, "native_unscoped": 101702}
    if counts != expected:
        raise RuntimeError(f"Frozen original model count mismatch {counts} != {expected}")
    return counts


def source_members(archive):
    if not archive.is_file() or sha256_file(archive) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("STOP: original publisher source hash mismatch")
    files = {}
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            if (member.startswith("__MACOSX/") or "/__MACOSX/" in member
                    or Path(member).name.startswith("._")
                    or not ("TC_BGR_Intersection_VA" in member
                            and "vehicle_tracks_" in member and member.endswith(".csv"))):
                continue
            matches = [j for j in range(4)
                       if member.endswith(f"vehicle_tracks_{j:03d}.csv")]
            if len(matches) != 1:
                continue
            rec = matches[0]
            if rec in files:
                raise RuntimeError("duplicate original publisher member")
            tracks = defaultdict(list)
            with z.open(member) as raw:
                reader = csv.DictReader(
                    io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                required = {"track_id", "frame_id", "timestamp_ms", "x", "y", "vx", "vy"}
                if not required.issubset(reader.fieldnames or []):
                    raise RuntimeError("publisher original source columns changed")
                for r in reader:
                    tid = str(r["track_id"]).strip()
                    frame, ms = int(r["frame_id"]), int(r["timestamp_ms"])
                    measured = tuple(float(r[k]) for k in ("x", "y", "vx", "vy"))
                    if (not tid or frame < 0 or ms < 0
                            or not all(map(math.isfinite, measured))):
                        raise RuntimeError("invalid real source actor measurement")
                    tracks[tid].append((frame, ms, *measured))
            files[rec] = (member, tracks)
    if set(files) != {0, 1, 2, 3}:
        raise RuntimeError("STOP: original recording cohort missing")
    return files


def consecutive(raw):
    parts, segment, seen = [], [], set()
    for row in sorted(raw, key=lambda x: (x[0], x[1])):
        ident = (row[0], row[1])
        if ident in seen:
            raise RuntimeError("duplicate original actor/frame/timestamp")
        seen.add(ident)
        if segment and (row[0] != segment[-1][0] + 1
                        or row[1] <= segment[-1][1]):
            parts.append(segment)
            segment = []
        segment.append(row)
    if segment:
        parts.append(segment)
    return parts


def original_cases(archive):
    recordings = source_members(archive)
    all_rows, misaligned = {}, defaultdict(int)
    for rec, (member, tracks) in sorted(recordings.items()):
        origins = defaultdict(list)
        for tid, track in tracks.items():
            for segment in consecutive(track):
                if len(segment) < HISTORY + max(FUTURES):
                    continue
                for i in range(HISTORY - 1, len(segment) - max(FUTURES)):
                    frame, ms = segment[i][:2]
                    if frame % STRIDE == 0:
                        origins[(frame, ms)].append((tid, segment, i))
        results = []
        for (frame, ms), actors in sorted(origins.items()):
            if len(actors) < 2:
                continue
            actors = sorted(
                actors, key=lambda v: (int(v[0]) if v[0].isdigit() else 10**12, v[0])
            )[:MAX_AGENTS]
            n = len(actors)
            histories = [track[i-HISTORY+1:i+1] for _, track, i in actors]
            real_frames = [row[:2] for row in histories[0]]
            if any([row[:2] for row in hist] != real_frames for hist in histories[1:]):
                misaligned[rec] += 1
                continue
            intervals = [(hist[-1][1] - hist[-2][1]) / 1000 for hist in histories]
            if not all(math.isfinite(delta) and delta > 0 for delta in intervals):
                misaligned[rec] += 1
                continue
            positions = torch.tensor(
                [[row[2:4] for row in hist] for hist in histories],
                dtype=torch.float32).permute(1, 0, 2)
            velocity = torch.tensor(
                [[row[4:6] for row in hist] for hist in histories],
                dtype=torch.float32).permute(1, 0, 2)
            neighbors_pos = (positions.sum(1, keepdim=True)-positions)/(n-1)-positions
            neighbors_vel = (velocity.sum(1, keepdim=True)-velocity)/(n-1)-velocity
            recorded_features = torch.cat(
                ((positions - positions[-1][None]) / 30,
                 velocity / 10, neighbors_pos / 30, neighbors_vel / 10), dim=-1)
            x = torch.zeros(HISTORY, MAX_AGENTS, 8)
            x[:, :n] = recorded_features
            active = torch.zeros(MAX_AGENTS)
            active[:n] = 1
            prior = torch.zeros(MAX_AGENTS, 2, 2)
            actual = torch.zeros(MAX_AGENTS, 2, 2)
            p0 = torch.zeros(MAX_AGENTS, 2, 2)
            p8 = torch.zeros(MAX_AGENTS, 2, 2)
            p256 = torch.zeros(MAX_AGENTS, 2, 2)
            for j, (_, track, i) in enumerate(actors):
                xy = torch.tensor(track[i][2:4], dtype=torch.float32)
                v_now = torch.tensor(track[i][4:6], dtype=torch.float32)
                v8 = torch.tensor(
                    [sum(row[4+k] for row in track[i-7:i+1]) / 8
                     for k in range(2)], dtype=torch.float32)
                v256 = torch.tensor(
                    [sum(row[4+k] for row in track[i-HISTORY+1:i+1]) / HISTORY
                     for k in range(2)], dtype=torch.float32)
                for ih, horizon in enumerate(FUTURES):
                    future = track[i+horizon]
                    if future[0] != frame+horizon or future[1] <= ms:
                        raise RuntimeError("source original physical future missing")
                    prior[j, ih] = xy + horizon*intervals[j]*v_now
                    actual[j, ih] = torch.tensor(future[2:4])
                    p0[j, ih] = xy
                    p8[j, ih] = xy + horizon*intervals[j]*v8
                    p256[j, ih] = xy + horizon*intervals[j]*v256
            results.append({"x": x, "prior": prior, "y": actual,
                            "mask": active, "p0": p0, "p8": p8, "p256": p256,
                            "actor_ids": [tid for tid, _, _ in actors],
                            "recording": rec, "frame": frame,
                            "timestamp_ms": ms, "source_member": member})
        if not results:
            raise RuntimeError(f"no genuine source-aligned multiactor samples rec {rec}")
        all_rows[rec] = results
    return all_rows, dict(misaligned)


def stack(rows):
    return tuple(torch.stack([item[k] for item in rows])
                 for k in ("x", "prior", "y", "mask"))


def physical_loss(pred, scale, actual, active):
    err = pred-actual
    score = (.7 * F.smooth_l1_loss(pred, actual, reduction="none", beta=.5)
             + .3 * (err.abs()/scale + torch.log(2*scale)))
    return (score * active[:, :, None, None]).sum() / (active.sum()*4)


def all_original_case_rows(model, dev, arm):
    model.eval()
    scored = []
    with torch.inference_mode():
        for i in range(0, len(dev), 16):
            subset = dev[i:i+16]
            x, prior, true, mask = stack(subset)
            pred, scale = model(x[:, -8:] if arm == "native_short" else x, prior, mask)
            for src, p, sigma in zip(subset, pred, scale):
                for j, actor_id in enumerate(src["actor_ids"]):
                    for hi, horizon in enumerate(FUTURES):
                        true_xy = src["y"][j,hi].tolist()
                        baseline = {k:src[k][j,hi].tolist()
                                    for k in ("prior","p0","p8","p256")}
                        pos = p[j,hi].tolist()
                        distance = lambda xy:math.hypot(
                            xy[0]-true_xy[0], xy[1]-true_xy[1])
                        scored.append({
                            "source_original_recording": src["recording"],
                            "source_original_member": src["source_member"],
                            "source_original_frame": src["frame"],
                            "source_original_timestamp_ms": src["timestamp_ms"],
                            "source_original_actor_id": actor_id,
                            "source_actor_cluster": f"{src['recording']}:{actor_id}",
                            "horizon_frames": horizon,
                            "true_original_future_xy_m": true_xy,
                            "observed_current_xy_m": baseline["p0"],
                            "model_predicted_xy_m": pos,
                            "predicted_laplace_scale_xy": sigma[j,hi].tolist(),
                            "original_fixed_physical_predictions": baseline,
                            "l2_meters": {"model":distance(pos),
                                          **{key:distance(xy) for key,xy in baseline.items()}},
                            "current_measured_source_velocity_norm_m_per_s":
                                math.hypot(
                                    (baseline["prior"][0]-baseline["p0"][0]) /
                                    (horizon*.1),
                                    (baseline["prior"][1]-baseline["p0"][1]) /
                                    (horizon*.1)),
                        })
    return scored


def actor_bootstrap(rows, source1, source2, seed, draws=2000):
    clusters=defaultdict(list)
    for r in rows:
        clusters[r["source_actor_cluster"]].append(r)
    ids=sorted(clusters)
    generator=random.Random(seed)
    bootstrap=[]
    for _ in range(draws):
        selected=[generator.choice(ids) for _ in ids]
        full=[r for key in selected for r in clusters[key]]
        bootstrap.append(sum(
            r["l2_meters"][source1]-r["l2_meters"][source2] for r in full)/len(full))
    bootstrap.sort()
    return [bootstrap[int(.025*draws)], bootstrap[int(.975*draws)-1]]


def score_one(rows, seed, arm, horizon, schedule_hash, checkpoint_hash, fit_seconds, start_loss, last_loss):
    selected=[r for r in rows if r["horizon_frames"]==horizon]
    avg=lambda k:sum(r["l2_meters"][k] for r in selected)/len(selected)
    stationary=[r for r in selected if r["current_measured_source_velocity_norm_m_per_s"] < .2]
    moving=[r for r in selected if r["current_measured_source_velocity_norm_m_per_s"] >= .2]
    average_group=lambda group,key:sum(r["l2_meters"][key] for r in group)/len(group) if group else None
    result = {
        "seed":seed,"architecture":arm,"horizon_original_frames":horizon,
        "original_dev_actor_predictions":len(selected),
        "distinct_original_actor_clusters":len({r["source_actor_cluster"] for r in selected}),
        "model_mean_L2_meters":avg("model"),
        "instantaneous_velocity_prior_L2_meters":avg("prior"),
        "persistence_prior_L2_meters":avg("p0"),
        "eight_frame_prior_L2_meters":avg("p8"),
        "twofiftysix_frame_prior_L2_meters":avg("p256"),
        "paired_original_actor_cluster_bootstrap_model_minus_P1_95pct":
            actor_bootstrap(selected,"model","prior",seed+horizon),
        "observed_speed_below_0_2_mps_count":len(stationary),
        "observed_speed_above_or_equal_0_2_mps_count":len(moving),
        "stationary_model_L2_meters":average_group(stationary,"model"),
        "stationary_prior_P1_L2_meters":average_group(stationary,"prior"),
        "moving_model_L2_meters":average_group(moving,"model"),
        "moving_prior_P1_L2_meters":average_group(moving,"prior"),
        "actual_fit_CPU_seconds_efficiency_only":fit_seconds,
        "original_fitted_checkpoint_sha256":checkpoint_hash,
        "first_training_loss":start_loss,"last_training_loss":last_loss,
        "identical_paired_source_schedule_sha256":schedule_hash,
        "actual_training_FLOPs_equal_certified":False,
    }
    return result


def fit(archive:Path, output:Path):
    param_counts=preflight()
    data, skipped=original_cases(archive)
    fit_rows=sum((data[rec] for rec in (0,1,2)),[])
    dev_rows=data[3]
    output.mkdir(parents=True, exist_ok=True)
    digest=hashlib.sha256(json.dumps(
        [(r["recording"],r["frame"],r["actor_ids"]) for r in fit_rows],
        separators=(",",":"),sort_keys=True).encode()).hexdigest()
    result={
        "study":"ESWAM012_GENUINE_JOINT256_REAL_PREDICTION_FIVE_FAMILIES",
        "original_publisher_archive_sha256":EXPECTED_ARCHIVE_SHA256,
        "original_one_intersection_site":"TC_BGR_Intersection_VA",
        "new_independent_physical_site_test":False,
        "source_FIT_recordings":[0,1,2],"source_EXPOSED_DEV_recording":3,
        "actual_real_source_only":True,"source_synthetic_fits":0,
        "observed_intervention_or_causal_do_action":False,
        "original_complete_history_frames":HISTORY,
        "futures_original_frames":list(FUTURES),
        "source_frame_stride":STRIDE,
        "fit_real_joint_source_anchors":len(fit_rows),
        "exposed_DEV_joint_source_anchors":len(dev_rows),
        "dropped_nonidentical_original_multiactor_time_alignment":skipped,
        "frozen_paired_seed_schedule_identifier_sha256":digest,
        "paired_seeds":list(SEEDS),"architectures":list(models()),
        "strict_live_active_parameter_counts":param_counts,
        "active_parameter_max_relative_gap":
            (max(param_counts.values())-min(param_counts.values()))/max(param_counts.values()),
        "identical_FIT_anchor_order_within_each_seed":True,
        "identical_FIT_real_future_targets_for_all_arms":True,
        "identical_physical_velocity_prior_and_readout_family":True,
        "optimizer_updates_each":UPDATES,"batch_size_each":BATCH,
        "adamw_learning_rate":LEARNING_RATE,"adamw_weight_decay":.00001,
        "gradient_clip":1.0,
        "DEV_opened_for_training_or_model_selection":False,
        "full_training_FLOPs_budget_parity_certified":False,
        "fair_scientific_architecture_superiority_eligible":False,
        "final_scores":[],
    }
    torch.set_num_threads(2)
    for seed in SEEDS:
        rng=random.Random(seed)
        order=[[rng.randrange(len(fit_rows)) for _ in range(BATCH)]
               for _ in range(UPDATES)]
        schedule_hash=hashlib.sha256(
            json.dumps(order,separators=(",",":")).encode()).hexdigest()
        for arm in models():
            torch.manual_seed(seed)
            model=models()[arm]
            if sum(p.numel() for p in model.parameters())!=param_counts[arm]:
                raise RuntimeError("live model changed after registered preflight")
            opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=.00001)
            model.train()
            losses=[]
            start=time.perf_counter()
            for step,indices in enumerate(order):
                x, prior, target, active=stack([fit_rows[k] for k in indices])
                opt.zero_grad(set_to_none=True)
                pred, scale=model(x[:, -8:] if arm=="native_short" else x,prior,active)
                loss=physical_loss(pred,scale,target,active)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite real training loss {arm}/{seed}/{step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            elapsed=time.perf_counter()-start
            ckpt={
                "seed":seed,"architecture":arm,"original_source_sha256":
                    EXPECTED_ARCHIVE_SHA256,
                "fit_real_sample_identity_sha256":digest,
                "identical_paired_training_schedule_sha256":schedule_hash,
                "exact_optimizer_updates":UPDATES,
                "weights":model.state_dict(),"optimizer":opt.state_dict(),
                "pytorch_rng_after_training":torch.get_rng_state(),
                "python_sampler_rng_after_schedule":rng.getstate(),
            }
            ckfile=output/f"{arm}-{seed}.pt"
            torch.save(ckpt,ckfile)
            ckhash=hashlib.sha256(ckfile.read_bytes()).hexdigest()
            scored=all_original_case_rows(model,dev_rows,arm)
            (output/f"{arm}-{seed}-ORIGINAL_DEV_CASES.json").write_text(
                json.dumps(scored,indent=2)+"\n")
            for h in FUTURES:
                summary=score_one(scored,seed,arm,h,schedule_hash,ckhash,
                                  elapsed,losses[0],losses[-1])
                print("ESWAM012_AUTHENTIC_TRAINED_RESULT="+
                      json.dumps(summary,sort_keys=True),flush=True)
                result["final_scores"].append(summary)
    (output/"FINAL_RESULTS.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("ESWAM012_AUTHENTIC_FIT_SUCCEEDED "+json.dumps({
        "training_model_count":len(result["final_scores"])//len(FUTURES),
        "source_fit_examples":len(fit_rows),
        "source_dev_examples":len(dev_rows),
        "all_trained_model_receipts_preserved":True},sort_keys=True),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=["preflight","source-summary","fit"])
    ap.add_argument("--archive",type=Path)
    ap.add_argument("--out",type=Path,default=BUNDLE)
    args=ap.parse_args()
    if args.mode=="preflight":
        print("ESWAM012_LIVE_PREFIT_PARAMETERS="+json.dumps(preflight(),sort_keys=True))
    elif args.mode=="source-summary":
        data,skipped=original_cases(args.archive)
        print("ESWAM012_STRICT_256_SOURCE_SAMPLES="+json.dumps({
            "eligible_original_real_anchors":{str(k):len(v) for k,v in data.items()},
            "strict_joint_timestamp_rejected":skipped},sort_keys=True))
    else:
        fit(args.archive,args.out)
