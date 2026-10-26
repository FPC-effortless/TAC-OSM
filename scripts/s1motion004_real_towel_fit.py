#!/usr/bin/env python3
"""S1MOTION004 one-shot registered, authentic-towel auxiliary-gradient experiment.

All 12 models trained and checkpoints/optimizer/RNG frozen BEFORE any DEV score.
Original historical experiments untouched. Measured high motion is a proxy,
not verified harm, success, policy or causal source action. NO TEST scoring.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.s1motion003_real_tape_fit import (
    prepare, windows, batch, score, sha256
)
from tac_osm.variants.s1motion004_gradient_routes import (
    IsolatedVelocitySystem1, VelocityGRUControl, losses, outputs
)
from tac_osm.variants.system1_velocity_anchored import VelocityAnchoredSystem1

STUDY="S1MOTION004"
SOURCE="lerobot/aloha_static_towel"
REV="13ad96f5ed0e219e48c72471626a2e3d0eb5aeff"
DATA_PATH="data/chunk-000/file-000.parquet"
BLOB_SHA="aa61a929598903401e10d9b677bd2f2df480da0c912b6a85a27541028155fa1b"
SEEDS=(28001,28002,28003)
FIT=tuple(range(25))
DEV=tuple(range(25,35))
ARMS=("tac_shared","tac_isolated","gru_shared","gru_isolated")
UPDATES=192
WINDOW=32
BATCH=8


def retrieve(url):
    req=urllib.request.Request(
        url,headers={"User-Agent":"TAC-OSM-S1MOTION004-registered-original/1.0"})
    with urllib.request.urlopen(req,timeout=120) as h:
        return h.read()


def real_source(cache):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    info=json.loads(retrieve(
        f"https://huggingface.co/api/datasets/{SOURCE}/revision/{REV}"))
    if info.get("sha")!=REV or DATA_PATH not in {
         x["rfilename"] for x in info.get("siblings",[])}:
        raise RuntimeError("S1MOTION004 pinned publisher SHA/file mismatch")
    fn=cache/"verified-aloha-towel.parquet"
    if not fn.exists():
        fn.write_bytes(retrieve(
            f"https://huggingface.co/datasets/{SOURCE}/resolve/{REV}/{DATA_PATH}"))
    contents=fn.read_bytes()
    digest=hashlib.sha256(contents).hexdigest()
    if digest!=BLOB_SHA:
        raise RuntimeError("S1MOTION004 SHA256 source mismatch, training blocked")
    t=pq.read_table(fn,columns=[
        "episode_index","frame_index","timestamp","observation.state","action"])
    if t.num_rows!=25000:
        raise RuntimeError("original publisher episode/frame census mismatch")
    # Exclude ALL protected TEST rows before building individual Python records.
    # DEV permitted only after ALL original model weights are frozen below.
    fit_table=t.filter(pc.less(t["episode_index"],pa.scalar(25)))
    dev_table=t.filter(pc.and_(
        pc.greater_equal(t["episode_index"],pa.scalar(25)),
        pc.less(t["episode_index"],pa.scalar(35))))
    def parse(filtered,range_ids):
        d=filtered.to_pydict()
        rows={ep:[] for ep in range_ids}
        for ep,frame,stamp,x,act in zip(*(d[k] for k in [
                "episode_index","frame_index","timestamp","observation.state","action"])):
            ep=int(ep)
            x=np.asarray(x,dtype=np.float64)
            act=np.asarray(act,dtype=np.float64)
            if (ep not in rows or x.shape!=(14,) or act.shape!=(14,) or
                not np.isfinite(x).all() or not np.isfinite(act).all() or
                not np.isfinite(stamp)):
                raise RuntimeError("unqualified authentic observation")
            if len(rows[ep])<257:
                rows[ep].append((int(frame),float(stamp),x,act))
        for ep,events in rows.items():
            if len(events)!=257:
                raise RuntimeError("source episode too short")
            if any(b[0]!=a[0]+1 or not 0 < b[1]-a[1] < .2
                    for a,b in zip(events,events[1:])):
                raise RuntimeError("frame/time authority failure")
        return rows
    # This method retrieves both partitions in the same method, but DEV scoring
    # is deferred until all train checkpoints are saved; no statistic uses DEV.
    fit_src=parse(fit_table,FIT)
    dev_src=parse(dev_table,DEV)
    return fit_src,dev_src,dict(
        name=SOURCE,revision=REV,parquet_sha256=digest,rows=t.num_rows,
        publisher_episode_count=50,fit_episode_ids=list(FIT),
        dev_episode_ids=list(DEV),test_episode_ids_excluded=list(range(35,50)),
        real_observations=True,whole_publisher_file_downloaded=True)


def initialization(seed,scale):
    """Identical initial tensor bytes within each paired architecture."""
    torch.manual_seed(seed)
    source_tac=VelocityAnchoredSystem1()
    source_tac.set_train_delta_scale(scale)
    isolated_tac=IsolatedVelocitySystem1()
    isolated_tac.load_state_dict(source_tac.state_dict(),strict=True)
    torch.manual_seed(seed+100_000)
    source_gru=VelocityGRUControl(scale,detach_auxiliary=False)
    isolated_gru=VelocityGRUControl(scale,detach_auxiliary=True)
    isolated_gru.load_state_dict(source_gru.state_dict(),strict=True)
    original={
        "tac_shared":source_tac,
        "tac_isolated":isolated_tac,
        "gru_shared":source_gru,
        "gru_isolated":isolated_gru,
    }
    assert all(sum(p.numel() for p in original[a].parameters())==95907
               for a in ("tac_shared","tac_isolated"))
    assert all(sum(p.numel() for p in original[a].parameters())==95869
               for a in ("gru_shared","gru_isolated"))
    return original


def study_fit(arm,model,seed,fit,plans,folder):
    isolated=arm.endswith("_isolated")
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    start=time.perf_counter()
    final_loss=None
    for selected,begins in plans:
        x,u,y,event,previous=batch(fit,selected,begins)
        optimizer.zero_grad(set_to_none=True)
        objective,parts=losses(model,x,u,y,event,previous,isolated=isolated)
        if not torch.isfinite(objective).item():
            raise RuntimeError("nonfinite real-source objective")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
        optimizer.step()
        final_loss=float(objective.detach())
    seconds=time.perf_counter()-start
    model.eval()
    snapshot=dict(arm=arm,seed=seed,
        model=model.state_dict(),optimizer=optimizer.state_dict(),
        torch_rng=torch.random.get_rng_state(),
        original_updates=UPDATES,batch=BATCH,window=WINDOW,
        source_sha=REV,source_parquet_sha256=BLOB_SHA)
    buffer=io.BytesIO();torch.save(snapshot,buffer)
    b=buffer.getvalue()
    folder.mkdir(parents=True,exist_ok=True)
    (folder/f"{arm}-{seed}.pt").write_bytes(b)
    return dict(arm=arm,seed=seed,
        params=sum(p.numel() for p in model.parameters()),
        checkpoint_sha256=sha256(b),training_seconds=seconds,
        optimizer_updates=UPDATES,frames_consumed=UPDATES*BATCH*WINDOW,
        last_fit_loss=final_loss,training_objective_scalar_unchanged=True,
        auxiliary_gradient_isolated=isolated,
        actor_and_clock_guard_external=True)


def incomplete_flops(model,fit,seed,isolated):
    """Diagnostic profiler counts only supported torch op subset."""
    try:
        from torch.profiler import profile,ProfilerActivity
        selected,begins=windows(fit,seed)[0]
        sample=batch(fit,selected,begins)
        model.train();model.zero_grad(set_to_none=True)
        with profile(activities=[ProfilerActivity.CPU],with_flops=True) as prof:
            l,_=losses(model,*sample,isolated=isolated)
            l.backward()
        model.zero_grad(set_to_none=True);model.eval()
        return dict(profiler_operator_subset_flops=sum(
          int(x.flops) for x in prof.key_averages() if x.flops),
          is_total_training_FLOPs=False,
          profiler_coverage="INCOMPLETE (supported matmul/conv only)")
    except Exception as exc:
        model.eval()
        return dict(profiler_operator_subset_flops=None,
                    is_total_training_FLOPs=False,error=type(exc).__name__)


def correction_decomposition(y,normalized_dev,normalization):
    state_sd=np.asarray(normalization["state_sd"])
    out=[]
    model=y
    with torch.no_grad():
        for ep in DEV:
            d=normalized_dev[ep]
            pred,_,_=outputs(model,d["x"][None],d["prior_u"][None],
                             d["previous_x"][:1])
            pred=pred[0].cpu().numpy().astype(np.float64)
            baseline=2*d["x"].numpy()-d["previous_x"].numpy()
            actual=d["y"].numpy()
            correction=(pred-baseline)*state_sd
            residual=(actual-baseline)*state_sd
            out.append(dict(episode=ep,
                correction_energy=float(np.mean(correction**2)),
                signed_residual_alignment=float(np.mean(correction*residual)),
                exact_mse_delta_vs_velocity=float(np.mean(
                    correction**2-2*correction*residual))))
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",default="/tmp/s1motion004-real-towel")
    parser.add_argument("--out",default="research/diagnostics/s1motion004-original")
    a=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fitsrc,devsrc,source=real_source(a.cache)
    fit,dev,norm=prepare(fitsrc,devsrc)
    output=ROOT/a.out;output.mkdir(parents=True,exist_ok=True)
    delta=torch.tensor(norm["train_delta_scale_normalized"],dtype=torch.float32)
    trained={}
    original_fit_receipts=[]
    for seed in SEEDS:
        pairs=initialization(seed,delta)
        sample_plans=windows(fit,seed)
        for arm in ARMS:
            learned=copy.deepcopy(pairs[arm])
            record=study_fit(arm,learned,seed,fit,sample_plans,output)
            trained[(seed,arm)]=learned
            original_fit_receipts.append(record)
            print("S1MOTION004_ORIGINAL_FIT_FROZEN",
                  json.dumps(record,sort_keys=True),flush=True)
    manifest=dict(
        study=STUDY,protocol="research/parallel-six/S1MOTION004-PROTOCOL.md",
        source=source,source_audit_original_run=40279375687,
        all_twelve_model_states_frozen_before_first_DEV=True,
        all_twelve_optimizer_rng_states_frozen_before_first_DEV=True,
        original_FIT_weights_only=True,trained_seed_list=list(SEEDS),
        model_fit_receipts=original_fit_receipts,
        train_stats_only=norm,
        full_training_FLOPs_matched=False,
        actual_adverse_or_task_success_labels_available=False,
        same_publisher_robot_not_independent_site=True,
        protected_test_scored=False)
    (output/"before-dev-score.json").write_text(
        json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("S1MOTION004_ALL_TWELVE_MODELS_SHA_FROZEN_BEFORE_DEV=YES",flush=True)
    result=[]
    for seed in SEEDS:
        scored={}
        for arm in ARMS:
            m=trained[(seed,arm)]
            scores=score(m,dev,norm)
            correction=correction_decomposition(m,dev,norm)
            scored[arm]=dict(dev_scores=scores,correction=correction)
            print("S1MOTION004_FIRST_DEV",json.dumps({
                "seed":seed,"arm":arm,
                "raw_joint_MSE":scores["raw_joint_MSE"],
                "raw_high_motion_MSE":scores["raw_joint_high_motion_MSE"],
                "proxy_AUPRC":scores["proxy_AUPRC"],
                "proxy_AUROC":scores["proxy_AUROC"],
                "proxy_Brier":scores["proxy_Brier"],
                "proxy_prevalence":scores["proxy_prevalence"],
                "real_scored_frames":scores["eval_real_frames"]},
                sort_keys=True),flush=True)
        iso=scored["tac_isolated"]["dev_scores"]
        shared=scored["tac_shared"]["dev_scores"]
        err=iso["raw_joint_MSE"]["tac"]
        high=iso["raw_joint_high_motion_MSE"]
        velocity=iso["raw_joint_MSE"]["velocity"]
        shared_err=shared["raw_joint_MSE"]["tac"]
        event_guard=iso["proxy_AUPRC"] >= .9*shared["proxy_AUPRC"]
        gate=(
            err<=.99*shared_err and
            err<=.99*velocity and
            high["tac"] is not None and
            high["velocity"] is not None and
            high["tac"]<=1.05*high["velocity"] and
            event_guard)
        result.append(dict(seed=seed,scores=scored,gate_pass=bool(gate),
            isolated_beats_tac_shared_by_1pct=bool(err<=.99*shared_err),
            isolated_beats_velocity_by_1pct=bool(err<=.99*velocity),
            high_motion_protection=bool(high["tac"] is not None and
                high["velocity"] is not None and
                high["tac"]<=1.05*high["velocity"]),
            proxy_AUPRC_protection=bool(event_guard)))
    for receipt in original_fit_receipts:
        key=(receipt["seed"],receipt["arm"])
        receipt["postfit_FIT_profiler"]=incomplete_flops(
            trained[key],fit,receipt["seed"],receipt["auxiliary_gradient_isolated"])
    report=dict(manifest,
        original_model_fit_receipts=original_fit_receipts,
        original_first_dev_real_episode_scores=result,
        g1_gradient_engineering_subject_to_CI=True,
        all_three_seed_preregistered_predictive_gate_pass=all(
            x["gate_pass"] for x in result),
        one_shot_original_scores_unmodified=True,
        source_is_genuine_recorded_real_joint_states=True,
        full_training_FLOP_parity_UNPROVEN=True,
        heldout_robot_site_and_task_safety_qualification=False,
        no_autonomous_policy=False)
    # no_autonomous_policy=False would misleadingly indicate autonomous policy;
    # explicit positive flag below is the truthful statement.
    report["no_autonomous_policy"]=True
    (output/"first-dev-scores.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("S1MOTION004_ORIGINAL_DEV_DONE=YES",flush=True)
    print("S1MOTION004_ORIGINAL_FULL_GATE="+(
        "PASS" if report["all_three_seed_preregistered_predictive_gate_pass"]
        else "FAIL"),flush=True)
    print("S1MOTION004_SAFETY_QUALIFIED=NO",flush=True)


if __name__=="__main__":
    main()
