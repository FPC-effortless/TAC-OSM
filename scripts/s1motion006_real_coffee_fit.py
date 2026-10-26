#!/usr/bin/env python3
"""S1MOTION006 original real coffee direct motion-feature paired fit.

Do not silently tune from historical screwdriver/towel/candy/tape DEV.
Only 25 SHA-pinned real coffee FIT episode files are acquired before saving
ALL six optimizer/model/RNG states. Real coffee DEV25..34 downloaded only AFTER
hash freeze; TEST35..49 forbidden by the source accessor.
"""
from __future__ import annotations
import argparse,copy,hashlib,io,json,math,sys,time,urllib.request
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.s1motion003_real_tape_fit import windows,batch,score,sha256
from scripts.s1motion004_real_towel_fit import prepare_fit_only,encode_source_episode
from tac_osm.variants.s1motion006_velocity_feature import VelocityEventSystem1
from tac_osm.variants.s1motion005_gradient_budget import step_model

LOCK=ROOT/"research/parallel-six/S1MOTION006-SOURCE-LOCK.json"
FIT=tuple(range(25));DEV=tuple(range(25,35));TEST=tuple(range(35,50))
ARMS=("core_prediction","core_plus_measured_velocity")
SEEDS=(36001,36002,36003)
UPDATES=192;BATCH=8;WINDOW=32

def _load_lock():
    lock=json.loads(LOCK.read_text())
    if (lock["dataset"]!="lerobot/aloha_static_coffee"
        or lock["revision_sha"]!="1c52c4c794b6f93d931489263cf15e4331600bfc"
        or lock["codebase_version"]!="v2.0"
        or lock["source_eligibility_run"]!=40289531876
        or lock["FIT"]!=list(FIT) or lock["DEV"]!=list(DEV)
        or lock["TEST"]!=list(TEST)
        or lock["DEV_episode_files_downloaded"]!=0
        or lock["TEST_episode_files_downloaded"]!=0
        or len(lock["FIT_receipts"])!=25):
        raise RuntimeError("source audit chain broken before fit")
    return lock

def _download(url):
    req=urllib.request.Request(url,headers={
        "User-Agent":"S1MOTION006-real-coffee-first-registered/1.0"})
    with urllib.request.urlopen(req,timeout=120) as r:return r.read()

def episode(ep,lock,cache,*,dev_released=False):
    if type(ep) is not int or ep not in (*FIT,*(DEV if dev_released else ())):
        raise ValueError("not a permitted source episode at this study stage")
    path=f"data/chunk-000/episode_{ep:06d}.parquet"
    cache.mkdir(parents=True,exist_ok=True)
    filename=cache/f"episode_{ep:06d}.parquet"
    if not filename.exists():
        filename.write_bytes(_download(
            f"https://huggingface.co/datasets/{lock['dataset']}/resolve/"
            f"{lock['revision_sha']}/{path}"))
    raw=filename.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if ep in FIT:
        expected=lock["FIT_receipts"][ep]
        if expected["episode"]!=ep or expected["sha256"]!=digest or expected["path"]!=path:
            raise RuntimeError("FIT source pre-registered SHA mismatch")
    table=pq.read_table(io.BytesIO(raw),columns=[
        "episode_index","frame_index","timestamp","observation.state","action"])
    if table.num_rows!=1100:raise RuntimeError("real task full original frame count changed")
    cols=table.to_pydict()
    first=[];last=None
    for i,(identity,frame,clock,x,u) in enumerate(zip(*(
       cols[k] for k in ("episode_index","frame_index","timestamp",
                         "observation.state","action")))):
        x=np.asarray(x,dtype=np.float64);u=np.asarray(u,dtype=np.float64)
        if (int(identity)!=ep or int(frame)!=i or x.shape!=(14,) or u.shape!=(14,)
            or not np.isfinite(x).all() or not np.isfinite(u).all()
            or not math.isfinite(float(clock))
            or (last is not None and not 0<float(clock)-last<.2)):
            raise RuntimeError("unqualified original robot clock/frame/sensor authority")
        if i<257:first.append((i,float(clock),x,u))
        last=float(clock)
    return first,dict(ep=ep,sha256=digest,path=path,
                      source_total_frames=1100,analyzed_frames=257)

def make_models(seed,delta):
    torch.manual_seed(seed)
    base=VelocityEventSystem1(feature_mode="core_prediction")
    base.set_train_delta_scale(delta)
    out={arm:VelocityEventSystem1(feature_mode=arm) for arm in ARMS}
    for model in out.values():model.load_state_dict(base.state_dict(),strict=True)
    for model in out.values():
        if sum(p.numel() for p in model.parameters())!=95907 or model.carried_fp32_bytes!=664:
            raise RuntimeError("source arm parameter/state capacity changed")
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache",default="/tmp/s1motion006-original-coffee")
    p.add_argument("--out",default="research/diagnostics/s1motion006-original")
    a=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    lock=_load_lock()
    cache=Path(a.cache);out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    src={};receipts=[]
    for ep in FIT:
        rows,r=episode(ep,lock,cache)
        src[ep]=rows;receipts.append(r)
    fit,norm=prepare_fit_only(src)
    delta=torch.tensor(norm["train_delta_scale_normalized"],dtype=torch.float32)
    frozen={};receipts_models=[]
    for seed in SEEDS:
        models=make_models(seed,delta)
        samples=windows(fit,seed)
        for arm in ARMS:
            model=models[arm].train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            t0=time.perf_counter(); training_checks=[]
            for selected,starts in samples:
                x,u,y,event,previous=batch(fit,selected,starts)
                loss,diagnostics=step_model(model,optimizer,
                    (x,u,y,event,previous),arm="tac_projected")
                if not math.isfinite(loss):raise RuntimeError("source point/event objective nonfinite")
                training_checks.append(diagnostics)
            model.eval();elapsed=time.perf_counter()-t0
            pkg=dict(study="S1MOTION006",arm=arm,seed=seed,
                source_revision=lock["revision_sha"],
                FIT_source_sha256=[r["sha256"] for r in receipts],
                model=model.state_dict(),optimizer=optimizer.state_dict(),
                torch_rng=torch.random.get_rng_state(),updates=UPDATES)
            b=io.BytesIO();torch.save(pkg,b)
            raw=b.getvalue()
            (out/f"{arm}-{seed}.pt").write_bytes(raw)
            frozen[(seed,arm)]=model
            item=dict(arm=arm,seed=seed,params=95907,
                sha256=hashlib.sha256(raw).hexdigest(),updates=UPDATES,
                unique_real_fit_episodes=25,
                sampled_real_frame_observations=UPDATES*BATCH*WINDOW,
                source_training_seconds=elapsed,
                actual_AdamW_point_direction_negative_count=sum(
                    z["actual_point_gradient_dot_parameter_delta"]<0 for z in training_checks),
                FIT_post_step_point_improved_count=sum(
                    z["post_point_fit"]<=z["pre_point_fit"] for z in training_checks),
                full_training_FLOPs_matched=False)
            receipts_models.append(item)
            print("S1MOTION006_FROZEN",json.dumps(item,sort_keys=True),flush=True)
    pre=dict(study="S1MOTION006",
        predeclared_protocol="research/parallel-six/S1MOTION006-PROTOCOL.md",
        locked_publisher_source=lock["revision_sha"],
        FIT_verified_receipts=receipts,
        normalizers_FIT_only=norm,
        all_SIX_model_optimizer_RNG_SHA_frozen_before_DEV=True,
        trained_models=receipts_models,
        no_test_files_downloaded=True,
        complete_training_FLOP_matching_NOT_verified=True,
        observational_high_motion_NOT_hazard=True)
    (out/"BEFORE-DEV-FREEZE.json").write_text(json.dumps(pre,indent=2,sort_keys=True)+"\n")
    print("S1MOTION006_ALL_6_ORIGINAL_SHA_FROZEN_BEFORE_DEV=PASS",flush=True)
    # The first DEV source row may be fetched ONLY after writing above receipt.
    dev_source={};dev_receipts=[]
    for ep in DEV:
        rows,r=episode(ep,lock,cache,dev_released=True)
        dev_source[ep]=rows;dev_receipts.append(r)
    dev={e:encode_source_episode(dev_source[e],norm) for e in DEV}
    result=[]
    for seed in SEEDS:
        arm_scores={}
        for arm in ARMS:
            model=frozen[(seed,arm)]
            with torch.no_grad():
                original=score(model,dev,norm)
            arm_scores[arm]=original
            print("S1MOTION006_REAL_FIRST_DEV",json.dumps(dict(seed=seed,arm=arm,
                MSE=original["raw_joint_MSE"],AUPRC=original["proxy_AUPRC"],
                Brier=original["proxy_Brier"],AUROC=original["proxy_AUROC"],
                proxy_prevalence=original["proxy_prevalence"]),
                sort_keys=True),flush=True)
        c,t=(arm_scores[k] for k in ARMS)
        pct=(t["proxy_AUPRC"]>=1.05*c["proxy_AUPRC"]
             and t["proxy_AUPRC"]>=t["proxy_prevalence"])
        m=t["raw_joint_MSE"]["tac"];v=t["raw_joint_MSE"]["velocity"]
        robust=t["raw_joint_high_motion_MSE"]
        pred=(m<=1.01*c["raw_joint_MSE"]["tac"] and m<=.99*v
              and robust["tac"] is not None
              and robust["velocity"] is not None
              and robust["tac"]<=1.05*robust["velocity"])
        result.append(dict(seed=seed,scores=arm_scores,
            motion_proxy_guard_pass=bool(pct),physical_state_guard_pass=bool(pred),
            original_seed_gate_pass=bool(pct and pred)))
    report=dict(pre,DEV_file_hashes_opened_after_FIT_model_freeze=dev_receipts,
                original_first_DEV_scores=result,
                full_three_seed_acceptance=all(z["original_seed_gate_pass"] for z in result),
                fully_equal_compute_not_demonstrated=True,
                authorized_safe_robot_control=False,
                true_adverse_outcome_labels=False)
    (out/"FIRST-REAL-DEV-SCORES.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("S1MOTION006_ORIGINAL_GATE="+("PASS" if report["full_three_seed_acceptance"] else "FAIL"),flush=True)
    print("S1MOTION006_COMPLETE_FIRST_DEV_SOURCE_SCORE=YES",flush=True)

if __name__=="__main__":
    main()
