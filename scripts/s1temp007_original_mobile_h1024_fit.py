#!/usr/bin/env python3
"""S1TEMP007 ORIGINAL authentic 2000-frame Mobile ALOHA Chair memory evaluation.

FIT-only real per-episode data before all 9 original optimizer/RNG/model SHAs.
Open DEV Parquet ep30..39 only AFTER all 6 checkpoint hashes are written.
Actual TEST ep40..54 is forbidden regardless of caller arguments.
NO simulator, action authority, incident or hazard labels.
"""
from __future__ import annotations
import argparse,hashlib,io,json,math,sys,time,urllib.request
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from tac_osm.variants.s1temp007_persistent_slow import VelocityAnchoredSlowSystem1
from tac_osm.variants.s1motion005_gradient_budget import step_model

LOCK=ROOT/"research/parallel-six/S1TEMP007-MOBILE-SOURCE-LOCK.json"
FIT=tuple(range(30));DEV=tuple(range(30,40));TEST=tuple(range(40,55))
SEEDS=(38001,38002,38003)
ARMS=("additive","slow16","multiscale16")
WINDOWS=(32,128,512,1024)
TRAIN_BATCH=4
TRAIN_UPDATES_PER_WINDOW=48
TERMINAL_FRAMES=(1100,1200,1300,1400,1500,1600,1700,1800)
REAL_EP_FRAMES=2000
NUM_EXAMPLES=REAL_EP_FRAMES-2

def lock():
    d=json.loads(LOCK.read_text())
    if (d["dataset"]!="lerobot/aloha_mobile_chair"
        or d["revision_sha"]!="7249d8a935db04f846f9e669b788f26c3a19b645"
        or d["codebase_version"]!="v2.0" or d["expected_frames_each"]!=2000
        or d["source_eligibility_run"]!=40296589314
        or d["FIT"]!=list(FIT) or d["DEV"]!=list(DEV)
        or d["TEST"]!=list(TEST) or len(d["FIT_receipts"])!=30
        or d["DEV_episode_files_downloaded"]!=0
        or d["TEST_episode_files_downloaded"]!=0):
        raise RuntimeError("publisher source lock missing or mutated")
    return d

def download(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={
      "User-Agent":"S1TEMP007-original-1500-frame-source-test/1.0"}),
      timeout=120) as r:return r.read()

def get_episode(ep,locked,cache,*,released_dev=False):
    if type(ep) is not int or (ep not in FIT and
        not (released_dev and ep in DEV)):
        raise ValueError("source cutoff disallows this episode")
    cache.mkdir(parents=True,exist_ok=True)
    path=f"data/chunk-000/episode_{ep:06d}.parquet"
    file=cache/f"episode_{ep:06d}.parquet"
    if not file.exists():
        file.write_bytes(download(
            f"https://huggingface.co/datasets/{locked['dataset']}/resolve/"
            f"{locked['revision_sha']}/{path}"))
    raw=file.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if ep in FIT and locked["FIT_receipts"][ep]["sha256"]!=sha:
        raise RuntimeError("original FIT source SHA mismatch")
    t=pq.read_table(io.BytesIO(raw),columns=["episode_index","frame_index",
       "timestamp","observation.state","action"])
    if t.num_rows!=REAL_EP_FRAMES:raise RuntimeError("source length changed")
    d=t.to_pydict();rows=[];clock=None
    for i,(e,f,ts,x,a) in enumerate(zip(*(d[k] for k in (
         "episode_index","frame_index","timestamp","observation.state","action")))):
        x=np.asarray(x,np.float64);a=np.asarray(a,np.float64)
        if (int(e)!=ep or int(f)!=i or x.shape!=(14,) or a.shape!=(14,)
            or not np.isfinite(x).all() or not np.isfinite(a).all()
            or not math.isfinite(float(ts))
            or (clock is not None and not (0<float(ts)-clock<.2))):
            raise RuntimeError("source temporal/physical validity violation")
        rows.append((i,float(ts),x,a));clock=float(ts)
    return rows,dict(episode=ep,path=path,sha256=sha,original_frames=REAL_EP_FRAMES)

def fit_norm(src):
    raw_x=np.concatenate([np.stack([r[2] for r in src[e]]) for e in FIT])
    raw_a=np.concatenate([np.stack([r[3] for r in src[e]]) for e in FIT])
    xm=raw_x.mean(0);xs=np.maximum(raw_x.std(0),1e-3)
    am=raw_a.mean(0);ass=np.maximum(raw_a.std(0),1e-3)
    movements=np.concatenate([((np.diff(np.stack(
      [r[2] for r in src[e]]),axis=0))**2).mean(-1) for e in FIT])
    threshold=float(np.quantile(movements,.90))
    delta=np.concatenate([(np.diff(np.stack(
      [r[2] for r in src[e]]),axis=0)/xs) for e in FIT])
    scale=np.maximum(np.quantile(np.abs(delta),.95,axis=0),1e-4)
    if threshold<=0 or not np.isfinite(scale).all():
        raise RuntimeError("invalid real FIT-only normalization")
    return dict(state_mean=xm.tolist(),state_sd=xs.tolist(),
        action_mean=am.tolist(),action_sd=ass.tolist(),
        train_delta_scale_normalized=scale.tolist(),
        high_motion_threshold_raw_squared_units=threshold)

def encode(rows,norm):
    raw_x=np.stack([r[2] for r in rows])
    raw_a=np.stack([r[3] for r in rows])
    xs=np.asarray(norm["state_sd"]);xm=np.asarray(norm["state_mean"])
    ass=np.asarray(norm["action_sd"]);am=np.asarray(norm["action_mean"])
    x=(raw_x-xm)/xs; a=(raw_a-am)/ass
    events=(((raw_x[2:]-raw_x[1:-1])**2).mean(-1)>=
            norm["high_motion_threshold_raw_squared_units"]).astype(np.float32)
    return dict(x=torch.tensor(x[1:-1],dtype=torch.float32),
       u=torch.tensor(a[:-2],dtype=torch.float32),
       previous=torch.tensor(x[:-2],dtype=torch.float32),
       y=torch.tensor(x[2:],dtype=torch.float32),
       event=torch.tensor(events,dtype=torch.float32),
       target_phys=raw_x[2:],obs_phys=raw_x[1:-1],
       previous_phys=raw_x[:-2])

def sample_plans(seed):
    rng=np.random.default_rng(seed)
    plans=[]
    # Interleave equal numbers of all training window lengths.
    for _ in range(TRAIN_UPDATES_PER_WINDOW):
        for W in WINDOWS:
            ids=[FIT[int(rng.integers(0,len(FIT)))] for _ in range(TRAIN_BATCH)]
            starts=[int(rng.integers(0,NUM_EXAMPLES-W+1))
                    for _ in range(TRAIN_BATCH)]
            plans.append((W,ids,starts))
    return plans

def make_batch(src,W,ids,starts):
    def stack(k):
        return torch.stack([src[e][k][s:s+W]
                            for e,s in zip(ids,starts)],dim=0)
    return stack("x"),stack("u"),stack("y"),stack("event"),torch.stack(
        [src[e]["previous"][s] for e,s in zip(ids,starts)],dim=0)

def model_pair(seed,delta):
    torch.manual_seed(seed)
    base=VelocityAnchoredSlowSystem1(memory_mode="additive")
    base.set_train_delta_scale(delta)
    variants={k:VelocityAnchoredSlowSystem1(memory_mode=k) for k in ARMS}
    for m in variants.values():
        m.load_state_dict(base.state_dict(),strict=True)
        if m.carried_fp32_bytes!=664 or sum(p.numel() for p in m.parameters())!=95907:
            raise RuntimeError("not an equal-capacity 166-float original model")
    return variants

def eval_context(model,real_source,normalizer,H):
    xs=np.asarray(normalizer["state_sd"]);xm=np.asarray(normalizer["state_mean"])
    cases=[]
    pred_events=[];proxy_scores=[]
    with torch.no_grad():
        for ep in DEV:
            z=real_source[ep]
            tids=TERMINAL_FRAMES
            observed=torch.stack([z["x"][t-H+1:t+1] for t in tids])
            actions=torch.stack([z["u"][t-H+1:t+1] for t in tids])
            previous=torch.stack([z["previous"][t-H+1] for t in tids])
            result=model(observed,actions,
                initial_previous_observation=previous)
            physical=(result.next_observation[:,-1].cpu().numpy()
                      .astype(np.float64)*xs+xm)
            actual=z["target_phys"][list(tids)]
            baseline=2*z["obs_phys"][list(tids)]-z["previous_phys"][list(tids)]
            motion=z["event"][list(tids)].numpy().astype(bool)
            probability=result.adverse_event_score[:,-1].cpu().numpy()
            pred_events.extend(motion.astype(np.int64).tolist())
            proxy_scores.extend(probability.tolist())
            cases.append(dict(episode=ep,
                original_terminal_frames=list(tids),
                source_MSE=float(((physical-actual)**2).mean()),
                velocity_MSE=float(((baseline-actual)**2).mean()),
                source_high_motion_MSE=float(((physical[motion]-actual[motion])**2).mean())
                     if motion.any() else None,
                velocity_high_motion_MSE=float(((baseline[motion]-actual[motion])**2).mean())
                     if motion.any() else None,
                moving_cases=int(motion.sum())))
    y=np.asarray(pred_events);p=np.asarray(proxy_scores)
    from scripts.s1motion003_real_tape_fit import auprc,auroc
    positive=int(y.sum())
    return dict(real_targets=len(y),positive_motion_cases=positive,
        proxy_prevalence=float(y.mean()),
        mean_source_MSE=float(np.mean([q["source_MSE"] for q in cases])),
        mean_velocity_MSE=float(np.mean([q["velocity_MSE"] for q in cases])),
        high_motion_MSE=(float(np.mean([q["source_high_motion_MSE"] for q in cases
                                  if q["source_high_motion_MSE"] is not None])) if positive else None),
        high_motion_velocity_MSE=(float(np.mean([
           q["velocity_high_motion_MSE"] for q in cases
           if q["velocity_high_motion_MSE"] is not None])) if positive else None),
        motion_proxy_AUPRC=float(auprc(y,p)) if 0<positive<len(y) else None,
        motion_proxy_AUROC=float(auroc(y,p)) if 0<positive<len(y) else None,
        motion_proxy_Brier=float(np.mean((p-y)**2)),
        genuine_episode_clusters=10,episodes=cases)

def main():
    args=argparse.ArgumentParser()
    args.add_argument("--cache",default="/tmp/s1temp007-original")
    args.add_argument("--out",default="research/diagnostics/s1temp007-original")
    a=args.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    src_lock=lock();cache=Path(a.cache);out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    raw_fit={};FIT_receipts=[]
    for ep in FIT:
        raw_fit[ep],r=get_episode(ep,src_lock,cache)
        FIT_receipts.append(r)
    norm=fit_norm(raw_fit)
    fit={e:encode(raw_fit[e],norm) for e in FIT}
    scale=torch.tensor(norm["train_delta_scale_normalized"],dtype=torch.float32)
    original_models={};checkpoint_receipts=[]
    for seed in SEEDS:
        models=model_pair(seed,scale);windows=sample_plans(seed)
        for arm in ARMS:
            model=models[arm].train()
            opt=torch.optim.AdamW(model.parameters(),lr=.001)
            t0=time.perf_counter();n=0;applied_negative=0
            for W,episodes,starts in windows:
                b=make_batch(fit,W,episodes,starts)
                _,audit=step_model(model,opt,b,arm="tac_projected")
                if not torch.isfinite(torch.tensor(audit["post_point_fit"])):
                    raise RuntimeError("physical gradient source nonfinite")
                applied_negative+=int(audit["actual_point_gradient_dot_parameter_delta"]<0)
                n+=1
            model.eval()
            receipt=dict(arm=arm,seed=seed,train_steps=n,active_params=95907,
                real_source_frames_exposed=sum(W*TRAIN_BATCH for W,_,__ in windows),
                source_FIT_episodes=30,source_CPU_seconds=time.perf_counter()-t0,
                AdamW_actual_point_descent_dot_count=applied_negative,
                complete_train_FLOPs_not_proven_equal=True)
            pkg=dict(arm=arm,seed=seed,model=model.state_dict(),
                optimizer=opt.state_dict(),torch_rng=torch.random.get_rng_state(),
                immutable_publisher_sha=src_lock["revision_sha"],
                FIT_original_hashes=[r["sha256"] for r in FIT_receipts],
                source_norm=norm,receipt=receipt)
            f=io.BytesIO();torch.save(pkg,f)
            binary=f.getvalue()
            (out/f"{arm}-{seed}.pt").write_bytes(binary)
            receipt["model_optimizer_rng_SHA256"]=hashlib.sha256(binary).hexdigest()
            checkpoint_receipts.append(receipt)
            original_models[(seed,arm)]=model
            print("S1TEMP007_FROZEN",json.dumps(receipt,sort_keys=True),flush=True)
    before=dict(study="S1TEMP007",
        source_revision=src_lock["revision_sha"],
        prospective_amendment="research/parallel-six/S1TEMP007-AMENDMENT-001.md",
        source_fit_Sha256=FIT_receipts,normalizers_from_only_original_FIT=norm,
        preregistered_seed_list=list(SEEDS),arms=list(ARMS),train_windows=list(WINDOWS),
        updates_per_window=TRAIN_UPDATES_PER_WINDOW,updates_total=192,
        real_training_batch=TRAIN_BATCH,
        registered_terminal_time_indices=list(TERMINAL_FRAMES),
        all_nine_original_model_optimizer_rng_SHA_frozen_before_DEV=True,
        TEST_40_to_54_NEVER_DOWNLOADED=True,
        original_checkpoint_receipts=checkpoint_receipts)
    (out/"BEFORE-DEV-SHA-FREEZE.json").write_text(json.dumps(
        before,indent=2,sort_keys=True)+"\n")
    print("S1TEMP007_ALL_9_CHECKPOINT_SHA_FROZEN_BEFORE_DEV=YES",flush=True)
    # All original unseen DEV files physically inaccessible before this point.
    real_DEV={};DEV_receipts=[]
    for ep in DEV:
        raw,r=get_episode(ep,src_lock,cache,released_dev=True)
        DEV_receipts.append(r);real_DEV[ep]=encode(raw,norm)

    results=[]
    for seed in SEEDS:
        scores={}
        for arm in ARMS:
            model=original_models[(seed,arm)]
            for H in (128,512,1024):
                val=eval_context(model,real_DEV,norm,H)
                scores[f"{arm}_H{H}"]=val
                print("S1TEMP007_FIRST_REAL_DEV",json.dumps(dict(
                    seed=seed,arm=arm,H=H,MSE=val["mean_source_MSE"],
                    velocity=val["mean_velocity_MSE"],
                    motion_AUPRC=val["motion_proxy_AUPRC"],
                    motion_prevalence=val["proxy_prevalence"]),sort_keys=True),flush=True)
        multi=scores["multiscale16_H1024"]
        m128=scores["multiscale16_H128"]
        mono=scores["slow16_H1024"]
        additive=scores["additive_H1024"]
        context=multi["mean_source_MSE"]<=.99*m128["mean_source_MSE"]
        both=(multi["mean_source_MSE"]<=.99*mono["mean_source_MSE"]
               and multi["mean_source_MSE"]<=.99*additive["mean_source_MSE"])
        motion=(multi["high_motion_MSE"] is not None
                and multi["high_motion_velocity_MSE"] is not None
                and multi["high_motion_MSE"]<=1.05*multi["high_motion_velocity_MSE"])
        ranking=(all(z["motion_proxy_AUPRC"] is not None
                     for z in (multi,mono,additive))
                 and multi["motion_proxy_AUPRC"]>=.9*max(
                     mono["motion_proxy_AUPRC"],additive["motion_proxy_AUPRC"]))
        gates=dict(context_gain_one_percent=bool(context),
                   versus_both_controls_gain_one_percent=bool(both),
                   high_motion_MSE_guard=bool(motion),
                   proxy_AUPRC_retention_guard=bool(ranking))
        results.append(dict(seed=seed,scores=scores,
                registered_original_gates=gates,
                original_seed_gate=all(gates.values())))
    report=dict(before,DEV_original_file_SHA_receipts=DEV_receipts,
        ORIGINAL_FIRST_DEV_context_test=results,
        complete_original_gate_pass=all(r["original_seed_gate"] for r in results),
        complete_training_FLOPs_equal_NOT_verified=True,
        real_adverse_hazard_event_labels_available=False,
        safe_robot_control_qualified=False,
        causal_action_policy=False,
        independent_robot_site_qualification=False)
    (out/"FIRST-ORIGINAL-REAL-DEV-SCORES.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\\n")
    print("S1TEMP007_ORIGINAL_GATE="+("PASS" if report["complete_original_gate_pass"] else "FAIL"),flush=True)
    print("S1TEMP007_REAL_SCORES_COMPLETE=YES",flush=True)

if __name__=="__main__":main()
