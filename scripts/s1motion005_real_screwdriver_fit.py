#!/usr/bin/env python3
"""Original S1MOTION005 genuine robotics five-arm source-pinned first test.

Physical per-episode source isolation: FIT IDs 0..24 downloaded/parsed
BEFORE original 15 checkpoint freeze; DEV IDs 25..34 ONLY downloaded AFTER
freeze. Original TEST IDs 35..49 NEVER downloaded. Not actual adverse risk.
"""
from __future__ import annotations
import argparse, copy, hashlib, io, json, math, sys, time, urllib.request
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.s1motion004_real_towel_fit import prepare_fit_only, encode_source_episode
from scripts.s1motion003_real_tape_fit import windows,batch,score,sha256
from tac_osm.variants.system1_velocity_anchored import VelocityAnchoredSystem1
from tac_osm.variants.s1motion004_gradient_routes import VelocityGRUControl,losses
from tac_osm.variants.s1motion005_gradient_budget import step_model,ARMS as TAC_ARMS,RHO,MAX_BETA

LOCK=ROOT/"research/parallel-six/S1MOTION005-SOURCE-LOCK.json"
SEEDS=(29001,29002,29003)
FIT=tuple(range(25));DEV=tuple(range(25,35))
ARMS=(*TAC_ARMS,"gru_shared","gru_isolated")
UPDATES=192;BATCH=8;WINDOW=32


def fetch(url):
    req=urllib.request.Request(url,headers={
        "User-Agent":"TAC-OSM-S1MOTION005-original-first-genuine/1.0"})
    with urllib.request.urlopen(req,timeout=120) as h:
        return h.read()


def episode_data(ep:int,source:dict,cache:Path,*,expected_sha:str|None):
    if type(ep) is not int or ep not in (*FIT,*DEV):
        raise ValueError("protected/unrecognized source episode access blocked")
    cache.mkdir(parents=True,exist_ok=True)
    key=f"{ep:06d}"
    path=f"data/chunk-000/episode_{key}.parquet"
    filename=cache/f"episode_{key}.parquet"
    if not filename.exists():
        data=fetch(f"https://huggingface.co/datasets/{source['dataset']}/resolve/"
                   f"{source['revision_sha']}/{path}")
        filename.write_bytes(data)
    blob=filename.read_bytes()
    digest=hashlib.sha256(blob).hexdigest()
    if expected_sha is not None and digest!=expected_sha:
        raise RuntimeError(f"FIT source checksum mismatch {ep}")
    table=pq.read_table(io.BytesIO(blob),columns=[
        "episode_index","frame_index","timestamp","observation.state","action"])
    if table.num_rows!=400:
        raise RuntimeError(f"source frame-count mismatch {ep}")
    cols=table.to_pydict()
    frames=[]
    previous_ts=None
    for i,(e,frame,stamp,x,u) in enumerate(zip(*(cols[k] for k in [
          "episode_index","frame_index","timestamp","observation.state","action"]))):
        x=np.asarray(x,dtype=np.float64);u=np.asarray(u,dtype=np.float64)
        if (int(e)!=ep or int(frame)!=i or
            x.shape!=(14,) or u.shape!=(14,) or
            not np.isfinite(x).all() or not np.isfinite(u).all() or
            not math.isfinite(float(stamp))):
            raise ValueError(f"non-source-authorized episode/frame {ep}:{i}")
        if previous_ts is not None and not (0<float(stamp)-previous_ts<.2):
            raise ValueError(f"real timestamp discontinuity {ep}:{i}")
        previous_ts=float(stamp)
        if i<257:frames.append((i,float(stamp),x,u))
    return frames,dict(episode=ep,path=path,sha256=digest,frames=400)


def approved_source_metadata(lock):
    if (lock["dataset"]!="lerobot/aloha_static_screw_driver"
        or lock["revision_sha"]!="0b873be59c9702f8d3de89022617121970db112f"
        or lock["source_codebase"]!="v2.0"
        or lock["fit_only_real_values_inspected"] is not True
        or lock["no_dev_or_test_episode_file_was_downloaded"] is not True
        or len(lock["fit_files"])!=25
        or [r["episode"] for r in lock["fit_files"]]!=list(FIT)
        or lock["original_source_only_run"]!=40282959710):
        raise RuntimeError("research source not fully pinned prior to fitting")


def original_models(seed,delta):
    torch.manual_seed(seed)
    tac=VelocityAnchoredSystem1()
    tac.set_train_delta_scale(delta)
    torch.manual_seed(seed+100_000)
    control=VelocityGRUControl(delta,False)
    initial={}
    for arm in ARMS:
        initial[arm]=copy.deepcopy(tac if arm.startswith("tac_") else control)
        if arm=="gru_isolated":
            initial[arm].detach_auxiliary=True
    return initial


def fit_arm(arm,model,seed,train,windows0,output,lock):
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    start=time.perf_counter()
    inspect=[]
    for index,(selected,starts) in enumerate(windows0):
        d=batch(train,selected,starts)
        if arm.startswith("tac_"):
            v,meta=step_model(model,opt,d,arm=arm)
        else:
            opt.zero_grad(set_to_none=True)
            value,_=losses(model,*d,isolated=arm=="gru_isolated")
            if not torch.isfinite(value).item():
                raise RuntimeError("nonfinite source loss")
            point_before=float((model(d[0],d[1],d[4])[0]-d[2]).square().mean().detach())
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            with torch.no_grad():
                point_after=float((model(d[0],d[1],d[4])[0]-d[2]).square().mean())
            v=float(value.detach())
            meta=dict(pre_point_fit=point_before,post_point_fit=point_after,
                      actual_AdamW_direction_logged=False)
        if not math.isfinite(v):
            raise RuntimeError("nonfinite source optimizer output")
        inspect.append(meta)
    elapsed=time.perf_counter()-start
    model.eval()
    snapshot=dict(model=model.state_dict(),optimizer=opt.state_dict(),
        torch_rng=torch.random.get_rng_state(),arm=arm,seed=seed,
        source_revision=lock["revision_sha"],
        source_FIT_checksums=[f["sha256"] for f in lock["fit_files"]],
        source_only_audit_run=lock["original_source_only_run"],
        updates=UPDATES,batch=BATCH,window=WINDOW)
    binary=io.BytesIO();torch.save(snapshot,binary)
    raw=binary.getvalue()
    output.mkdir(parents=True,exist_ok=True)
    (output/f"{arm}-{seed}.pt").write_bytes(raw)
    before=sum(1 for x in inspect if x.get("post_point_fit",0.) <= x.get("pre_point_fit",float("inf")))
    descending=sum(1 for x in inspect if x.get("actual_grad_dot_parameter_delta",0.)<0)
    record=dict(arm=arm,seed=seed,params=sum(p.numel() for p in model.parameters()),
        training_seconds=elapsed,model_SHA256=sha256(raw),
        updates=UPDATES,frames_exposed=UPDATES*BATCH*WINDOW,
        identical_FIT_source_plan=True,
        within_Tac_equal_two_autograd_calls=arm.startswith("tac_"),
        actual_FIT_poststep_point_improvement_steps=before,
        actual_FIT_AdamW_gradient_descent_direction_steps=(descending if arm.startswith("tac_") else None),
        mean_FIT_poststep_point_delta=float(np.mean([
            x["post_point_fit"]-x["pre_point_fit"] for x in inspect])),
        last_loss=v)
    (output/f"{arm}-{seed}-fit-updates.json").write_text(json.dumps(
        inspect,indent=2,sort_keys=True)+"\n")
    return record


def main():
    args=argparse.ArgumentParser()
    args.add_argument("--cache",default="/tmp/s1motion005-original")
    args.add_argument("--out",default="research/diagnostics/s1motion005-original")
    a=args.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lock=json.loads(LOCK.read_text());approved_source_metadata(lock)
    cache=Path(a.cache);out=ROOT/a.out
    # PHYSICAL no-download guarantee: this block accesses only FIT episode ids.
    fit_source={}; source_receipts=[]
    for r in lock["fit_files"]:
        ep=r["episode"]
        x,receipt=episode_data(ep,lock,cache,expected_sha=r["sha256"])
        fit_source[ep]=x;source_receipts.append(receipt)
    fit,norm=prepare_fit_only(fit_source)
    delta=torch.tensor(norm["train_delta_scale_normalized"],dtype=torch.float32)
    trained={}; records=[]
    for seed in SEEDS:
        models=original_models(seed,delta)
        sample=windows(fit,seed)
        for arm in ARMS:
            record=fit_arm(arm,models[arm],seed,fit,sample,out,lock)
            trained[(seed,arm)]=models[arm]
            records.append(record)
            print("S1MOTION005_FIT_FROZEN",json.dumps(record,sort_keys=True),flush=True)
    pre=dict(study="S1MOTION005",
        preregistered_protocol="research/parallel-six/S1MOTION005-PROTOCOL.md",
        source_locked_revision=lock["revision_sha"],
        source_license_provenance=lock["license_evidence_scope"],
        FIT_authorized_receipts=source_receipts,
        FIT_ep_ids=list(FIT),DEV_ep_ids=list(DEV),
        TEST_ep_ids_never_downloaded=list(range(35,50)),
        normalization_FIT_only=norm,
        five_arms=list(ARMS),seeds=list(SEEDS),frozen_models=records,
        all_15_model_optimizer_RNG_state_blobs_SHA_frozen_before_any_DEV_file=True,
        full_training_FLOP_parity_not_proven=True,
        actual_adverse_event_labels_available=False)
    out.mkdir(parents=True,exist_ok=True)
    (out/"BEFORE-DEV-ALL-FROZEN.json").write_text(json.dumps(pre,indent=2,sort_keys=True)+"\n")
    print("S1MOTION005_ALL_15_MODELS_SHA_FROZEN_BEFORE_DEV_DOWNLOAD=YES",flush=True)
    # After all 15 original models are frozen, permit only DEV files 25..34;
    # protected TEST bytes are NEVER downloaded by any function in this run.
    dev_source={};dev_receipts=[]
    for ep in DEV:
        x,receipt=episode_data(ep,lock,cache,expected_sha=None)
        dev_source[ep]=x;dev_receipts.append(receipt)
    dev={ep:encode_source_episode(dev_source[ep],norm) for ep in DEV}
    results=[]
    for seed in SEEDS:
        scores={}
        for arm in ARMS:
            model=trained[(seed,arm)]
            observed=score(model,dev,norm)
            scores[arm]=observed
            print("S1MOTION005_FIRST_DEV",json.dumps(dict(
                seed=seed,arm=arm,raw_joint_MSE=observed["raw_joint_MSE"],
                high_motion_MSE=observed["raw_joint_high_motion_MSE"],
                proxy_AUPRC=observed["proxy_AUPRC"],
                proxy_AUROC=observed["proxy_AUROC"],
                proxy_Brier=observed["proxy_Brier"],
                proxy_prevalence=observed["proxy_prevalence"]),
                sort_keys=True),flush=True)
        p=scores["tac_projected"];shared=scores["tac_shared"];isolated=scores["tac_isolated"]
        m=p["raw_joint_MSE"]["tac"];v=p["raw_joint_MSE"]["velocity"]
        h=p["raw_joint_high_motion_MSE"]
        g2=m<=.99*v and m<=min(shared["raw_joint_MSE"]["tac"],isolated["raw_joint_MSE"]["tac"])
        g3=h["tac"] is not None and h["velocity"] is not None and h["tac"]<=1.05*h["velocity"]
        g4=p["proxy_AUPRC"]>=.9*shared["proxy_AUPRC"]
        results.append(dict(seed=seed,score=scores,
            g2_predictive_pass=bool(g2),g3_high_motion_pass=bool(g3),
            g4_proxy_ranking_pass=bool(g4),original_seed_gate_pass=bool(g2 and g3 and g4)))
    report=dict(pre,dev_downloaded_only_after_ALL_models_frozen=dev_receipts,
        original_first_dev_results=results,
        full_original_predictive_gate_pass=all(x["original_seed_gate_pass"] for x in results),
        all_15_frozen_before_DEV=True,
        independent_site_or_actor_qualification=False,
        protected_TEST_files_downloaded=False,
        real_adverse_or_collision_risk_labels_available=False,
        equal_training_full_FLOPs_verified=False)
    (out/"ORIGINAL-FIRST-DEV-SCORES.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("S1MOTION005_ORIGINAL_FIRST_DEV_COMPLETE=YES",flush=True)
    print("S1MOTION005_ORIGINAL_GATE="+("PASS" if report["full_original_predictive_gate_pass"] else "FAIL"),flush=True)
    print("S1MOTION005_REAL_WORLD_SAFETY_QUALIFIED=NO",flush=True)

if __name__=="__main__":
    main()
