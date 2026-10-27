"""PAR-MEM-011: original real-source paired training for lawful historical recall.

Two genuinely TRAINED full 98,326-parameter TAC models per seed:
same source-trained backbone init, learned content state, optimizer/data/batch
and computational graph. Only the final historical source query address
is changed: correct observed old event vs the next genuinely observed
adjacent *wrong* event. This is an ADDRESS control, not stateless GRU.

Precommit each source-owned 129-event history schedule BEFORE any fitting;
freeze ALL 10 originals (weights, AdamW, RNG, source plan) BEFORE opening
historically exposed original DEV source targets. Save every predicted
real source target and cold clean-checkout replay afterwards.

No new protected source year, independent TEST, action policy, or full
train-operator FLOP parity claim. Diagnostic training performance remains
exploratory until independent genuinely new cohorts and complete controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.partime002_real_source import SHA256, score, SITES
from scripts.partime005_continuous_real_training import (
    source_and_times, frozen, verify_originals, sha as sha_file
)
from tac_osm.variants.source_aligned_parallel_world import SourceAlignedParallelWorld

ROOT=Path(__file__).resolve().parents[1]
PROTO=ROOT/"research/parallel-six/PAR-MEM-011-PROTOCOL.json"
PLAN=ROOT/"research/parallel-six/PAR-MEM-011-FROZEN-PLAN.json"
OUT=ROOT/"research/replay-bundles/PAR-MEM-011"
OLD=ROOT/"research/replay-bundles/PAR-TIME-005"
SEEDS=(271028,271029,271030,271031,271032)
ARMS=("correct_shared_source_key","adjacent_wrong_source_key")
LAGS=(16,64,128)
ANCHORS=(160,192,224,256,288,320,352,384,416,448,480,512,544,576,608,640,672,704,736)
DEV_ANCHORS=(160,224,288,352,416,480,544,608,672,736)
STEPS=18
BATCH=4
T=129
LR=.0005

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def ready(source):
    cfg=json.loads(PROTO.read_text())
    checks=dict(study_id="PAR-MEM-011-REAL-SOURCE-TRAINED-SHARED-WRITE-READ",
                source_original_zip_sha256=SHA256,
                seeds=list(SEEDS),arms=list(ARMS),
                train_lags=list(LAGS),updates_per_seed_per_arm=STEPS,
                training_batch_sites=BATCH,
                registered_history_events_per_case=T,
                train_anchor_allowed_indices=list(ANCHORS),
                evaluation_indices=list(DEV_ANCHORS))
    for k,expected in checks.items():
        if cfg.get(k)!=expected:
            raise RuntimeError("frozen real-source protocol drift: "+k)
    plan=frozen(source)
    receipt=verify_originals(plan)
    data,fit,dev=source_and_times(source)
    if any(len(z)<=max(DEV_ANCHORS) for z in fit+dev):
        raise RuntimeError("incomplete genuine longitudinal source cohort")
    return data,fit,dev,plan,receipt

def sample(seed,fit):
    rng=random.Random(seed+11011)
    seq=[]
    for step in range(STEPS):
        lag=LAGS[step%len(LAGS)]
        sites=rng.sample(range(len(SITES)),BATCH)
        anchors=[rng.choice(ANCHORS) for _ in sites]
        seq.append(dict(lag=lag,sites=sites,anchors=anchors,
                        owner_source_clocks=[fit[s][a] for s,a in zip(sites,anchors)],
                        true_query_past_event_source_clocks=[
                            fit[s][a-lag] for s,a in zip(sites,anchors)]))
    return seq

def publish_plan(source):
    if PLAN.exists() or OUT.exists():
        raise RuntimeError("training plan/checkpoints already exist: no overwrite")
    _,fit,dev,base,receipt=ready(source)
    plan=dict(study_id="PAR-MEM-011-REAL-SOURCE-TRAINED-SHARED-WRITE-READ",
              source_SHA256=SHA256,source_training_clock_SHA256=base["train_source_clock_SHA256"],
              source_previously_exposed_DEV_clock_SHA256=base["exposed_DEV_source_clock_SHA256"],
              seeds=list(SEEDS),arms=list(ARMS),
              genuine_train_schedules_by_seed={str(s):sample(s,fit) for s in SEEDS},
              registered_DEV_source_clock_indices_by_site=[
                  [row[a] for a in DEV_ANCHORS] for row in dev],
              registration_before_any_optimizer_steps=True,
              developer_knows_exposed_DEV_not_independent_TEST=True,
              full_train_FLOP_parity_certified=False)
    PLAN.write_text(json.dumps(plan,sort_keys=True,indent=2)+"\n")
    print("PAR_MEM_011_SOURCE_PLAN_FROZEN_BEFORE_ANY_FIT="+json.dumps({
        "source":SHA256,"seeds":len(SEEDS),
        "registered_genuine_fit_cases":len(SEEDS)*STEPS*BATCH,
        "per_seed_sample_order_SHA256":{str(s):digest(plan["genuine_train_schedules_by_seed"][str(s)]) for s in SEEDS},
        "optimizer_steps_executed":0},sort_keys=True),flush=True)

def fixed(source):
    if not PLAN.exists():raise RuntimeError("original source query plan not precommitted")
    data,fit,dev,base,receipt=ready(source)
    p=json.loads(PLAN.read_text())
    if (p["study_id"]!="PAR-MEM-011-REAL-SOURCE-TRAINED-SHARED-WRITE-READ"
        or p["source_SHA256"]!=SHA256
        or p["source_training_clock_SHA256"]!=base["train_source_clock_SHA256"]
        or p["source_previously_exposed_DEV_clock_SHA256"]!=base["exposed_DEV_source_clock_SHA256"]
        or p["seeds"]!=list(SEEDS)
        or p["arms"]!=list(ARMS)
        or p["genuine_train_schedules_by_seed"]!={str(s):sample(s,fit) for s in SEEDS}
        or p["registered_DEV_source_clock_indices_by_site"]!=[
            [row[a] for a in DEV_ANCHORS] for row in dev]):
        raise RuntimeError("original source chronological plan changed after freeze")
    return p,data,fit,dev,receipt

def neural(seed,original):
    torch.manual_seed(seed+11011)
    model=SourceAlignedParallelWorld()
    c=model.contract()
    if c["learned_trainable_parameters"]!=98326 or c["total_fp32_per_stream"]!=410:
        raise RuntimeError("new source-aligned trained model geometry drift")
    exact=next((row for row in original["original_checkpoints"] if row["seed"]==seed),None)
    if exact is None:raise RuntimeError("trained original source backbone missing")
    source_path=OLD/f"seed-{seed}-original-continuous.pt"
    if sha_file(source_path)!=exact["original_checkpoint_SHA256"]:
        raise RuntimeError("original real-trained checkpoint has been changed")
    raw=torch.load(source_path,map_location="cpu",weights_only=False)
    model.backbone.load_state_dict(raw["model_state"],strict=True)
    return model

def source_batch(data,rows,sites,anchors,lag,arm):
    if arm not in ARMS or lag not in LAGS or len(sites)!=len(anchors):
        raise ValueError("unregistered historical query input")
    observations=[]; covariates=[]; gaps=[]; ordinals=[];queries=[];hist=[]
    nxt=[];valid=[]
    for site,anchor in zip(sites,anchors):
        if (site<0 or site>=12 or anchor not in ANCHORS+DEV_ANCHORS
            or anchor<T-1 or anchor>=len(rows[site])):
            raise ValueError("invalid source registered owner/anchor")
        clocks=torch.tensor(rows[site][anchor-T+1:anchor+1],dtype=torch.long)
        if clocks.shape!=(T,) or (clocks[1:]<=clocks[:-1]).any():
            raise RuntimeError("true source observation order changed")
        v=data["values"][site,clocks]
        ok=data["valid"][site,clocks]
        if not ok[:,0].all().item() or not (ok.sum(-1)>=12).all().item():
            raise RuntimeError("unobserved original PM2.5 used as source write")
        observations.append(torch.cat((v,ok.float().mean(-1,keepdim=True)),dim=-1))
        known_hour=(clocks.remainder(24)).float()*2*math.pi/24
        covariates.append(torch.cat((ok.float(),known_hour.sin().unsqueeze(-1)),dim=-1))
        gaps.append(torch.cat((torch.zeros(1),clocks[1:].sub(clocks[:-1]).float()),0))
        ordinals.append(clocks)
        # Source-known past position relative to current anchor is 128-lag
        # (not source hour arithmetic: real missing observations create gaps).
        older=T-1-lag
        true_clock=int(clocks[older])
        wrong_clock=int(clocks[older+1])
        if not true_clock<wrong_clock<int(clocks[-1]):
            raise RuntimeError("invalid historical source query visibility")
        q=(clocks-1).clamp_min(0)
        q[-1]=true_clock if arm==ARMS[0] else wrong_clock
        if not (q<clocks).all().item():
            raise RuntimeError("historical query used current/future observation")
        queries.append(q)
        hist.append(v[older,0])
        # Actual recorded next clock-hour outcome, AFTER all source inputs.
        target_ix=int(clocks[-1])+1
        target=data["values"][site,target_ix]
        mask=data["valid"][site,target_ix]
        nxt.append(torch.cat((target,mask.float().mean().view(1)),dim=0))
        valid.append(mask)
    return dict(obs=torch.stack(observations),
                action=torch.stack(covariates),
                dt=torch.stack(gaps),indexes=torch.stack(ordinals),
                query=torch.stack(queries),
                owners=torch.tensor(sites,dtype=torch.long),
                past=torch.stack(hist),next=torch.stack(nxt),
                next_valid=torch.stack(valid))

def forward_loss(model,batch):
    pred,trace=model(batch["obs"],batch["action"],
          elapsed_dt=batch["dt"],
          write_mask=torch.ones_like(batch["dt"],dtype=torch.bool),
          source_ids=batch["owners"],event_index=batch["indexes"],
          query_event_index=batch["query"],return_trace=True)
    historical=model.historical_content(trace["aligned_old_read"][:,-1])[:,0]
    hist_loss=F.smooth_l1_loss(historical,batch["past"])
    forecast_loss=score(pred[:,-1],batch["next"],batch["next_valid"])
    total=hist_loss+.15*forecast_loss
    if not (torch.isfinite(total).item() and historical.isfinite().all().item()):
        raise RuntimeError("source history content loss not finite")
    return total,historical,pred[:,-1]

def train(source):
    if OUT.exists():raise RuntimeError("original neural source output directory already frozen")
    p,data,fit,_,original=fixed(source)
    OUT.mkdir(parents=True,exist_ok=False)
    rows=[]
    for seed in SEEDS:
        # Paired arms reuse EXACT SAME initialized weights/backbone.
        # Both train all genuinely reachable weights on same source cases.
        for arm in ARMS:
            random.seed(seed+11011)
            model=neural(seed,original)
            model.train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=LR,
                  weight_decay=.001,foreach=False,fused=False)
            before={n:w.detach().clone() for n,w in model.named_parameters()}
            losses=[];first_grads_checked=False
            start=time.monotonic()
            for update,case in enumerate(p["genuine_train_schedules_by_seed"][str(seed)]):
                batch=source_batch(data,fit,case["sites"],case["anchors"],case["lag"],arm)
                optimizer.zero_grad(set_to_none=True)
                total,historical,forecast=forward_loss(model,batch)
                total.backward()
                if any(v.grad is None or not v.grad.isfinite().all().item()
                       for v in model.parameters()):
                    raise RuntimeError("genuine source training contains unused/frozen/nonfinite learned weights")
                first_grads_checked=True
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                losses.append(float(total.detach()))
                if len(losses)!=update+1:raise RuntimeError("missing original optimizer step")
            if not first_grads_checked or not all(
                not torch.equal(p0,v.detach()) for n,v in model.named_parameters()
                for p0 in [before[n]]):
                raise RuntimeError("source model is not fully trained or has unused parameter")
            path=OUT/f"seed-{seed}-{arm}-original.pt"
            torch.save(dict(
              study_id="PAR-MEM-011-REAL-SOURCE-TRAINED-SHARED-WRITE-READ",
              seed=seed,arm=arm,model_state=model.state_dict(),
              optimizer_state=optimizer.state_dict(),
              torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
              exact_source_SHA256=SHA256,
              fit_schedule_digest=digest(p["genuine_train_schedules_by_seed"][str(seed)]),
              pretrained_backbone_SHA256=next(x["original_checkpoint_SHA256"]
                 for x in original["original_checkpoints"] if x["seed"]==seed),
              live_parameters=98326,full_FP32_state_per_owner=410,
              model_source_SHA256=sha_file(ROOT/"src/tac_osm/variants/source_aligned_parallel_world.py"),
              runner_source_SHA256=sha_file(ROOT/"scripts/parmem011_genuine_source_train.py"),
              optimizer_updates=STEPS,
              no_protected_numerical_source_used=True,
              exposed_development_not_yet_scored=True,
              source_training_loss_trace=losses),path)
            row=dict(seed=seed,arm=arm,
                original_checkpoint_SHA256=sha_file(path),
                start_loss=losses[0],last_loss=losses[-1],
                avg_last_six=sum(losses[-6:])/6,
                training_source_cases=STEPS*BATCH,
                elligible_training_steps=STEPS,
                wall_CPU_seconds_not_compute_parity=time.monotonic()-start)
            rows.append(row)
            print("PAR_MEM_011_FIVE_SEED_ORIGINAL_REAL_SOURCE_FIT="+json.dumps(row,sort_keys=True),flush=True)
    (OUT/"TRAIN_RECEIPT.json").write_text(json.dumps(dict(
       original_models=len(rows),source_SHA256=SHA256,
       all_original_checkpoints_GIT_FROZEN_BEFORE_DEV=True,
       original_seed_trainable_param_count=98326,
       registered_sample_order_source_SHA256=digest(p["genuine_train_schedules_by_seed"]),
       full_training_OPERATOR_FLOP_parity_certified=False,
       real_2014_DEV_previously_exposed=True,records=rows),indent=2,sort_keys=True)+"\n")

def frozen_models(p,original):
    receipt=json.loads((OUT/"TRAIN_RECEIPT.json").read_text())
    if (receipt["original_models"]!=10 or
        receipt["registered_sample_order_source_SHA256"]!=digest(p["genuine_train_schedules_by_seed"])
        or sorted((z["seed"],z["arm"]) for z in receipt["records"])!=
            sorted((seed,arm) for seed in SEEDS for arm in ARMS)):
        raise RuntimeError("original ten original real-source query weights not frozen")
    for row in receipt["records"]:
        path=OUT/f"seed-{row['seed']}-{row['arm']}-original.pt"
        if sha_file(path)!=row["original_checkpoint_SHA256"]:
            raise RuntimeError("old original model checkpoint byte identity drift")
        ck=torch.load(path,map_location="cpu",weights_only=False)
        if (ck["exact_source_SHA256"]!=SHA256
            or ck["live_parameters"]!=98326
            or ck["full_FP32_state_per_owner"]!=410
            or ck["optimizer_updates"]!=STEPS
            or ck["fit_schedule_digest"]!=digest(p["genuine_train_schedules_by_seed"][str(row["seed"])])
            or ck["model_source_SHA256"]!=sha_file(ROOT/"src/tac_osm/variants/source_aligned_parallel_world.py")
            or ck["runner_source_SHA256"]!=sha_file(ROOT/"scripts/parmem011_genuine_source_train.py")):
            raise RuntimeError("original source-trained model/optimizer protocol modified")
    return receipt

def evaluate(source,*,replay):
    p,data,_,dev,original=fixed(source)
    receipt=frozen_models(p,original)
    output=OUT/"DEV_RESULTS.json"
    if output.exists()!=replay:
        raise RuntimeError("exposed real DEV results create-once, replay-only")
    reports=[]
    for seed in SEEDS:
        for arm in ARMS:
            source_ck=OUT/f"seed-{seed}-{arm}-original.pt"
            snapshot=torch.load(source_ck,map_location="cpu",weights_only=False)
            model=neural(seed,original)
            model.load_state_dict(snapshot["model_state"],strict=True)
            model.eval()
            preds=[];true=[];recent=[]; future_pred=[];future_true=[]; future_valid=[]
            lags=[];ids=[]
            with torch.no_grad():
                for anchor in DEV_ANCHORS:
                    for lag in LAGS:
                        b=source_batch(data,dev,list(range(12)),[anchor]*12,lag,arm)
                        _,historical,forecast=forward_loss(model,b)
                        preds.append(historical.detach())
                        true.append(b["past"].detach())
                        recent.append(b["obs"][:,-1,0].detach())
                        future_pred.append(forecast[:,0].detach())
                        future_true.append(b["next"][:,0].detach())
                        future_valid.append(b["next_valid"][:,0].detach())
                        lags.extend([lag]*12)
                        ids.extend([(site,anchor,lag) for site in range(12)])
            bundle=dict(prediction=torch.cat(preds),truth=torch.cat(true),
                 source_current=torch.cat(recent),
                 forecast_PM25=torch.cat(future_pred),
                 next_true_PM25=torch.cat(future_true),
                 next_PM25_valid=torch.cat(future_valid),
                 lag=torch.tensor(lags,dtype=torch.int64),
                 source_identity=torch.tensor(ids,dtype=torch.int64))
            archive=OUT/f"seed-{seed}-{arm}-original_DEV_tensors.pt"
            if replay:
                saved=torch.load(archive,map_location="cpu",weights_only=True)
                for k,v in bundle.items():
                    torch.testing.assert_close(saved[k],v,rtol=1e-5,atol=1e-6)
            else:torch.save(bundle,archive)
            metrics={}
            for lag in LAGS:
                mask=bundle["lag"]==lag
                err=(bundle["prediction"][mask]-bundle["truth"][mask]).abs().mean()
                now=(bundle["source_current"][mask]-bundle["truth"][mask]).abs().mean()
                fm=bundle["next_PM25_valid"] & mask
                physical=(bundle["forecast_PM25"][fm]-bundle["next_true_PM25"][fm]).abs().mean()
                metrics[str(lag)]=dict(
                    historical_original_source_observed_queries=int(mask.sum()),
                    normalized_true_historical_PM25_MAE=float(err),
                    normalized_source_current_copy_historical_baseline_MAE=float(now),
                    next_source_PM25_labels=int(fm.sum()),
                    normalized_next_clock_hour_PM25_MAE=float(physical))
            reports.append(dict(seed=seed,arm=arm,
                 original_checkpoint_SHA256=sha_file(source_ck),
                 historical_genuine_source_lag_results=metrics))
            print("PAR_MEM_011_REAL_ORIGINAL_DEV="+json.dumps(reports[-1],sort_keys=True),flush=True)
    result=dict(study_id="PAR-MEM-011-REAL-SOURCE-TRAINED-SHARED-WRITE-READ",
      source_original_SHA256=SHA256,
      ten_full_neural_models_original_from_precommitted_training_schedule=True,
      original_record_count=len(reports),original_seed_count=5,
      true_observed_source_events_and_values=True,
      historical_query_target_never_passed_as_query_metadata=True,
      previously_exposed_2014_DEV_not_independent=True,
      full_optimizer_and_backward_FLOP_parity_not_yet_qualified=True,
      no_independent_architecture_superiority_claim=True,
      original_source_development_records=reports)
    if replay:
        old=json.loads(output.read_text())
        def compare(a,b,k="root"):
            if type(a)!=type(b):raise RuntimeError("source replay schema drift: "+k)
            if isinstance(a,dict):
                if set(a)!=set(b):raise RuntimeError("replay field drift: "+k)
                for item in a:compare(a[item],b[item],k+"."+str(item))
            elif isinstance(a,list):
                if len(a)!=len(b):raise RuntimeError("replay seed count drift: "+k)
                for i,(v,w) in enumerate(zip(a,b)):compare(v,w,k+f".{i}")
            elif isinstance(a,float):
                if abs(a-b)>1e-5*max(1.,abs(a)):
                    raise RuntimeError("source real replay metric drift: "+k)
            elif a!=b:raise RuntimeError("source history replay provenance drift: "+k)
        compare(old,result)
        print("PAR_MEM_011_CLEAN_INDEPENDENT_TEN_REAL_TRAINED_MODELS_REPLAY_PASS",flush=True)
    else:output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",required=True,choices=("plan","fit","eval","replay"))
    parser.add_argument("--source",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.mode=="plan":publish_plan(args.source)
    elif args.mode=="fit":train(args.source)
    else:evaluate(args.source,replay=args.mode=="replay")
