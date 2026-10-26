"""LATFUSE002 original UCI501 real-world development fit and clean replay.

No synthetic observations/actions. Future-known 6D calendar covariates are
non-interventional conditioning. 2016-17 numerical target rows never parsed.
Report DEV2015 already historically exposed; this cannot confer qualification.
Source normalization is fit on 2013-03..2014-08 ONLY. Exact train plan,
seed/update budget and held-out origins are registered BEFORE first optimizer.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import zipfile

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from tac_osm.latent_fusion_002 import FusedLatentFixedState,state_contract
from scripts.validate_research_parity import check_manifest

SOURCE_SHA="d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8"
SITES=("Aotizhongxin","Changping","Dingling","Dongsi","Guanyuan","Gucheng",
       "Huairou","Nongzhanguan","Shunyi","Tiantan","Wanliu","Wanshouxigong")
CHANNELS=("PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM")
WIND=("N","NNE","NE","ENE","E","ESE","SE","S","SSW","SW","WSW","W","WNW","NW","NNW")
START=dt.datetime(2013,3,1)
TRAIN_END=dt.datetime(2014,9,1)
VAL_END=dt.datetime(2015,1,1)
DEV_END=dt.datetime(2016,1,1)
H=24
HORIZON=6
BATCH=8
STEPS=128
SEEDS=(202601,202602)
OUT=ROOT/"research/replay-bundles/LATFUSE002"
DOC=ROOT/"research/mathematical-engineering/latfuse002"
REG=DOC/"FAIR_COMPARISON.json"
EXPECTED_PARAMS=98775


def hour(date):
    return int((date-START).total_seconds()//3600)


def source_frames(path):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise RuntimeError("original source SHA mismatch")
    records=[]
    with zipfile.ZipFile(path) as archive:
        for site in SITES:
            names=[p for p in archive.namelist() if p.endswith(f"PRSA_Data_{site}_20130301-20170228.csv")]
            if len(names)!=1:raise RuntimeError("original station file unresolved")
            vals=[]
            with archive.open(names[0]) as fp:
                rdr=csv.DictReader(io.TextIOWrapper(fp,encoding="utf-8-sig"))
                for row in rdr:
                    # CUT OFF BEFORE parsing protected 2016-2017 numeric fields.
                    if int(row["year"])>=2016:break
                    if row["station"]!=site:raise RuntimeError("source station spoof")
                    when=dt.datetime(*(int(row[k]) for k in ("year","month","day","hour")))
                    if hour(when)!=len(vals):raise RuntimeError("source chronological order/gap mismatch")
                    x=[float(row[k]) if row[k]!="NA" else float("nan") for k in CHANNELS]
                    w=row["wd"]
                    if w=="NA":x.extend((float("nan"),float("nan")))
                    else:
                        if w not in WIND:raise RuntimeError("unregistered source wind")
                        a=2*math.pi*WIND.index(w)/16
                        x.extend((math.sin(a),math.cos(a)))
                    if not all(math.isfinite(q) or math.isnan(q) for q in x):
                        raise RuntimeError("invalid infinite source input")
                    if any(q<0 for q in x[:6] if math.isfinite(q)):
                        raise RuntimeError("invalid negative pollutant observation")
                    vals.append(x)
            if len(vals)!=hour(DEV_END):raise RuntimeError("unexpected original source truncation")
            records.append(vals)
    raw=torch.tensor(records,dtype=torch.float32)
    if raw.shape!=(12,hour(DEV_END),13):raise RuntimeError("source contract changed")
    x=raw.clone()
    x[:,:,:6]=torch.where(torch.isfinite(raw[:,:,:6]),
                          torch.log1p(raw[:,:,:6].clamp_min(0)),raw[:,:,:6])
    cutoff=hour(TRAIN_END)
    fit=x[:,:cutoff]
    observed=torch.isfinite(fit)
    count=observed.sum((0,1)).clamp_min(1)
    center=torch.where(observed,fit,0).sum((0,1))/count
    spread=(torch.where(observed,fit-center,0).square().sum((0,1))/count).clamp_min(1e-6).sqrt()
    normalized=(x-center)/spread
    normalobs=torch.isfinite(normalized)
    safe=torch.where(normalobs,normalized,torch.zeros_like(normalized))
    # Compute source-last-value and elapsed since last observation
    # causally from t=0. No future target may refresh these caches.
    last=torch.zeros_like(safe)
    elapsed=torch.empty_like(safe)
    recent=torch.zeros((12,13),dtype=torch.float32)
    age=torch.full((12,13),256.,dtype=torch.float32)
    for t in range(raw.shape[1]):
        last[:,t]=recent
        mask=normalobs[:,t]
        age=torch.where(mask,torch.zeros_like(age),torch.clamp(age+1,max=256.))
        elapsed[:,t]=age
        recent=torch.where(mask,safe[:,t],recent)
    return dict(raw=raw,values=safe,valid=normalobs,last=last,age=elapsed,
                center=center,spread=spread,source_sha=SOURCE_SHA)


def calendar(hours):
    """Known timestamp only, no actual weather or future sensor reading."""
    rows=[]
    for h in hours:
        when=START+dt.timedelta(hours=int(h))
        a=[2*math.pi*when.hour/24,2*math.pi*when.weekday()/7,
           2*math.pi*(when.timetuple().tm_yday-1)/366]
        rows.append([math.sin(z) for z in a]+[math.cos(z) for z in a])
    return torch.tensor(rows,dtype=torch.float32)


def registered_plan():
    plan=json.loads(REG.read_text())
    check_manifest(plan,"preflight")
    if (plan["study_id"]!="LATFUSE002-REAL-PREDICTION-SINGLE" or
        tuple(x["seed"] for x in plan["models"])!=SEEDS or
        any(x["trainable_parameters"]!=EXPECTED_PARAMS or x["deployed_parameters"]!=EXPECTED_PARAMS for x in plan["models"])):
        raise RuntimeError("registered study source/seed/size changed")
    protocol=plan["protocol"]
    if (protocol["updates"]!=STEPS or protocol["batch_size"]!=BATCH or
        protocol["history"]!=H or protocol["horizon"]!=HORIZON):
        raise RuntimeError("resource plan mismatch")
    return plan


def sample_origins(seed):
    """Immutable input IDs and order; independent of targets and outcomes."""
    rng=random.Random(seed)
    return [(rng.randrange(12),rng.randrange(H-1,hour(TRAIN_END)-HORIZON))
            for _ in range(STEPS*BATCH)]


def eval_origins(a,b):
    # Completely chronological and deterministic; never based on target availability.
    return [(site,t) for t in range(a+H, b-HORIZON,336) for site in range(12)]


def evaluate(model,d,origins,*,store=False):
    model.eval()
    store_pred=[];store_truth=[];store_valid=[];store_persist=[]
    with torch.no_grad():
        for offset in range(0,len(origins),8):
            batch=origins[offset:offset+8]
            sid=torch.tensor([p[0] for p in batch],dtype=torch.long)
            time_ix=torch.tensor([p[1] for p in batch],dtype=torch.long)
            memory=model.initial(len(batch))
            for h in range(H):
                ts=time_ix-H+1+h
                memory=model.step(memory,d["values"][sid,ts],d["valid"][sid,ts],
                                  d["age"][sid,ts],d["last"][sid,ts],sid)
            q=calendar([int(t+HORIZON) for t in time_ix]).unsqueeze(1)
            pred,scale=model.forecast(memory,sid,q)
            target=d["values"][sid,time_ix+HORIZON]
            eligible=d["valid"][sid,time_ix+HORIZON]
            baseline=d["last"][sid,time_ix+1] # last legally observed at or before t
            store_pred.append(torch.stack((pred[:,0,:],scale[:,0,:]),dim=1))
            store_truth.append(target)
            store_valid.append(eligible)
            store_persist.append(baseline)
    if not origins:raise RuntimeError("empty fixed real chronological cohort")
    forecast=torch.cat(store_pred)
    truth=torch.cat(store_truth)
    valid=torch.cat(store_valid)
    persist=torch.cat(store_persist)
    mean,sd=forecast[:,0,:],forecast[:,1,:]
    safe=torch.where(valid,truth,mean)
    # Numeric Gaussian model on normalized 13D source, NOT censored-normal law.
    nll=.5*(math.log(2*math.pi)+2*torch.log(sd)+((safe-mean)/sd).square())
    nll=float(torch.where(valid,nll,torch.zeros_like(nll)).sum()/valid.sum().clamp_min(1))
    mae=float(torch.where(valid,(mean-safe).abs(),torch.zeros_like(mean)).sum()/valid.sum().clamp_min(1))
    base_mae=float(torch.where(valid,(persist-safe).abs(),torch.zeros_like(mean)).sum()/valid.sum().clamp_min(1))
    pm=valid[:,0]
    physical=(torch.expm1((mean[:,0]*d["spread"][0]+d["center"][0]).clamp(-12,15))).clamp_min(0)
    physical_y=torch.expm1((safe[:,0]*d["spread"][0]+d["center"][0]).clamp(-12,15)).clamp_min(0)
    baseline_pm=torch.expm1((persist[:,0]*d["spread"][0]+d["center"][0]).clamp(-12,15)).clamp_min(0)
    pmae=float((physical[pm]-physical_y[pm]).abs().mean()) if bool(pm.any()) else None
    pbmae=float((baseline_pm[pm]-physical_y[pm]).abs().mean()) if bool(pm.any()) else None
    cover=float(torch.where(valid,((safe-mean).abs()<=1.6448536269514722*sd),
                            torch.zeros_like(valid)).sum()/valid.sum().clamp_min(1))
    result=dict(n_origins=len(origins),n_valid_channels=int(valid.sum()),nll=nll,
                normalized_mae=mae,persistence_normalized_mae=base_mae,
                pm25_mae=pmae,persistence_pm25_mae=pbmae,nominal_90_coverage=cover)
    if store:
        return result,dict(origin=torch.tensor(origins),forecast=forecast,truth=truth,
                           eligible=valid,persistence=persist)
    return result


def strict_budget(model):
    contract=state_contract(model)
    if (contract["trainable_parameters"]!=EXPECTED_PARAMS or
        contract["deployed_learned_parameters"]!=EXPECTED_PARAMS or
        contract["nominal_fp32_bytes"]!=6800):
        raise RuntimeError("100k or exact FP32 state budget broken: "+str(contract))
    return contract


def step_batch(model,d,examples):
    sid=torch.tensor([q[0] for q in examples],dtype=torch.long)
    ti=torch.tensor([q[1] for q in examples],dtype=torch.long)
    s=model.initial(len(examples))
    for h in range(H):
        t=ti-H+1+h
        s=model.step(s,d["values"][sid,t],d["valid"][sid,t],d["age"][sid,t],
                     d["last"][sid,t],sid)
    # LEGAL future calendar, not future observed conditions, outcomes or reward.
    candidate=calendar([int(q+HORIZON) for q in ti]).unsqueeze(1)
    mean,scale=model.forecast(s,sid,candidate)
    # True future observations can be read only after the forecast was made.
    y=d["values"][sid,ti+HORIZON]
    valid=d["valid"][sid,ti+HORIZON]
    mu=mean[:,0,:];sd=scale[:,0,:]
    finite=torch.where(valid,y,mu.detach())
    loss_mat=.5*(math.log(2*math.pi)+2*torch.log(sd)+((finite-mu)/sd).square())
    absolute=(mu-finite).abs()
    loss=((.8*loss_mat+.2*absolute)*valid).sum()/valid.sum().clamp_min(1)
    return loss


def run_fit(source):
    registered_plan()
    if OUT.exists():
        raise RuntimeError("FIT artifact directory already exists: no overwrites/retries")
    t0=time.monotonic()
    d=source_frames(source)
    val=eval_origins(hour(TRAIN_END),hour(VAL_END))
    dev=eval_origins(hour(VAL_END),hour(DEV_END))
    OUT.mkdir(parents=True,exist_ok=False)
    sha=lambda p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    src_hashes={p:sha(p) for p in ("src/tac_osm/latent_fusion_001.py",
      "src/tac_osm/latent_fusion_002.py","scripts/latfuse002_real_fit.py",
      "scripts/validate_research_parity.py")}
    results=[]
    for seed in SEEDS:
        torch.manual_seed(seed)
        random.seed(seed)
        model=FusedLatentFixedState()
        contract=strict_budget(model)
        if not all(p.dtype==torch.float32 for p in model.parameters()):
            raise RuntimeError("trained weights not FP32")
        plan=sample_origins(seed)
        plan_bytes=json.dumps(plan,separators=(",",":")).encode()
        plan_sha=hashlib.sha256(plan_bytes).hexdigest()
        # Exact mandatory preoptimizer budget/source/inventory and source cutoffs.
        opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
        if seed==SEEDS[0]:
            diag=dict(status="PREFIT_REGISTERED_ORIGINAL_SOURCE",
                      registered_manifest_sha256=hashlib.sha256(REG.read_bytes()).hexdigest(),
                      source_sha256=SOURCE_SHA,source_files_sha256=src_hashes,
                      train_origins_sha256=plan_sha,actual_contract=contract,
                      frozen_fixed_val_origin_count=len(val),frozen_dev_origin_count=len(dev),
                      training_steps=STEPS,batch_size=BATCH,history=H,horizon=HORIZON,
                      no_future_outcome_in_model_inputs=True,real_source_cutoffs=[str(TRAIN_END),str(VAL_END),str(DEV_END)],
                      registered_before_first_optimizer=True)
            (OUT/"PREFIT.json").write_text(json.dumps(diag,indent=2,sort_keys=True)+"\n")
        before_model={name:p.detach().clone() for name,p in model.named_parameters()}
        losses=[]
        for step in range(STEPS):
            examples=plan[step*BATCH:(step+1)*BATCH]
            model.train()
            opt.zero_grad(set_to_none=True)
            loss=step_batch(model,d,examples)
            if not torch.isfinite(loss):raise RuntimeError("nonfinite source FIT training loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("missing/nonfinite source FIT model gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            losses.append(float(loss.detach()))
        contract2=strict_budget(model)
        changed=sum(int(not torch.equal(before_model[name],p.detach())) for name,p in model.named_parameters())
        if not changed:raise RuntimeError("no actual model fitting occurred")
        # Freeze original trained bytes BEFORE reading already-exposed 2015 development data.
        folder=OUT/f"seed-{seed}"
        folder.mkdir(parents=True)
        cp_path=folder/"final.pt"
        checkpoint=dict(model=model.state_dict(),optimizer=opt.state_dict(),seed=seed,
                        steps=STEPS,batch_size=BATCH,history=H,horizon=HORIZON,
                        torch_rng=torch.get_rng_state(),train_plan_sha256=plan_sha,
                        original_source_sha256=SOURCE_SHA,
                        source_file_sha256=src_hashes,
                        normalizer=dict(center=d["center"],spread=d["spread"]),
                        live_state_contract=contract2,
                        train_loss_first=losses[0],train_loss_last=losses[-1])
        torch.save(checkpoint,cp_path)
        checkpoint_sha=hashlib.sha256(cp_path.read_bytes()).hexdigest()
        # Freeze hash and training facts in repository-owned files.
        fit_receipt=dict(seed=seed,steps=STEPS,batch_size=BATCH,train_plan_sha256=plan_sha,
                         live_state=contract2,original_checkpoint_sha256=checkpoint_sha,
                         train_loss_first=losses[0],train_loss_last=losses[-1],
                         changed_param_tensors=changed,training_finished_before_DEV=True)
        (folder/"FIT_RECEIPT.json").write_text(json.dumps(fit_receipt,indent=2,sort_keys=True)+"\n")
        # 2014 internal out-of-time validation; 2015 exposed DEV only after original bytes frozen.
        validation=evaluate(model,d,val)
        development,arrays=evaluate(model,d,dev,store=True)
        torch.save(arrays,folder/"DEV_PREDICTIONS.pt")
        results.append(dict(**fit_receipt,validation_2014=validation,
                            development_2015=development,
                            scored_DEV_persisted_original=True))
        print("LATFUSE002_SEED_JSON="+json.dumps(results[-1],sort_keys=True),flush=True)
    result=dict(study_id="LATFUSE002-REAL-PREDICTION-SINGLE",phase="exposed_2015_DEV_only",
                original_source_sha256=SOURCE_SHA,source_files_sha256=src_hashes,
                research_category="real-world observational forecasting; no causal policy",
                trained_seeds=list(SEEDS),actual_training_updates=STEPS*len(SEEDS),
                source_time_cutoffs=dict(FIT_start=str(START),training_before=str(TRAIN_END),
                    validation_before=str(VAL_END),DEV_before=str(DEV_END),
                    protected_2016_2017="never parsed"),
                fit_pipeline_wall_seconds=time.monotonic()-t0,
                results=results,competitive_architecture_claim=False,independent_TEST_qualification=False)
    (OUT/"RESULTS.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("LATFUSE002_RESULT_JSON="+json.dumps(result,sort_keys=True),flush=True)


def replay(source):
    d=source_frames(source)
    dev=eval_origins(hour(VAL_END),hour(DEV_END))
    report=json.loads((OUT/"RESULTS.json").read_text())
    original_source=report["original_source_sha256"]
    if original_source!=SOURCE_SHA:raise RuntimeError("wrong original source")
    for row in report["results"]:
        seed=row["seed"]
        folder=OUT/f"seed-{seed}"
        pt=folder/"final.pt"
        digest=hashlib.sha256(pt.read_bytes()).hexdigest()
        if digest!=row["original_checkpoint_sha256"]:
            raise RuntimeError("original checkpoint byte SHA mismatch")
        cp=torch.load(pt,map_location="cpu",weights_only=False)
        m=FusedLatentFixedState()
        m.load_state_dict(cp["model"],strict=True)
        strict_budget(m)
        a=torch.load(folder/"DEV_PREDICTIONS.pt",map_location="cpu",weights_only=True)
        real,computed=evaluate(m,d,dev,store=True)
        for k in ("origin","forecast","truth","eligible","persistence"):
            if not torch.equal(a[k],computed[k]):
                raise RuntimeError("independent stored development array replay mismatch: "+k)
        for metric in real:
            x=real[metric];y=row["development_2015"][metric]
            if x is None:
                if y is not None: raise RuntimeError("missing-score mismatch")
            elif isinstance(x,float):
                if abs(x-y)>1e-8:raise RuntimeError("metric mismatch: "+metric)
            elif x!=y:raise RuntimeError("metric value mismatch: "+metric)
    print("LATFUSE002_REPLAY_PASS="+json.dumps({"seeds":list(SEEDS),"clean_weights_no_refit":True,
        "original_sha_verified":True,"forecast_arrays_bit_identical":True,
        "source_and_code_require_pinned_checkout":True},sort_keys=True))




def run_prefit(source):
    """Real source/shape/gradient preflight; no optimizer, no score-driven tuning."""
    plan=registered_plan()
    d=source_frames(source)
    model=FusedLatentFixedState()
    contract=strict_budget(model)
    if plan["real_world_source"]["sha256"]!=SOURCE_SHA:
        raise RuntimeError("manifest source authenticity mismatch")
    if not all(p.dtype==torch.float32 for p in model.parameters()):
        raise RuntimeError("FP32-only training preregistration violated")
    before=[p.detach().clone() for p in model.parameters()]
    original=sample_origins(SEEDS[0])
    if len(original)!=STEPS*BATCH:
        raise RuntimeError("wrong sampler exposure length")
    loss=step_batch(model,d,original[:BATCH])
    loss.backward()
    if (not torch.isfinite(loss) or
        not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())):
        raise RuntimeError("original source prefit gradient not finite/complete")
    if any(not torch.equal(p.detach(),b) for p,b in zip(model.parameters(),before)):
        raise RuntimeError("prefit changed weights")
    ret=dict(status="PASS_PREFIT_NO_OPTIMIZER",source_sha256=SOURCE_SHA,
             authorized_stations=len(SITES),train_until=str(TRAIN_END),
             validation_until=str(VAL_END),development_until=str(DEV_END),
             history=H,horizon=HORIZON,steps_per_seed=STEPS,
             seed_count=len(SEEDS),actual_state=contract,loss=float(loss.detach()),
             src_valid_training_channels=int(d["valid"][:,:hour(TRAIN_END)].sum()),
             sampled_training_origin_digest=hashlib.sha256(json.dumps(
                 original,separators=(",",":")).encode()).hexdigest(),
             weights_unchanged=True,optimizer_steps=0)
    print("LATFUSE002_PREFIT_JSON="+json.dumps(ret,sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("prefit","fit","replay"),required=True)
    p.add_argument("--source",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.mode=="prefit":run_prefit(a.source)
    elif a.mode=="fit":run_fit(a.source)
    else:replay(a.source)


if __name__=="__main__":
    main()
