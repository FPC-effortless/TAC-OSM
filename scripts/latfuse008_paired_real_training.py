"""LATFUSE008 real-source paired historical query exploratory fit+clean replay.

Two fixed 99,390-weight/6,800-byte architectures × two original seeds,
32 updates each. Real observed original March–April PM2.5 source-event
label from RAW concentration thresholds. Training-only chronological
2013Mar–2014Feb normalizer; May–June genuine source evaluation.
This is NOT a qualified full-train-FLOP-matched architecture comparison.
No numeric year>=2015 original source rows or protected 2016–17 opened.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F

from scripts import latfuse002_real_fit as source
from scripts.validate_research_parity import check_manifest
from tac_osm.latent_fusion_007_historic_query import SourceTimedEventQuery
from tac_osm.latent_fusion_008_nonassoc_control import SourceTimedNonassociativeQuery

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse008/REAL_FIT_REGISTRATION.json"
PREF=ROOT/"research/mathematical-engineering/latfuse008/NOFIT_RESULTS.json"
OUT=ROOT/"research/replay-bundles/LATFUSE008"
ARMS={"assoc_historic_query":SourceTimedEventQuery,
      "fast_slow_historic_query":SourceTimedNonassociativeQuery}
SEEDS=(80701,80702)
TRAIN_START=source.hour(dt.datetime(2014,3,1))
TRAIN_STOP=source.hour(dt.datetime(2014,4,18))
EVAL_START=source.hour(dt.datetime(2014,5,1))
EVAL_STOP=source.hour(dt.datetime(2014,6,10))
NORM_STOP=TRAIN_START
LAGS=(16,64,128,512)
TRAIN_LAG=64
HISTORY=24
STEPS=32
BATCH=8

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def registered():
    r=json.loads(REG.read_text())
    assert check_manifest(r,"preflight")["status"]=="PASS_PREFIT_SINGLE_MODEL"
    p=json.loads(PREF.read_text())
    if p["source_sha256"]!=source.SOURCE_SHA or not p["partial_profiled_compute_within_one_pct"]:
        raise RuntimeError("original physically observed source prefit not passed")
    if (set(x["architecture"] for x in r["models"])!=set(ARMS) or
        set(x["seed"] for x in r["models"])!=set(SEEDS) or len(r["models"])!=4):
        raise RuntimeError("registered original 4 true-source models or seeds invalid")
    if r["real_world_source"]["sha256"]!=source.SOURCE_SHA:
        raise RuntimeError("unmatched source archive")
    if OUT.exists():
        raise RuntimeError("LATFUSE008 original checkpoints already exist; never overwrite")
    return r

def source_observations(zip_path):
    # Original source loader parses numerical rows only before 2015,
    # NEVER the original protected 2016–17 or exposed 2015 source.
    d=source.source_frames(zip_path,phase="fit")
    raw=d["raw"]
    x=raw.clone()
    x[:,:,:6]=torch.where(torch.isfinite(x[:,:,:6]),
               torch.log1p(x[:,:,:6].clamp_min(0)),x[:,:,:6])
    if NORM_STOP!=source.hour(dt.datetime(2014,3,1)):
        raise RuntimeError("prospective past normalization source cutoff drift")
    historic=x[:,:NORM_STOP]
    observed=torch.isfinite(historic)
    count=observed.sum((0,1)).clamp_min(1)
    center=torch.where(observed,historic,0).sum((0,1))/count
    spread=(torch.where(observed,historic-center,0).square().sum((0,1))/count).clamp_min(1e-6).sqrt()
    norm=(x-center)/spread
    ok=torch.isfinite(norm)
    values=torch.where(ok,norm,torch.zeros_like(norm))
    age=torch.full((12,13),256.,dtype=torch.float32)
    recent=torch.zeros_like(age)
    last=torch.zeros_like(values)
    elapsed=torch.zeros_like(values)
    for t in range(values.shape[1]):
        last[:,t]=recent
        age=torch.where(ok[:,t],torch.zeros_like(age),(age+1).clamp(max=256))
        elapsed[:,t]=age
        recent=torch.where(ok[:,t],values[:,t],recent)
    return dict(raw=raw,values=values,valid=ok,last=last,age=elapsed,
                center=center,spread=spread,source_sha=source.SOURCE_SHA)

def select_pairs(d,start,stop,max_per_site):
    """Source-measured raw ug/m³; source station/hour matched. No model labels."""
    pairs=[]
    for site in range(12):
        recorded=d["raw"][site,:,0]
        high=[]
        low=[]
        for t in range(start,stop):
            if torch.isfinite(recorded[t]) and float(recorded[t])>=150 and (
                not high or t-high[-1]>=24):
                high.append(t)
                if len(high)>=max_per_site:break
        for h in high:
            candidates=[t for t in range(start,stop) if
                torch.isfinite(recorded[t]) and float(recorded[t])<=35 and
                t%24==h%24 and abs(t-h)>=24 and
                all(abs(t-j)>=24 for j in low)]
            if not candidates:continue
            normal=min(candidates,key=lambda t:(abs(t-h),t))
            low.append(normal)
            pairs.extend([(site,h,1),(site,normal,0)])
    if len(pairs)<24 or len(pairs)%2:
        raise RuntimeError("source-real >=12 complete high/ordinary pairs required")
    for i in range(0,len(pairs),2):
        p,q=pairs[i:i+2]
        if not(p[0]==q[0] and p[1]%24==q[1]%24 and p[2]==1 and q[2]==0):
            raise RuntimeError("genuine station/clock pair invalid")
    if any(t-HISTORY+1<0 or t+max(LAGS)+6>=source.hour(source.TRAIN_END)
           for _,t,_ in pairs):
        raise RuntimeError("original heldout or full historic source availability violated")
    return pairs

def plan(rows,seed):
    rng=random.Random(seed)
    return [rows[rng.randrange(len(rows))] for _ in range(STEPS*BATCH)]

def create_model(seed,name):
    torch.manual_seed(seed)
    model=ARMS[name]()
    c=model.original_state_and_weight_contract()
    if not(c["trainable_parameters"]==c["deployed_parameters"]==99390 and
           c["original_nominal_memory_bytes"]==6800):
        raise RuntimeError("pretrained H80 FP32 and live 99,390 parameter budget violation")
    if any(p.dtype!=torch.float32 or not p.requires_grad for p in model.parameters()):
        raise RuntimeError("original full FP32 trainable parameter contract invalid")
    return model

def forward_case(model,d,cases,lag,*,with_truth):
    b=len(cases)
    site=torch.tensor([x[0] for x in cases],dtype=torch.long)
    anchor=torch.tensor([x[1] for x in cases],dtype=torch.long)
    state=model.initial(b)
    for step in range(-HISTORY+1,lag+1):
        t=anchor+step
        state=model.step(state,d["values"][site,t],d["valid"][site,t],
            d["age"][site,t],d["last"][site,t],site)
    past_clock=source.calendar([int(v) for v in anchor]).unsqueeze(1)
    logit=model.historic_event_logits(state,site,past_clock)[:,0]
    future=source.calendar([int(v+lag+6) for v in anchor]).unsqueeze(1)
    mean,sd=model.forecast(state,site,future)
    if not with_truth:
        return logit,mean[:,0],sd[:,0],None
    # All labeled physical source outcomes are looked up AFTER both forward
    # branches and computed from original physically recorded source.
    true=d["raw"][site,anchor,0]
    if not all(torch.isfinite(true[i]) and (
        float(true[i])>=150 if row[2]==1 else float(true[i])<=35)
        for i,row in enumerate(cases)):
        raise RuntimeError("historical event category source target mismatch")
    label=(true>=150).float()
    yt=d["values"][site,anchor+lag+6]
    valid=d["valid"][site,anchor+lag+6]
    mu=mean[:,0];scale=sd[:,0]
    clean=torch.where(valid,yt,mu.detach())
    gauss=.5*(math.log(2*math.pi)+2*scale.log()+((clean-mu)/scale).square())
    forecast=((.8*gauss+.2*(clean-mu).abs())*valid).sum()/valid.sum().clamp_min(1)
    event=F.binary_cross_entropy_with_logits(logit,label)
    return logit,mu,scale,(forecast+event,forecast,event,yt,valid,label)

def metrics(model,d,rows,lag):
    predictions=[];target=[];forecast=[];outcome=[];availability=[];persist=[]
    model.eval()
    with torch.no_grad():
        for i in range(0,len(rows),8):
            segment=rows[i:i+8]
            y,m,sd,truth=forward_case(model,d,segment,lag,with_truth=True)
            predictions.append(torch.sigmoid(y))
            target.append(truth[-1])
            forecast.append(torch.stack((m,sd),dim=1))
            outcome.append(truth[3])
            availability.append(truth[4])
            site=torch.tensor([r[0] for r in segment])
            anchor=torch.tensor([r[1] for r in segment])
            persist.append(d["last"][site,anchor+lag+1])
    p=torch.cat(predictions)
    lab=torch.cat(target)
    mu_sd=torch.cat(forecast)
    actual=torch.cat(outcome)
    mask=torch.cat(availability)
    historical=torch.cat(persist)
    if not torch.equal(lab[::2],torch.ones_like(lab[::2])) or not torch.equal(
            lab[1::2],torch.zeros_like(lab[1::2])):
        raise RuntimeError("source event-control true order drift")
    p1=p[::2];p0=p[1::2]
    rank=float(((p1>p0).float()+.5*(p1==p0).float()).mean())
    pred=(p>=.5).float()
    balanced=.5*(float(pred[lab==1].mean())+float((1-pred[lab==0]).mean()))
    brier=float((p-lab).square().mean())
    mu=mu_sd[:,0,:];sd=mu_sd[:,1,:]
    safe=torch.where(mask,actual,mu)
    nll=.5*(math.log(2*math.pi)+2*sd.log()+((safe-mu)/sd).square())
    denom=mask.sum().clamp_min(1)
    vmask=mask.to(mu.dtype)
    mae=float(((safe-mu).abs()*vmask).sum()/denom)
    persist_mae=float(((safe-historical).abs()*vmask).sum()/denom)
    covered=(safe-mu).abs()<=1.6448536269514722*sd
    scores=dict(n_source_matched_pairs=len(rows)//2,
        balanced_accuracy=balanced,physically_observed_event_Brier=brier,
        event_pair_ranking_accuracy=rank,
        fixed_prior_Brier=.25,
        real_source_all_channel_normalized_mae=mae,
        real_source_all_channel_persistence_mae=persist_mae,
        gaussian_nll=float((nll*vmask).sum()/denom),
        nominal_90_coverage=float((covered*vmask).sum()/denom))
    return scores,dict(p=p,labels=lab,forecast=mu_sd,truth=actual,valid=mask,
                       source_last_observed=historical)

def fit(src):
    reg=registered()
    d=source_observations(src)
    tr=select_pairs(d,TRAIN_START,TRAIN_STOP,8)
    val=select_pairs(d,EVAL_START,EVAL_STOP,2)
    if max(t for _,t,_ in tr)+TRAIN_LAG+6>=min(t for _,t,_ in val):
        raise RuntimeError("cross-chronological original source label time leakage")
    if len(tr)<24 or len(val)<24:raise RuntimeError("real cohort insufficient")
    cohort_hash=lambda c:hashlib.sha256(json.dumps(c,separators=(",",":")).encode()).hexdigest()
    codes=["src/tac_osm/latent_fusion_007_historic_query.py",
           "src/tac_osm/latent_fusion_008_nonassoc_control.py",
           "src/tac_osm/latent_fusion_004_factorial.py",
           "scripts/latfuse008_paired_real_training.py"]
    hashed={p:sha(ROOT/p) for p in codes}
    OUT.mkdir(parents=True,exist_ok=False)
    records=[]
    for seed in SEEDS:
        selected=plan(tr,seed)
        order_sha=cohort_hash(selected)
        for arm in ARMS:
            m=create_model(seed,arm)
            opt=torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=.001)
            original=[x.detach().clone() for x in m.parameters()]
            losses=[]
            started=time.monotonic()
            for i in range(STEPS):
                cases=selected[i*BATCH:(i+1)*BATCH]
                opt.zero_grad(set_to_none=True)
                m.train()
                _,_,_,truth=forward_case(m,d,cases,TRAIN_LAG,with_truth=True)
                loss=truth[0]
                if not torch.isfinite(loss):raise RuntimeError("original real source nonfinite loss")
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):
                    raise RuntimeError("original source recurrent tensor disconnected")
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
                opt.step()
                losses.append(float(loss.detach()))
            if not any(not torch.equal(a,b) for a,b in zip(original,m.parameters())):
                raise RuntimeError("model not trained on genuine source")
            directory=OUT/f"seed-{seed}"/arm
            directory.mkdir(parents=True)
            pt=directory/"final.pt"
            torch.save(dict(original_model=m.state_dict(),optimizer=opt.state_dict(),
                 seed=seed,arm=arm,original_source_sha256=source.SOURCE_SHA,
                 past_normalizer_center=d["center"],past_normalizer_spread=d["spread"],
                 original_case_order_sha256=order_sha,
                 original_code_shas=hashed,source_train_matched_cohort_sha256=cohort_hash(tr),
                 source_eval_matched_cohort_sha256=cohort_hash(val),
                 torch_rng=torch.get_rng_state(),
                 full_original_training_trace=losses,optimizer_updates=STEPS,
                 model_contract=m.original_state_and_weight_contract()),pt)
            checkpoint_sha=sha(pt)
            records.append(dict(seed=seed,arm=arm,
                original_checkpoint_sha256=checkpoint_sha,
                optimizer_updates=STEPS,train_examples=STEPS*BATCH,
                original_source_case_order_sha256=order_sha,
                training_loss_first=losses[0],training_loss_last=losses[-1],
                CPU_wall_seconds=time.monotonic()-started,
                model_contract=m.original_state_and_weight_contract()))
            print("LATFUSE008_TRAINED_ORIGINAL="+json.dumps(dict(seed=seed,arm=arm,
                original_checkpoint_sha256=checkpoint_sha,
                optimizer_steps=STEPS,loss_final=losses[-1]),sort_keys=True),flush=True)
    # Absolutely ALL four original checkpoints stored SHA BEFORE opening May-June
    for z in records:
        cp=OUT/f"seed-{z['seed']}"/z["arm"]/"final.pt"
        if sha(cp)!=z["original_checkpoint_sha256"]:
            raise RuntimeError("original model SHA before chronological evaluation invalid")
    for z in records:
        cp=OUT/f"seed-{z['seed']}"/z["arm"]/"final.pt"
        original=torch.load(cp,map_location="cpu",weights_only=False)
        m=create_model(z["seed"],z["arm"])
        m.load_state_dict(original["original_model"],strict=True)
        m.eval()
        z["chronological_2014_internal_retrieval"]={}
        for delay in LAGS:
            result,arrays=metrics(m,d,val,delay)
            z["chronological_2014_internal_retrieval"][str(delay)]=result
            torch.save(arrays,OUT/f"seed-{z['seed']}"/z["arm"]/f"EVAL_{delay}.pt")
    report=dict(study_id=reg["study_id"],source_sha256=source.SOURCE_SHA,
          source_train_pairs=len(tr)//2,source_eval_pairs=len(val)//2,
          source_train_cohort_sha256=cohort_hash(tr),
          source_eval_cohort_sha256=cohort_hash(val),
          train_period="March1-Apr18 2014 original",eval_period="May1-Jun10 2014 original FIT",
          normalizer_period="original 2013Mar1 to2014Mar1 (strictly before March label training cohort)",
          original_training_horizon=TRAIN_LAG,observed_eval_delays=list(LAGS),
          original_optimizer_steps_per_model=STEPS,total_optimizer_steps=4*STEPS,
          original_checkpoint_weights_frozen_before_out_of_time_metrics=True,
          source_original_true_observed_sensor_labels=True,
          original_protected_2016_2017_numeric_never_parsed=True,
          original_2015_DEV_numeric_never_parsed=True,
          source_result_is_exploratory_not_independent_qualification=True,
          complete_training_FLOPs_parity_not_established=True,
          no_architecture_superiority_claim=True,
          model_results=records)
    (OUT/"RESULTS.json").write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LATFUSE008_ORIGINAL_FIT_DONE="+json.dumps({
        "models":len(records),"updates":4*STEPS,
        "train_pairs":len(tr)//2,"eval_pairs":len(val)//2,
        "fair_architecture_comparison":False},sort_keys=True))

def replay(src):
    d=source_observations(src)
    out=json.loads((OUT/"RESULTS.json").read_text())
    rows=select_pairs(d,EVAL_START,EVAL_STOP,2)
    if (out["source_eval_cohort_sha256"]!=hashlib.sha256(
        json.dumps(rows,separators=(",",":")).encode()).hexdigest()):
        raise RuntimeError("original chronology changes across checkpoint reload")
    for z in out["model_results"]:
        directory=OUT/f"seed-{z['seed']}"/z["arm"]
        pt=directory/"final.pt"
        if sha(pt)!=z["original_checkpoint_sha256"]:
            raise RuntimeError("real-trained original source checkpoint byte changed")
        cp=torch.load(pt,map_location="cpu",weights_only=False)
        m=create_model(z["seed"],z["arm"])
        m.load_state_dict(cp["original_model"],strict=True)
        for lag in LAGS:
            actual,values=metrics(m,d,rows,lag)
            previous=torch.load(directory/f"EVAL_{lag}.pt",map_location="cpu",weights_only=True)
            for k in values:
                if not torch.equal(values[k],previous[k]):
                    raise RuntimeError("clean checkpoint source predictions not exact: "+k)
            for k in actual:
                if abs(actual[k]-z["chronological_2014_internal_retrieval"][str(lag)][k])>1e-8:
                    raise RuntimeError("original numerical score replay differs")
    print("LATFUSE008_CLEAN_EVAL_REPLAY_PASS="+json.dumps({
        "models":len(out["model_results"]),"delays":list(LAGS),
        "no_optimizer":True,"full_source_pred_arrays_bitwise":True},sort_keys=True))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("fit","replay"),required=True)
    p.add_argument("--source",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    if sha(a.source)!=source.SOURCE_SHA:raise RuntimeError("physical source ZIP changed")
    if a.mode=="fit":fit(a.source)
    else:replay(a.source)
if __name__=="__main__":main()
