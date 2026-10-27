"""LATFUSE016 original real-source CONTENT reconstruction loss NO-FIT preflight.

Same original 99,390 parameter, 6,800 byte recurrent TAC models, one uses
actual associative historic read, one original matched fastslow/clock head.
No optimizer instantiation, no model fit, no unseen-source capability claims.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from scripts import latfuse002_real_fit as source
from scripts import latfuse008_paired_real_training as registered
from scripts.validate_research_parity import check_manifest

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse016/REGISTERED_CONTENT_NOFIT.json"
SEEDS=(16016,16017)
ARMS=tuple(registered.ARMS)
H=64

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def actual_rows(d):
    start=source.hour(dt.datetime(2014,3,5))
    stop=source.hour(dt.datetime(2014,3,14))
    rows=[]
    for s in range(8):
        sample=d["raw"][s,:,0]
        found=[]
        for t in range(start,stop):
            if bool(torch.isfinite(sample[t])) and (not found or t-found[-1]>=24):
                found.append(t)
                if len(found)==2:break
        if len(found)!=2:
            raise RuntimeError("eligible original physical sensor was missing real observations")
        rows.extend((s,t) for t in found)
    if len(rows)!=16 or max(t for _,t in rows)+H+6>=source.hour(source.TRAIN_END):
        raise RuntimeError("registered original source time/observability violated")
    return rows

def original_source_forward_loss(m,d,rows):
    b=len(rows)
    source_id=torch.tensor([x[0] for x in rows],dtype=torch.long)
    anchor=torch.tensor([x[1] for x in rows],dtype=torch.long)
    state=m.initial(b)
    for step in range(-23,H+1):
        t=anchor+step
        state=m.step(state,d["values"][source_id,t],d["valid"][source_id,t],
            d["age"][source_id,t],d["last"][source_id,t],source_id)
    # Historic head gets ONLY the timestamp of original earlier source event,
    # not old PM2.5 magnitude or its write-key oracle.
    past=source.calendar([int(x) for x in anchor])[:,None,:]
    yhat=m.historic_event_logits(state,source_id,past)[:,0]
    candidate=source.calendar([int(x+H+6) for x in anchor])[:,None,:]
    future_mu,future_sd=m.forecast(state,source_id,candidate)
    # Ground truth and masks consulted only after BOTH neural outputs exist.
    historical=d["values"][source_id,anchor,0]
    historical_obs=d["valid"][source_id,anchor,0]
    if not bool(historical_obs.all()):
        raise RuntimeError("source historic regression target not physically observed")
    reconstruction=F.smooth_l1_loss(yhat,historical,beta=1)
    actual=d["values"][source_id,anchor+H+6]
    observed=d["valid"][source_id,anchor+H+6]
    mu=future_mu[:,0];sd=future_sd[:,0]
    yt=torch.where(observed,actual,mu.detach())
    nll=.5*(math.log(2*math.pi)+2*sd.log()+((yt-mu)/sd).square())
    forecast=((.8*nll+.2*(yt-mu).abs())*observed).sum()/observed.sum().clamp_min(1)
    return reconstruction+forecast,reconstruction,forecast,historical

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise RuntimeError("cannot replace original source gradient evidence")
    torch.set_num_threads(1)
    j=json.loads(REG.read_text())
    check_manifest(j,"preflight")
    if j["study_id"]!="LATFUSE016-AUTHENTIC-PAST-CONTENT-LOSS-NOFIT":
        raise RuntimeError("registered objective drift")
    if sha(a.source)!=source.SOURCE_SHA:
        raise RuntimeError("original observed sensor archive not source-pinned")
    data=registered.source_observations(a.source)
    rows=actual_rows(data)
    results=[]
    for seed in SEEDS:
        for arm in ARMS:
            m=registered.create_model(seed,arm)
            before=[p.detach().clone() for p in m.parameters()]
            m.train()
            m.zero_grad(set_to_none=True)
            total,content,forecast,historic=original_source_forward_loss(m,data,rows)
            if not all(torch.isfinite(x).all() for x in (total,content,forecast,historic)):
                raise RuntimeError("nonfinite source-grounded content and future loss")
            total.backward()
            grads=dict(m.named_parameters())
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in grads.values()):
                raise RuntimeError("source objective disconnected or nonfinite original gradient")
            if any(not torch.equal(p,q) for p,q in zip(m.parameters(),before)):
                raise RuntimeError("no-fit original live model unexpectedly changed")
            contract=m.original_state_and_weight_contract()
            if (contract["trainable_parameters"]!=99390 or
               contract["original_nominal_memory_bytes"]!=6800):
                raise RuntimeError("content objective changes real source recurrent/parameter budget")
            results.append(dict(seed=seed,arm=arm,
               original_genuine_source_history_examples=len(rows),
               total_source_content_objective_before_training=float(total.detach()),
               actual_historic_content_smoothL1_before_training=float(content.detach()),
               original_physical_future_forecast_loss_before_training=float(forecast.detach()),
               source_trainable_tensors_with_finite_grad=len(grads),
               original_observation_target_only_in_loss_not_clock=True,
               99390_active_source_parameters=True,
               recurrent_stream_6800_bytes=True,
               optimizer_steps=0,no_model_fit=True))
            print("LATFUSE016_REAL_SOURCE_NOFIT="+json.dumps(results[-1],sort_keys=True),flush=True)
    out=dict(study_id=j["study_id"],source_sha256=source.SOURCE_SHA,
       source_eligible_case_time_SHA=hashlib.sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest(),
       true_physical_source_cases=len(rows),
       original_models_checked=len(results),same_original_source_forward_query_horizon=H,
       original_PNDS_exact_state_1696_fp32_16flags=True,
       no_optimization_or_new_model_checkpoints=True,
       full_training_forward_AD_optimizer_FLOP_parity_NOT_verified=True,
       original_real_source_training_and_independent_eval_NOT_performed=True,
       no_original_2015_DEV_or_protected_2016_17_numeric_source=True,
       source_true_historic_PM25_and_future_targets_read_only_after_forward=True,
       sidecar_H64_native_RECALL_ability_NOT_claimed=True,models=results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("LATFUSE016_PHYSICAL_CONTENT_OBJECTIVE_PREFIT_PASS="+json.dumps(
       {"models":len(results),"valid_source_rows":len(rows),"neural_optimizer_steps":0,
        "training_fairness":False},sort_keys=True))
if __name__=="__main__":main()
