"""LATFUSE005: genuine FIT-source, pre-registered event-conditioned long-delay memory.

Eight original SHA-pinned trained LATFUSE004 checkpoints. Absolutely NO fits,
checkpoint modification, future target inputs, exposed 2015 DEV or protected
2016–2017 source. All actual hourly samples and real +6 outcomes on 2014
original FIT only. A read at the ORIGINAL WRITE KEY is NOT the model's learned
future forecasting read query, nor proof of factual historical event retrieval.
The zero-prefix-memory continuation is OOD mechanistic sensitivity, not a
causal or independently matched architecture comparison.
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

from scripts import latfuse002_real_fit as data
from scripts import latfuse004_factorial_training as trained
from scripts.validate_research_parity import check_manifest
from tac_osm.latent_fusion_004_factorial import FactorialFusion,ARMS,contract

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse005/REGISTERED_PROBE.json"
ANCHOR_FIRST=data.hour(dt.datetime(2014,7,1))
ANCHOR_STOP=data.hour(dt.datetime(2014,8,1))
HISTORY=24
DELAYS=(16,64,128,512)
DELAY_MAX=max(DELAYS)
SOURCE_EVENTS_PER_SITE=2

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def source_only_cohort(d):
    """No model/target outcome access: genuinely observed original PM2.5 anchors."""
    picks=[]
    for station in range(12):
        have=d["valid"][station,:,0]
        raw=d["values"][station,:,0]
        events=[]
        for t in range(ANCHOR_FIRST,ANCHOR_STOP):
            if bool(have[t]) and float(raw[t])>=1.25 and (
                not events or t-events[-1]>=24):
                events.append(t)
                if len(events)==SOURCE_EVENTS_PER_SITE:break
        normal=[]
        for spike in events:
            eligible=[t for t in range(ANCHOR_FIRST,ANCHOR_STOP)
               if bool(have[t]) and abs(float(raw[t]))<=.25
               and t%24==spike%24 and abs(t-spike)>=24
               and all(abs(t-used)>=24 for used in normal)]
            if not eligible:continue
            candidate=min(eligible,key=lambda t:(abs(t-spike),t))
            normal.append(candidate)
            picks.append((station,spike,"observed_pm25_high"))
            picks.append((station,candidate,"same_station_clock_ordinary"))
    if len(picks)<24 or len(picks)%2:
        raise RuntimeError("original cohort lacks >=12 recorded source event/control pairs")
    if any(not (ANCHOR_FIRST<=t<ANCHOR_STOP and t>=HISTORY-1
             and t+DELAY_MAX+6<data.hour(data.TRAIN_END))
           for _,t,_ in picks):
        raise RuntimeError("cohort crosses lawful original historical FIT cutoff")
    # Pair within each station and UTC hour, neither outcome nor model-dependent.
    for j in range(0,len(picks),2):
        a,b=picks[j:j+2]
        if a[0]!=b[0] or a[1]%24!=b[1]%24:
            raise RuntimeError("matched physical observed event provenance broken")
    return picks

def source_step(m,s,d,station,when):
    return m.step(s,d["values"][station,when],d["valid"][station,when],
                  d["age"][station,when],d["last"][station,when],station)

def event_key(m,d,station,when):
    u=m.encode(d["values"][station,when],d["valid"][station,when],
               d["age"][station,when],d["last"][station,when])
    keys=F.normalize(m.key_map(u).reshape(len(station),m.head_count,m.key_width),
                     dim=-1,eps=1e-8)
    v=torch.tanh(m.val_map(u))
    return keys,v

def head_read(m,memory,key):
    matrix=memory.reshape(len(key),16,m.head_count,m.key_width).permute(0,2,1,3)
    return torch.einsum("bhvk,bhk->bhv",matrix,key)

def masked_source_scores(mu,scale,truth,valid,persist,d):
    safe=torch.where(valid,truth,mu)
    error=(mu-safe).abs()
    nll=.5*(math.log(2*math.pi)+2*scale.log()+((safe-mu)/scale).square())
    diff=torch.where(valid,error,torch.zeros_like(error))
    mask=valid.to(diff.dtype)
    physical=torch.expm1((mu[:,0]*d["spread"][0]+d["center"][0]).clamp(-12,15)).clamp_min(0)
    physical_y=torch.expm1((safe[:,0]*d["spread"][0]+d["center"][0]).clamp(-12,15)).clamp_min(0)
    physical_p=torch.expm1((persist[:,0]*d["spread"][0]+d["center"][0]).clamp(-12,15)).clamp_min(0)
    return dict(all_error=diff,all_nll=torch.where(valid,nll,0),mask=mask,
                pm25_error=(physical-physical_y).abs(),
                pm25_persistence_error=(physical_p-physical_y).abs(),
                pm25_valid=valid[:,0])

def summarize_by_type(labels,stats,stats_zero,ratio,anch_read,read,key,v,source):
    result={}
    h=key.shape[1]
    orig_recon=(anch_read-v[:,None,:]).square().mean(dim=(-1,-2)).sqrt()
    now_recon=(read-v[:,None,:]).square().mean(dim=(-1,-2)).sqrt()
    for label in ("observed_pm25_high","same_station_clock_ordinary"):
        idx=torch.tensor([z==label for z in labels],dtype=torch.bool)
        valid=stats["mask"][idx]
        total=int(valid.sum())
        pm=stats["pm25_valid"][idx]
        if total==0 or not bool(pm.any()):
            result[label]={"n_cases":int(idx.sum()),"n_channel_labels":total,
                           "status":"INSUFFICIENT_TRUE_SOURCE_LABELS"}
            continue
        ae=float(stats["all_error"][idx].sum()/valid.sum())
        ab=float(stats_zero["all_error"][idx].sum()/valid.sum())
        nel=float(stats["all_nll"][idx].sum()/valid.sum())
        nz=float(stats_zero["all_nll"][idx].sum()/valid.sum())
        physical=float(stats["pm25_error"][idx][pm].mean())
        zphysical=float(stats_zero["pm25_error"][idx][pm].mean())
        persistence=float(stats["pm25_persistence_error"][idx][pm].mean())
        result[label]=dict(n_cases=int(idx.sum()),observed_target_channels=total,
            normalized_forecast_mae=ae,zero_prior_memory_forecast_mae=ab,
            associative_prefix_memory_net_mae_benefit=ab-ae,
            original_forecast_normalized_gaussian_nll=nel,
            zero_prior_memory_normalized_gaussian_nll=nz,
            associative_prefix_memory_net_nll_benefit=nz-nel,
            physical_pm25_mae=physical,
            zero_prior_memory_pm25_mae=zphysical,
            original_last_observed_persistence_pm25_mae=persistence,
            historical_anchor_key_reconstruction_rmse_mean=float(now_recon[idx].mean()),
            immediate_anchor_key_reconstruction_rmse_mean=float(orig_recon[idx].mean()),
            prefix_matrix_difference_norm_retained=float(ratio[idx].mean()))
    return result

def run_one(m,d,cohort):
    b=len(cohort)
    station=torch.tensor([x[0] for x in cohort],dtype=torch.long)
    at=torch.tensor([x[1] for x in cohort],dtype=torch.long)
    labels=[x[2] for x in cohort]
    state=m.initial(b)
    for j in range(HISTORY):
        now=at-HISTORY+1+j
        state=source_step(m,state,d,station,now)
    keys,v=event_key(m,d,station,at)
    immediate=head_read(m,state.associative,keys)
    prefix=state.associative.clone()
    # Zeroing the entire 24h prefix matrix does NOT remove only this spike:
    # it is an OOD, historical-predecessor state lesion. We never mislabel it
    # as event-specific deletion or factual-retrieval success.
    untouched=state
    erased=state._replace(associative=torch.zeros_like(prefix))
    result={}
    for j in range(1,DELAY_MAX+1):
        now=at+j
        untouched=source_step(m,untouched,d,station,now)
        erased=source_step(m,erased,d,station,now)
        if j not in DELAYS:continue
        got=head_read(m,untouched.associative,keys)
        source_trace=head_read(m,untouched.associative-erased.associative,keys)
        denom=immediate.flatten(1).norm(dim=1).clamp_min(1e-7)
        trace=source_trace.flatten(1).norm(dim=1)/denom
        if any(not bool(torch.isfinite(t).all()) for t in (trace,got,source_trace)):
            raise RuntimeError("nonfinite long-delay read trace")
        q=data.calendar([int(t+j+6) for t in at]).unsqueeze(1)
        forecast,sd=m.forecast(untouched,station,q)
        control,sd0=m.forecast(erased,station,q)
        # Exactly here after both forecast branches: label source read.
        actual=d["values"][station,at+j+6]
        observed=d["valid"][station,at+j+6]
        last_observed=d["last"][station,at+j+1]
        a=masked_source_scores(forecast[:,0],sd[:,0],actual,observed,last_observed,d)
        z=masked_source_scores(control[:,0],sd0[:,0],actual,observed,last_observed,d)
        result[str(j)]=dict(
            source_hour_since_event=j,
            mean_prefix_memory_trace_at_original_event_write_key=float(trace.mean()),
            mean_matrix_prefix_difference_fraction=float(
                ((untouched.associative-erased.associative).flatten(1).norm(dim=1)/
                   prefix.flatten(1).norm(dim=1).clamp_min(1e-7)).mean()),
            mean_abs_original_vs_erased_forecast_change=float(
                (forecast-control).abs().mean()),
            data_group=summarize_by_type(labels,a,z,trace,immediate,got,keys,v,d))
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    args=ap.parse_args()
    if args.output.exists():raise RuntimeError("source diagnostic already recorded; no overwrite")
    torch.set_num_threads(1)
    registration=json.loads(REG.read_text())
    check_manifest(registration,"preflight")
    if registration["study_id"]!="LATFUSE005-GENUINE-EVENT-LONG-DELAY-FROZEN-AUDIT":
        raise RuntimeError("original source-truth delay registration not matched")
    if sha(args.source)!=data.SOURCE_SHA:
        raise RuntimeError("original UCI source fingerprint invalid")
    d=data.source_frames(args.source,phase="fit")
    cohort=source_only_cohort(d)
    footprint=hashlib.sha256(json.dumps(cohort,separators=(",",":")).encode()).hexdigest()
    original=json.loads((trained.OUT/"RESULTS.json").read_text())
    if (original["total_original_optimizer_updates"]!=1024 or
        original["source_sha256"]!=data.SOURCE_SHA):
        raise RuntimeError("frozen original LATFUSE004 source checkpoint register incomplete")
    evidence=[]
    with torch.no_grad():
        for seed in trained.SEEDS:
            for arm in ARMS:
                record=next((r for r in original["models"] if r["seed"]==seed
                     and r["arm"]==arm),None)
                if not record:raise RuntimeError("original 2x2 trained family incomplete")
                p=trained.OUT/f"seed-{seed}"/arm/"final.pt"
                check=sha(p)
                if check!=record["original_checkpoint_sha256"]:
                    raise RuntimeError("original checkpoint SHA mismatch")
                weights=torch.load(p,map_location="cpu",weights_only=False)
                if (weights["seed"]!=seed or weights["arm"]!=arm or
                    weights["original_source_sha256"]!=data.SOURCE_SHA):
                    raise RuntimeError("frozen original trained source provenance mismatch")
                m=FactorialFusion(arm)
                m.load_state_dict(weights["model"],strict=True)
                m.eval()
                if contract(arm)["per_stream_nominal_bytes"]!=6800:
                    raise RuntimeError("original FP32 model/state budget mismatch")
                before=[p.detach().clone() for p in m.parameters()]
                findings=run_one(m,d,cohort)
                if sha(p)!=check or any(not torch.equal(a,b) for a,b
                                          in zip(before,m.parameters())):
                    raise RuntimeError("trained checkpoint modified by memory diagnosis")
                e=dict(seed=seed,arm=arm,original_checkpoint_sha256=check,
                       delay_diagnostics=findings)
                evidence.append(e)
                print("LATFUSE005_LONG_DELAY_ARM="+json.dumps(
                    {"seed":seed,"arm":arm,
                     "memory_prefix_decay":{h:z["mean_matrix_prefix_difference_fraction"]
                       for h,z in findings.items()},
                     "PM25_ablation_benefit":{h:
                         z["data_group"]["observed_pm25_high"].get(
                            "associative_prefix_memory_net_mae_benefit")
                         for h,z in findings.items()}},sort_keys=True),flush=True)
    result=dict(study_id="LATFUSE005-GENUINE-EVENT-LONG-DELAY-FROZEN-AUDIT",
        true_original_archive_sha256=data.SOURCE_SHA,
        original_fit_only_July2014=True,original_unmodified_model_count=len(evidence),
        source_authentic_station_count=12,source_cohort_count=len(cohort),
        source_high_event_matched_pairs=len(cohort)//2,
        cohort_sha256=footprint,cohort=cohort,
        source_labels_origin="physically measured original PM2.5 at anchor; threshold1.25 FIT-normalized, matching ordinary abs<=.25 same source and hour",
        delays_hours=list(DELAYS),
        evidence_tier="original fitted models, original observed source 2014 FIT + genuine source +6 forecast labels, NO training",
        anchor_key_is_write_key_not_actual_future_forecast_query=True,
        intervention="entire pre-suffix associative matrix set zero; all other latent fields and all original source inputs identical. Mechanistic OOD only, NOT event-only deletion.",
        source_history_reconstruction_is_NOT_proven=True,
        no_new_training=True,no_original_weights_changed=True,
        no_2015_or_protected_numeric_read=True,
        no_architecture_superiority_or_causal_claim=True,
        models=evidence)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LATFUSE005_GENUINE_LONG_DELAY_RESULT="+json.dumps(
        {"status":"PASS_SOURCE_FROZEN_EVENT_CONDITIONAL",
         "cohort_pairs":len(cohort)//2,
         "checkpoint_count":len(evidence),"no_fit":True},sort_keys=True))
if __name__=="__main__":
    main()
