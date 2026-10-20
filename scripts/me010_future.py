#!/usr/bin/env python3
"""ME010F first-exposure April6–13 real change-mixture vs direct control. ZERO model fits."""
from __future__ import annotations
import csv,hashlib,io,json,urllib.request,zipfile,math
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from scripts.me004_native_real_forecast import FEATURES,HIST,HORIZONS,make_model,score

URL="https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip"
SHA="aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a"
START=datetime.fromisoformat("2020-04-06 00:00:00")
END=datetime.fromisoformat("2020-04-13 00:00:00")

def source(out):
    path=out/"metropt3-verified.zip";sha=hashlib.sha256();n=0
    req=urllib.request.Request(URL,headers={"User-Agent":"TAC-OSM-ME010F/1.0"})
    with urllib.request.urlopen(req,timeout=120) as stream,path.open("wb") as f:
        if getattr(stream,'status',200)!=200:raise ValueError('source HTTP')
        while (chunk:=stream.read(2**20)):
            n+=len(chunk)
            if n>500_000_000:raise ValueError('archive too big')
            sha.update(chunk);f.write(chunk)
    if sha.hexdigest()!=SHA:raise ValueError('source hash mismatch: no data exposure')
    timestamps=[];values=[];future_rows=0
    with zipfile.ZipFile(path) as z:
        names=[name for name in z.namelist()
               if Path(name).name.lower()=='metropt3(aircompressor).csv']
        if len(names)!=1:raise ValueError('ambiguous CSV member')
        with z.open(names[0]) as stream:
            reader=csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8-sig',newline=''))
            if not set(FEATURES+('timestamp',)).issubset(set(reader.fieldnames or [])):
                raise ValueError('wrong schema')
            for row in reader:
                stamp=datetime.fromisoformat(row['timestamp'].strip())
                if stamp>=END:break
                if timestamps and stamp < timestamps[-1]:raise ValueError('not chronological')
                vals=[]
                for sensor in FEATURES:
                    try:val=float(row[sensor])
                    except (TypeError,ValueError):val=float('nan')
                    vals.append(val)
                values.append(vals);timestamps.append(stamp)
                if stamp>=START:future_rows+=1
    path.unlink(missing_ok=True)
    if future_rows<10000:raise ValueError('insufficient later real source data')
    return np.asarray(values,dtype='float32'),np.asarray(timestamps,dtype='datetime64[s]').astype('int64'),{
        'source_sha256':SHA,'publisher_zip_bytes':n,'rows_before_end':len(values),
        'future_week_rows_read':future_rows,'future_start':str(START),'future_end_exclusive':str(END),
        'no_records_after_test_end_read':True,'no_model_training':True}

def prep(data,t,cohort):
    mu=np.asarray(cohort['mean_train_only'],dtype=np.float32)
    std=np.asarray(cohort['std_train_only'],dtype=np.float32)
    if not ((std>0).all() and np.isfinite(std).all()):raise ValueError('invalid train scale')
    m=np.isfinite(data)
    x=np.clip(np.nan_to_num((data-mu)/std,nan=0,posinf=0,neginf=0),-10,10)
    all_features=np.concatenate((x,m.astype('float32')),axis=-1).astype('float32')
    bad=(np.diff(t)<=0)|(np.diff(t)>30)
    cumulative=np.concatenate([[0],np.cumsum(bad.astype('int32'))])
    idx=np.arange(HIST-1,len(data)-max(HORIZONS),12)
    idx=idx[cumulative[idx+max(HORIZONS)]==cumulative[idx-HIST+1]]
    ix_start=int(np.datetime64(START,'s').astype('int64'))
    ix_end=int(np.datetime64(END,'s').astype('int64'))
    idx=idx[(t[idx]>=ix_start)&(t[idx+max(HORIZONS)]<ix_end)]
    idx=idx[np.all(m[idx[:,None]+np.asarray(HORIZONS)[None,:]],axis=(1,2))]
    if len(idx)<500:raise ValueError('future test too sparse')
    past=np.arange(HIST)[None,:]-HIST+1
    inputs=all_features[idx[:,None]+past]
    targets=np.stack([x[idx+h]-x[idx] for h in HORIZONS],axis=1)
    threshold=np.asarray(cohort['threshold_train_only'],dtype='float32')
    return torch.from_numpy(inputs),torch.from_numpy(targets),torch.from_numpy(threshold),{
        'future_week_examples':len(idx),'sample_stride':12,
        'first_scored_timestamp':str(np.datetime64(int(t[idx[0]]),'s')),
        'last_scored_timestamp':str(np.datetime64(int(t[idx[-1]]),'s'))}






from scripts.me007_train_matched_gru import MatchedGRU
from scripts.me009_reliability import ReliabilityGate
from scripts.me009_train_gate import anomaly_metrics
from scripts.me010_event_dynamics import EventDynamics, physical_event_metrics

def load_verified(folder,entry,checkpoint=None):
    path=Path(folder)/entry["checkpoint"]
    digest=entry.get("sha256",entry.get("checkpoint_sha256"))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError("frozen checkpoint hash failed BEFORE future source opened")
    return torch.load(path,map_location="cpu",weights_only=True)

def audit(original,gru,gate,new,out):
    original,gru,gate,new,out=map(Path,(original,gru,gate,new,out))
    out.mkdir(parents=True,exist_ok=True)
    a=json.loads((original/"results.json").read_text())
    b=json.loads((gru/"results.json").read_text())
    c=json.loads((gate/"results.json").read_text())
    d=json.loads((new/"results.json").read_text())
    if (a["study"],b["study"],c["study"],d["study"])!=(
       "ME008_NATIVE_EQUAL_STEPS","ME007_CAPACITY_MATCHED_GRU",
       "ME009_NATIVE_RELIABILITY_CALIBRATION","ME010_REAL_EVENT_CONDITIONED_DYNAMICS"):
        raise ValueError("require original immutable fit ledgers")
    assert len(a["fits"])==len(b["fits"])==len(c["fits"])==3
    assert len(d["original_fits"])==6
    assert not d["model_promoted"] and d["future_april6_13_unopened"]
    native={};controls={};older={};arms=[]
    for fitgroup in a["fits"]:
        for fit in fitgroup["checkpoints"]:
            checkpoint=load_verified(original,fit)
            if fit["steps"]==1000:
                assert checkpoint["seed"]==fit["seed"] and checkpoint["steps"]==1000
                native[fit["seed"]]=(fit,checkpoint)
    for fitgroup in b["fits"]:
        for fit in fitgroup["checkpoints"]:
            checkpoint=load_verified(gru,fit)
            if fit["steps"]==1000:
                assert checkpoint["seed"]==fit["seed"] and checkpoint["model"]=="gru41"
                controls[fit["seed"]]=(fit,checkpoint)
    for fit in c["fits"]:
        checkpoint=load_verified(gate,fit)
        assert checkpoint["base_seed"]==fit["base_seed"] and checkpoint["steps"]==220
        older[fit["base_seed"]]=(fit,checkpoint)
    for fit in d["original_fits"]:
        checkpoint=load_verified(new,fit)
        assert checkpoint["mode"]==fit["mode"] and checkpoint["seed"]==fit["seed"]
        assert checkpoint["base_seed"]==fit["base_seed"] and checkpoint["steps"]==260
        arms.append((fit,checkpoint))
    assert set(native)==set(controls)==set(older)=={240401,240402,240403}
    assert {(fit["mode"],fit["base_seed"]) for fit,ck in arms}=={
       (mode,seed) for mode in ("direct","mixture") for seed in native}
    for fit,ck in arms:
        assert ck["native_source_sha256"]==native[fit["base_seed"]][0]["sha256"]
    for seed,(fit,ck) in older.items():
        assert ck["original_sha256"]==native[seed][0]["sha256"]
    # All *original* checkpoint digests were tested before opening the later source.
    data,timestamps,source_manifest=source(out)
    x,y,threshold,cohort=prep(data,timestamps,a["cohort"])
    torch.set_num_threads(1)
    results=[]
    def measure(name,seed,pred,digest,prob=None,old_gate=None):
        metrics=score(pred,y,threshold)
        metrics["quiet_diagnostic"]=anomaly_metrics(pred,y,threshold)
        if prob is not None:
            rates=torch.tensor(d["historical_event_probability_by_horizon_sensor"],dtype=torch.float32)
            actual=(y.abs()>threshold[None,:,:]).float()
            base_prob=rates.unsqueeze(0).expand_as(actual)
            metrics["historical_event_rate_brier"]=float((base_prob-actual).square().mean())
            if prob.shape!=actual.shape:raise ValueError("event-probability shape mismatch")
            event=(actual>0.5)
            alarm=(prob>=.5)
            tp=(alarm&event).sum()
            bins=torch.linspace(0,1,11)
            metrics.update({
                "event_brier":float((prob-actual).square().mean()),
                "event_log_loss":float(torch.nn.functional.binary_cross_entropy(prob.clamp(1e-7,1-1e-7),actual)),
                "event_prevalence":float(actual.mean()),
                "event_probability_mean":float(prob.mean()),
                "event_precision":float(tp/alarm.sum()) if bool(alarm.any()) else 0.,
                "event_recall":float(tp/event.sum()) if bool(event.any()) else 0.,
                "event_ece_10_bins":float(sum(
                    (m.float().mean()*torch.abs(prob[m].mean()-actual[m].mean())).item()
                    for i in range(10)
                    for m in [((prob>=i/10)&(prob<(i+1)/10))] if bool(m.any())))})
        if old_gate is not None:metrics["mean_previous_gate_alpha"]=old_gate
        results.append({"model":name,"seed":seed,"checkpoint_sha256":digest,
                        "score":metrics,
                        "per_channel_mae":(pred-y).abs().double().mean((0,1)).tolist()})
        print(json.dumps({"model":name,"seed":seed,"MAE":metrics["macro_mae"],
            "event_brier":metrics.get("event_brier"),"quiet_alarm":metrics["quiet_diagnostic"]["false_movement_rate"]},sort_keys=True),flush=True)
    def predict(m,mode="normal"):
        m.eval();forecast=[];prob=[];gate=[]
        with torch.inference_mode():
            for i in range(0,len(x),256):
                xx=x[i:i+256]
                if mode=="event":
                    v=m(xx,details=True)
                    forecast.append(v["prediction"])
                    if v["p_change"] is not None:prob.append(v["p_change"])
                elif mode=="reliability":
                    v=m(xx,details=True)
                    forecast.append(v["prediction"]);gate.append(v["alpha"])
                else:forecast.append(m(xx))
        return torch.cat(forecast),torch.cat(prob) if prob else None,torch.cat(gate).mean().item() if gate else None
    measure("persistence",-1,torch.zeros_like(y),"no-weights")
    for seed,(fit,ck) in sorted(native.items()):
        model=make_model("tac_two_timescale")
        model.load_state_dict(ck["state_dict"],strict=True)
        pred,_,_=predict(model)
        measure("native_tac1000",seed,pred,fit["sha256"])
        gfit,gck=controls[seed]
        gru_model=MatchedGRU()
        gru_model.load_state_dict(gck["state_dict"],strict=True)
        pred,_,_=predict(gru_model)
        measure("gru41_1000",seed,pred,gfit["sha256"])
        efit,eck=older[seed]
        nm=make_model("tac_two_timescale")
        nm.load_state_dict(ck["state_dict"],strict=True)
        gate_model=ReliabilityGate(nm)
        gate_model.load_state_dict(eck["state_dict"],strict=True)
        pred,_,alpha=predict(gate_model,"reliability")
        measure("me009_reliability",seed,pred,efit["sha256"],old_gate=alpha)
    for fit,ck in arms:
        native_fit,native_ck=native[fit["base_seed"]]
        model_base=make_model("tac_two_timescale")
        model_base.load_state_dict(native_ck["state_dict"],strict=True)
        m=EventDynamics(model_base,mode=fit["mode"])
        m.load_state_dict(ck["state_dict"],strict=True)
        pred,prob,_=predict(m,"event")
        measure("me010_"+fit["mode"],fit["seed"],pred,fit["sha256"],prob=prob)
    assert len(results)==16
    record={"study":"ME010F_EVENT_MIXTURE_FROZEN_FIRST_EXPOSURE",
        "source":source_manifest,"cohort":cohort,
        "period_start":"2020-04-06","period_end_exclusive":"2020-04-13",
        "fits":results,"future_optimizer_updates":0,"future_after_end_read":False,
        "model_promoted":False,"original_native_run":39382850579,
        "original_gru_run":39382074165,"original_gate_run":39384177806,
        "original_event_run":None,
        "scope":"Single physical compressor, genuine observational forecasting; no logged action-effect conclusions."}
    (out/"results.json").write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"ORIGINAL_FIRST_FROZEN_REAL_TEST_COMPLETE",
           "forecast_windows":len(x),"scored_models":len(results),"refits":0},sort_keys=True))
    return record

if __name__=="__main__":
    audit("results/me008","results/me007","results/me009","results/me010","results/me010f")
