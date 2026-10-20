#!/usr/bin/env python3
"""ME008F first-exposure March 23-30 unchanged native TAC vs GRU. ZERO model fits."""
from __future__ import annotations
import csv,hashlib,io,json,urllib.request,zipfile,math
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from scripts.me004_native_real_forecast import FEATURES,HIST,HORIZONS,make_model,score

URL="https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip"
SHA="aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a"
START=datetime.fromisoformat("2020-03-23 00:00:00")
END=datetime.fromisoformat("2020-03-30 00:00:00")

def source(out):
    path=out/"metropt3-verified.zip";sha=hashlib.sha256();n=0
    req=urllib.request.Request(URL,headers={"User-Agent":"TAC-OSM-ME008F/1.0"})
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




from scripts.me006_causal_relevance import CausalRelevanceAdapter
from scripts.me007_train_matched_gru import MatchedGRU
from scripts.me008_train_native_1000 import SEEDS as NATIVE_SEEDS

def read_verified(folder,fit):
    path=Path(folder)/fit["checkpoint"]
    expected=fit.get("sha256",fit.get("checkpoint_sha256"))
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:raise ValueError("checkpoint source hash mismatch "+str(path))
    return torch.load(path,weights_only=True,map_location="cpu")

def run(original,adapter,gru,native,out):
    original,adapter,gru,native,out=map(Path,(original,adapter,gru,native,out))
    out.mkdir(parents=True,exist_ok=True)
    manifests=[json.loads((r/"results.json").read_text()) for r in (original,adapter,gru,native)]
    o,a,g,n=manifests
    if (o["study"],a["study"],g["study"],n["study"]) != (
         "ME004","ME006_REAL_CAUSAL_RELEVANCE","ME007_CAPACITY_MATCHED_GRU",
         "ME008_NATIVE_EQUAL_STEPS"):raise ValueError("wrong immutable study source")
    if len(o["fits"])!=6 or len(a["fits"])!=4 or len(g["fits"])!=3 or len(n["fits"])!=3:
        raise ValueError("incomplete historical fit matrix")
    originals={}
    for fit in o["fits"]:
        ck=read_verified(original,fit)
        if fit["model"]=="tac_two_timescale":originals[fit["seed"]]=(fit,ck)
    assert set(originals)=={240401,240402}
    adapters=[]
    for fit in a["fits"]:
        ck=read_verified(adapter,fit)
        if fit["mode"]=="direct":adapters.append((fit,ck))
    assert len(adapters)==2
    candidates=[]
    for study,folder,kind in ((g,gru,"gru"),(n,native,"tac")):
        for seed_record in study["fits"]:
            for fit in seed_record["checkpoints"]:
                ck=read_verified(folder,fit)
                if fit["seed"] not in NATIVE_SEEDS or fit["steps"] not in (450,1000):
                    raise ValueError("unregistered step/seed")
                if ck["seed"]!=fit["seed"] or ck["steps"]!=fit["steps"]:
                    raise ValueError("checkpoint mismatched content")
                expected_kind="gru41" if kind=="gru" else "native2time"
                if ck["model"]!=expected_kind:
                    raise ValueError("wrong model identifier")
                candidates.append((kind,fit,ck))
    if len(candidates)!=12:raise ValueError("12 controls required")
    # No future source accessed until every original trained checkpoint is verified.
    values,ts,source_meta=source(out)
    xx,yy,thresh,cohort=prep(values,ts,o["cohort"])
    torch.set_num_threads(1);rows=[]
    def store(model,seed,pred,checkpoint,parameters=None):
        metrics=score(pred,yy,thresh)
        rows.append({"model":model,"seed":seed,"score":metrics,
            "per_sensor_mae":(pred-yy).abs().double().mean((0,1)).tolist(),
            "checkpoint_sha256":checkpoint,"parameters":parameters})
        print(json.dumps({"model":model,"seed":seed,"mae":metrics["macro_mae"]}),flush=True)
    def predict(m):
        m.eval()
        with torch.inference_mode():
            return torch.cat([m(xx[i:i+256]) for i in range(0,len(xx),256)])
    store("persistence",-1,torch.zeros_like(yy),"not_trained",0)
    for fit,ck in adapters:
        original_fit,original_ck=originals[fit["base_seed"]]
        m=make_model("tac_two_timescale")
        m.load_state_dict(original_ck["state_dict"],strict=True)
        ad=CausalRelevanceAdapter(m,mode="direct")
        ad.load_state_dict(ck["state_dict"],strict=True)
        store("tac_direct",fit["seed"],predict(ad),fit["checkpoint_sha256"],10450)
    for kind,fit,ck in candidates:
        m=MatchedGRU() if kind=="gru" else make_model("tac_two_timescale")
        m.load_state_dict(ck["state_dict"],strict=True)
        store(("gru41_" if kind=="gru" else "native2time_")+str(fit["steps"]),
              fit["seed"],predict(m),fit["sha256"],fit["parameters"])
    assert len(rows)==15
    summary={"study":"ME008F_FIRST_FROZEN_REAL_COMPARISON",
        "period_start":"2020-03-23","period_end_exclusive":"2020-03-30",
        "source":source_meta,"cohort":cohort,
        "fits":rows,"optimizer_steps_in_future_eval":0,
        "future_after_end_not_read":True,"architecture_promoted":False,
        "original_gru_run_id":39382074165,
        "original_tac_run_id":None,
        "scope":"one-site real continuous time-series forecasts, not action effects"}
    (out/"results.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"FROZEN_REAL_EVAL_COMPLETE","forecasts":len(xx),
                      "models":len(rows),"new_optimizer_updates":0},sort_keys=True))
    return summary

if __name__=="__main__":
    run("results/me004","results/me006","results/me007","results/me008","results/me008f")
