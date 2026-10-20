#!/usr/bin/env python3
"""ME006F independent March 9-16 frozen relevance forecast. ZERO model fits."""
from __future__ import annotations
import csv,hashlib,io,json,urllib.request,zipfile,math
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from scripts.me004_native_real_forecast import FEATURES,HIST,HORIZONS,make_model,score

URL="https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip"
SHA="aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a"
START=datetime.fromisoformat("2020-03-09 00:00:00")
END=datetime.fromisoformat("2020-03-16 00:00:00")

def source(out):
    path=out/"metropt3-verified.zip";sha=hashlib.sha256();n=0
    req=urllib.request.Request(URL,headers={"User-Agent":"TAC-OSM-ME006F/1.0"})
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

def replay(original, adapters, out):
    original=Path(original);adapters=Path(adapters);out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    pre=json.loads((original/"results.json").read_text())
    fitted=json.loads((adapters/"results.json").read_text())
    if pre["study"]!="ME004" or fitted["study"]!="ME006_REAL_CAUSAL_RELEVANCE":
        raise ValueError("original frozen ledgers required")
    if len(pre["fits"])!=6 or len(fitted["fits"])!=4:
        raise ValueError("incomplete original model matrix")
    original_by_seed={}
    for f in pre["fits"]:
        if hashlib.sha256((original/f["checkpoint"]).read_bytes()).hexdigest()!=f["checkpoint_sha256"]:
            raise ValueError("ME004 checkpoint hash mismatch")
        if f["model"]=="tac_two_timescale":
            original_by_seed[f["seed"]]=f
    for f in fitted["fits"]:
        if hashlib.sha256((adapters/f["checkpoint"]).read_bytes()).hexdigest()!=f["checkpoint_sha256"]:
            raise ValueError("ME006 adapter checkpoint hash mismatch")
    torch.set_num_threads(1)
    values,ts,source_manifest=source(out)
    xx,yy,threshold,cohort=prep(values,ts,pre["cohort"])
    baseline=score(torch.zeros_like(yy),yy,threshold)
    original_scores={};records=[]
    with torch.inference_mode():
        for seed,old in sorted(original_by_seed.items()):
            checkpoint=torch.load(original/old["checkpoint"],weights_only=True,map_location="cpu")
            model=make_model("tac_two_timescale")
            if checkpoint["kind"]!="tac_two_timescale" or checkpoint["seed"]!=seed:
                raise ValueError("wrong original checkpoint")
            model.load_state_dict(checkpoint["state_dict"],strict=True);model.eval()
            prediction=torch.cat([model(xx[i:i+256]) for i in range(0,len(xx),256)])
            original_scores[str(seed)]=score(prediction,yy,threshold)
        for f in fitted["fits"]:
            bseed=f["base_seed"]
            if bseed not in original_by_seed:raise ValueError("missing paired original seed")
            old=original_by_seed[bseed]
            original_model=make_model("tac_two_timescale")
            original_ck=torch.load(original/old["checkpoint"],weights_only=True,map_location="cpu")
            original_model.load_state_dict(original_ck["state_dict"],strict=True)
            adapter=CausalRelevanceAdapter(original_model,mode=f["mode"])
            ck=torch.load(adapters/f["checkpoint"],weights_only=True,map_location="cpu")
            if ck["mode"]!=f["mode"] or ck["seed"]!=f["seed"] or ck["base_seed"]!=bseed or ck["steps"]!=160:
                raise ValueError("adapter checkpoint identity mismatch")
            adapter.load_state_dict(ck["state_dict"],strict=True);adapter.eval()
            predictions=[];base_preds=[]
            for i in range(0,len(xx),256):
                d=adapter(xx[i:i+256],return_details=True)
                predictions.append(d["forecast"]);base_preds.append(d["baseline"])
            prediction=torch.cat(predictions);base_pred=torch.cat(base_preds)
            scores=score(prediction,yy,threshold)
            matched=score(base_pred,yy,threshold)
            if not math.isclose(matched["macro_mae"],original_scores[str(bseed)]["macro_mae"],
                                rel_tol=3e-6,abs_tol=3e-6):
                raise AssertionError("reused pretrained original outputs differ")
            records.append({"mode":f["mode"],"seed":f["seed"],"original_seed":bseed,
                            "score":scores,"original_frozen":matched,
                            "checkpoint_sha256":f["checkpoint_sha256"]})
            print(json.dumps({"mode":f["mode"],"seed":f["seed"],
                              "MAE":scores["macro_mae"],"base_MAE":matched["macro_mae"]}),flush=True)
    result={"study":"ME006F_ONE_TIME_FROZEN_FUTURE","period_start":"2020-03-09",
            "period_end_exclusive":"2020-03-16","source":source_manifest,
            "cohort":cohort,"persistence":baseline,"original_scores":original_scores,
            "adapters":records,"train_updates":0,"real_future_outcomes_only_for_evaluation":True,
            "checkpoint_original_run":39380424029,
            "model_promoted":False,"new_future_after_end_not_read":True}
    (out/"future-results.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"FROZEN_EVAL_COMPLETE","windows":len(xx),"refits":0}),flush=True)
    return result

if __name__=="__main__":
    replay("results/me004","results/me006","results/me006-future")
