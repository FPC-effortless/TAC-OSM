#!/usr/bin/env python3
"""ME009F first-exposure March30–April6 real reliability gate. ZERO model fits."""
from __future__ import annotations
import csv,hashlib,io,json,urllib.request,zipfile,math
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from scripts.me004_native_real_forecast import FEATURES,HIST,HORIZONS,make_model,score

URL="https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip"
SHA="aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a"
START=datetime.fromisoformat("2020-03-30 00:00:00")
END=datetime.fromisoformat("2020-04-06 00:00:00")

def source(out):
    path=out/"metropt3-verified.zip";sha=hashlib.sha256();n=0
    req=urllib.request.Request(URL,headers={"User-Agent":"TAC-OSM-ME009F/1.0"})
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

def read_verified(folder,entry):
    p=Path(folder)/entry["checkpoint"]
    expected=entry.get("sha256",entry.get("checkpoint_sha256"))
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
        raise ValueError("checkpoint integrity failure before real source exposure")
    return torch.load(p,map_location="cpu",weights_only=True)

def run(native_dir,gru_dir,gate_dir,out):
    native_dir,gru_dir,gate_dir,out=map(Path,(native_dir,gru_dir,gate_dir,out))
    out.mkdir(parents=True,exist_ok=True)
    n=json.loads((native_dir/"results.json").read_text())
    g=json.loads((gru_dir/"results.json").read_text())
    m=json.loads((gate_dir/"results.json").read_text())
    assert n["study"]=="ME008_NATIVE_EQUAL_STEPS"
    assert g["study"]=="ME007_CAPACITY_MATCHED_GRU"
    assert m["study"]=="ME009_NATIVE_RELIABILITY_CALIBRATION"
    assert len(n["fits"])==3 and len(g["fits"])==3 and len(m["fits"])==3
    native={};grus={};gates={}
    for group in n["fits"]:
        for f in group["checkpoints"]:
            ck=read_verified(native_dir,f)
            if f["steps"]==1000:native[f["seed"]]=(f,ck)
    for group in g["fits"]:
        for f in group["checkpoints"]:
            ck=read_verified(gru_dir,f)
            if f["steps"]==1000:grus[f["seed"]]=(f,ck)
    for f in m["fits"]:
        ck=read_verified(gate_dir,f)
        gates[f["base_seed"]]=(f,ck)
    assert set(native)==set(grus)==set(gates)=={240401,240402,240403}
    for seed,(fit,ck) in gates.items():
        assert ck["kind"]=="reliability-gate"
        assert ck["base_seed"]==seed and ck["steps"]==220
        assert ck["original_sha256"]==native[seed][0]["sha256"]
    # No genuine future data opened until all registered source/weight checks pass.
    data,timestamps,src=source(out)
    xx,yy,threshold,cohort=prep(data,timestamps,n["cohort"])
    torch.set_num_threads(1);records=[]
    def save(kind,seed,pred,ckhash,gate_mean=None):
        metrics=score(pred,yy,threshold)
        metrics["quiet_diagnostic"]=anomaly_metrics(pred,yy,threshold)
        records.append({"model":kind,"seed":seed,"score":metrics,
            "per_sensor_mae":(pred-yy).abs().double().mean((0,1)).tolist(),
            "checkpoint_sha256":ckhash,"mean_reliability_gate":gate_mean})
        print(json.dumps({"model":kind,"seed":seed,"macro_mae":metrics["macro_mae"],
                          "quiet_alarm":metrics["quiet_diagnostic"]["false_movement_rate"]},sort_keys=True),flush=True)
    def predict(net,with_gate=False):
        net.eval()
        with torch.inference_mode():
            batches=[]
            gated=[]
            for i in range(0,len(xx),256):
                if with_gate:
                    details=net(xx[i:i+256],details=True)
                    batches.append(details["prediction"]);gated.append(details["alpha"])
                else:batches.append(net(xx[i:i+256]))
            return torch.cat(batches),torch.cat(gated).mean().item() if gated else None
    save("persistence",-1,torch.zeros_like(yy),"not-trained")
    for seed,(f,ck) in sorted(native.items()):
        model=make_model("tac_two_timescale")
        assert ck["model"]=="native2time" and ck["seed"]==seed and ck["steps"]==1000
        model.load_state_dict(ck["state_dict"],strict=True)
        original,_=predict(model)
        save("native_tac1000",seed,original,f["sha256"])
        save("fixed_half_tac",seed,original*.5,f["sha256"])
        fgate,ckgate=gates[seed]
        native_again=make_model("tac_two_timescale")
        native_again.load_state_dict(ck["state_dict"],strict=True)
        gate=ReliabilityGate(native_again)
        gate.load_state_dict(ckgate["state_dict"],strict=True)
        dynamic,alpha=predict(gate,with_gate=True)
        save("learned_reliability_tac",seed,dynamic,fgate["sha256"],alpha)
        gfit,gck=grus[seed]
        grumodel=MatchedGRU();grumodel.load_state_dict(gck["state_dict"],strict=True)
        gpred,_=predict(grumodel)
        save("gru41_1000",seed,gpred,gfit["sha256"])
    assert len(records)==13
    result={"study":"ME009F_FIRST_FROZEN_REAL_RELIABILITY_TEST",
         "source":src,"cohort":cohort,"fits":records,
         "period_start":"2020-03-30","period_end_exclusive":"2020-04-06",
         "test_optimizer_updates":0,"source_after_end_read":False,
         "model_promoted":False,"base_native_original_run":39382850579,
         "gate_original_run":39384177806,"baseline_gru_original_run":39382074165,
         "evidence":"one-site real out-of-time testing, no real action effect"}
    (out/"results.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"FROZEN_RELIABILITY_REAL_TEST_DONE",
                      "forecast_windows":len(xx),"arms":len(records),"refits":0},sort_keys=True))
    return result

if __name__=="__main__":
    run("results/me008","results/me007","results/me009","results/me009f")
