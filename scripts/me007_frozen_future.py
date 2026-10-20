#!/usr/bin/env python3
"""ME007F original first-exposure March 16-23 matched-control audit. ZERO model fits."""
from __future__ import annotations
import csv,hashlib,io,json,urllib.request,zipfile,math
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from scripts.me004_native_real_forecast import FEATURES,HIST,HORIZONS,make_model,score

URL="https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip"
SHA="aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a"
START=datetime.fromisoformat("2020-03-16 00:00:00")
END=datetime.fromisoformat("2020-03-23 00:00:00")

def source(out):
    path=out/"metropt3-verified.zip";sha=hashlib.sha256();n=0
    req=urllib.request.Request(URL,headers={"User-Agent":"TAC-OSM-ME007F/1.0"})
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
from scripts.me007_train_matched_gru import MatchedGRU, SEEDS as GRU_SEEDS

def checked(path, digest):
    actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual!=digest:raise ValueError("frozen checkpoint digest mismatch: "+str(path))
    return torch.load(path,map_location="cpu",weights_only=True)

def run(original,adapters,controls,output):
    original,adapters,controls,output=map(Path,(original,adapters,controls,output))
    output.mkdir(parents=True,exist_ok=True)
    base_manifest=json.loads((original/"results.json").read_text())
    adapter_manifest=json.loads((adapters/"results.json").read_text())
    ctrl_manifest=json.loads((controls/"results.json").read_text())
    assert base_manifest["study"]=="ME004" and len(base_manifest["fits"])==6
    assert adapter_manifest["study"]=="ME006_REAL_CAUSAL_RELEVANCE" and len(adapter_manifest["fits"])==4
    assert ctrl_manifest["study"]=="ME007_CAPACITY_MATCHED_GRU" and len(ctrl_manifest["fits"])==3
    assert not ctrl_manifest["future_period_ever_opened_by_this_script"]
    assert not adapter_manifest["future_reserved_test_read"]
    originals={}
    for fit in base_manifest["fits"]:
        path=original/fit["checkpoint"]
        checked(path,fit["checkpoint_sha256"])
        if fit["model"]=="tac_two_timescale":
            originals[fit["seed"]]=fit
    assert set(originals)=={240401,240402}
    for fit in adapter_manifest["fits"]:
        checked(adapters/fit["checkpoint"],fit["checkpoint_sha256"])
    ctrl_entries=[]
    for seed_rec in ctrl_manifest["fits"]:
        assert seed_rec["seed"] in GRU_SEEDS
        for entry in seed_rec["checkpoints"]:
            assert entry["steps"] in (450,1000) and entry["parameters"]==9249
            checked(controls/entry["checkpoint"],entry["sha256"])
            ctrl_entries.append(entry)
    assert len(ctrl_entries)==6
    # Crucially, no source is opened until all 16 pretrained/fixed checkpoint
    # references (6 + 4 + 6) have been checked.
    values,ts,src=source(output)
    xx,yy,threshold,cohort=prep(values,ts,base_manifest["cohort"])
    torch.set_num_threads(1)
    persistence=score(torch.zeros_like(yy),yy,threshold)
    records=[]
    def record(name,seed,pred,metadata):
        result=score(pred,yy,threshold)
        # Explicit source-time selection comparison, NEVER used for fitting.
        channel_error=(pred-yy).abs().double().mean((0,1)).tolist()
        row={"model":name,"seed":seed,"score":result,"per_sensor_mae":channel_error,**metadata}
        records.append(row)
        print(json.dumps({"model":name,"seed":seed,"mae":result["macro_mae"]},sort_keys=True),flush=True)
    with torch.inference_mode():
        for seed,entry in sorted(originals.items()):
            ck=checked(original/entry["checkpoint"],entry["checkpoint_sha256"])
            assert ck["seed"]==seed and ck["kind"]=="tac_two_timescale"
            model=make_model("tac_two_timescale")
            model.load_state_dict(ck["state_dict"],strict=True);model.eval()
            pred=torch.cat([model(xx[i:i+256]) for i in range(0,len(xx),256)])
            record("original_tac",seed,pred,{"checkpoint_sha256":entry["checkpoint_sha256"],
                                           "parameters":9368})
        for fit in adapter_manifest["fits"]:
            if fit["mode"]!="direct":continue
            bseed=fit["base_seed"]
            base=originals[bseed]
            original_ck=checked(original/base["checkpoint"],base["checkpoint_sha256"])
            net=make_model("tac_two_timescale")
            net.load_state_dict(original_ck["state_dict"],strict=True)
            model=CausalRelevanceAdapter(net,mode="direct")
            adapter_ck=checked(adapters/fit["checkpoint"],fit["checkpoint_sha256"])
            assert adapter_ck["mode"]=="direct" and adapter_ck["seed"]==fit["seed"] and adapter_ck["base_seed"]==bseed
            model.load_state_dict(adapter_ck["state_dict"],strict=True);model.eval()
            pred=torch.cat([model(xx[i:i+256]) for i in range(0,len(xx),256)])
            record("tac_direct",fit["seed"],pred,{"base_seed":bseed,
                  "checkpoint_sha256":fit["checkpoint_sha256"],
                  "backbone_checkpoint_sha256":base["checkpoint_sha256"],
                  "parameters_total":9368+fit["trainable_parameters"]})
        for fit in ctrl_entries:
            ck=checked(controls/fit["checkpoint"],fit["sha256"])
            assert ck["model"]=="gru41" and ck["steps"]==fit["steps"] and ck["seed"]==fit["seed"]
            model=MatchedGRU()
            model.load_state_dict(ck["state_dict"],strict=True);model.eval()
            pred=torch.cat([model(xx[i:i+256]) for i in range(0,len(xx),256)])
            record(f"gru41_{fit['steps']}",fit["seed"],pred,
                   {"checkpoint_sha256":fit["sha256"],"parameters":9249,
                    "original_train_wall_seconds":fit["train_wall_seconds_cumulative"]})
    assert len(records)==10
    results={"study":"ME007F_CAPACITY_MATCHED_ONE_TIME_REAL",
      "original_publication":"FIRST_EXPOSURE","source":src,"cohort":cohort,
      "period_start":"2020-03-16","period_end_exclusive":"2020-03-23",
      "persistence":persistence,"fits":records,
      "pretrained_me004_run_id":39357593422,
      "pretrained_me006_run_id":39380424029,
      "total_optimizer_updates_on_test":0,"source_after_test_end_read":False,
      "not_external_site_replication":True,"architecture_promoted":False,
      "interpretation":"Capacity nearly matched; compute and training exposure not provably equal"}
    (output/"results.json").write_text(json.dumps(results,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"FROZEN_REAL_TEST_COMPLETE","n":len(xx),
                      "evaluated":len(records),"refits":0},sort_keys=True),flush=True)
    return results

if __name__=="__main__":
    run("results/me004","results/me006","results/me007","results/me007f")
