#!/usr/bin/env python3
"""ME004F one-time genuinely later real-world forecast test. ZERO model fits."""
from __future__ import annotations
import csv,hashlib,io,json,urllib.request,zipfile,math
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from scripts.me004_native_real_forecast import FEATURES,HIST,HORIZONS,make_model,score

URL="https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip"
SHA="aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a"
START=datetime.fromisoformat("2020-03-02 00:00:00")
END=datetime.fromisoformat("2020-03-09 00:00:00")

def source(out):
    path=out/"metropt3-verified.zip";sha=hashlib.sha256();n=0
    req=urllib.request.Request(URL,headers={"User-Agent":"TAC-OSM-ME004F/1.0"})
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

def run(original,out):
    original=Path(original);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report=json.loads((original/'results.json').read_text())
    if report['study']!='ME004' or len(report['fits'])!=6:raise ValueError('wrong original weights')
    if not report['official_future_test_unopened']:raise ValueError('bad original source contract')
    for f in report['fits']:
        sha=hashlib.sha256((original/f['checkpoint']).read_bytes()).hexdigest()
        if sha!=f['checkpoint_sha256']:raise ValueError('frozen original checkpoint mismatch')
    physical,time_points,src=source(out)
    xx,yy,threshold,cohort=prep(physical,time_points,report['cohort'])
    base=score(torch.zeros_like(yy),yy,threshold)
    per_channel=[]
    fits=[];torch.set_num_threads(1)
    for f in report['fits']:
        ck=torch.load(original/f['checkpoint'],map_location='cpu',weights_only=True)
        if ck['kind']!=f['model'] or ck['seed']!=f['seed'] or ck['steps']!=450:
            raise ValueError('checkpoint metadata mismatch')
        model=make_model(f['model'])
        model.load_state_dict(ck['state_dict'],strict=True);model.eval()
        with torch.inference_mode():
            p=torch.cat([model(xx[j:j+256]) for j in range(0,len(xx),256)],dim=0)
        scores=score(p,yy,threshold)
        if not math.isfinite(scores['macro_mae']):raise ValueError('score nonfinite')
        sensor_error=(p-yy).abs().double().mean(0).cpu().numpy()
        per_channel.append({'model':f['model'],'seed':f['seed'],
                            'horizon_feature_mae':sensor_error.tolist()})
        fits.append({'model':f['model'],'seed':f['seed'],'score':scores,
                     'checkpoint_sha256':f['checkpoint_sha256']})
        print(json.dumps({'model':f['model'],'seed':f['seed'],'future_week':scores}),flush=True)
    results={'study':'ME004F_FROZEN_FUTURE_WEEK','source':src,'cohort':cohort,
      'baseline_persistence':base,'fits':fits,'per_channel':per_channel,
      'train_steps_performed':0,'original_model_sha':'954baafe63d1450e4968cc1b49dac02a5e8f6f1d',
      'original_fitting_run':39357593422,'original_fit_artifact':12235023980,
      'future_test_first_and_only_exposure':True,'no_action_outcome_claim':True,
      'architecture_promoted':False,'reserved_later_period_unopened':True}
    (out/'future-week-results.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'FROZEN_OUT_OF_TIME_COMPLETE','n':len(xx),
                      'models_scored':len(fits),'refits':0}),flush=True)
    return results

if __name__=='__main__':
    run('results/me004','results/me004-future')
