#!/usr/bin/env python3
"""ME004: direct real MetroPT-3 forecasting of authentic compressor observations.
Explore GRU versus native TAC identity-first and two-timescale recurrence.
Not an SSM conversion. Original TRAIN-era data only; no action/causal claims.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,time,urllib.request,zipfile
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
URL='https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip'
ZIP_SHA='aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a'
FEATURES=('TP2','TP3','H1','DV_pressure','Reservoirs','Oil_temperature',
          'Motor_current','COMP','DV_eletric','Towers','MPG','Pressure_switch')
HIST=32;HORIZONS=(12,60);SEEDS=(240401,240402);STEPS=450;BATCH=128

def acquire(out):
    path=out/'source.zip';digest=hashlib.sha256();size=0
    req=urllib.request.Request(URL,headers={'User-Agent':'TAC-OSM-ME004/1.0'})
    with urllib.request.urlopen(req,timeout=120) as response,path.open('wb') as dest:
        if getattr(response,'status',200)!=200:raise ValueError('source status')
        while (chunk:=response.read(2**20)):
            size+=len(chunk)
            if size>500_000_000:raise ValueError('oversized source')
            digest.update(chunk);dest.write(chunk)
    if digest.hexdigest()!=ZIP_SHA:raise ValueError('publisher source digest changed')
    values=[];times=[]
    with zipfile.ZipFile(path) as z:
        matched=[n for n in z.namelist() if Path(n).name.lower()=='metropt3(aircompressor).csv']
        if len(matched)!=1:raise ValueError('missing source member')
        with z.open(matched[0]) as f:
            reader=csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline=''))
            if not set(FEATURES+('timestamp',)).issubset(set(reader.fieldnames or [])):
                raise ValueError('wrong real telemetry schema')
            first=None
            for row in reader:
                t=datetime.fromisoformat(row['timestamp'].strip())
                if first is None:first=t
                if t>=first+timedelta(days=25):break
                if times and t<times[-1]:raise ValueError('source not chronological')
                obs=[]
                for sensor in FEATURES:
                    try:value=float(row[sensor])
                    except (TypeError,ValueError):value=float('nan')
                    obs.append(value)
                values.append(obs);times.append(t)
    path.unlink(missing_ok=True)
    if len(values)<90000:raise ValueError('insufficient real readouts')
    return np.asarray(values,dtype='float32'),np.asarray(times,dtype='datetime64[s]').astype('int64'),{
        'source_url':URL,'publisher_archive_sha256':digest.hexdigest(),'archive_bytes':size,
        'rows_read':len(values),'first_timestamp':str(times[0]),'last_timestamp':str(times[-1]),
        'future_after_day25_not_read':True,'source_member':matched[0],'synthetic_training':False}

def prepare(data,t):
    train_cut=t[0]+20*86400
    train_obs=np.where(t[:,None]<train_cut,data,np.nan)
    mean=np.nanmean(train_obs,axis=0)
    std=np.maximum(np.nanstd(train_obs,axis=0),1e-3)
    mask=np.isfinite(data)
    v=np.clip(np.nan_to_num((data-mean)/std,nan=0,posinf=0,neginf=0),-10,10).astype('float32')
    f=np.concatenate([v,mask.astype('float32')],axis=1)
    gaps=np.diff(t)
    invalid=(gaps<=0)|(gaps>30)
    cum=np.concatenate([[0],np.cumsum(invalid.astype('int32'))])
    idx=np.arange(HIST-1,len(v)-max(HORIZONS),12)
    idx=idx[cum[idx+max(HORIZONS)]==cum[idx-HIST+1]]
    idx=idx[np.all(mask[idx[:,None]+np.array(HORIZONS)[None,:]],axis=(1,2))]
    tr=idx[t[idx+max(HORIZONS)]<train_cut]
    va=idx[(t[idx]>=train_cut)&(t[idx+max(HORIZONS)]<t[0]+25*86400)]
    if len(tr)<4000 or len(va)<500:raise ValueError('insufficient time separated examples')
    offsets=np.arange(HIST)[None,:]-HIST+1
    def pack(ix):
        xx=f[ix[:,None]+offsets].copy()
        yy=np.stack([v[ix+h]-v[ix] for h in HORIZONS],axis=1)
        return torch.from_numpy(xx),torch.from_numpy(yy)
    training,validation=pack(tr),pack(va)
    threshold=np.maximum(np.quantile(np.abs(training[1].numpy()),.90,axis=0),.04).astype('float32')
    return training,validation,torch.from_numpy(threshold),{
        'train_windows':len(tr),'validation_windows':len(va),'feature_names':list(FEATURES),
        'time_history':HIST,'horizons_in_samples':list(HORIZONS),
        'train_cut_timestamp':str(np.datetime64(int(train_cut),'s')),
        'mean_train_only':mean.tolist(),'std_train_only':std.tolist(),
        'threshold_train_only':threshold.tolist(),'sample_stride':12}

class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(2*len(FEATURES),32,batch_first=True)
        self.head=nn.Linear(32,len(FEATURES)*len(HORIZONS))
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        return self.head(self.gru(x)[0][:,-1]).reshape(-1,2,len(FEATURES))

class NativeTAC(nn.Module):
    def __init__(self,kind):
        super().__init__()
        self.kind=kind;h=32;d=2*len(FEATURES)
        self.encoder=nn.Linear(d,h);self.recur=nn.Linear(h,h,bias=False)
        self.gate=nn.Linear(d+h,h);nn.init.constant_(self.gate.bias,-2.5)
        if kind=='tac_two_timescale':
            self.slow=nn.Linear(h*2,h);self.slow_gate=nn.Linear(h*2,h)
            nn.init.constant_(self.slow_gate.bias,-4)
        self.head=nn.Linear(h*(2 if kind=='tac_two_timescale' else 1),len(FEATURES)*2)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        b,T,_=x.shape
        state=x.new_zeros(b,32);slow=state.clone()
        for step in range(T):
            u=x[:,step];proposal=torch.tanh(self.encoder(u)+self.recur(state))
            g=torch.sigmoid(self.gate(torch.cat((u,state),dim=-1)))
            state=(1-g)*state+g*proposal
            if self.kind=='tac_two_timescale':
                context=torch.cat((slow,state),dim=-1)
                sg=torch.sigmoid(self.slow_gate(context))
                slow=(1-sg)*slow+sg*torch.tanh(self.slow(context))
        output=torch.cat((state,slow),dim=-1) if self.kind=='tac_two_timescale' else state
        return self.head(output).reshape(-1,2,len(FEATURES))

def make_model(kind):
    return GRU() if kind=='gru' else NativeTAC(kind) if kind in ('tac_identity','tac_two_timescale') else None

def score(pred,actual,threshold):
    error=(pred-actual).abs().double();p=actual.abs().double()
    changes=actual.abs()>threshold[None,:,:]
    up=(actual>threshold[None,:,:]);down=(actual< -threshold[None,:,:])
    return {'macro_mae':float(error.mean()),'persistence_macro_mae':float(p.mean()),
      'per_horizon_mae':error.mean((0,2)).tolist(),
      'per_horizon_persistence_mae':p.mean((0,2)).tolist(),
      'changed_mae':float(error[changes].mean()) if changes.any() else None,
      'changed_persistence_mae':float(p[changes].mean()) if changes.any() else None,
      'changed_count':int(changes.sum()),'positive_changes':int(up.sum()),'negative_changes':int(down.sum()),
      'change_direction_accuracy':float(((torch.sign(pred)==torch.sign(actual))&changes).sum()/changes.sum()) if changes.any() else None}

def evaluate(model,data,threshold):
    model.eval();x,y=data
    with torch.inference_mode():
        pred=torch.cat([model(x[i:i+256]) for i in range(0,len(x),256)],dim=0)
    return score(pred,y,threshold)

def fit(kind,seed,train,val,threshold,out,steps):
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=make_model(kind);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(seed+200)
    x,y=train;curve=[];tic=time.perf_counter()
    for j in range(steps):
        ix=torch.randint(len(x),(128,),generator=generator)
        prediction=model(x[ix]);loss=F.smooth_l1_loss(prediction,y[ix],beta=.2)
        if not bool(torch.isfinite(loss)):raise ValueError('nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        if not bool(torch.isfinite(gn)):raise ValueError('nonfinite grad')
        opt.step()
        if j+1 in (1,steps//2,steps):curve.append({'step':j+1,'loss':float(loss.detach())})
    result=evaluate(model,val,threshold)
    checkpoint=out/'checkpoints'/f'{kind}-{seed}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'kind':kind,'seed':seed,'steps':steps,'state_dict':model.state_dict()},checkpoint)
    return {'model':kind,'seed':seed,'parameter_count':sum(p.numel() for p in model.parameters()),
      'train_steps':steps,'train_seconds':round(time.perf_counter()-tic,2),
      'validation':result,'curve':curve,'checkpoint':str(checkpoint.relative_to(out)),
      'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}

def run(out,steps=STEPS):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data,t,source=acquire(out)
    train,val,threshold,cohort=prepare(data,t)
    baseline=score(torch.zeros_like(val[1]),val[1],threshold)
    report={'study':'ME004','stage':'ORIGINAL_REAL_TRAIN_ERA_EXPLORATORY','source':source,
        'cohort':cohort,'baseline_persistence':baseline,'fits':[],
        'official_future_test_unopened':True,'architecture_promoted':False,
        'decision_utility_tested':False,'commercial_qualified':False}
    for seed in SEEDS:
        for kind in ('gru','tac_identity','tac_two_timescale'):
            r=fit(kind,seed,train,val,threshold,out,steps)
            print(json.dumps({'model':kind,'seed':seed,'metrics':r['validation'],
                              'train_seconds':r['train_seconds']},sort_keys=True),flush=True)
            report['fits'].append(r)
    (out/'results.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (out/'source.json').write_text(json.dumps(source,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':'COMPLETE','fits':len(report['fits']),
                     'persistence_mae':baseline['macro_mae']}),flush=True)

def selftest():
    torch.set_num_threads(1);torch.manual_seed(42)
    x=torch.randn(4,HIST,24)
    for k in ('gru','tac_identity','tac_two_timescale'):
        model=make_model(k);y=model(x)
        assert y.shape==(4,2,len(FEATURES)) and bool(torch.isfinite(y).all())
        y.square().mean().backward()
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        assert any(bool(torch.any(p.grad!=0)) for p in model.parameters() if p.grad is not None)
        print(k, sum(p.numel() for p in model.parameters()),flush=True)
    print('ME004_SOFTWARE_PREFLIGHT_PASS')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='results/me004')
    parser.add_argument('--steps',type=int,default=STEPS)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    selftest() if args.self_test else run(args.out,args.steps)
