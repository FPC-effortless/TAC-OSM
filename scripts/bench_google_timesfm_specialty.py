#!/usr/bin/env python3
"""BENCH-FM-002-A: native TAC versus TimesFM3 on real ETTh1 24/96-hour forecasting."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,time,urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

URL="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
COLS=("HUFL","HULL","MUFL","MULL","LUFL","LULL","OT")
CONTEXT=96; STEPS=(24,96); N_TEST=48; TRAIN_END=8760; DEV_END=11640
SEEDS=(35001,35002); SHARED_UPDATES=250

def data():
    req=urllib.request.Request(URL,headers={"User-Agent":"TAC-OSM-BENCH-FM-002-Research"})
    with urllib.request.urlopen(req,timeout=120) as r:
        raw=r.read(8_000_001)
    if len(raw)>8_000_000:raise ValueError("Unexpected dataset size")
    sha=hashlib.sha256(raw).hexdigest()
    rd=csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if tuple(rd.fieldnames or ())!=("date",)+COLS:raise ValueError("Wrong official ETTh1 columns")
    rows=[];dates=[]
    from datetime import datetime
    for row in rd:
        d=datetime.fromisoformat(row["date"])
        if dates and (d-dates[-1]).total_seconds()!=3600:raise ValueError("Nonhourly source")
        v=[float(row[k]) for k in COLS]
        if not np.isfinite(v).all():raise ValueError("Nonfinite source row")
        rows.append(v);dates.append(d)
    a=np.asarray(rows,dtype=np.float32)
    if len(a)<17000:raise ValueError("Truncated official source")
    mean=a[:TRAIN_END].mean(0,dtype=np.float64).astype(np.float32)
    std=np.maximum(a[:TRAIN_END].std(0,dtype=np.float64),.001).astype(np.float32)
    z=(a-mean)/std
    ixtrain=np.arange(CONTEXT-1,TRAIN_END-max(STEPS),3,dtype=np.int64)
    ixtest=np.linspace(DEV_END,len(z)-max(STEPS)-1,N_TEST,dtype=np.int64)
    if len(np.unique(ixtest))!=N_TEST:raise ValueError("duplicate test origins")
    assert ixtrain.max()+max(STEPS)<TRAIN_END and ixtest.min()>=DEV_END
    def pack(idx,ctx):
        offsets=np.arange(-ctx+1,1)[None,:]
        x=z[idx[:,None]+offsets].copy()
        y=z[idx[:,None]+np.asarray(STEPS)[None,:]].copy()
        return torch.from_numpy(x),torch.from_numpy(y)
    tr=pack(ixtrain,CONTEXT)
    te=pack(ixtest,CONTEXT)
    longx,_=pack(ixtest,512)
    meta={"official_url":URL,"source_sha256":sha,"real_observations":len(a),
          "training_rows":TRAIN_END,"previously_inspected_dev_end":DEV_END,
          "train_origins":len(ixtrain),"test_origins":len(ixtest),
          "test_first_origin":int(ixtest[0]),"test_last_origin":int(ixtest[-1]),
          "horizons_hours":list(STEPS),"channels":list(COLS),
          "fit_only_mean":mean.tolist(),"fit_only_std":std.tolist(),
          "identical_test_targets":True}
    return tr,te,longx,meta

class NativeTAC(nn.Module):
    def __init__(self):
        super().__init__()
        import math
        self.enc=nn.Linear(7,32);self.transition=nn.Linear(32,32,bias=False)
        self.gate=nn.Linear(39,32)
        self.slow=nn.Linear(64,32);self.slow_gate=nn.Linear(64,32)
        self.direct=nn.Linear(42,14);self.mix=nn.Linear(106,96);self.output=nn.Linear(48,14)
        nn.init.orthogonal_(self.transition.weight,gain=.9)
        with torch.no_grad():
            fast=torch.logspace(math.log10(2.),math.log10(32.),32)
            slow=torch.logspace(math.log10(32.),math.log10(256.),32)
            self.gate.bias.copy_(-torch.log(fast-1.))
            self.slow_gate.bias.copy_(-torch.log(slow-1.))
        nn.init.normal_(self.slow.weight,std=.01);nn.init.zeros_(self.slow.bias)
        nn.init.normal_(self.output.weight,std=.001);nn.init.zeros_(self.output.bias)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        mean=x.mean(1,keepdim=True)
        scale=torch.sqrt((x-mean).square().mean(1,keepdim=True)+.0025)
        z=(x-mean)/scale
        f=z.new_zeros((z.shape[0],32));s=f.clone()
        for k in range(z.shape[1]):
            u=z[:,k];proposal=torch.tanh(self.enc(u)+self.transition(f))
            g=torch.sigmoid(self.gate(torch.cat((u,f),-1)))
            f=f+g*(proposal-f)
            sv=torch.sigmoid(self.slow_gate(torch.cat((f,s),-1)))
            innovation=torch.tanh(self.slow(torch.cat((f-s,s),-1)))
            candidate=torch.tanh(s+innovation)
            s=s+sv*(candidate-s)
        state=torch.cat((f/torch.sqrt(f.square().mean(-1,keepdim=True)+.01),
                         s/torch.sqrt(s.square().mean(-1,keepdim=True)+.01)),-1)
        context=torch.cat((z[:,-1],z[:,-6:].mean(1),z[:,-24:].mean(1),
            z[:,-1]-z[:,-6:].mean(1),mean[:,0],torch.log(scale[:,0])),-1)
        a,b=self.mix(torch.cat((context,state),-1)).chunk(2,-1)
        residual=(self.direct(context)+self.output(F.silu(a)*b)).reshape(-1,2,7)
        return x[:,-1:]+residual*scale

class ReferenceGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(7,64,batch_first=True)
        self.head=nn.Sequential(nn.Linear(64,64),nn.SiLU(),nn.Linear(64,14))
    def forward(self,x):
        y,_=self.gru(x)
        return x[:,-1:]+self.head(y[:,-1]).reshape(-1,2,7)

def score(yhat,y):
    diff=(np.asarray(yhat,dtype=np.float64)-np.asarray(y,dtype=np.float64))
    if diff.shape!=(N_TEST,2,7) or not np.isfinite(diff).all():raise ValueError("Invalid forecast")
    return {"MSE":float((diff**2).mean()),"MAE":float(np.abs(diff).mean()),
      "horizon_MAE":np.abs(diff).mean((0,2)).tolist(),
      "channel_MAE":np.abs(diff).mean((0,1)).tolist(),"n_test_origins":N_TEST}

def save(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")

def train_eval(out):
    torch.set_num_threads(1)
    (x,y),(xt,yt),xl,meta=data()
    test=yt.numpy()
    result={"study":"BENCH-FM-002-A","status":"real_test_completed",
       "training_budget_updates":SHARED_UPDATES,"batch":64,"source":meta,"models":{},
       "native_is_newly_trained":True,"old_me035_checkpoint_not_used":True,
       "architecture_winner_qualified":False}
    for seed in SEEDS:
        for name in ("TAC_native","GRU"):
            torch.manual_seed(seed)
            model=torch.jit.script(NativeTAC() if name=="TAC_native" else ReferenceGRU())
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
            g=torch.Generator().manual_seed(seed+200)
            t=time.process_time()
            for step in range(SHARED_UPDATES):
                ix=torch.randint(len(x),(64,),generator=g)
                opt.zero_grad(set_to_none=True)
                pred=model(x[ix]);loss=F.mse_loss(pred,y[ix])
                if not torch.isfinite(loss):raise ArithmeticError("Nonfinite real MSE")
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
            cpu=time.process_time()-t
            ck=out/f"{name}-{seed}.pt"
            torch.save({"weights":model.state_dict(),"seed":seed,"name":name,"updates":SHARED_UPDATES,"source_sha256":meta["source_sha256"]},ck)
            with torch.inference_mode():
                model.eval();p=torch.cat([model(xt[i:i+24]) for i in range(0,len(xt),24)]).numpy()
            result["models"][f"{name}-{seed}"]={**score(p,test),"training_cpu_seconds":cpu,
                "n_parameters":sum(t.numel() for t in model.parameters()),
                "checkpoint_sha256":hashlib.sha256(ck.read_bytes()).hexdigest()}
            print(json.dumps({"completed":f"{name}-{seed}","metrics":result["models"][f"{name}-{seed}"]}),flush=True)
    persistence=xt[:,-1:].expand(-1,2,-1).numpy()
    seasonal=np.stack([xl[:,-24].numpy(),xl[:,-24].numpy()],axis=1)
    result["models"]["persistence"]=score(persistence,test)
    result["models"]["seasonal_lag24"]=score(seasonal,test)
    save(out/"tac-gru-reference-results.json",result)
    print("BENCH_FM_002_TRAINED_REAL_TSCOMPLETE",flush=True)

def google_timesfm(out):
    from timesfm3 import TimesFM3Evaluator,ModelConfig
    torch.set_num_threads(2)
    (_, _),(xt,yt),xl,meta=data()
    reference_file=out/"tac-gru-reference-results.json"
    if not reference_file.exists():raise ValueError("Reference checkpoint manifest missing")
    ref=json.loads(reference_file.read_text())
    if ref["source"]["source_sha256"]!=meta["source_sha256"]:raise ValueError("Source version changed")
    engine=TimesFM3Evaluator(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",per_core_batch_size=4,device="cpu"))
    results={"study":"BENCH-FM-002-A","checkpoint":"google/timesfm-3.0-pytorch",
             "source_sha256":meta["source_sha256"],"models":{}}
    for ctx,xx in [(96,xt),(512,xl)]:
        pred=[];t=time.perf_counter()
        for i in range(0,len(xx),4):
            chunk=[np.ascontiguousarray(a.T,dtype=np.float32) for a in xx[i:i+4].numpy()]
            outputs=list(engine.predict_batch(contexts=chunk,horizon=96,return_quantiles=False,use_symmetric_averaging=False))
            for o in outputs:
                a=np.asarray(o.forecast,dtype=np.float64)
                if a.shape!=(7,96):raise ValueError("Wrong TimesFM3 output "+str(a.shape))
                pred.append(a[:,np.array(STEPS)-1].T)
        r={**score(np.asarray(pred),yt.numpy()),"inference_seconds_excluding_load":time.perf_counter()-t,
           "zero_shot":True,"past_context_hours":ctx}
        results["models"][f"TimesFM3_context{ctx}"]=r
        save(out/"timesfm3-results.json",results)
        print(json.dumps({"model":f"TimesFM3_context{ctx}","score":r}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["train","timesfm"],required=True)
    ap.add_argument("--out",type=Path,default=Path("results/bench-fm-002/timeseries"))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.stage=="train":train_eval(args.out)
    else:google_timesfm(args.out)

if __name__=="__main__":main()
