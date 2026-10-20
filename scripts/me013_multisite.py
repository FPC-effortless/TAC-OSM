#!/usr/bin/env python3
"""ME013 REAL 12-site UCI atmospheric observations; original TRAIN sites/time only.

No model training on future/site TEST. Real outcomes, not simulated benchmarks.
Native TAC remains a two-timescale predictive recurrence, NOT an SSM.
"""
from __future__ import annotations
import csv,hashlib,io,json,math,time,urllib.request,zipfile
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SOURCE="https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip"
ALL_SITES=("Aotizhongxin","Changping","Dingling","Dongsi","Guanyuan",
           "Gucheng","Huairou","Nongzhanguan","Shunyi","Tiantan","Wanliu","Wanshouxigong")
UNSEEN=("Shunyi","Wanshouxigong")
FIT_SITES=tuple(s for s in ALL_SITES if s not in UNSEEN)
NUMERIC=("PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM")
FIT_CUT=datetime(2016,1,1);DEV_CUT=datetime(2016,7,1)
FINAL_CUT=datetime(2017,3,1)
HISTORY=48;HORIZON=24;STRIDE=12;D=26
SEEDS=(13001,13002,13003);UPDATES=500;BATCH=64

def source_zip():
    req=urllib.request.Request(SOURCE,headers={"User-Agent":"TAC-OSM-ME013-real-research/1.0"})
    chunks=[];size=0;sha=hashlib.sha256()
    with urllib.request.urlopen(req,timeout=120) as resp:
        if getattr(resp,"status",200)!=200:raise ValueError("official dataset HTTP error")
        while (part:=resp.read(1<<20)):
            size+=len(part)
            if size>25_000_000:raise ValueError("unexpected source size")
            sha.update(part);chunks.append(part)
    return b"".join(chunks),{"source":SOURCE,"sha256":sha.hexdigest(),"download_bytes":size}

def station_archive_members(z):
    expected={s:f"PRSA_Data_{s}_20130301-20170228.csv" for s in ALL_SITES}
    catalog={Path(n).name:n for n in z.namelist() if not n.endswith("/")}
    if not set(expected.values()).issubset(set(catalog)):
        raise ValueError("canonical 12 real monitoring station CSVs missing")
    return {site:catalog[name] for site,name in expected.items()}

def parse_site(z,member,site,cutoff):
    """NEVER read/open a held-out CSV when caller only supplied FIT_SITES."""
    values=[];times=[];hour_clock=[]
    with z.open(member) as fh:
        reader=csv.DictReader(io.TextIOWrapper(fh,encoding="utf-8-sig",newline=""))
        if not set(NUMERIC+("year","month","day","hour","station")).issubset(set(reader.fieldnames or [])):
            raise ValueError("air-quality source schema mismatch")
        for row in reader:
            t=datetime(int(row["year"]),int(row["month"]),int(row["day"]),int(row["hour"]))
            if t>=cutoff:break
            if row["station"]!=site:raise ValueError("station member identity mismatch")
            if times and t-times[-1]!=timedelta(hours=1):
                raise ValueError("nonchronological station source")
            obs=[]
            for name in NUMERIC:
                try:v=float(row[name])
                except (ValueError,TypeError):v=float("nan")
                obs.append(v if math.isfinite(v) else float("nan"))
            values.append(obs);times.append(t)
            hour_clock.append([math.sin(2*math.pi*t.hour/24),
                               math.cos(2*math.pi*t.hour/24),
                               math.sin(2*math.pi*t.timetuple().tm_yday/366),
                               math.cos(2*math.pi*t.timetuple().tm_yday/366)])
    if len(times)<10000:raise ValueError("station source too short")
    return np.asarray(values,dtype=np.float32),np.asarray(times,dtype="datetime64[s]"),np.asarray(hour_clock,dtype=np.float32)

def read_allowed(allowed,cutoff):
    allowed=tuple(allowed)
    if len(set(allowed))!=len(allowed) or not set(allowed).issubset(set(ALL_SITES)):
        raise ValueError("unregistered station names")
    raw,meta=source_zip()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        files=station_archive_members(z)
        parsed={site:parse_site(z,files[site],site,cutoff) for site in allowed}
    meta.update({"parsed_stations":list(allowed),
                 "parsed_rows_by_station":{s:len(v[0]) for s,v in parsed.items()},
                 "source_rows_at_or_after_cutoff_parsed":False,
                 "source_cutoff_exclusive":str(cutoff)})
    return parsed,meta

def global_train_stats(sites):
    array=np.concatenate([v[0][v[1]<np.datetime64(FIT_CUT)] for v in sites.values()])
    mean=np.nanmean(array,axis=0);std=np.maximum(np.nanstd(array,axis=0),1e-3)
    if not (np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all()):
        raise ValueError("unusable genuine physical training statistics")
    return {"train_only_mu":mean.tolist(),"train_only_sigma":std.tolist(),
            "physical_columns":list(NUMERIC),"input_dim":D,"history":HISTORY,
            "future_horizon_hours":HORIZON,"stride_hours":STRIDE}

def windows(parsed,stats,start,end):
    mu=np.asarray(stats["train_only_mu"],dtype="float32")
    std=np.asarray(stats["train_only_sigma"],dtype="float32")
    X=[];Y=[];B=[];sites=[]
    for site,(values,times,calendar) in parsed.items():
        mask=np.isfinite(values)
        norm=np.where(mask,(values-mu)/std,0)
        bounded=np.clip(norm,-10,10).astype("float32")
        xx=np.concatenate((bounded,mask.astype("float32"),calendar),axis=1)
        timestamps=times.astype("datetime64[s]")
        max_i=len(xx)-HORIZON
        start_i=max(HISTORY-1,int(np.searchsorted(timestamps,np.datetime64(start))))
        anchors=np.arange(start_i,max_i,STRIDE)
        cutoff=np.datetime64(end)
        anchors=anchors[timestamps[anchors+HORIZON]<cutoff]
        anchors=anchors[mask[anchors,0]&mask[anchors+HORIZON,0]]
        if not len(anchors):continue
        for i in anchors:
            X.append(xx[i-HISTORY+1:i+1])
            Y.append(float((values[i+HORIZON,0]-values[i,0])/std[0]))
            B.append([float(values[i,0]),float(values[i-24,0]) if mask[i-24,0] else float("nan")])
            sites.append(site)
    if not X:raise ValueError("no eligible genuine forecast targets")
    return (torch.tensor(np.asarray(X,dtype="float32")),
            torch.tensor(np.asarray(Y,dtype="float32")[:,None]),
            torch.tensor(np.asarray(B,dtype="float32")),
            sites)

class NativeTAC(nn.Module):
    def __init__(self,routed=False):
        super().__init__();self.routed=routed;h=32
        self.enc=nn.Linear(D,h);self.recur=nn.Linear(h,h,bias=False)
        self.gate=nn.Linear(D+h,h)
        nn.init.constant_(self.gate.bias,-2.5)
        self.slow=nn.Linear(2*h,h);self.slow_gate=nn.Linear(2*h,h)
        nn.init.constant_(self.slow_gate.bias,-4.)
        self.head=nn.Linear(2*h,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        if routed:
            self.query=nn.Linear(2*h,16)
            self.anchors=nn.Parameter(torch.randn(4,16)*.05)
            self.experts=nn.ModuleList([nn.Linear(2*h,1,bias=False) for _ in range(4)])
            for e in self.experts:nn.init.zeros_(e.weight)

    def forward(self,x,return_route=False):
        if x.ndim!=3 or x.shape[1:]!=(HISTORY,D):raise ValueError("native causal input shape")
        z=x.new_zeros(len(x),32);s=z.clone()
        for t in range(x.shape[1]):
            u=x[:,t]
            q=torch.tanh(self.enc(u)+self.recur(z))
            g=torch.sigmoid(self.gate(torch.cat((u,z),dim=-1)))
            z=(1-g)*z+g*q
            ctx=torch.cat((s,z),dim=-1)
            gs=torch.sigmoid(self.slow_gate(ctx))
            s=(1-gs)*s+gs*torch.tanh(self.slow(ctx))
        state=torch.cat((z,s),dim=-1)
        y=self.head(state)
        route=None
        if self.routed:
            route=torch.softmax(torch.tanh(self.query(state))@self.anchors.T/4,dim=-1)
            candidates=torch.cat([expert(state) for expert in self.experts],dim=-1)
            y=y+(candidates*route).sum(-1,keepdim=True)
        if return_route:return y,route
        return y

class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(D,44,batch_first=True)
        self.head=nn.Linear(44,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x,return_route=False):
        if x.ndim!=3 or x.shape[1:]!=(HISTORY,D):raise ValueError("GRU causal input shape")
        y=self.head(self.gru(x)[0][:,-1])
        return (y,None) if return_route else y

def model_for(kind):
    return NativeTAC(kind=="native_routed") if kind in ("native_plain","native_routed") else GRU() if kind=="gru" else None

def evaluate(model,hold,stats):
    x,y,baselines,names=hold
    model.eval();all_predictions=[];routes=[]
    with torch.inference_mode():
        for i in range(0,len(x),256):
            p,r=model(x[i:i+256],return_route=True)
            all_predictions.append(p)
            if r is not None:routes.append(r)
    pred=torch.cat(all_predictions).reshape(-1)
    y=y.reshape(-1)
    physical=float(stats["train_only_sigma"][0])
    error=(pred-y).abs()
    answer={"mae_ug_m3":float(error.mean()*physical),
            "persistence_mae_ug_m3":float(y.abs().mean()*physical),
            "sample_count":len(names),"per_site_mae_ug_m3":{},
            "large_change_mae_ug_m3":None,"quiet_mae_ug_m3":None}
    threshold=1.0
    changed=y.abs()>=threshold
    if bool(changed.any()):answer["large_change_mae_ug_m3"]=float(error[changed].mean()*physical)
    if bool((~changed).any()):answer["quiet_mae_ug_m3"]=float(error[~changed].mean()*physical)
    for site in sorted(set(names)):
        selector=torch.tensor([s==site for s in names],dtype=torch.bool)
        answer["per_site_mae_ug_m3"][site]=float(error[selector].mean()*physical)
    if routes:
        rt=torch.cat(routes)
        entropy=-(rt*rt.clamp_min(1e-9).log()).sum(-1).mean()
        answer["route_entropy_nats"]=float(entropy)
        answer["route_weights_by_station"]={
          site:rt[torch.tensor([s==site for s in names])].mean(0).tolist()
          for site in sorted(set(names))}
    return answer

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Original first exposure opens ONLY ten registered known stations through DEV_CUT.
    parsed,manifest=read_allowed(FIT_SITES,DEV_CUT)
    if tuple(sorted(manifest["parsed_stations"]))!=tuple(sorted(FIT_SITES)):
        raise AssertionError("heldout station leak")
    stats=global_train_stats(parsed)
    train=windows(parsed,stats,datetime(2013,3,1),FIT_CUT)
    dev=windows(parsed,stats,FIT_CUT,DEV_CUT)
    torch.set_num_threads(1)
    report={"study":"ME013_UCI_REAL_MULTI_STATION","source":manifest,
      "stats":stats,"splits":{"training_end":str(FIT_CUT),
      "dev_end":str(DEV_CUT),"test_unopened":True,"heldout_sites_unopened":True},
      "train_windows":len(train[0]),"dev_windows":len(dev[0]),
      "fits":[],"future_test_read":False,"architecture_promoted":False}
    x,y=train[:2]
    for seed in SEEDS:
      for kind in ("native_plain","native_routed","gru"):
        torch.manual_seed(seed)
        model=model_for(kind)
        parameters=sum(v.numel() for v in model.parameters())
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        rng=torch.Generator().manual_seed(seed+2026)
        trace=[];start=time.perf_counter()
        for step in range(1,UPDATES+1):
            idx=torch.randint(len(x),(BATCH,),generator=rng)
            pred=model(x[idx]);loss=F.smooth_l1_loss(pred,y[idx],beta=.2)
            if not bool(torch.isfinite(loss)):raise ArithmeticError("training nonfinite")
            opt.zero_grad(set_to_none=True);loss.backward()
            g=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not bool(torch.isfinite(g)):raise ArithmeticError("gradient nonfinite")
            opt.step()
            if step in (1,UPDATES//2,UPDATES):
                trace.append({"update":step,"train_batch_loss":float(loss.detach())})
        trained=evaluate(model,dev,stats)
        ck=out/"checkpoints"/f"{kind}-{seed}.pt";ck.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"kind":kind,"seed":seed,"steps":UPDATES,
                    "source_sha256":manifest["sha256"],"model_state":model.state_dict()},ck)
        record={"model":kind,"seed":seed,"trainable_parameters":parameters,
         "train_wall_seconds":round(time.perf_counter()-start,3),
         "historical_dev":trained,"curve":trace,"checkpoint":str(ck.relative_to(out)),
         "checkpoint_sha256":hashlib.sha256(ck.read_bytes()).hexdigest()}
        report["fits"].append(record)
        print(json.dumps({"model":kind,"seed":seed,
          "historic_mae":trained["mae_ug_m3"],"params":parameters},sort_keys=True),flush=True)
    (out/"results.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"ME013_REAL_ORIGINAL_FITS_DONE",
      "models":len(report["fits"]),"heldout_sites_read":False,
      "later_time_rows_parsed":False,"promoted":False}),flush=True)

def preflight():
    torch.set_num_threads(1);torch.manual_seed(1326)
    x=torch.randn(3,HISTORY,D)
    y=torch.randn(3,1)
    for kind in ("native_plain","native_routed","gru"):
        m=model_for(kind)
        p,route=m(x,return_route=True)
        assert p.shape==(3,1) and bool(torch.isfinite(p).all())
        assert torch.equal(p,torch.zeros_like(p))
        if kind=="native_routed":
            assert route.shape==(3,4)
            assert torch.allclose(route.sum(1),torch.ones(3),atol=1e-6)
        F.smooth_l1_loss(p,y).backward()
        assert all(bool(torch.isfinite(v.grad).all()) for v in m.parameters() if v.grad is not None)
        print(json.dumps({"kind":kind,"parameters":sum(v.numel() for v in m.parameters())}))
    print("ME013_NATIVE_BOUNDS_ROUTING_GRADIENT_SOFTWARE_PASS")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--out",default="results/me013")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args();preflight() if a.self_test else run(a.out)
