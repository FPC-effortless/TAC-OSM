#!/usr/bin/env python3
"""ME011: native TAC vs GRU probabilistic real-transformer forecasting.

Official ETDataset ETTh1 early train/calibration only. No access to later
chronological TEST nor ETTh2 cross-site records in this training script.
The native recurrence remains distinct from SSMs.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math,time,urllib.request
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

COLUMNS=("HUFL","HULL","MUFL","MULL","LUFL","LULL","OT")
HISTORY=96
HORIZONS=(24,168)
QUANTILES=(.1,.5,.9)
TRAIN_END=8760
CAL_END=11640
SEEDS=(11001,11002,11003)
UPDATES=450
BATCH=64
URL="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
# Documented by independent ETTh1 package; abort if upstream mutates.
PIN_MD5="8381763947c85f4be6ac456c508460d6"

def acquire(out, url=URL, expected_md5=PIN_MD5, cutoff=CAL_END):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if cutoff>CAL_END:
        raise ValueError("Original trainer may not read prospective TEST source")
    request=urllib.request.Request(url,headers={"User-Agent":"TAC-OSM-ME011-source/1.0"})
    digest=hashlib.md5();sha=hashlib.sha256();raw=bytearray()
    with urllib.request.urlopen(request,timeout=90) as stream:
        if getattr(stream,"status",200)!=200:raise ValueError("official upstream HTTP status")
        while (data:=stream.read(2**20)):
            raw.extend(data);digest.update(data);sha.update(data)
            if len(raw)>10_000_000:raise ValueError("unexpected source file size")
    if expected_md5 and digest.hexdigest()!=expected_md5:
        raise ValueError("upstream authentic ETTh1 checksum changed")
    reader=csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if tuple(reader.fieldnames or ())!=("date",)+COLUMNS:raise ValueError("ETT schema changed")
    vals=[];timestamps=[];seen=0
    for row in reader:
        if seen>=cutoff:break
        when=datetime.fromisoformat(row["date"])
        if timestamps and (when-timestamps[-1]).total_seconds()!=3600:
            raise ValueError("non-hourly source discontinuity")
        v=[float(row[c]) for c in COLUMNS]
        if not all(math.isfinite(x) for x in v):raise ValueError("unexpected missing ETTh fields")
        vals.append(v);timestamps.append(when);seen+=1
    if seen!=cutoff:raise ValueError(f"authentic early source rows incomplete: {seen}")
    return np.array(vals,dtype=np.float32),{
        "url":url,"publisher_md5":digest.hexdigest(),"publisher_sha256":sha.hexdigest(),
        "source_total_bytes":len(raw),"rows_parsed":seen,
        "first_timestamp":str(timestamps[0]),"last_timestamp":str(timestamps[-1]),
        "cutoff_exclusive_row":cutoff,
        "final_test_labels_parsed":False,"cross_site_labels_parsed":False}

def pack(data,stats,first,last,stride=6):
    # t is the LAST observed sample. Targets t+24 and t+168 are hidden inputs.
    mu=np.asarray(stats["train_mu"],dtype=np.float32)
    scale=np.asarray(stats["train_std"],dtype=np.float32)
    scaled=((data-mu)/scale).astype(np.float32)
    clipped=np.clip(scaled,-10,10)
    mask=np.isfinite(scaled).astype(np.float32)
    feat=np.concatenate((np.nan_to_num(clipped,nan=0),mask),axis=-1)
    idx=np.arange(max(HISTORY-1,first),last-max(HORIZONS),stride)
    if len(idx)<100:raise ValueError("too few windows")
    windows=feat[idx[:,None]+np.arange(-HISTORY+1,1)[None,:]]
    # Target oil temperature only, in train-only physical standard-deviation units.
    delta=(scaled[idx[:,None]+np.array(HORIZONS)[None,:],6]-
           scaled[idx,None,6]).astype(np.float32)
    return torch.from_numpy(windows.copy()),torch.from_numpy(delta.copy()),{
        "windows":len(idx),"min_anchor_idx":int(idx[0]),"max_anchor_idx":int(idx[-1]),
        "first_requested_anchor":first,"last_target_boundary_exclusive":last,
        "stride":stride}

def source_and_splits(out):
    rows,src=acquire(out)
    train=rows[:TRAIN_END]
    mu=train.mean(0,dtype=np.float64)
    std=np.maximum(train.std(0,dtype=np.float64),1e-3)
    stats={"train_mu":mu.tolist(),"train_std":std.tolist(),
           "train_rows":TRAIN_END,"historical_calibration_cutoff":CAL_END,
           "columns":list(COLUMNS),"history":HISTORY,"horizons":list(HORIZONS)}
    a=pack(rows,stats,HISTORY-1,TRAIN_END)
    b=pack(rows,stats,TRAIN_END,CAL_END)
    return a,b,stats,src

class QuantileHead(nn.Module):
    """Noncrossing q10<=q50<=q90 in train-normalized delta units."""
    def __init__(self,in_dim):
        super().__init__()
        self.linear=nn.Linear(in_dim,len(HORIZONS)*3)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        with torch.no_grad():
            self.linear.bias.view(2,3)[:,1:].fill_(-1.5)
    def forward(self,h):
        raw=self.linear(h).reshape(-1,2,3)
        center=raw[:,:,0]
        left=F.softplus(raw[:,:,1])
        right=F.softplus(raw[:,:,2])
        return torch.stack((center-left,center,center+right),dim=-1)

class NativeTACETT(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        self.enc=nn.Linear(14,width)
        self.recur=nn.Linear(width,width,bias=False)
        self.gate=nn.Linear(14+width,width)
        nn.init.constant_(self.gate.bias,-2.5)
        self.slow=nn.Linear(width*2,width)
        self.slow_gate=nn.Linear(width*2,width)
        nn.init.constant_(self.slow_gate.bias,-4)
        self.head=QuantileHead(width*2)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(HISTORY,14):raise ValueError("bad input dimensions")
        z=x.new_zeros(len(x),self.recur.out_features);s=z.clone()
        for t in range(x.shape[1]):
            u=x[:,t]
            proposal=torch.tanh(self.enc(u)+self.recur(z))
            g=torch.sigmoid(self.gate(torch.cat((u,z),dim=-1)))
            z=(1-g)*z+g*proposal
            q=torch.cat((s,z),dim=-1)
            sg=torch.sigmoid(self.slow_gate(q))
            s=(1-sg)*s+sg*torch.tanh(self.slow(q))
        return self.head(torch.cat((z,s),dim=-1))

class GRUETT(nn.Module):
    def __init__(self,width=40):
        super().__init__()
        self.gru=nn.GRU(14,width,batch_first=True)
        self.head=QuantileHead(width)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(HISTORY,14):raise ValueError("bad input dimensions")
        return self.head(self.gru(x)[0][:,-1])

def model_for(name):
    return NativeTACETT() if name=="native_tac" else GRUETT() if name=="gru" else None

def pinball(pred,target):
    if pred.shape!=target.shape+(3,):raise ValueError("quantile shape mismatch")
    e=target[:,:,None]-pred
    q=pred.new_tensor(QUANTILES).view(1,1,3)
    return torch.maximum(q*e,(q-1)*e).mean()

def metrics(pred,y,stats):
    err=(pred[:,:,1]-y).abs()
    cover=(y>=pred[:,:,0])&(y<=pred[:,:,2])
    width=(pred[:,:,2]-pred[:,:,0])
    sd=float(stats["train_std"][6])
    return {"median_mae_train_norm":float(err.mean()),
        "median_mae_degrees_c":float(err.mean()*sd),
        "per_horizon_mae_deg_c":(err.mean(0)*sd).tolist(),
        "pinball_train_norm":float(pinball(pred,y)),
        "interval_80_coverage":float(cover.float().mean()),
        "per_horizon_coverage":cover.float().mean(0).tolist(),
        "mean_interval_width_degrees_c":float(width.mean()*sd),
        "predicted_crossings":int(((pred[:,:,0]>pred[:,:,1])|(pred[:,:,1]>pred[:,:,2])).sum())}

def evaluate(model,hold,stats):
    x,y=hold[:2];model.eval();chunks=[]
    with torch.inference_mode():
        for i in range(0,len(x),256):
            chunks.append(model(x[i:i+256]))
    return metrics(torch.cat(chunks),y,stats)

def fit(arm,seed,train,valid,stats,out):
    torch.manual_seed(seed);torch.set_num_threads(1)
    model=model_for(arm)
    params=sum(p.numel() for p in model.parameters())
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    rng=torch.Generator().manual_seed(seed+50)
    x,y=train[:2];t0=time.monotonic();curve=[]
    before=evaluate(model,valid,stats)
    for step in range(1,UPDATES+1):
        ix=torch.randint(len(x),(BATCH,),generator=rng)
        pred=model(x[ix]);loss=pinball(pred,y[ix])
        if not bool(torch.isfinite(loss)):raise ValueError("nonfinite loss")
        opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(grad)):raise ValueError("nonfinite gradient")
        opt.step()
        if step in (1,UPDATES//2,UPDATES):curve.append({"step":step,"pinball_train_batch":float(loss.detach())})
    scores=evaluate(model,valid,stats)
    dest=out/"checkpoints"/f"{arm}-{seed}.pt"
    dest.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"kind":arm,"seed":seed,"steps":UPDATES,
                "model_state":model.state_dict()},dest)
    return {"model":arm,"seed":seed,"steps":UPDATES,"trainable_params":params,
        "training_cpu_seconds":round(time.monotonic()-t0,3),
        "historical_validation_before":before,"historical_validation":scores,
        "loss_curve":curve,"checkpoint":str(dest.relative_to(out)),
        "sha256":hashlib.sha256(dest.read_bytes()).hexdigest()}

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    train,val,stats,src=source_and_splits(out)
    report={"study":"ME011_REAL_TRANSFORMER_QUANTILES",
      "source":src,"normalization":stats,"train_windows":train[2],
      "calibration_windows":val[2],"fits":[],
      "official_future_labels_parsed":False,"etth2_source_opened":False,
      "models_promoted":False,"optimizer_updates_per_fit":UPDATES}
    for seed in SEEDS:
        for kind in ("native_tac","gru"):
            rec=fit(kind,seed,train,val,stats,out)
            report["fits"].append(rec)
            print(json.dumps({"model":kind,"seed":seed,"historical_scores":rec["historical_validation"],
                              "seconds":rec["training_cpu_seconds"]},sort_keys=True),flush=True)
    (out/"results.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"ME011_EARLY_ETTH1_TRAIN_ONLY_COMPLETE",
             "fits":6,"future_or_cross_site_test_opened":False}),flush=True)

def preflight():
    torch.manual_seed(123);torch.set_num_threads(1)
    x=torch.randn(2,HISTORY,14)
    y=torch.randn(2,2)
    for kind in ("native_tac","gru"):
        m=model_for(kind)
        p=m(x)
        assert p.shape==(2,2,3)
        assert bool((p[:,:,0]<=p[:,:,1]).all() and (p[:,:,1]<=p[:,:,2]).all())
        assert bool(torch.isfinite(p).all())
        loss=pinball(p,y);loss.backward()
        assert all(bool(torch.isfinite(v.grad).all()) for v in m.parameters() if v.grad is not None)
        assert m.head.linear.weight.grad.abs().sum()>0
        print(json.dumps({"model":kind,"params":sum(v.numel() for v in m.parameters())}))
    print("ME011_QUANTILE_AND_CAUSAL_SHAPES_PASS")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="results/me011")
    p.add_argument("--self-test",action="store_true");a=p.parse_args()
    preflight() if a.self_test else run(a.out)
