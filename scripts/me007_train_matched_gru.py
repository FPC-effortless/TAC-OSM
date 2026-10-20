#!/usr/bin/env python3
"""ME007 strong capacity-matched GRU controls, authentic previously-used TRAIN only.

Use frozen registration, no later-week opening here.
TAC-OSM is a native world model and is NOT converted into an SSM or GRU.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from scripts.me004_native_real_forecast import acquire,prepare,FEATURES,HORIZONS,HIST,score

SEEDS=(240401,240402,240403)
TRAIN_STEPS=(450,1000)
HIDDEN=41
BATCH=128

class MatchedGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(len(FEATURES)*2,HIDDEN,batch_first=True)
        self.head=nn.Linear(HIDDEN,len(FEATURES)*len(HORIZONS))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self,x):
        if x.ndim!=3 or x.shape[-2:]!=(HIST,2*len(FEATURES)):
            raise ValueError("expected [batch,32,24] observed historical sensor tokens")
        return self.head(self.gru(x)[0][:,-1]).reshape(-1,len(HORIZONS),len(FEATURES))

def evaluate(model,data,threshold):
    model.eval();x,y=data
    with torch.inference_mode():
        pred=torch.cat([model(x[i:i+256]) for i in range(0,len(x),256)])
    return score(pred,y,threshold)

def fit(seed,train,val,threshold,out):
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=MatchedGRU()
    n=sum(p.numel() for p in model.parameters())
    if n!=9249:raise AssertionError(f"parameter shape changed: {n}")
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(seed+200)
    x,y=train;records=[];curve=[];t0=time.monotonic()
    for step in range(1,max(TRAIN_STEPS)+1):
        indices=torch.randint(len(x),(BATCH,),generator=generator)
        pred=model(x[indices])
        loss=F.smooth_l1_loss(pred,y[indices],beta=.2)
        if not torch.isfinite(loss):raise ValueError("training loss nonfinite")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        if not torch.isfinite(grad):raise ValueError("nonfinite gradient")
        optim.step()
        if step in (1,450,1000):
            curve.append({"steps":step,"train_loss":float(loss.detach())})
        if step in TRAIN_STEPS:
            ck=out/"checkpoints"/f"gru41-{seed}-steps{step}.pt"
            ck.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model":"gru41","seed":seed,"steps":step,"state_dict":model.state_dict()},ck)
            records.append({"model":"gru41","seed":seed,"steps":step,"parameters":n,
                 "train_wall_seconds_cumulative":round(time.monotonic()-t0,4),
                 "exploratory_existing_val":evaluate(model,val,threshold),
                 "checkpoint":str(ck.relative_to(out)),
                 "sha256":hashlib.sha256(ck.read_bytes()).hexdigest()})
            print(json.dumps(records[-1],sort_keys=True),flush=True)
    return {"seed":seed,"curve":curve,"checkpoints":records}

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    observations,timestamps,source=acquire(out)
    training,validation,threshold,cohort=prepare(observations,timestamps)
    result={"study":"ME007_CAPACITY_MATCHED_GRU","status":"authentic_train_only",
       "source":source,"cohort":cohort,
       "seeds":list(SEEDS),"budgets":list(TRAIN_STEPS),
       "fits":[],"future_period_ever_opened_by_this_script":False,
       "tac_implementation_modified":False,"promotion":False}
    for seed in SEEDS:
        result["fits"].append(fit(seed,training,validation,threshold,out))
    (out/"results.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"ORIGINAL_STRONG_CONTROL_TRAIN_DONE",
                      "seeds":3,"checkpoints":6,"later_test_read":False}),flush=True)

def preflight():
    torch.set_num_threads(1);torch.manual_seed(17)
    net=MatchedGRU()
    assert sum(p.numel() for p in net.parameters())==9249
    x=torch.randn(2,HIST,len(FEATURES)*2)
    x[:,:,len(FEATURES):]=(torch.rand(2,HIST,len(FEATURES))>.25).float()
    z=net(x)
    assert z.shape==(2,2,len(FEATURES)) and bool(torch.isfinite(z).all())
    ((z-1).square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert net.head.weight.grad.abs().sum()>0
    print("ME007_SOFTWARE_PREFLIGHT_PASS")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/me007")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    preflight() if args.self_test else run(args.out)
