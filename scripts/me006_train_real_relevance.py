#!/usr/bin/env python3
"""ME006 four-fit authentic MetroPT relevance ablation; reused exploratory split.
Never opens original future-week test, no policy/action claims.
"""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from scripts.me004_native_real_forecast import acquire,prepare
from scripts.me005_unified_native_tac import load_me004_backbone
from scripts.me006_causal_relevance import CausalRelevanceAdapter,future_gain

SEEDS=(606001,606002)
BACKBONE_SEEDS=(240401,240402)
ARMS=("direct","associative")
STEPS=160
BATCH=64

def evaluate(model,cohort):
    model.eval();x,y=cohort
    absolute=[];base_errors=[];gains=[]
    with torch.no_grad():
        for i in range(0,len(x),128):
            xx,yy=x[i:i+128],y[i:i+128]
            d=model(xx,return_details=True)
            absolute.append((d["forecast"]-yy).abs())
            base_errors.append((d["baseline"]-yy).abs())
            gains.append(future_gain(d["forecast"],yy,d["baseline"])*len(xx))
    return {"macro_mae":float(torch.cat(absolute).mean()),
            "base_mae":float(torch.cat(base_errors).mean()),
            "future_gaussian_gain":float(torch.stack(gains).sum()/len(x))}

def run(out,original,updates=STEPS):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    original=Path(original)
    ledger=json.loads((original/"results.json").read_text())
    if ledger["study"]!="ME004" or len(ledger["fits"])!=6:
        raise ValueError("ME004 original train record required")
    for f in ledger["fits"]:
        path=original/f["checkpoint"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=f["checkpoint_sha256"]:
            raise ValueError("unverified pretrained checkpoint")
    values,timestamps,source=acquire(out)
    if source!=ledger["source"]:raise ValueError("official original source differs")
    train,val,threshold,cohort=prepare(values,timestamps)
    if cohort!=ledger["cohort"]:raise ValueError("original source/split changed")
    torch.set_num_threads(1)
    result={"study":"ME006_REAL_CAUSAL_RELEVANCE",
            "evidence":"ORIGINAL_REAL_TRAIN_ERA_REUSED_EXPLORATORY",
            "future_reserved_test_read":False,"original_pretrained_refits":0,
            "source":source,"cohort":cohort,"fits":[],"promoted":False}
    for seed,bseed in zip(SEEDS,BACKBONE_SEEDS):
        for mode in ARMS:
            torch.manual_seed(seed)
            base=load_me004_backbone(original/"checkpoints"/f"tac_two_timescale-{bseed}.pt",
                                    original/"results.json",bseed)
            model=CausalRelevanceAdapter(base,mode=mode)
            before=evaluate(model,val)
            trainable=[p for p in model.parameters() if p.requires_grad]
            opt=torch.optim.AdamW(trainable,lr=.001,weight_decay=1e-4)
            gen=torch.Generator().manual_seed(seed+100)
            x,y=train;curve=[];tic=time.monotonic()
            for step in range(updates):
                ix=torch.randint(len(x),(BATCH,),generator=gen)
                pred=model(x[ix])
                loss=F.smooth_l1_loss(pred,y[ix],beta=.2)
                if not torch.isfinite(loss):raise ArithmeticError("nonfinite loss")
                opt.zero_grad(set_to_none=True);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(trainable,1)
                if not torch.isfinite(grad):raise ArithmeticError("nonfinite gradient")
                opt.step()
                if step+1 in (1,updates//2,updates):
                    curve.append({"step":step+1,"loss":float(loss.detach())})
            after=evaluate(model,val)
            path=out/"checkpoints"/f"{mode}-{seed}.pt"
            path.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"mode":mode,"seed":seed,"base_seed":bseed,"steps":updates,
                        "state_dict":model.state_dict()},path)
            fit={"mode":mode,"seed":seed,"base_seed":bseed,
                 "trainable_parameters":sum(p.numel() for p in trainable),
                 "before":before,"after":after,"loss":curve,
                 "cpu_seconds":round(time.monotonic()-tic,2),
                 "checkpoint":str(path.relative_to(out)),
                 "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            result["fits"].append(fit)
            print(json.dumps(fit,sort_keys=True),flush=True)
    (out/"results.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"EXPLORATORY_COMPLETE","fits":len(result["fits"]),
                      "refits_of_pretrained":0,"future_test_read":False}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="results/me006")
    p.add_argument("--original",default="results/me004");p.add_argument("--updates",type=int,default=STEPS)
    a=p.parse_args();run(a.out,a.original,a.updates)
