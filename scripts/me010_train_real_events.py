#!/usr/bin/env python3
"""ME010 real MetroPT-3: future-trained event mixture vs equal-capacity direct head.

Use ONLY previously opened days 20-25 as supervised calibration. DO NOT read
registered new April 6-13 TEST here. All original native TAC-1000 weights frozen.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import torch
from scripts.me004_native_real_forecast import acquire,prepare,make_model,score
from scripts.me010_event_dynamics import EventDynamics,train_loss,physical_event_metrics

MODEL_SEEDS=(101001,101002,101003)
BASE_SEEDS=(240401,240402,240403)
ARMS=("direct","mixture")
UPDATES=260
BATCH=128

def eval_historical(model,data,threshold):
    model.eval();x,y=data
    with torch.inference_mode():
        d={}
        outputs=[]
        scores=[]
        for i in range(0,len(x),256):
            outputs.append(model(x[i:i+256],details=True))
        pred=torch.cat([v["prediction"] for v in outputs])
        p=torch.cat([v["p_change"] for v in outputs]) if model.mode=="mixture" else None
        record={"prediction":pred,"p_change":p}
        metrics=score(pred,y,threshold)
        # Compute genuine event scores only with authentic outcome labels.
        metrics.update(physical_event_metrics(model,record,y,threshold))
    return metrics

def main(out,original):
    out,original=Path(out),Path(original)
    out.mkdir(parents=True,exist_ok=True)
    ledger=json.loads((original/"results.json").read_text())
    if ledger["study"]!="ME008_NATIVE_EQUAL_STEPS" or len(ledger["fits"])!=3:
        raise ValueError("requires frozen original native-1000 training source")
    base={}
    for group in ledger["fits"]:
        for item in group["checkpoints"]:
            digest=hashlib.sha256((original/item["checkpoint"]).read_bytes()).hexdigest()
            if digest!=item["sha256"]:raise ValueError("original checkpoint SHA mismatch")
            if item["steps"]==1000:base[item["seed"]]=item
    if set(base)!=set(BASE_SEEDS):raise ValueError("missing native base seeds")
    observed,timestamps,source=acquire(out)
    if source!=ledger["source"]:raise ValueError("authentic source changed")
    training,historical,threshold,cohort=prepare(observed,timestamps)
    if cohort!=ledger["cohort"]:raise ValueError("training normalizer/split changed")
    torch.set_num_threads(1)
    manifest={"study":"ME010_REAL_EVENT_CONDITIONED_DYNAMICS",
        "evidence":"HISTORICALLY_REUSED_DAYS20_25_TRAINING_ONLY",
        "source":source,"cohort":cohort,
        "base_native_optimizer_steps_added":0,"future_april6_13_unopened":True,
        "original_fits":[],"model_promoted":False,
        "experiment_reuses_historical_labels":True}
    x,y=historical
    for seed,bseed in zip(MODEL_SEEDS,BASE_SEEDS):
        for mode in ARMS:
            torch.manual_seed(seed)
            native=make_model("tac_two_timescale")
            fit=base[bseed]
            ck=torch.load(original/fit["checkpoint"],weights_only=True,map_location="cpu")
            if ck["seed"]!=bseed or ck["steps"]!=1000 or ck["model"]!="native2time":
                raise ValueError("wrong pretrained source")
            native.load_state_dict(ck["state_dict"],strict=True)
            model=EventDynamics(native,mode=mode)
            params=[p for p in model.parameters() if p.requires_grad]
            opt=torch.optim.AdamW(params,lr=.001,weight_decay=.0001)
            gen=torch.Generator().manual_seed(seed+100)
            before=eval_historical(model,historical,threshold)
            tic=time.monotonic()
            losses=[]
            for step in range(1,UPDATES+1):
                ix=torch.randint(len(x),(BATCH,),generator=gen)
                model.train()
                outinfo=model(x[ix],details=True)
                loss,diagnostic=train_loss(model,outinfo,y[ix],threshold)
                if not torch.isfinite(loss):raise ValueError("nonfinite objective")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(params,1)
                if not torch.isfinite(grad):raise ValueError("nonfinite gradient")
                opt.step()
                if step in (1,UPDATES//2,UPDATES):
                    losses.append({"step":step,"loss":float(loss.detach()),**diagnostic})
            model.eval()
            after=eval_historical(model,historical,threshold)
            ckpath=out/"checkpoints"/f"{mode}-{seed}.pt"
            ckpath.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"mode":mode,"seed":seed,"base_seed":bseed,
                        "native_source_sha256":fit["sha256"],"steps":UPDATES,
                        "state_dict":model.state_dict()},ckpath)
            row={"mode":mode,"seed":seed,"base_seed":bseed,
                 "native_sha256":fit["sha256"],
                 "trainable_parameters":sum(p.numel() for p in params),
                 "before":before,"after_historical_training":after,
                 "train_wall_seconds":round(time.monotonic()-tic,3),
                 "loss_trace":losses,"steps":UPDATES,
                 "checkpoint":str(ckpath.relative_to(out)),
                 "sha256":hashlib.sha256(ckpath.read_bytes()).hexdigest()}
            manifest["original_fits"].append(row)
            print(json.dumps({"arm":mode,"seed":seed,"historical_mae":after["macro_mae"],
                 "historical_event_brier":after.get("event_brier"),
                 "original_mae":before["macro_mae"]},sort_keys=True),flush=True)
    (out/"results.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"ME010_HISTORICAL_REAL_TRAIN_COMPLETE","checkpoint_count":6,
                      "future_test_read":False,"base_refits":0}),flush=True)

def preflight():
    torch.manual_seed(17);torch.set_num_threads(1)
    x=torch.randn(4,32,24)
    x[:,:,12:]=(torch.rand(4,32,12)>.2).float()
    y=torch.randn(4,2,12)
    threshold=torch.full((2,12),.3)
    for mode in ARMS:
        model=EventDynamics(make_model("tac_two_timescale"),mode)
        d=model(x,details=True)
        assert d["prediction"].shape==(4,2,12)
        assert torch.equal(d["prediction"],d["baseline"])
        assert all(p.requires_grad is False for p in model.native.parameters())
        corrupted=x.clone()
        corrupted[:,:,:12]=torch.where(x[:,:,12:]>0,x[:,:,:12],1e7)
        assert torch.equal(model(corrupted),d["prediction"])
        if mode=="mixture":
            assert bool((d["p_change"]>=0).all() and (d["p_change"]<=1).all())
            assert torch.equal(d["prediction"],d["p_change"]*d["mean_if_change"]+
                               (1-d["p_change"])*d["mean_if_quiet"])
        loss,stats=train_loss(model,d,y,threshold)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(model.trunk[-1].weight.grad).all()
        assert model.trunk[-1].weight.grad.abs().sum()>0
        assert all(p.grad is None for p in model.native.parameters())
    print("ME010_SOFTWARE_AND_CAUSAL_ORDER_PREFLIGHT_PASS")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--original",default="results/me008")
    parser.add_argument("--out",default="results/me010")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    preflight() if args.self_test else main(args.out,args.original)
