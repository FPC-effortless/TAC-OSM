#!/usr/bin/env python3
"""ME009: calibrate a small reliability gate on HISTORICAL real calibration only.
Frozen native-1000 learned weights are never retrained. New future TEST stays unopened.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import torch
from scripts.me004_native_real_forecast import acquire,prepare,make_model,score
from scripts.me009_reliability import ReliabilityGate,loss_on_authentic_outcomes

SEEDS=(909001,909002,909003)
BASE_SEEDS=(240401,240402,240403)
STEPS=220
BATCH=64

def anomaly_metrics(pred,target,threshold):
    quiet=target.abs()<=threshold.unsqueeze(0)
    flagged=pred.abs()>threshold.unsqueeze(0)
    if not quiet.any():return {"quiet_count":0,"false_movement_rate":None}
    return {"quiet_count":int(quiet.sum()),
            "false_movement_rate":float((quiet&flagged).sum()/quiet.sum()),
            "quiet_abs_predicted_delta":float(pred.abs()[quiet].mean())}

def evaluate(model,history,threshold):
    model.eval();x,y=history
    with torch.inference_mode():
        p=torch.cat([model(x[i:i+256]) for i in range(0,len(x),256)])
        o=torch.cat([model.native(x[i:i+256]) for i in range(0,len(x),256)])
    result=score(p,y,threshold)
    result["original_native_macro_mae"]=score(o,y,threshold)["macro_mae"]
    result["anomaly"]=anomaly_metrics(p,y,threshold)
    return result

def fit(seed,base_seed,base_record,train_data,threshold,original,out):
    torch.manual_seed(seed);torch.set_num_threads(1)
    native=make_model("tac_two_timescale")
    ck_path=original/base_record["checkpoint"]
    digest=hashlib.sha256(ck_path.read_bytes()).hexdigest()
    if digest!=base_record["sha256"]:raise ValueError("base checksum mismatch")
    ck=torch.load(ck_path,map_location="cpu",weights_only=True)
    if ck["seed"]!=base_seed or ck["model"]!="native2time" or ck["steps"]!=1000:
        raise ValueError("not the registered original native checkpoint")
    native.load_state_dict(ck["state_dict"],strict=True)
    model=ReliabilityGate(native)
    assert sum(p.numel() for p in model.net.parameters())==161
    opt=torch.optim.AdamW(model.net.parameters(),lr=.002,weight_decay=0)
    x,y=train_data
    gen=torch.Generator().manual_seed(seed+77)
    history=[];start=time.monotonic()
    before=evaluate(model,train_data,threshold)
    for step in range(1,STEPS+1):
        ids=torch.randint(len(x),(BATCH,),generator=gen)
        model.train()
        pred=model(x[ids])
        loss=loss_on_authentic_outcomes(pred,y[ids],threshold)
        if not torch.isfinite(loss):raise ValueError("loss not finite")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g=torch.nn.utils.clip_grad_norm_(model.net.parameters(),1)
        if not torch.isfinite(g):raise ValueError("grad not finite")
        opt.step()
        if step in (1,STEPS//2,STEPS):
            history.append({"update":step,"loss":float(loss.detach())})
    after=evaluate(model,train_data,threshold)
    path=out/"checkpoints"/f"reliability-{seed}.pt"
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"kind":"reliability-gate","seed":seed,"base_seed":base_seed,
                "steps":STEPS,"original_sha256":digest,"state_dict":model.state_dict()},path)
    return {"seed":seed,"base_seed":base_seed,"original_sha256":digest,
            "trainable_gate_params":161,"calibration_updates":STEPS,
            "historical_calibration_before":before,"historical_calibration_after":after,
            "loss_trace":history,"train_wall_seconds":round(time.monotonic()-start,3),
            "checkpoint":str(path.relative_to(out)),
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

def run(original,out):
    original=Path(original);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ledger=json.loads((original/"results.json").read_text())
    if ledger["study"]!="ME008_NATIVE_EQUAL_STEPS":
        raise ValueError("exact original ME008 source needed")
    registered={}
    for source_record in ledger["fits"]:
        for candidate in source_record["checkpoints"]:
            path=original/candidate["checkpoint"]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=candidate["sha256"]:
                raise ValueError("original checkpoint digest mismatch")
            if candidate["steps"]==1000:
                registered[candidate["seed"]]=candidate
    assert set(registered)==set(BASE_SEEDS)
    vals,ts,source=acquire(out)
    if source!=ledger["source"]:raise ValueError("official TRAIN source changed")
    train,calibration,threshold,cohort=prepare(vals,ts)
    if cohort!=ledger["cohort"]:raise ValueError("original cohort changed")
    record={"study":"ME009_NATIVE_RELIABILITY_CALIBRATION",
            "evidence":"REUSED_HISTORICAL_CALIBRATION_DAYS_20_25",
            "source":source,"cohort":cohort,"calibration_samples":len(calibration[0]),
            "fits":[],"future_test_read":False,"base_weights_refit":False,
            "architecture_promoted":False}
    for seed,bseed in zip(SEEDS,BASE_SEEDS):
        z=fit(seed,bseed,registered[bseed],calibration,threshold,original,out)
        record["fits"].append(z)
        print(json.dumps({"seed":seed,"mae":z["historical_calibration_after"]["macro_mae"],
                          "native":z["historical_calibration_after"]["original_native_macro_mae"],
                          "quiet_false_move_rate":z["historical_calibration_after"]["anomaly"]["false_movement_rate"]}),flush=True)
    (out/"results.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"HISTORICAL_CALIBRATION_DONE","fits":3,
                      "future_test_read":False,"new_base_fits":0}),flush=True)

def preflight():
    torch.manual_seed(29);torch.set_num_threads(1)
    native=make_model("tac_two_timescale")
    m=ReliabilityGate(native)
    assert sum(p.numel() for p in m.net.parameters())==161
    x=torch.randn(2,32,24)
    x[:,:,12:]=(torch.rand(2,32,12)>.2).float()
    y=m(x,details=True)
    assert y["prediction"].shape==(2,2,12)
    assert torch.all(y["alpha"]>=0) and torch.all(y["alpha"]<=1)
    assert torch.allclose(y["prediction"],y["original"]*.5,atol=1e-7)
    corrupted=x.clone();corrupted[:,:,:12]+=1000*(1-x[:,:,12:])
    assert torch.allclose(m(corrupted),y["prediction"],atol=1e-7)
    target=torch.randn_like(y["prediction"])
    loss=loss_on_authentic_outcomes(y["prediction"],target,torch.ones(2,12)*.4)
    loss.backward()
    assert torch.isfinite(m.net[-1].weight.grad).all()
    print("ME009_CAUSAL_GATE_SOFTWARE_PREFLIGHT_PASS")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",default="results/me008")
    p.add_argument("--out",default="results/me009")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    preflight() if a.self_test else run(a.original,a.out)
