#!/usr/bin/env python3
"""ME006 hash-verified research-only future sensor inference. No action authorization."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scripts.me004_native_real_forecast import FEATURES,HIST,HORIZONS,make_model
from scripts.me006_causal_relevance import CausalRelevanceAdapter

def predict(base_report,base_checkpoint,adapter_report,adapter_checkpoint,history):
    old=json.loads(Path(base_report).read_text())
    new=json.loads(Path(adapter_report).read_text())
    if old["study"]!="ME004" or new["study"]!="ME006_REAL_CAUSAL_RELEVANCE":
        raise ValueError("expected genuine original training score manifests")
    bd=hashlib.sha256(Path(base_checkpoint).read_bytes()).hexdigest()
    ad=hashlib.sha256(Path(adapter_checkpoint).read_bytes()).hexdigest()
    base_ck=torch.load(base_checkpoint,map_location="cpu",weights_only=True)
    adapt_ck=torch.load(adapter_checkpoint,map_location="cpu",weights_only=True)
    if not any(f["model"]=="tac_two_timescale" and f["seed"]==base_ck["seed"] and
               f["checkpoint_sha256"]==bd for f in old["fits"]):
        raise ValueError("unverified original base checkpoint")
    if not any(f["mode"]==adapt_ck["mode"] and f["seed"]==adapt_ck["seed"] and
               f["base_seed"]==base_ck["seed"] and f["checkpoint_sha256"]==ad for f in new["fits"]):
        raise ValueError("unverified original relevance adapter")
    if not isinstance(history,list) or len(history)!=HIST:
        raise ValueError("exactly 32 chronological real sensor observations required")
    mu=np.array(old["cohort"]["mean_train_only"],dtype=np.float32)
    std=np.array(old["cohort"]["std_train_only"],dtype=np.float32)
    if not np.isfinite(mu).all() or not np.isfinite(std).all() or not (std>0).all():
        raise ValueError("invalid training-only normalizer")
    observations=np.asarray([[float(row.get(f,float("nan"))) for f in FEATURES]
                             for row in history],dtype=np.float32)
    mask=np.isfinite(observations)
    if not mask[-1].all():raise ValueError("last observation must have all sensor values")
    scaled=np.clip(np.nan_to_num((observations-mu)/std,nan=0,posinf=0,neginf=0),-10,10)
    x=np.concatenate([scaled,mask.astype(np.float32)],axis=1)[None]
    base=make_model("tac_two_timescale");base.load_state_dict(base_ck["state_dict"],strict=True)
    model=CausalRelevanceAdapter(base,mode=adapt_ck["mode"])
    model.load_state_dict(adapt_ck["state_dict"],strict=True);model.eval()
    with torch.inference_mode():delta=model(torch.from_numpy(x))[0].numpy()
    future=observations[-1][None]+delta*std[None]
    if not np.isfinite(future).all():raise ValueError("nonfinite forecast")
    return {"mode":adapt_ck["mode"],"base_sha256":bd,"adapter_sha256":ad,
            "horizon_units":"future observed time steps, not minutes",
            "predictions":{str(h):{f:float(v) for f,v in zip(FEATURES,future[i])}
                           for i,h in enumerate(HORIZONS)},
            "trained_on_genuine_operational_data":True,
            "commercially_qualified":False,"authorized_for_real_actions":False}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--base-report",required=True);p.add_argument("--base",required=True)
    p.add_argument("--adapter-report",required=True);p.add_argument("--adapter",required=True)
    p.add_argument("--observations",required=True)
    a=p.parse_args()
    print(json.dumps(predict(a.base_report,a.base,a.adapter_report,a.adapter,
                       json.loads(Path(a.observations).read_text())),sort_keys=True,indent=2))
