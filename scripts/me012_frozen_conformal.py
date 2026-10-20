#!/usr/bin/env python3
"""ME012: frozen post-hoc CQR vs conformal persistence on real ETT data.

Known-site ETTh1 calibration ONLY at [8760,11640). Prospective test
[14520,17420) on ETTh1/ETTh2, zero model fits or TEST selection. Published
conformal coverage guarantees require exchangeable scores; that assumption is
NOT established for autocorrelated/cross-site telemetry.
"""
from __future__ import annotations
import csv,hashlib,io,json,math,urllib.request
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from scripts.me011_ett_quantiles import model_for,pack,COLUMNS,SEEDS,metrics
CAL_START=8760
CAL_END=11640
TEST_START=14520
TEST_END=17420
QUANTILE_TARGET=.80
URL_ROOT="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
PINS={
 "ETTh1":{"sha256":"f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
          "md5":"8381763947c85f4be6ac456c508460d6"},
 "ETTh2":{"sha256":"a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
          "md5":"51a229a3fc13579dd939364fefe9c7ab"}
}

def load_real_site(site):
    if site not in PINS:raise ValueError("not preregistered")
    p=PINS[site];url=URL_ROOT+site+".csv";buf=bytearray()
    request=urllib.request.Request(url,headers={"User-Agent":"TAC-OSM-ME012-original/1.0"})
    with urllib.request.urlopen(request,timeout=120) as reply:
        if getattr(reply,"status",200)!=200:raise ValueError("publisher HTTP error")
        while (chunk:=reply.read(1<<20)):
            buf.extend(chunk)
            if len(buf)>10_000_000:raise ValueError("publisher source unusually big")
    actual_sha=hashlib.sha256(buf).hexdigest()
    if actual_sha!=p["sha256"] or hashlib.md5(buf).hexdigest()!=p["md5"]:
        raise ValueError("published real source checksum mismatch")
    reader=csv.DictReader(io.StringIO(buf.decode("utf-8-sig")))
    if tuple(reader.fieldnames or ())!=("date",)+COLUMNS:raise ValueError("unexpected telemetry schema")
    samples=[];prev=None
    for ix,row in enumerate(reader):
        if ix==TEST_END:break
        ts=datetime.fromisoformat(row["date"])
        if prev is not None and (ts-prev).total_seconds()!=3600:
            raise ValueError("hourly continuity broken")
        prev=ts
        vals=[float(row[f]) for f in COLUMNS]
        if not all(math.isfinite(v) for v in vals):raise ValueError("unexpected nonfinite source field")
        samples.append(vals)
    if len(samples)!=TEST_END:raise ValueError("insufficient authentic source history")
    return np.array(samples,dtype=np.float32),{
      "site":site,"source_url":url,"source_sha256":actual_sha,
      "source_md5":p["md5"],"parsed_rows":len(samples),
      "latest_parsed_timestamp":str(prev),"later_rows_parsed":False}

def finite_sample_correction(scores,nominal=QUANTILE_TARGET):
    """The standard ceil((n+1)*(1-alpha))th ascending score."""
    scores=scores.reshape(-1)
    if not bool(torch.isfinite(scores).all()):raise ValueError("nonfinite calibration score")
    n=len(scores)
    if n<10:raise ValueError("insufficient calibration")
    k=min(n,max(1,math.ceil((n+1)*nominal)))
    return torch.sort(scores).values[k-1],{"n":n,"rank_1_based":k}

def apply_corrected_interval(pred,correction):
    if tuple(pred.shape[-2:])!=(2,3) or correction.shape!=(2,):
        raise ValueError("bad conformal output shapes")
    adjusted=pred.clone()
    adjusted[:,:,0]=pred[:,:,0]-correction[None,:]
    adjusted[:,:,2]=pred[:,:,2]+correction[None,:]
    return adjusted

def model_predictions(model,x):
    model.eval();chunks=[]
    with torch.inference_mode():
        for first in range(0,len(x),256):
            chunks.append(model(x[first:first+256]))
    return torch.cat(chunks)

def coverage_and_width(q,y,stats):
    good=(y>=q[:,:,0])&(y<=q[:,:,2])
    cross=((q[:,:,0]>q[:,:,1])|(q[:,:,1]>q[:,:,2]))
    return {"coverage":float(good.float().mean()),
       "coverage_by_horizon":good.float().mean(0).tolist(),
       "mean_width_celsius":float((q[:,:,2]-q[:,:,0]).mean()*stats["train_std"][6]),
       "violations_of_median_order":int(cross.sum()),
       "point_mae_celsius":float((q[:,:,1]-y).abs().mean()*stats["train_std"][6])}

def run(original,out):
    original,out=Path(original),Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((original/"results.json").read_text())
    if manifest["study"]!="ME011_REAL_TRANSFORMER_QUANTILES" or len(manifest["fits"])!=6:
        raise ValueError("requires fixed original ME011 model ledger")
    if manifest["official_future_labels_parsed"] or manifest["etth2_source_opened"]:
        raise ValueError("original source was not sealed")
    checkpoints=[]
    for entry in manifest["fits"]:
        p=original/entry["checkpoint"]
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest!=entry["sha256"]:raise ValueError("trained source weights modified")
        ck=torch.load(p,map_location="cpu",weights_only=True)
        if ck["kind"]!=entry["model"] or ck["seed"]!=entry["seed"] or ck["steps"]!=450:
            raise ValueError("unregistered checkpoint metadata")
        model=model_for(entry["model"]);model.load_state_dict(ck["model_state"],strict=True)
        checkpoints.append((entry,model))
    assert {(e["model"],e["seed"]) for e,m in checkpoints}=={
         (kind,seed) for kind in ("native_tac","gru") for seed in SEEDS}
    stats=manifest["normalization"]
    if manifest["source"]["publisher_sha256"]!=PINS["ETTh1"]["sha256"]:
        raise ValueError("real train site source changed")
    torch.set_num_threads(1)
    # Download of the publisher file is unavoidable to verify full-file digest,
    # but only previously opened ETTh1 calibration indices enter calibration.
    etth1,source1=load_real_site("ETTh1")
    cal=pack(etth1[:CAL_END],stats,CAL_START,CAL_END)
    xx,yy=cal[:2]
    baseline_offset=[]
    for h in range(2):
        c,_=finite_sample_correction(yy[:,h].abs())
        baseline_offset.append(c)
    base=torch.stack(baseline_offset)
    offsets={};cal_metrics={}
    for entry,model in checkpoints:
        q=model_predictions(model,xx)
        residual_scores=torch.maximum(q[:,:,0]-yy,yy-q[:,:,2])
        correction=[]
        rank=[]
        for h in range(2):
            value,rank_info=finite_sample_correction(residual_scores[:,h])
            correction.append(value);rank.append(rank_info)
        correction=torch.stack(correction)
        name=f"{entry['model']}-{entry['seed']}"
        offsets[name]=correction
        cal_metrics[name]={"offset_normalized":correction.tolist(),"rank":rank,
                           "raw":coverage_and_width(q,yy,stats),
                           "corrected":coverage_and_width(apply_corrected_interval(q,correction),yy,stats)}
    # Only after calibration is frozen do we access the registered heldout ranges.
    report={"study":"ME012_FROZEN_CONFORMAL_REAL_ETT",
      "training_original_run":39388044790,
      "calibration_rows":[CAL_START,CAL_END],
      "final_test_rows":[TEST_START,TEST_END],
      "nominal_interval_coverage":QUANTILE_TARGET,
      "calibration_reference":"ETTh1 historical only, all scores before test",
      "model_calibration":cal_metrics,
      "persistence_calibration_offset_normalized":base.tolist(),
      "sites":{},"new_optimizer_updates":0,
      "has_valid_exchangeability_guarantee":False,
      "architecture_promoted":False}
    for site in ("ETTh1","ETTh2"):
        data,source=(etth1,source1) if site=="ETTh1" else load_real_site("ETTh2")
        hold=pack(data,stats,TEST_START,TEST_END)
        x,y=hold[:2]
        zero=torch.zeros_like(y)
        base_intervals=torch.stack((zero-base[None,:],zero,zero+base[None,:]),dim=-1)
        site_result={
          "official_source":source,"window_metadata":hold[2],
          "conformal_persistence":coverage_and_width(base_intervals,y,stats),
          "models":[]}
        for entry,model in checkpoints:
            name=f"{entry['model']}-{entry['seed']}"
            raw=model_predictions(model,x)
            calibrated=apply_corrected_interval(raw,offsets[name])
            before=coverage_and_width(raw,y,stats)
            after=coverage_and_width(calibrated,y,stats)
            if abs(before["point_mae_celsius"]-after["point_mae_celsius"])>1e-6:
                raise AssertionError("posthoc interval correction changed point forecast")
            site_result["models"].append({
              "model":entry["model"],"seed":entry["seed"],
              "trained_weights_sha256":entry["sha256"],
              "calibration_correction_normalized":offsets[name].tolist(),
              "raw":before,"corrected":after})
            print(json.dumps({"site":site,"model":entry["model"],"seed":entry["seed"],
                "raw_coverage":before["coverage"],"corrected_coverage":after["coverage"],
                "corrected_width_c":after["mean_width_celsius"]},sort_keys=True),flush=True)
        report["sites"][site]=site_result
    (out/"first-frozen-conformal-results.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"ME012_FIRST_FROZEN_CQR_COMPLETE",
       "site_count":len(report["sites"]),
       "checkpoints":len(checkpoints),
       "test_optimizer_updates":0,
       "not_formally_exchangeable":True},sort_keys=True),flush=True)
    return report

def preflight():
    # This is a software/math check, NOT synthetic real-world evidence.
    score=torch.tensor([-2.,0.,1.,2.,3.,4.,5.,6.,7.,8.])
    v,r=finite_sample_correction(score,.8)
    assert r["rank_1_based"]==9 and float(v)==7.
    q=torch.tensor([[[0.,1.,2.],[3.,4.,5.]]])
    adj=apply_corrected_interval(q,torch.tensor([.5,.25]))
    assert torch.equal(q[:,:,1],adj[:,:,1])
    assert torch.equal(adj[:,:,0],torch.tensor([[-.5,2.75]]))
    assert torch.equal(adj[:,:,2],torch.tensor([[2.5,5.25]]))
    print("ME012_ORDER_STATISTIC_AND_MEDIAN_PRESERVATION_PREFLIGHT_PASS")

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--original",default="results/me011")
    parser.add_argument("--out",default="results/me012")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    preflight() if args.self_test else run(args.original,args.out)
