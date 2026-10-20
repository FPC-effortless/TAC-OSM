#!/usr/bin/env python3
"""ME011F one-time original model TEST on frozen ETTh1 and cross-site ETTh2.

ZERO optimizer updates; original model checksum verification BEFORE data access.
"""
from __future__ import annotations
import csv,hashlib,io,json,math,urllib.request
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from scripts.me011_ett_quantiles import (COLUMNS,SEEDS,model_for,metrics,pinball,pack,
                                          TRAIN_END,CAL_END)
TEST_END=14520
OTHER_SOURCE_URL="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
OTHER_MD5="51a229a3fc13579dd939364fefe9c7ab"
SOURCE_ETTH1="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"

def get_site(name,expected_sha,expected_md5):
    if name not in ("ETTh1","ETTh2"):raise ValueError("unregistered source")
    url=SOURCE_ETTH1 if name=="ETTh1" else OTHER_SOURCE_URL
    raw=bytearray()
    with urllib.request.urlopen(urllib.request.Request(
            url,headers={"User-Agent":"TAC-OSM-ME011F/1.0"}),timeout=90) as stream:
        if getattr(stream,"status",200)!=200:raise ValueError("bad publisher HTTP status")
        while (part:=stream.read(2**20)):
            raw.extend(part)
            if len(raw)>10_000_000:raise ValueError("oversized source")
    md5=hashlib.md5(raw).hexdigest();sha=hashlib.sha256(raw).hexdigest()
    if expected_md5!=md5 or expected_sha is not None and expected_sha!=sha:
        raise ValueError("official source digest failed")
    reader=csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if tuple(reader.fieldnames or ())!=("date",)+COLUMNS:raise ValueError("source schema changed")
    rows=[];previous=None;first=None;last=None
    for idx,row in enumerate(reader):
        if idx>=TEST_END:break
        dt=datetime.fromisoformat(row["date"])
        if previous is not None and (dt-previous).total_seconds()!=3600:
            raise ValueError("source timestamps not hourly")
        previous=dt
        if first is None:first=dt
        last=dt
        values=[float(row[k]) for k in COLUMNS]
        if not all(math.isfinite(k) for k in values):raise ValueError("source nonfinite real value")
        rows.append(values)
    if len(rows)!=TEST_END:raise ValueError("official real source does not contain registered period")
    return np.asarray(rows,dtype=np.float32),{
      "official_url":url,"official_md5":md5,"official_sha256":sha,
      "parsed_rows":len(rows),"physical_site":name,
      "first_timestamp":str(first),"last_scored_source_timestamp":str(last),
      "no_rows_after_test_cutoff_parsed":True}

def benchmark(pred,target,stats):
    return metrics(pred,target,stats)

def baseline_scores(x,y,stats):
    # No future covariates. Uses just known past OT readings, in ETTh1 TRAIN scale.
    baseline=torch.zeros_like(y)
    # 24-hour rolling-mean of observed past OT, compared to latest.
    # Target signed delta is future oil temp minus last oil temp.
    roll=x[:,-24:,6].mean(-1)-x[:,-1,6]
    roll=roll[:,None].expand(-1,2)
    # Degenerate intervals are *not* probabilistic calibration baselines.
    def deterministic(z):
        return z.unsqueeze(-1).repeat(1,1,3)
    return {"persistence":{"median_only":benchmark(deterministic(baseline),y,stats),
                           "probabilistic_reference":False},
            "trailing_24h_mean":{"median_only":benchmark(deterministic(roll),y,stats),
                                   "probabilistic_reference":False}}

def evaluate(original,out):
    original,out=Path(original),Path(out);out.mkdir(parents=True,exist_ok=True)
    ledger=json.loads((original/"results.json").read_text())
    if ledger["study"]!="ME011_REAL_TRANSFORMER_QUANTILES" or len(ledger["fits"])!=6:
        raise ValueError("wrong original training manifest")
    if ledger["official_future_labels_parsed"] or ledger["etth2_source_opened"]:
        raise ValueError("future site had been opened by original trainer")
    if ledger["source"]["rows_parsed"]!=CAL_END:raise ValueError("wrong training source cutoff")
    models=[]
    for entry in ledger["fits"]:
        p=original/entry["checkpoint"]
        if hashlib.sha256(p.read_bytes()).hexdigest()!=entry["sha256"]:
            raise ValueError("checkpoint original SHA mismatch")
        ck=torch.load(p,map_location="cpu",weights_only=True)
        if ck["kind"]!=entry["model"] or ck["seed"]!=entry["seed"] or ck["steps"]!=450:
            raise ValueError("original identity mismatch")
        m=model_for(entry["model"])
        m.load_state_dict(ck["model_state"],strict=True);m.eval()
        models.append((entry,m))
    assert {(e["model"],e["seed"]) for e,m in models} =={
        (arm,seed) for arm in ("native_tac","gru") for seed in SEEDS}
    stats=ledger["normalization"]
    results={"study":"ME011F_REAL_TRANSFORMER_FROZEN_CROSS_SITE",
       "source_train_checksum":ledger["source"]["publisher_sha256"],
       "original_checkpoint_count":len(models),
       "original_model_fit_evidence_only":True,
       "sites":{},"test_time_optimizer_updates":0,
       "no_source_after_registered_end_parsed":True,
       "architecture_promoted":False}
    torch.set_num_threads(1)
    for site in ("ETTh1","ETTh2"):
        values,source=get_site(site,
            ledger["source"]["publisher_sha256"] if site=="ETTh1" else None,
            ledger["source"]["publisher_md5"] if site=="ETTh1" else OTHER_MD5)
        hold=pack(values,stats,CAL_END,TEST_END)
        x,y,win=hold
        baselines=baseline_scores(x,y,stats)
        site_result={"source":source,"window_selection":win,"baselines":baselines,"models":[]}
        for entry,net in models:
            forecasts=[]
            with torch.inference_mode():
                for start in range(0,len(x),256):
                    forecasts.append(net(x[start:start+256]))
            estimate=torch.cat(forecasts)
            result=benchmark(estimate,y,stats)
            row={"model":entry["model"],"seed":entry["seed"],
                 "source_checkpoint_sha256":entry["sha256"],
                 "metrics":result,
                 "original_cpu_training_s":entry["training_cpu_seconds"],
                 "parameter_count":entry["trainable_params"]}
            site_result["models"].append(row)
            print(json.dumps({"site":site,"model":entry["model"],"seed":entry["seed"],
                 "median_MAE_degreesC":result["median_mae_degrees_c"],
                 "coverage_80":result["interval_80_coverage"],
                 "quantile_pinball":result["pinball_train_norm"]},sort_keys=True),flush=True)
        results["sites"][site]=site_result
    (out/"first-frozen-cross-site-results.json").write_text(
        json.dumps(results,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"ME011F_FROZEN_REAL_CROSS_SITE_DONE",
          "sites":list(results["sites"]),"new_optimizer_steps":0,
          "total_forecasts":sum(s["window_selection"]["windows"] for s in results["sites"].values())}),flush=True)
    return results

if __name__=="__main__":
    evaluate("results/me011","results/me011f")
