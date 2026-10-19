#!/usr/bin/env python3
"""SEQ009-S: train and evaluate REAL SCANIA v3 next-readout dynamics baselines.

Train one sensor-wise affine autoregression using only publisher TRAIN truck
readouts. Measure it against persistence on the final available transition of
each publisher VALIDATION truck. No TEST files opened, no simulated events,
no repair-risk/causal utility claims. This is an auxiliary real-data task.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import time
import urllib.request
import zipfile

import numpy as np

ARCHIVE_URL = "https://api.researchdata.se/dataset/2024-34/3/file/zip"
FILES = ("train_operational_readouts.csv", "validation_operational_readouts.csv")
ARCHIVE_LIMIT = 2_600_000_000
FILE_LIMIT = 2_000_000_000

def signed_log(v):
    return np.sign(v) * np.log1p(np.abs(v))

def download_archive(path):
    h=hashlib.sha256()
    count=0
    req=urllib.request.Request(ARCHIVE_URL,
          headers={"User-Agent":"TAC-OSM-SEQ009-real-sequence/1.0",
                   "Accept":"application/zip"})
    with urllib.request.urlopen(req,timeout=120) as response, path.open("wb") as out:
        if response.status != 200:
            raise ValueError("Official ZIP endpoint rejected")
        while True:
            data=response.read(2*1024*1024)
            if not data: break
            count += len(data)
            if count > ARCHIVE_LIMIT: raise ValueError("Unexpected archive size")
            h.update(data)
            out.write(data)
    return {"url":ARCHIVE_URL,"size":count,"sha256":h.hexdigest()}

def extract_allowed(archive_path, directory):
    out={}
    with zipfile.ZipFile(archive_path) as z:
        matches={}
        for e in z.infolist():
            leaf=Path(e.filename).name
            if leaf not in FILES: continue
            if leaf in matches: raise ValueError("Ambiguous archive members " + leaf)
            if e.is_dir() or e.file_size <= 0 or e.file_size > FILE_LIMIT:
                raise ValueError("Invalid source size " + leaf)
            matches[leaf]=e
        if set(matches)!=set(FILES):
            raise ValueError(f"Missing approved operational file(s): {set(FILES)-set(matches)}")
        for name,e in matches.items():
            h=hashlib.sha256()
            size=0
            dest=directory/name
            with z.open(e) as inp,dest.open("wb") as dst:
                while True:
                    data=inp.read(2*1024*1024)
                    if not data:break
                    size+=len(data)
                    if size>FILE_LIMIT:raise ValueError("Decompressed size exceeded")
                    h.update(data)
                    dst.write(data)
            if size!=e.file_size: raise ValueError("Truncated member")
            out[name]={"zip_member":e.filename,"bytes":size,"sha256":h.hexdigest()}
    return out

def parsed_record(row, indices):
    def parse(value):
        value=value.strip()
        if value in ("", "NA", "NaN", "nan", "null", "None"):
            return float("nan")
        return float(value)
    return signed_log(np.fromiter((parse(row[i]) for i in indices),
                      dtype=np.float64,count=len(indices)))

def fit_and_evaluate(train_path,val_path):
    # File contract discovered from authentic TRAIN header, no label-driven feature choice.
    with train_path.open(newline="", encoding="utf-8-sig") as stream:
        header=next(csv.reader(stream))
    if len(set(header)) != len(header):
        raise ValueError("Duplicate headers")
    if "vehicle_id" not in header or "time_step" not in header:
        raise ValueError(f"Required fields missing: {header[:8]}")
    ix_id=header.index("vehicle_id")
    ix_time=header.index("time_step")
    sensors=[i for i,col in enumerate(header) if col not in ("vehicle_id","time_step")]
    names=[header[i] for i in sensors]
    d=len(sensors)
    if d < 20:
        raise ValueError("Unexpectedly few sensor columns")
    n=np.zeros(d,dtype=np.float64)
    sx=np.zeros(d);sy=np.zeros(d);sxx=np.zeros(d);sxy=np.zeros(d)
    change=np.zeros(d)
    train_ids=set()
    prev={}
    rows_train,pairs_train=0,0
    with train_path.open(newline="",encoding="utf-8-sig") as stream:
        reader=csv.reader(stream)
        assert next(reader)==header
        for row in reader:
            if len(row)!=len(header):raise ValueError("Bad train row length")
            vid=row[ix_id]
            if not vid:raise ValueError("Blank train vehicle")
            try:t=float(row[ix_time])
            except ValueError as e:raise ValueError("Bad train time") from e
            if not math.isfinite(t):raise ValueError("Nonfinite time")
            x=parsed_record(row,sensors)
            train_ids.add(vid)
            rows_train+=1
            old=prev.get(vid)
            if old is not None:
                old_time,p=old
                if t<=old_time:raise ValueError("Non-increasing train time within vehicle")
                ok=np.isfinite(p)&np.isfinite(x)
                n[ok]+=1
                sx[ok]+=p[ok]; sy[ok]+=x[ok]
                sxx[ok]+=p[ok]*p[ok]
                sxy[ok]+=p[ok]*x[ok]
                change[ok]+=np.abs(x[ok]-p[ok])
                pairs_train+=1
            prev[vid]=(t,x)
    if (n<100).any():raise ValueError("Unobserved/rare sensor prevents fair comparison")
    mx=sx/n; my=sy/n
    ridge=1e-3*n
    slope=(sxy-sx*sy/n)/(sxx-sx*sx/n+ridge)
    slope=np.clip(slope,-1.0,2.0)
    intercept=my-slope*mx
    # Scale based on TRAIN next-readout change; cap tiny scales.
    scale=np.maximum(change/n,0.05)
    errors={"persistence":np.zeros(d),"affine_autoreg":np.zeros(d)}
    normalized={"persistence":np.zeros(d),"affine_autoreg":np.zeros(d)}
    counts=np.zeros(d)
    rows_val=0
    prev={}
    last_pairs={}
    val_ids=set()
    with val_path.open(newline="",encoding="utf-8-sig") as stream:
        reader=csv.reader(stream)
        val_header=next(reader)
        if val_header!=header:
            raise ValueError("Validation header differs from training; do not remap silently")
        for row in reader:
            if len(row)!=len(header):raise ValueError("Bad val row length")
            vid=row[ix_id]
            if not vid:raise ValueError("Blank val vehicle")
            t=float(row[ix_time])
            if not math.isfinite(t):raise ValueError("Nonfinite val time")
            x=parsed_record(row,sensors)
            rows_val+=1
            val_ids.add(vid)
            old=prev.get(vid)
            if old is not None:
                old_time,p=old
                if t<=old_time:raise ValueError("Non-increasing validation time within vehicle")
                last_pairs[vid]=(p,x)
            prev[vid]=(t,x)
    if train_ids & val_ids:
        raise ValueError("Vehicle identity leakage across source partitions")
    for vid,(p,y) in last_pairs.items():
        ok=np.isfinite(p)&np.isfinite(y)
        pred0=p
        pred1=slope*p+intercept
        err0=np.abs(pred0[ok]-y[ok])
        err1=np.abs(pred1[ok]-y[ok])
        errors["persistence"][ok]+=err0
        errors["affine_autoreg"][ok]+=err1
        normalized["persistence"][ok]+=err0/scale[ok]
        normalized["affine_autoreg"][ok]+=err1/scale[ok]
        counts[ok]+=1
    if counts.min()<=0:raise ValueError("No observable validation targets for some sensors")
    cells=float(counts.sum())
    def summarize(key):
        # Mean over features (macro) avoids domination by histogram bin counts.
        per_sensor=errors[key]/counts
        per_sensor_norm=normalized[key]/counts
        return {
            "signed_log_mae_micro":float(errors[key].sum()/cells),
            "signed_log_mae_macro":float(per_sensor.mean()),
            "train_change_scaled_mae_macro":float(per_sensor_norm.mean()),
            "per_feature_observed_targets_min":int(counts.min()),
        }
    return {
        "source_type":"real_operational_sensors",
        "train_vehicle_count":len(train_ids),"validation_vehicle_count":len(val_ids),
        "train_readouts":rows_train,"validation_readouts":rows_val,
        "train_next_readout_pairs":pairs_train,
        "validation_last_readout_pairs":len(last_pairs),
        "feature_columns":names,"feature_count":d,
        "calibration_target":"next genuine readout sensor vector, not repair risk",
        "timestamp_used_as_feature":False,
        "training_target_source":"real consecutive training readouts",
        "test_set_opened":False,"test_labels_opened":False,
        "metrics":{"persistence":summarize("persistence"),
                   "affine_autoreg":summarize("affine_autoreg")},
        "method":"signed-log sensor transform; sensor-wise diagonal ridge AR(1); last validation pair per real vehicle",
        "limitations":["VALIDATION only, not final TEST","No real repair-risk labels trained",
           "Irregular next-readout gap not known as input","Sensor-column-wise affine autoregression is not TAC-OSM",
           "Only vehicles with two or more readouts have targets",
           "Missing predictor/target values are excluded symmetrically"],
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/seq009-real-sensors")
    args=p.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    archive=root/"official-v3.zip"
    try:
        manifest={"doi":"10.5878/bnh5-ka77","version":3,
                  "test_entries_opened":False,"archive":download_archive(archive)}
        manifest["files"]=extract_allowed(archive,root)
    finally:
        archive.unlink(missing_ok=True)
    (root/"source.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    result=fit_and_evaluate(root/FILES[0],root/FILES[1])
    (root/"results.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":"REAL_NEXT_READOUT_VALIDATION_COMPLETE",
        "train_vehicles":result["train_vehicle_count"],
        "validation_vehicles":result["validation_vehicle_count"],
        "metrics":result["metrics"],
        "no_test":not result["test_set_opened"],
    },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
