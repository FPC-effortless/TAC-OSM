#!/usr/bin/env python3
"""MODEL02 tiny original-source classical risk lookup; EXPOSED DEV DIAGNOSTIC ONLY.

Reads 46 already-inspected publisher days. Current-day SMART 5 and SMART 197
binary flags form a 9-cell Beta-smoothed risk lookup. Source time availability
and label adjudication are NOT independently verified. This FIT study does
NOT meet qualified-model training gates; no neural TAC model is fitted.
"""
from __future__ import annotations
from collections import defaultdict
import csv
from datetime import timedelta
import hashlib
import io
import json
import math
from pathlib import Path,PurePosixPath
import sys
import zipfile

import backblaze_2013_cohort_feasibility as period
import backblaze_2013_source_probe as pub
import backblaze_rolling_admission_feasibility as rolling

ORIGINAL_SHA=period.SOURCE_SHA
FEATURES=("smart_5_raw","smart_197_raw")
SMOOTHING=.5

class EvidenceError(ValueError):
    pass

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)

def bin_value(raw):
    if raw is None or raw == "":
        return "MISSING"
    try:
        x=float(raw)
    except ValueError as exc:
        raise EvidenceError("invalid numeric SMART source observation") from exc
    if not math.isfinite(x) or x<0:
        raise EvidenceError("invalid source SMART range")
    return "POSITIVE" if x>0 else "ZERO"

def bucket(values):
    return "|".join(values)

def records_from_original(data):
    if pub.sha(data)!=ORIGINAL_SHA:
        raise EvidenceError("publisher-original ZIP must match frozen source SHA256")
    wanted={d.isoformat()+".csv":d for d in period.days()}
    timelines=defaultdict(dict)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        members={}
        for meta in z.infolist():
            if meta.is_dir():continue
            path=PurePosixPath(meta.filename)
            if ".." in path.parts or meta.filename.startswith("/"):
                raise EvidenceError("unsafe ZIP member")
            if path.name in wanted:
                if path.name in members or meta.file_size>pub.MAX_MEMBER:
                    raise EvidenceError("duplicate/oversize selected daily CSV")
                members[path.name]=meta
        if set(members)!=set(wanted):
            raise EvidenceError("publisher daily exposure not complete")
        for day in period.days():
            name=day.isoformat()+".csv"
            with z.open(members[name]) as zf:
                raw=zf.read(pub.MAX_MEMBER+1)
            if len(raw)>pub.MAX_MEMBER:
                raise EvidenceError("daily decompression budget exceeded")
            reader=csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
            if not set(("date","serial_number","model","failure")+FEATURES).issubset(reader.fieldnames or []):
                raise EvidenceError("required original publisher SMART fields missing")
            seen=set()
            for row in reader:
                key=row["serial_number"],row["model"]
                if not all(key) or key in seen or row["date"]!=day.isoformat() or row["failure"] not in ("0","1"):
                    raise EvidenceError("duplicate/invalid original source drive-day")
                seen.add(key)
                if len(seen)>pub.MAX_DAILY_ROWS:raise EvidenceError("daily source count budget")
                b=bucket([bin_value(row[f]) for f in FEATURES])
                timelines[key][day]={"failed":row["failure"]=="1","cell":b}
    for timeline in timelines.values():
        fail_days=[d for d,v in timeline.items() if v["failed"]]
        if len(fail_days)>1 or (fail_days and any(d>fail_days[0] for d in timeline)):
            raise EvidenceError("source physical drive failure/resurrection ambiguity")
    return timelines

def case_rows(timelines, start, end):
    counts=defaultdict(lambda:[0,0])  # [fail-window, complete-window]
    positive_drives=set()
    eligible=0;censored=0
    for key,series in timelines.items():
        observed={date:row["failed"] for date,row in series.items()}
        for d in sorted(series):
            if not start<=d<end or not rolling.admitted_at(observed,d):
                continue
            eligible+=1
            target=rolling.label_future(observed,d,end)
            if target=="CENSORED":
                censored+=1;continue
            is_positive=int(target=="POSITIVE")
            counts[series[d]["cell"]][0]+=is_positive
            counts[series[d]["cell"]][1]+=1
            if is_positive:positive_drives.add(key)
    return counts,{"eligible_windows":eligible,"right_censored_windows":censored,
                   "matured_windows":eligible-censored,
                   "distinct_drives_with_positive_windows":len(positive_drives)}

def fit(fit_counts):
    all_cases=sum(n for pos,n in fit_counts.values())
    positives=sum(pos for pos,n in fit_counts.values())
    if not 0<positives<all_cases:
        raise EvidenceError("both FIT classes required")
    prior=(positives+SMOOTHING)/(all_cases+2*SMOOTHING)
    cells={}
    for a in ("MISSING","ZERO","POSITIVE"):
        for b in ("MISSING","ZERO","POSITIVE"):
            key=bucket((a,b))
            pos,n=fit_counts.get(key,(0,0))
            cells[key]={"n":n,"pos":pos,
                        "p":(pos+SMOOTHING)/(n+2*SMOOTHING) if n>0 else prior}
    return {"schema":"model02-classical-smart-binary-original-v1",
            "model_type":"FIT_BINNED_BETA_SMOOTHED_REFERENCE_NOT_TAC",
            "features":list(FEATURES),"prior_fit":prior,
            "fit_windows":all_cases,"fit_positive_windows":positives,
            "cells":cells,"source_zip_sha256":ORIGINAL_SHA,
            "fit_period":[str(period.FIRST),str(period.FIT_END-timedelta(days=1))],
            "horizon_days":period.HORIZON_DAYS,
            "source_visibility_status":"UNVERIFIED_ORIGINAL_OPERATIONAL_AVAILABLE_AT",
            "scientific_qualification":"EXPLORATORY_POST_EXPOSURE_NOT_QUALIFIED"}

def risk(model, cell):
    if model.get("schema")!="model02-classical-smart-binary-original-v1":
        raise EvidenceError("invalid checkpoint schema")
    if cell not in model["cells"]:
        raise EvidenceError("unregistered source feature cell")
    p=model["cells"][cell]["p"]
    if not 0<p<1 or not math.isfinite(p):raise EvidenceError("invalid risk value")
    return p

def score(model, counts, n_unique):
    n=0;pos=0;sum_brier=sum_ll=base_brier=base_ll=0.
    prior=model["prior_fit"]
    for cell,(events,amount) in counts.items():
        p=risk(model,cell)
        n+=amount;pos+=events
        sum_brier+=events*(1-p)**2+(amount-events)*p*p
        sum_ll+=-(events*math.log(p)+(amount-events)*math.log1p(-p))
        base_brier+=events*(1-prior)**2+(amount-events)*prior*prior
        base_ll+=-(events*math.log(prior)+(amount-events)*math.log1p(-prior))
    if n==0:raise EvidenceError("no mature target windows")
    return {"matured_windows":n,"positive_overlapping_windows":pos,
            "distinct_positive_drives":n_unique,"observed_positive_fraction":pos/n,
            "brier":sum_brier/n,"log_loss":sum_ll/n,
            "fit_only_prior_baseline_brier":base_brier/n,
            "fit_only_prior_baseline_log_loss":base_ll/n}

def run(source_bytes):
    timelines=records_from_original(source_bytes)
    counts_fit,denom_fit=case_rows(timelines,period.FIRST,period.FIT_END)
    counts_dev,denom_dev=case_rows(timelines,period.FIT_END,period.EXPOSED_END)
    if (denom_fit["distinct_drives_with_positive_windows"]!=35 or
        denom_dev["distinct_drives_with_positive_windows"]!=65):
        raise EvidenceError("original source eligibility differs from preserved official census")
    model=fit(counts_fit)
    return model,{"schema":"model02-real-exposed-risk-development-v1",
                  "source_zip_sha256":ORIGINAL_SHA,
                  "FIT":{**denom_fit,**score(model,counts_fit,denom_fit["distinct_drives_with_positive_windows"])},
                  "DEV":{**denom_dev,**score(model,counts_dev,denom_dev["distinct_drives_with_positive_windows"])},
                  "trained_classical_reference":True,
                  "trained_TAC_model":False,
                  "independent_evaluation":False,
                  "protected_future_member_contents_opened":False,
                  "scientific_qualification":"NOT_QUALIFIED",
                  "warnings":["original source day is not independently authenticated prediction-time availability",
                              "rolling admission and features were chosen after exposure of same historical source",
                              "source recorded failure is not independent service-life hazard adjudication",
                              "many repeated drive-day overlapping labels make metrics heavily correlated",
                              "censoring and selection can bias complete-case Brier/log-loss",
                              "no unseen source evaluation, calibration certification, or useful-memory advantage"]}

def main():
    ck=Path("model02_original_exploratory_checkpoint.json")
    receipt=Path("model02_original_exploratory_dev.json")
    if ck.exists() or receipt.exists():
        raise EvidenceError("refusing overwrite of original evidence")
    data=pub.fetch_zip()
    model,report=run(data)
    ck.write_text(json.dumps(model,indent=2,sort_keys=True)+"\n")
    receipt.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("ORIGINAL_MODEL02_CHECKPOINT="+canonical(model))
    print("ORIGINAL_MODEL02_DEV="+canonical(report))
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except (EvidenceError, OSError, ValueError, TypeError, UnicodeError, zipfile.BadZipFile) as exc:
        print(canonical({"status":"BLOCKED_OR_FAILED","reason":str(exc)}),file=sys.stderr)
        sys.exit(2)
