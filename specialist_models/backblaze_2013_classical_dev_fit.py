#!/usr/bin/env python3
"""FIRST Backblaze 2013 original-source classical risk DEV diagnostic.

All source/label windows are ALREADY EXPOSED (2013-10-01 to 2013-11-15).
Learns only on dated FIT 2013-10, scores only exposed 2013-11-01..15 DEV,
and never opens any publisher ZIP member on or after 2013-11-16.
This is NOT an as-of-time qualified, independent or production risk predictor.
No TAC-OSM neural fitting, architecture superiority or actionable warning.
"""
from __future__ import annotations
import argparse
import csv
from datetime import timedelta
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile

import backblaze_rolling_admission_feasibility as admission
import backblaze_2013_cohort_feasibility as source
import backblaze_2013_source_probe as original

SOURCE_SHA = source.SOURCE_SHA
FIELDS = ("smart_5_raw", "smart_9_raw", "smart_197_raw")
DIMS = 8
FIT_BOUNDARY = source.FIT_END
DEV_BOUNDARY = source.EXPOSED_END
EPOCHS = 55
LR = 0.4
L2 = 0.01

class StudyError(ValueError):
    pass

def canonical(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), allow_nan=False)

def feature(raw, previous):
    def read(field):
        value = raw.get(field, "")
        if value is None or value == "":
            return 0., 0.
        try:
            number=float(value)
        except (TypeError, ValueError) as e:
            raise StudyError("non-numeric original SMART attribute") from e
        if not 0 <= number < float("inf") or not math.isfinite(number):
            raise StudyError("invalid original SMART magnitude")
        return math.log1p(number), 1.
    v5,p5=read(FIELDS[0])
    v9,p9=read(FIELDS[1])
    v197,p197=read(FIELDS[2])
    if previous is not None:
        old5,was5=read_previous(previous,FIELDS[0])
        old197,was197=read_previous(previous,FIELDS[2])
    else:
        old5=old197=was5=was197=0.
    return [v5,p5,v9,p9,v197,p197,
            v5-old5 if p5 and was5 else 0.,
            v197-old197 if p197 and was197 else 0.]

def read_previous(previous, field):
    value = previous.get(field, "")
    if value in ("", None):
        return 0., 0.
    try:
        number = float(value)
    except ValueError as e:
        raise StudyError("invalid previous SMART") from e
    if not math.isfinite(number) or number < 0:
        raise StudyError("invalid previous SMART")
    return math.log1p(number), 1.

def raw_smart_history(payload, allowed_entities):
    """Open and validate only named exposed daily publisher CSV members."""
    if original.sha(payload) != SOURCE_SHA:
        raise StudyError("source ZIP SHA differs from original evidence")
    wanted={d.isoformat()+".csv":d for d in source.days()}
    history={key:{} for key in allowed_entities}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members={}
        for info in archive.infolist():
            if info.is_dir():
                continue
            name=PurePosixPath(info.filename)
            if ".." in name.parts or info.filename.startswith("/"):
                raise StudyError("unsafe publisher member")
            if name.name in wanted:
                if name.name in members or info.file_size>original.MAX_MEMBER:
                    raise StudyError("duplicate or oversized original source day")
                members[name.name]=info
        if set(members)!=set(wanted):
            raise StudyError("source day set incomplete")
        for name, day in wanted.items():
            with archive.open(members[name]) as stream:
                blob=stream.read(original.MAX_MEMBER+1)
            if len(blob)>original.MAX_MEMBER:
                raise StudyError("decompressed original source over budget")
            reader=csv.DictReader(io.StringIO(blob.decode("utf-8-sig")))
            cols=set(reader.fieldnames or [])
            if not {"date","serial_number","model","failure",*FIELDS}.issubset(cols):
                raise StudyError("mandatory source feature missing")
            selected=set()
            for row in reader:
                key=(row["serial_number"],row["model"])
                if key not in allowed_entities:
                    continue
                if row["date"]!=day.isoformat() or row["failure"] not in ("0","1"):
                    raise StudyError("publisher original case mismatch")
                if key in selected:
                    raise StudyError("duplicate source entity-day")
                selected.add(key)
                history[key][day]={k:row.get(k) for k in FIELDS}
                for field in FIELDS:
                    value = row.get(field)
                    if value not in ("", None):
                        try:
                            parsed = float(value)
                        except (TypeError, ValueError) as error:
                            raise StudyError("invalid source SMART numeric field") from error
                        if not math.isfinite(parsed) or parsed < 0:
                            raise StudyError("nonfinite/negative original source SMART value")
    return history

def cohorts(payload):
    if original.sha(payload)!=SOURCE_SHA:
        raise StudyError("source ZIP SHA mismatch")
    flags,daily=admission.read_timeline(payload)
    history=raw_smart_history(payload,set(flags))
    result={}
    for tag,start,end in (("FIT",source.FIRST,FIT_BOUNDARY),
                          ("DEV",FIT_BOUNDARY,DEV_BOUNDARY)):
        xs,ys=[],[]
        seen_positive=set()
        seen_drives=set()
        censored=0
        for ident,timeline in flags.items():
            record=history[ident]
            for day in sorted(timeline):
                if not start<=day<end or not admission.admitted_at(timeline,day):
                    continue
                label=admission.label_future(timeline,day,end)
                if label=="CENSORED":
                    censored+=1
                    continue
                if day not in record:
                    raise StudyError("missing original source SMART for admitted day")
                past=record.get(day-timedelta(days=1))
                x=feature(record[day],past)
                if len(x)!=DIMS or any(not math.isfinite(v) for v in x):
                    raise StudyError("invalid original feature graph")
                xs.append(x)
                y=int(label=="POSITIVE")
                ys.append(y)
                seen_drives.add(ident)
                if y:
                    seen_positive.add(ident)
        if not xs or len(set(ys)) != {0,1}:
            raise StudyError(tag+": FIT/DEV has insufficient source-positive cases")
        result[tag]={"xs":xs,"ys":ys,"censored":censored,
                     "unique_scored_drives":len(seen_drives),
                     "unique_drives_with_positive_windows":len(seen_positive)}
    return result, daily

def sigmoid(z):
    if z>=0:
        return 1/(1+math.exp(-z))
    q=math.exp(z)
    return q/(1+q)

def fit(examples):
    x,y=examples["xs"],examples["ys"]
    n=len(x)
    pos=sum(y)
    if not 0<pos<n:
        raise StudyError("both source classes required")
    means=[sum(row[j] for row in x)/n for j in range(DIMS)]
    scales=[max(math.sqrt(sum((row[j]-means[j])**2 for row in x)/n),.00001)
            for j in range(DIMS)]
    z=[tuple(max(-15.,min(15.,(row[j]-means[j])/scales[j])) for j in range(DIMS))
       for row in x]
    w=[0.]*DIMS
    bias=math.log(pos/(n-pos))
    for _ in range(EPOCHS):
        gb=0.
        gw=[0.]*DIMS
        for row,target in zip(z,y):
            error=sigmoid(bias+sum(row[j]*w[j] for j in range(DIMS)))-target
            gb+=error
            for j in range(DIMS):
                gw[j]+=error*row[j]
        bias-=LR*gb/n
        for j in range(DIMS):
            w[j]-=LR*(gw[j]/n+L2*w[j])
    return {"schema":"model02-2013-original-classical-lr-DEV-001",
            "model_type":"CLASSICAL_LOGISTIC_NOT_TAC",
            "source_archive_sha256":SOURCE_SHA,
            "FIT_days":[source.FIRST.isoformat(),(FIT_BOUNDARY-timedelta(days=1)).isoformat()],
            "DEV_exposed_days":[FIT_BOUNDARY.isoformat(),(DEV_BOUNDARY-timedelta(days=1)).isoformat()],
            "feature_fields":list(FIELDS),
            "feature_names":["5_log","5_present","9_log","9_present",
                             "197_log","197_present","5_delta","197_delta"],
            "input_normalizer_fit_means":means, "input_normalizer_fit_scales":scales,
            "coefficients":w,"bias":bias,
            "fit_examples":n,"fit_positive_windows":pos,
            "updates":EPOCHS,"step":LR,"regularizer":L2,
            "qualification":"EXPOSED_SOURCE_DEV_ONLY_NO_ASOF_EVIDENCE",
            "training_from_protected_test":False}

def predict(model,x):
    w=model["coefficients"];mean=model["input_normalizer_fit_means"]
    scale=model["input_normalizer_fit_scales"]
    if len(x)!=DIMS or len(w)!=DIMS or len(mean)!=DIMS or len(scale)!=DIMS:
        raise StudyError("original model shape mismatch")
    z=[max(-15.,min(15.,(x[j]-mean[j])/scale[j])) for j in range(DIMS)]
    return sigmoid(model["bias"]+sum(z[j]*w[j] for j in range(DIMS)))

def metrics(y,p):
    n=len(y)
    positives=sum(y)
    if not n or positives==0:
        raise StudyError("missing independent physical positives in scoring population")
    brier=sum((prediction-target)**2 for prediction,target in zip(p,y))/n
    nll=-sum(target*math.log(max(prediction,1e-15))+
             (1-target)*math.log(max(1-prediction,1e-15))
             for target,prediction in zip(y,p))/n
    # Average precision over case-windows is exploratory. Its denominator
    # includes repeated target events for the same physical drive.
    ordered=sorted(zip(p,y),key=lambda pair:pair[0],reverse=True)
    hits=0;ap=0.
    for rank,(_,target) in enumerate(ordered,1):
        if target:
            hits+=1
            ap+=hits/rank
    return {"n_scored_windows":n,"positive_windows":positives,
            "positive_window_fraction":positives/n,
            "brier":brier,"log_loss":nll,"average_precision_OVERLAPPING_WINDOWS":ap/positives}

def evaluate(checkpoint, cohorts_, daily):
    fit_,dev_=cohorts_["FIT"],cohorts_["DEV"]
    p=[predict(checkpoint,x) for x in dev_["xs"]]
    base=checkpoint["fit_positive_windows"]/checkpoint["fit_examples"]
    base_metrics=metrics(dev_["ys"],[base]*len(p))
    model_metrics=metrics(dev_["ys"],p)
    return {"study":"MODEL02-2013-ORIGINAL-CLASSICAL-RISK-EXPOSED-DEV-001",
            "tier":"GENUINE_PUBLISHER_SOURCE_TRAINED_CLASSICAL_EXPOSED_DEV_DIAGNOSTIC",
            "source_original_zip_sha256":SOURCE_SHA,
            "fit_examples":checkpoint["fit_examples"],
            "fit_positive_windows":checkpoint["fit_positive_windows"],
            "fit_unique_positive_physical_drives":fit_["unique_drives_with_positive_windows"],
            "dev_unique_positive_physical_drives":dev_["unique_drives_with_positive_windows"],
            "dev_unique_scored_physical_drives":dev_["unique_scored_drives"],
            "dev_censored_windows_excluded":dev_["censored"],
            "dev":model_metrics,"fit_prevalence_reference_dev":base_metrics,
            "publisher_source_original_days_parsed":len(daily),
            "model_type":"CLASSICAL_NOT_TAC",
            "source_published_asof_verified":False,
            "trained_on_protected_TEST":False,
            "independent_evaluation":False,
            "conclusion":"DEV_ONLY_NO_REAL_WORLD_QUALIFICATION",
            "cautions":[
                "publication and original SMART available-at clock not independently verified",
                "training/DEV both already exposed as part of preceding source-feasibility work",
                "overlapping drive-day windows artificially increase sample counts",
                "drive failure flags are publisher-reported and retirements are right-censored",
                "an observed DEV metric does not guarantee future alert precision/utility",
                "new model families must obey 100k neural parameter gate before comparative claims",
                "original trained weights plus independent clean replay required for model evidence"]}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-model",type=Path,required=True)
    p.add_argument("--out-report",type=Path,required=True)
    p.add_argument("--replay",action="store_true")
    args=p.parse_args(argv)
    try:
        if args.out_report.exists() or (not args.replay and args.out_model.exists()):
            raise StudyError("refusing to overwrite original checkpoint or receipt")
        archive=original.fetch_zip()
        data,daily=cohorts(archive)
        if args.replay:
            trained=json.loads(args.out_model.read_text(encoding="utf-8"))
            if trained.get("source_archive_sha256")!=SOURCE_SHA or trained.get("schema")!="model02-2013-original-classical-lr-DEV-001":
                raise StudyError("wrong frozen risk model checkpoint")
        else:
            trained=fit(data["FIT"])
            args.out_model.parent.mkdir(parents=True,exist_ok=True)
            with args.out_model.open("x",encoding="utf-8") as out:
                json.dump(trained,out,indent=2,sort_keys=True,allow_nan=False)
        result=evaluate(trained,data,daily)
        args.out_report.parent.mkdir(parents=True,exist_ok=True)
        with args.out_report.open("x",encoding="utf-8") as out:
            json.dump(result,out,indent=2,sort_keys=True,allow_nan=False)
        print(canonical(result))
        print("ORIGINAL_MODEL02_CHECKPOINT_JSON="+canonical(trained))
        return 0
    except (StudyError, admission.SourceError, original.SourceError, OSError,
            ValueError,TypeError,zipfile.BadZipFile,UnicodeError) as error:
        print(canonical({"status":"BLOCKED_OR_FAILED","reason":str(error)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    sys.exit(main())
