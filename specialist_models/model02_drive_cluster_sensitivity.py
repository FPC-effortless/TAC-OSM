#!/usr/bin/env python3
"""Original Backblaze 2013 exposed DEV drive-level score sensitivity.

No refitting or future source access. Compares the original nine-cell lookup
against the frozen FIT-only prevalence probability. Equal-weight DRIVE averages
and one-first-matured-window/drive supplement prior overlapping-window metrics.
Exploratory, post-exposure, and censor-biased; NOT qualified calibration.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import timedelta
import json
import math
from pathlib import Path
import sys

import model02_backblaze_exploratory as exp
import backblaze_rolling_admission_feasibility as rolling

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/"research/replay-bundles/PRODUCT-MODEL02-BACKBLAZE-CLASSICAL"
MODEL=ORIGINAL/"ORIGINAL_CHECKPOINT.json"
RESULT=ORIGINAL/"ORIGINAL_EXPOSED_DEV.json"


def measures(prob, outcome):
    return ((prob-outcome)**2,
            -math.log(prob) if outcome else -math.log1p(-prob))


def summarize(values):
    if not values:
        raise exp.EvidenceError("no physical drives to score")
    n=len(values)
    avg=sum(values)/n
    var=sum((x-avg)**2 for x in values)/(n-1) if n>1 else 0.
    return {"n_distinct_drives":n,"mean":avg,"standard_error_by_drive":math.sqrt(var/n),
            "min":min(values),"max":max(values)}


def aggregate(model, timelines):
    prior=model["prior_fit"]
    if not 0<prior<1:
        raise exp.EvidenceError("invalid FIT-only source prior")
    macro={"lookup_brier":[],"prior_brier":[],"difference_brier":[],
           "lookup_logloss":[],"prior_logloss":[],"difference_logloss":[]}
    first={"lookup_brier":[],"prior_brier":[],"difference_brier":[],
           "lookup_logloss":[],"prior_logloss":[],"difference_logloss":[]}
    total_eligible=total_censored=total_matured=total_positive=0
    source_event_drives=0
    scores_by_drive={}
    for physical_key, series in timelines.items():
        observed={d:r["failed"] for d,r in series.items()}
        entries=[]
        any_positive=False
        for day in sorted(series):
            if not exp.period.FIT_END<=day<exp.period.EXPOSED_END:
                continue
            if not rolling.admitted_at(observed,day):
                continue
            total_eligible+=1
            target=rolling.label_future(observed,day,exp.period.EXPOSED_END)
            if target=="CENSORED":
                total_censored+=1
                continue
            y=int(target=="POSITIVE")
            total_positive+=y
            any_positive=any_positive or bool(y)
            p=exp.risk(model,series[day]["cell"])
            b1,l1=measures(p,y)
            b0,l0=measures(prior,y)
            entries.append((b1,b0,b1-b0,l1,l0,l1-l0))
            total_matured+=1
        if not entries:
            continue
        source_event_drives+=int(any_positive)
        scores_by_drive[physical_key]=len(entries)
        for idx,name in enumerate(("lookup_brier","prior_brier","difference_brier",
                                   "lookup_logloss","prior_logloss","difference_logloss")):
            macro[name].append(sum(v[idx] for v in entries)/len(entries))
            first[name].append(entries[0][idx])
    original=json.loads(RESULT.read_bytes())
    if (total_eligible!=original["DEV"]["eligible_windows"] or
        total_censored!=original["DEV"]["right_censored_windows"] or
        total_matured!=original["DEV"]["matured_windows"] or
        total_positive!=original["DEV"]["positive_overlapping_windows"] or
        source_event_drives!=original["DEV"]["distinct_positive_drives"]):
        raise exp.EvidenceError("drive-level grouping changed original source denominators")
    n_drive=len(scores_by_drive)
    return {"schema":"model02-drive-equal-source-dev-sensitivity-v1",
            "source_original_zip_sha256":exp.ORIGINAL_SHA,
            "original_model_source":"ORIGINAL_CHECKPOINT.json",
            "source_period":"EXPOSED_ORIGINAL_2013_DEV_NOV1_15",
            "original_exposed_dev_eligible_windows":total_eligible,
            "original_exposed_dev_matured_windows":total_matured,
            "original_exposed_dev_censored_windows":total_censored,
            "original_exposed_dev_positive_windows":total_positive,
            "distinct_dev_drives_with_matured_windows":n_drive,
            "distinct_dev_drives_with_positive_windows":source_event_drives,
            "observed_matured_windows_per_drive":{
                "min":min(scores_by_drive.values()),"max":max(scores_by_drive.values()),
                "mean":total_matured/n_drive},
            "equal_drive_weighted":{k:summarize(v) for k,v in macro.items()},
            "one_earliest_matured_window_per_drive":{k:summarize(v) for k,v in first.items()},
            "original_overlapping_window_brier_lookup":original["DEV"]["brier"],
            "original_overlapping_window_brier_fit_prior":original["DEV"]["fit_only_prior_baseline_brier"],
            "original_overlapping_window_logloss_lookup":original["DEV"]["log_loss"],
            "original_overlapping_window_logloss_fit_prior":original["DEV"]["fit_only_prior_baseline_log_loss"],
            "fit_performed_this_run":False,"trained_tac_model":False,
            "protected_future_member_contents_parsed":False,
            "independent_evaluation":False,"scientific_qualification":"NOT_QUALIFIED",
            "warning":["These are post-exposure sensitivity scores, not prospective prediction validity",
                       "Within-drive means only equalize case weights; retention/censoring and selection bias remain",
                       "Earliest matured window is selected by future label maturity, not a validated prospective sampling design",
                       "Drive-level SEs treat drives as independent but may miss shared model/site/operation correlations",
                       "Source historical available-at and adjudicated outcomes unverified; no TAC neural training or superiority"]}


def main():
    model=json.loads(MODEL.read_bytes())
    if (model.get("schema")!="model02-classical-smart-binary-original-v1" or
        model.get("source_zip_sha256")!=exp.ORIGINAL_SHA):
        raise exp.EvidenceError("original trained model missing or modified")
    records=exp.records_from_original(exp.pub.fetch_zip())
    outcome=aggregate(model,records)
    print(exp.canonical(outcome))


if __name__=="__main__":
    try:main()
    except (exp.EvidenceError,OSError,ValueError,TypeError) as exc:
        print(exp.canonical({"status":"BLOCKED_OR_FAILED","reason":str(exc)}),file=sys.stderr)
        sys.exit(2)
