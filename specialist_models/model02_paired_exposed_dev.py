#!/usr/bin/env python3
"""Paired no-refit score of two Git-owned classical risk references, exposed DEV.

All legitimate original source eligible (entity, day, H=7) cases are paired
before looking at model score. Reports correlated window metrics separately
from the first matured decision per source-owned drive. NOT independent
evaluation, censor-aware survival estimation, TAC comparison or useful memory.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import timedelta
import hashlib
import json
import math
import random
from pathlib import Path
import sys

import backblaze_2013_classical_dev_fit as lr
import backblaze_2013_cohort_feasibility as boundary
import backblaze_2013_source_probe as pub
import backblaze_rolling_admission_feasibility as source
import model02_backblaze_exploratory as lookup

ROOT=Path(__file__).resolve().parents[1]
LR_FILE=ROOT/"research/replay-bundles/PRODUCT-MODEL02-BACKBLAZE-2013-CLASSICAL/ORIGINAL_CHECKPOINT.json"
LOOKUP_FILE=ROOT/"research/replay-bundles/PRODUCT-MODEL02-BACKBLAZE-CLASSICAL/ORIGINAL_CHECKPOINT.json"
LR_ORIGINAL_HASH="4c4cd931a8ec19aba67ea4a3e6a51e16941c7ad21632eec86f81cbd834d81d63"
LOOKUP_ORIGINAL_HASH="5f8b731ddebaf0704637a9e070835317a04d38b215c579f53ca47c7e7684a8fd"
SOURCE_HASH=boundary.SOURCE_SHA
MAX_CASES=600000

class PairError(ValueError):
    pass

def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)

def load_models():
    raw1=LR_FILE.read_bytes()
    raw2=LOOKUP_FILE.read_bytes()
    if (hashlib.sha256(raw1).hexdigest()!=LR_ORIGINAL_HASH or
        hashlib.sha256(raw2).hexdigest()!=LOOKUP_ORIGINAL_HASH):
        raise PairError("original Git checkpoint digests differ from committed run evidence")
    logistic=json.loads(raw1)
    table=json.loads(raw2)
    if (logistic.get("source_archive_sha256")!=SOURCE_HASH or
        table.get("source_zip_sha256")!=SOURCE_HASH or
        logistic.get("fit_examples")!=112115 or
        table.get("fit_windows")!=112115):
        raise PairError("original source and FIT cohort mismatch")
    if (logistic.get("schema")!="model02-2013-original-classical-lr-DEV-001" or
        table.get("schema")!="model02-classical-smart-binary-original-v1"):
        raise PairError("original model schema is not the approved classical reference")
    return logistic,table

def extract_rows(source_bytes, logistic, table):
    if pub.sha(source_bytes)!=SOURCE_HASH:
        raise PairError("original publisher ZIP mismatch")
    flags,daily=source.read_timeline(source_bytes)
    histories=lr.raw_smart_history(source_bytes,set(flags))
    cases=[]
    censored=eligible=0
    positive_drives=set()
    scored_drives=set()
    for entity,timeline in flags.items():
        readings=histories[entity]
        for day in sorted(timeline):
            if not boundary.FIT_END<=day<boundary.EXPOSED_END or not source.admitted_at(timeline,day):
                continue
            eligible+=1
            result=source.label_future(timeline,day,boundary.EXPOSED_END)
            if result=="CENSORED":
                censored+=1
                continue
            if day not in readings:
                raise PairError("missing source-time SMART features")
            cur=readings[day]
            x=lr.feature(cur,readings.get(day-timedelta(days=1)))
            p1=lr.predict(logistic,x)
            cell=lookup.bucket([lookup.bin_value(cur.get(f)) for f in lookup.FEATURES])
            p2=lookup.risk(table,cell)
            y=int(result=="POSITIVE")
            if y:positive_drives.add(entity)
            scored_drives.add(entity)
            if not all(math.isfinite(p) and 0<p<1 for p in (p1,p2)):
                raise PairError("nonfinite or boundary model probability")
            cases.append((entity,day,y,p1,p2))
            if len(cases)>MAX_CASES:
                raise PairError("source cohort larger than registered audit case budget")
    if len(cases)!=204373 or sum(c[2] for c in cases)!=397 or len(positive_drives)!=65 or censored!=180532:
        raise PairError("original source case, censor and outcome denominators changed")
    return cases,{"eligible_windows":eligible,"matured_windows":len(cases),
                  "right_censored_windows":censored,
                  "distinct_scored_drives":len(scored_drives),
                  "distinct_drives_with_positive_windows":len(positive_drives),
                  "positive_windows":397}

def summarize(cases, score_column):
    if not cases:
        return {"status":"NO_SCORED_CASES","n":0}
    n=len(cases);pos=sum(c[2] for c in cases)
    brier=sum((c[score_column]-c[2])**2 for c in cases)/n
    nll=-sum(c[2]*math.log(c[score_column])+(1-c[2])*math.log1p(-c[score_column])
             for c in cases)/n
    return {"n":n,"positive":pos,"brier":brier,"log_loss":nll,
            "observed_positive_fraction":pos/n}

def per_drive_first_matured(cases):
    """Earliest eligible mature-case decision per real drive; never outcome-selected."""
    first={}
    for row in cases:
        key=row[0]
        if key not in first or row[1]<first[key][1]:
            first[key]=row
    return list(first.values())

def first_drive_bootstrap_posthoc(first, replicates=300, seed=20261027):
    """POST-HOC exploratory paired drive resampling, not confidence proof.

    One matured decision per source-owned drive. Assumes exchangeable physical
    drives even though manufacturer, site and calendar clusters may matter.
    """
    if len(first)<2 or type(replicates) is not int or not 50<=replicates<=5000:
        raise PairError("invalid first-drive bootstrap input")
    diffs=[(r[4]-r[2])**2 - (r[3]-r[2])**2 for r in first]
    mean=sum(diffs)/len(diffs)
    rng=random.Random(seed)
    sampled=[]
    n=len(diffs)
    for _ in range(replicates):
        sampled.append(sum(diffs[rng.randrange(n)] for j in range(n))/n)
    sampled.sort()
    return {"status":"POST_HOC_EXPLORATORY_NOT_INDEPENDENT_CONFIRMATION",
            "unit":"first matured labeled prediction per physical drive",
            "sampled_drive_count":n,"positive_decisions":sum(c[2] for c in first),
            "seed":seed,"replicates":replicates,
            "delta_definition":"two_feature_lookup_brier_minus_eight_feature_logistic_brier",
            "paired_delta":mean,
            "percentile_95_bounds":[sampled[int((replicates-1)*.025)],
                                    sampled[int((replicates-1)*.975)]],
            "caveats":["added after inspecting initial paired exposed development results",
                       "conditional on noncensored first mature cases, not censor-aware",
                       "manufacturer/site/date correlated and unequal exposure, not iid proof",
                       "not a fresh source or architecture-superiority uncertainty interval"]}

def pair_report(cases, source_denominators, logistic, lookup_model):
    if source_denominators["matured_windows"]!=len(cases):
        raise PairError("source case count mismatch")
    if not cases:
        raise PairError("no legal paired cases")
    first=per_drive_first_matured(cases)
    # First-per-drive outcome is the H-day label for that FIRST decision, not
    # "eventually failed anywhere in November", avoiding outcome-picked times.
    by_day=defaultdict(list)
    for c in cases:by_day[str(c[1])].append(c)
    paired={}
    for label,subset in (("all_matured_correlated_windows",cases),
                         ("first_matured_forecast_per_drive",first)):
        paired[label]={
            "eight_feature_logistic":summarize(subset,3),
            "two_feature_smoothed_lookup":summarize(subset,4),
            "fit_only_constant_prior":summarize(
                [(key,day,y,logistic["fit_positive_windows"]/logistic["fit_examples"],
                  lookup_model["prior_fit"]) for key,day,y,*_ in subset],3)
        }
    return {
        "schema":"model02-paired-original-source-no-fit-references-v1",
        "evidence_tier":"ORIGINAL_EXPOSED_DEV_PAIRED_REFERENCE_DIAGNOSTIC_ONLY",
        "source_archive_sha256":SOURCE_HASH,
        "original_model_git_shas":{"eight_feature_logistic":LR_ORIGINAL_HASH,
                                    "two_feature_lookup":LOOKUP_ORIGINAL_HASH},
        "source_denominators":source_denominators,
        "paired_results":paired,
        "first_per_drive_brier_delta_exploratory_posthoc":first_drive_bootstrap_posthoc(first),
        "dev_day_metrics":[{"date":date,"n":len(subset),
                            "positive_windows":sum(c[2] for c in subset),
                            "logistic_brier":summarize(subset,3)["brier"],
                            "lookup_brier":summarize(subset,4)["brier"]}
                           for date,subset in sorted(by_day.items())],
        "earliest_decision_rule":"first chronologically matured noncensored eligible prediction per physical drive",
        "fit_performed_during_comparison":False,
        "matched_observations_and_outcome_definitions":True,
        "architecture_superiority_claim":False,
        "independent_evaluation":False,
        "protected_future_after_2013_11_15_parsed":False,
        "scientific_qualification":"NOT_QUALIFIED",
        "limitations":[
            "the same November 2013 development cases were already exposed before model designs",
            "two models have different original FIT capacities/features; not a matched TAC architecture comparison",
            "daily windows strongly correlated and censoring selective; first-per-drive metric is only a sensitivity view",
            "first mature decision is not an unbiased incident-based prospective hazard evaluation",
            "publisher as-of time and true failure adjudication remain unverified",
            "risk differences are descriptive without paired cluster uncertainty or held-out confirmation",
            "no TAC memory model was trained or scored"
        ]}

def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    try:
        if args.out.exists():
            raise PairError("do not overwrite original comparison receipt")
        model1,model2=load_models()
        original=pub.fetch_zip()
        cases,denom=extract_rows(original,model1,model2)
        report=pair_report(cases,denom,model1,model2)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
        print(canonical({"schema":report["schema"],
                         "paired_results":report["paired_results"],
                         "source_denominators":denom,
                         "scientific_qualification":"NOT_QUALIFIED"}))
        return 0
    except (PairError,OSError,ValueError,KeyError,TypeError,
            lr.StudyError,source.SourceError,pub.SourceError) as exc:
        print(canonical({"status":"BLOCKED_OR_FAILED","reason":str(exc)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
