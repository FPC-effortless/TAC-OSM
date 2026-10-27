#!/usr/bin/env python3
"""Original Backblaze source: first eligible real-time decision, including censoring.

NO FIT. Both pinned existing classical checkpoints score at the FIRST eligible
DEV observation per source physical drive, irrespective of future maturity.
The 7-day outcome is separately POSITIVE, NEGATIVE or UNKNOWN/CENSORED.
For each source-prespecified review budget, report sharp conditional precision
bounds [observed positive/k, (positive+unknown)/k] over all selected alerts.
Bounds do not impute unknown outcomes; they are NOT calibrated interval bounds
or causal performance. Already exposed, retrospective source, no promotion.
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import sys

import model02_paired_exposed_dev as frozen

H=7
BUDGETS=(.01,.05)
MAX_DRIVES=40000

class EvaluationError(ValueError):
    pass


def cases_from_source(original, logistic, lookup):
    if frozen.pub.sha(original)!=frozen.SOURCE_HASH:
        raise EvaluationError("original source SHA256 changed")
    flags,_=frozen.source.read_timeline(original)
    histories=frozen.lr.raw_smart_history(original,set(flags))
    records=[]
    for entity,timeline in flags.items():
        for day in sorted(timeline):
            if not frozen.boundary.FIT_END<=day<frozen.boundary.EXPOSED_END:
                continue
            if not frozen.source.admitted_at(timeline,day):
                continue
            readings=histories.get(entity)
            if readings is None or day not in readings:
                raise EvaluationError("missing original drive-day features")
            now=readings[day]
            prior=readings.get(day-timedelta(days=1))
            p_log=frozen.lr.predict(logistic,frozen.lr.feature(now,prior))
            cell=frozen.lookup.bucket([frozen.lookup.bin_value(now.get(f)) for f in frozen.lookup.FEATURES])
            p_lookup=frozen.lookup.risk(lookup,cell)
            state=frozen.source.label_future(timeline,day,frozen.boundary.EXPOSED_END)
            if state not in ("POSITIVE","NEGATIVE","CENSORED"):
                raise EvaluationError("unrecognized source status")
            if not all(math.isfinite(p) and 0<p<1 for p in (p_log,p_lookup)):
                raise EvaluationError("invalid original model probability")
            records.append((entity,day,state,p_log,p_lookup))
            break  # First eligible decision, NOT first known-positive or first matured.
        if len(records)>MAX_DRIVES:
            raise EvaluationError("entity capacity exceeded")
    if not records or len({r[0] for r in records})!=len(records):
        raise EvaluationError("empty/duplicate original enrolled drive")
    return records


def bounded_alerts(first, budgets=BUDGETS):
    if not first or len(set(r[0] for r in first))!=len(first):
        raise EvaluationError("one decision per distinct physical drive required")
    total=len(first)
    if any(r[2] not in ("POSITIVE","NEGATIVE","CENSORED") for r in first):
        raise EvaluationError("invalid outcome category")
    status_counts={s:sum(r[2]==s for r in first) for s in ("POSITIVE","NEGATIVE","CENSORED")}
    def tie(r):
        return hashlib.sha256(("\x1f".join(r[0])).encode()).hexdigest()
    out=[]
    for fraction in budgets:
        if type(fraction)!=float or not .0<fraction<1:
            raise EvaluationError("review fraction invalid")
        k=max(1,int(total*fraction))
        item={"review_fraction":fraction,"review_capacity_drives":k,"models":{}}
        for name,column in (("eight_feature_logistic",3),("two_feature_lookup",4)):
            chosen=sorted(first,key=lambda r:(-r[column],tie(r)))[:k]
            pos=sum(r[2]=="POSITIVE" for r in chosen)
            neg=sum(r[2]=="NEGATIVE" for r in chosen)
            censored=sum(r[2]=="CENSORED" for r in chosen)
            if pos+neg+censored!=k:
                raise EvaluationError("alert partition mismatch")
            item["models"][name]={
                "reviewed_drives":k,
                "known_positive":pos,
                "known_negative":neg,
                "unknown_censored":censored,
                "precision_lower_bound_if_unknown_negative":pos/k,
                "precision_upper_bound_if_unknown_positive":(pos+censored)/k,
                "false_alerts_lower_bound":neg,
                "false_alerts_upper_bound":neg+censored,
                "positive_capture_among_all_known_first_decisions": (
                    pos/status_counts["POSITIVE"] if status_counts["POSITIVE"] else None),
                "recall_over_all_physical_failures":"NOT_IDENTIFIABLE_FROM_CENSORING"
            }
        out.append(item)
    return {"schema":"model02-original-censor-inclusive-first-decision-bounds-v1",
            "source_zip_sha256":frozen.SOURCE_HASH,
            "original_git_checkpoint_sha256":{"logistic":frozen.LR_ORIGINAL_HASH,
                                               "lookup":frozen.LOOKUP_ORIGINAL_HASH},
            "first_eligible_distinct_drives":total,
            "first_decision_outcomes":status_counts,
            "budgets":out,
            "first_decision_rule":"earliest chronologically eligible DEV day, no future outcome gating",
            "tie_breaking":"source-owned physical identity SHA256, no label access",
            "exposed_development_only":True,
            "fitting_or_calibration_during_evaluation":False,
            "independent_evaluation":False,
            "prospective_prediction_time_visibility_proven":False,
            "scientific_qualification":"NOT_QUALIFIED",
            "interpretation":"worst/best labels of outcome-unknown reviewed drives, not confidence interval, identified survival probability, or utility",
            "warnings":[
                "source and models have already been inspected during prior DEV experiments",
                "censoring is unresolved and could depend strongly on manufacturer, age and model",
                "first eligible DEV decision is no longer filtered for future completeness",
                "unknown future outcomes are never silently relabeled negative",
                "rankings reflect historical publisher records, no live alert or intervention",
                "original publisher available-at and legal source-time semantics unverified",
                "binary best-worst bounds can be broad and are not probabilistic estimates",
                "no TAC neural architecture comparison or proof of useful long memory"
            ]}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    try:
        if args.out.exists():
            raise EvaluationError("refuse to overwrite original evidence")
        logistic,lookup=frozen.load_models()
        original=frozen.pub.fetch_zip()
        rows=cases_from_source(original,logistic,lookup)
        receipt=bounded_alerts(rows)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
        print(frozen.canonical(receipt))
        return 0
    except (EvaluationError,frozen.PairError,OSError,ValueError,KeyError,TypeError,
            frozen.lr.StudyError, frozen.source.SourceError) as e:
        print(frozen.canonical({"status":"BLOCKED_OR_FAILED","reason":str(e)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
