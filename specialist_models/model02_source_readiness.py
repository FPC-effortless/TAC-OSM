#!/usr/bin/env python3
"""MODEL02 fail-closed source/train-readiness screen using Git-owned publisher receipts.

Produces factual blockers; does NOT authorize training, release protected data,
verify publisher time, compute clinical/safety significance or train a risk model.
Thresholds are provisional resource-planning filters, NOT scientific evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"research/source-evidence"
FIXED=SOURCE/"MODEL02-BACKBLAZE-2013-46DAY-FEASIBILITY.json"
ROLLING=SOURCE/"MODEL02-BACKBLAZE-2013-ROLLING-EXPOSED-FEASIBILITY.json"
ORIGINAL=SOURCE/"MODEL02-BACKBLAZE-2013-EXPOSED-SOURCE-AUDIT.json"
EXPECTED_ZIP_SHA="7f5a53e79b16e695b4b034955806bb3bb194534b169f6eca460dfd3dc48096fe"
MIN_DISTINCT_FIT_FAILURES=50  # planning threshold, not a power analysis
MAX_CENSOR_FRACTION=.50  # planning threshold, not an inference guarantee

class AuditError(ValueError):
    pass

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False)

def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    data=json.loads(path.read_bytes())
    if not isinstance(data,dict):
        raise AuditError("source receipt is not JSON object")
    return data

def integer(value):
    return type(value) is int and value >= 0

def evaluate(fixed,rolling,original):
    if (fixed.get("schema")!="model02-backblaze-46day-original-source-feasibility-summary-v1"
        or rolling.get("schema")!="model02-backblaze-post-exposure-rolling-admission-summary-v1"
        or original.get("schema")!="product-model02-publisher-exposed-source-audit-condensed-v1"):
        raise AuditError("unexpected source receipt schema")
    hashes=[fixed.get("source_original_zip_sha256"),rolling.get("source_original_zip_sha256"),
            original.get("original_zip_sha256")]
    if hashes!=[EXPECTED_ZIP_SHA]*3:
        raise AuditError("original source archive mismatch")
    if (rolling.get("dates_exposed") != ["2013-10-01","2013-11-15"] or
        rolling.get("heldout_start_not_parsed")!="2013-11-16" or
        fixed.get("date_range_exposed")!=rolling["dates_exposed"] or
        fixed.get("future_test_member_contents_parsed") is not False or
        rolling.get("selected_horizon_days")!=7):
        raise AuditError("wrong source cohort or protected boundary")
    if original.get("exposed_end_inclusive")!="2013-10-14":
        raise AuditError("original source audit wrong frozen historical span")
    fixed_dev=fixed.get("exposed_dev")
    fit=rolling.get("fit_exposed")
    dev=rolling.get("dev_exposed")
    if not all(isinstance(x,dict) for x in (fixed_dev,fit,dev)):
        raise AuditError("cohort denominators missing")
    if fixed_dev.get("provisional_positive_windows")!=0:
        raise AuditError("fixed cohort result changed")
    totals={}
    for name,record in (("fit",fit),("dev",dev)):
        keys=("source_eligible_seven_day_warmup_windows",
              "observed_future_failure_windows",
              "complete_observed_nonfailure_windows","right_censored_windows",
              "unique_drives_with_future_failure_windows","unique_admitted_drives")
        if any(not integer(record.get(k)) for k in keys):
            raise AuditError("invalid source denominators")
        n=record[keys[0]]
        if n!=sum(record[k] for k in keys[1:4]):
            raise AuditError("overlapping/censored counts fail partition")
        if record["unique_drives_with_future_failure_windows"]>record["observed_future_failure_windows"]:
            raise AuditError("independent events greater than positive windows")
        if record["unique_drives_with_future_failure_windows"]>record["unique_admitted_drives"]:
            raise AuditError("unique failing drives exceed admitted drives")
        totals[name]={"source_windows":n,
                      "unique_admitted_drives":record["unique_admitted_drives"],
                      "positive_overlapping_windows":record["observed_future_failure_windows"],
                      "distinct_positive_drives":record["unique_drives_with_future_failure_windows"],
                      "censored_windows":record["right_censored_windows"],
                      "censor_fraction":record["right_censored_windows"]/n if n else 1.}
    if fit.get("unique_drives_with_future_failure_windows") != 35 or dev.get("unique_drives_with_future_failure_windows") != 65:
        raise AuditError("historical exposure case counts changed; inspect before updating")
    if not isinstance(original.get("daily"),list) or len(original["daily"])!=14:
        raise AuditError("original daily source member hash lineage absent")
    if any(not isinstance(r,dict) or not isinstance(r.get("original_member_sha256"),str)
           or re.fullmatch(r"[0-9a-f]{64}",r["original_member_sha256"]) is None
           for r in original["daily"]):
        raise AuditError("source daily hash invalid")
    blockers=[]
    if rolling.get("real_time_source_available_at_verified") is not True:
        blockers.append("SOURCE_PREDICTION_TIME_AVAILABILITY_UNVERIFIED")
    if rolling.get("source_rights_and_original_failure_adjudication_verified") is not True:
        blockers.append("SOURCE_RIGHTS_AND_OUTCOME_ADJUDICATION_UNVERIFIED")
    if rolling.get("independent_evaluation") is not True:
        blockers.append("NO_INDEPENDENT_HELDOUT_OR_PREEXPOSURE_REGISTRATION")
    if totals["fit"]["distinct_positive_drives"]<MIN_DISTINCT_FIT_FAILURES:
        blockers.append("FIT_DISTINCT_FAILURE_EVENTS_BELOW_PLANNING_MINIMUM")
    if totals["fit"]["censor_fraction"]>MAX_CENSOR_FRACTION:
        blockers.append("FIT_CENSOR_FRACTION_ABOVE_PLANNING_LIMIT")
    if totals["dev"]["censor_fraction"]>MAX_CENSOR_FRACTION:
        blockers.append("DEV_CENSOR_FRACTION_ABOVE_PLANNING_LIMIT")
    # This post-exposure source cannot qualify by merely changing a JSON flag.
    if rolling.get("study") != "POST_EXPOSURE_ROLLING_ENTITY_ADMISSION_DIAGNOSTIC":
        raise AuditError("source study provenance invalid")
    blockers.append("POST_EXPOSURE_ADMISSION_RULE_REQUIRES_NEW_SOURCE_COHORT")
    return {"schema":"model02-real-source-train-readiness-v1",
            "original_zip_sha256":EXPECTED_ZIP_SHA,
            "source_period":"EXPOSED_OCT_NOV_2013",
            "outcomes_by_phase":totals,
            "status":"BLOCKED_FOR_QUALIFIED_TRAINING",
            "qualified_fit_authorized":False,
            "training_performed":False,"protected_test_opened":False,
            "blockers":blockers,
            "planning_gates_only":{"min_distinct_fit_failure_events":MIN_DISTINCT_FIT_FAILURES,
                                    "max_censor_fraction":MAX_CENSOR_FRACTION},
            "disclosure":"Thresholds are planning filters, not formal sample-size or valid survival-effect assurance. Flag changes cannot confer source authority."}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path)
    args=p.parse_args(argv)
    try:
        result=evaluate(read(FIXED),read(ROLLING),read(ORIGINAL))
        result["input_file_sha256"]={x.name:sha_file(x) for x in (FIXED,ROLLING,ORIGINAL)}
        if args.out:
            if args.out.exists():raise AuditError("refusing overwrite")
            args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print(canonical(result))
        return 0  # Audit ran; training readiness remains BLOCKED.
    except (AuditError,OSError,ValueError,TypeError,KeyError) as exc:
        print(canonical({"status":"SOURCE_AUDIT_INVALID","reason":str(exc)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
