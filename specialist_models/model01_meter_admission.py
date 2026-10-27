#!/usr/bin/env python3
"""Model01 pilot meter ingestion: strict point-in-time admission, not qualification.

For an UNQUALIFIED meter/site, returns an advisory no-training persistence
reference or DEFER, never a claimed calibrated original-UCI ridge forecast.
The original model has not generalized to a new compatible customer site.
No authorization to control equipment, issue alarms or bill customers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import sys


SCHEMA="model01-meter-asof-ingestion-v1"
TARGET="appliances_energy_wh_per_10min"
WINDOW=145
STEP=timedelta(minutes=10)
MAX_FRESHNESS=timedelta(minutes=20)

class AdmissionError(ValueError):
    pass


def parse_utc(value, field):
    if not isinstance(value,str) or not value.endswith("Z"):
        raise AdmissionError(field+" must be canonical UTC with Z")
    try:
        parsed=datetime.fromisoformat(value[:-1]+"+00:00")
    except ValueError as exc:
        raise AdmissionError(field+" invalid ISO 8601") from exc
    if parsed.utcoffset()!=timedelta(0) or parsed.second or parsed.microsecond:
        raise AdmissionError(field+" must be exact minute UTC")
    if parsed.isoformat(timespec="seconds")[:-6]+"Z" != value:
        # Require YYYY-MM-DDTHH:MM:00Z for stable idempotency.
        raise AdmissionError(field+" requires exact second precision")
    return parsed


def score_request(message):
    if not isinstance(message,dict) or set(message)!={
        "schema","meter_id","target","sampling_minutes","timezone",
        "decision_time_utc","samples","allow_advisory_persistence"
    }:
        raise AdmissionError("exact meter-asof contract required")
    if message["schema"]!=SCHEMA:
        raise AdmissionError("unsupported meter ingestion version")
    ident=message["meter_id"]
    if not isinstance(ident,str) or not 1<=len(ident)<=64 or not ident.isascii() or any(
        not(c.isalnum() or c in "-_.") for c in ident
    ):
        raise AdmissionError("meter id must be opaque limited ASCII")
    if message["target"]!=TARGET:
        return {"schema":"model01-meter-decision-v1","decision":"DEFER",
                "reason":"TARGET_NOT_APPLIANCES_WH_10MIN",
                "model_invoked":False,"authorized_action":False,
                "independent_qualification":"NOT_ESTABLISHED"}
    if type(message["sampling_minutes"]) is not int or message["sampling_minutes"]!=10:
        raise AdmissionError("meter input must be 10-minute energy accumulation")
    if message["timezone"]!="UTC":
        raise AdmissionError("non-UTC input not accepted without mapping")
    decision=parse_utc(message["decision_time_utc"],"decision_time_utc")
    samples=message["samples"]
    if not isinstance(samples,list) or len(samples)!=WINDOW:
        raise AdmissionError("exact 145 consecutive samples required")
    if type(message["allow_advisory_persistence"]) is not bool:
        raise AdmissionError("explicit advisory fallback selection required")

    last_end=None
    for i,row in enumerate(samples):
        if not isinstance(row,dict) or set(row)!={"window_end_utc","available_at_utc","energy_wh"}:
            raise AdmissionError("exact immutable sensor sample fields required")
        end=parse_utc(row["window_end_utc"],"window_end_utc")
        available=parse_utc(row["available_at_utc"],"available_at_utc")
        # A complete 10-minute accumulated interval becomes meaningful only at
        # window_end. Available-at must not precede complete interval creation.
        if available<end or available>decision:
            raise AdmissionError("observation availability falls outside causal decision time")
        if end>decision or (last_end is not None and end-last_end != STEP):
            raise AdmissionError("timestamp history contains gap, duplicate or future window")
        last_end=end
        value=row["energy_wh"]
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1_000_000:
            raise AdmissionError("energy_wh must be finite, nonnegative accumulated Wh")
    if last_end is None:
        raise AdmissionError("no valid source history")
    if decision-last_end>MAX_FRESHNESS:
        return {"schema":"model01-meter-decision-v1","decision":"DEFER",
                "reason":"LAST_COMPLETE_READING_STALE",
                "age_minutes":(decision-last_end).total_seconds()/60,
                "model_invoked":False,"authorized_action":False,
                "independent_qualification":"NOT_ESTABLISHED"}
    if not message["allow_advisory_persistence"]:
        return {"schema":"model01-meter-decision-v1","decision":"DEFER",
                "reason":"ADVISORY_BASELINE_NOT_REQUESTED",
                "model_invoked":False,"authorized_action":False,
                "independent_qualification":"NOT_ESTABLISHED"}
    return {
        "schema":"model01-meter-decision-v1",
        "decision":"ADVISORY_BASELINE_ONLY",
        "prediction_target":TARGET,
        "prediction_horizon_minutes":60,
        "forecast_reference":"last_completed_observed_10min_appliance_energy",
        "prediction_wh":samples[-1]["energy_wh"],
        "forecast_at_decision_time_utc":message["decision_time_utc"],
        "source_last_complete_window_end_utc":samples[-1]["window_end_utc"],
        "original_ridge_invoked":False,
        "calibrated_interval":None,
        "authorized_action":False,
        "independent_qualification":"NOT_ESTABLISHED",
        "warning":"Persistence reference is not externally qualified; explicit site use and calibration required"
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request",type=Path,required=True)
    args=p.parse_args(argv)
    try:
        body=json.loads(args.request.read_text(encoding="utf-8"),
                        parse_constant=lambda _: (_ for _ in ()).throw(AdmissionError("nonfinite JSON")))
        result=score_request(body)
        print(json.dumps(result,sort_keys=True,allow_nan=False))
        return 0
    except (AdmissionError,OSError,ValueError,TypeError) as exc:
        print(json.dumps({"status":"INPUT_REJECTED","reason":str(exc)},sort_keys=True))
        return 2


if __name__=="__main__":
    sys.exit(main())
