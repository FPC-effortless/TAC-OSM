"""Actual Backblaze Drive Stats observation-time and future-label trust law.

No physically executed maintenance action is available in this source.
An individual daily row has source serial, date, SMART and an outcome
indicator `failure`. The indicator is NEVER included in neural inputs.

Future one-week status must be constructed from OTHER later source rows
for the SAME serial. A negative requires seven consecutive subsequent
observed days with no failure; missing snapshots/partition tail are CENSORED.
This module cannot acquire network data or secretly open DEV/TEST archives.
"""
from __future__ import annotations
from datetime import date,timedelta
import math
import numpy as np

SMART_FEATURES=(
    "smart_5_raw","smart_9_raw","smart_187_raw","smart_188_raw",
    "smart_197_raw","smart_198_raw","smart_1_raw","smart_7_raw",
    "smart_193_raw","smart_194_raw","smart_3_raw","smart_4_raw",
    "smart_12_raw","smart_199_raw"
)
PRIORITY_COUNTERS=("smart_5_raw","smart_187_raw","smart_197_raw","smart_198_raw")
WINDOW_DAYS=7
CENSORED=None

def outcome_status(day:date,observed_days:set[date],failure_days:set[date]):
    """1 true future failure; 0 complete subsequent healthy followup; None censor.

    Does not inspect any snapshot beyond the dataset partition boundary;
    must be invoked only after source allowed an entire FIT partition read.
    """
    if not isinstance(day,date) or day not in observed_days:
        raise ValueError("prediction date absent from trusted source")
    if day in failure_days:
        return CENSORED   # same-day known failing row never prediction input
    if not failure_days.issubset(observed_days):
        raise ValueError("failure row missing from source date set")
    after=tuple(day+timedelta(days=i) for i in range(1,WINDOW_DAYS+1))
    if any(x in failure_days for x in after):
        return 1
    if all(x in observed_days for x in after):
        return 0
    return CENSORED

def _smart_val(cell):
    if cell is None or str(cell).strip()=="":
        return None
    try:
        x=float(cell)
    except (ValueError,TypeError):
        return None
    if not math.isfinite(x) or x<0:
        return None
    return x

def authorized_observation(record:dict):
    """14D known-log-SMART and 14D observed-presence, priority known at t."""
    if not isinstance(record,dict):
        raise ValueError("actual source record required")
    # Reject target-bearing input dictionaries, not just ignore unexpected
    # sources. The source reader must pass just the explicit SMART subset.
    extra=set(record)-set(SMART_FEATURES)
    if extra:
        raise ValueError(f"unauthorized non-SMART source paths: {sorted(extra)}")
    values=[];presence=[]
    for key in SMART_FEATURES:
        val=_smart_val(record.get(key))
        values.append(math.log1p(val) if val is not None else 0.)
        presence.append(float(val is not None))
    priority=float(any(
        _smart_val(record.get(key)) is not None and
        _smart_val(record.get(key))>0
        for key in PRIORITY_COUNTERS))
    return np.asarray(values,dtype=np.float32),np.asarray(
        presence,dtype=np.float32),priority

def validate_source_row(source_row,*,archive_day):
    if not isinstance(source_row,dict) or source_row.get("date")!=archive_day.isoformat():
        raise ValueError("snapshot date not authenticated")
    serial=source_row.get("serial_number")
    if not isinstance(serial,str) or len(serial)<1:
        raise ValueError("missing actual entity identity")
    if source_row.get("failure") not in ("0","1"):
        raise ValueError("unverified true observed failure indicator")
    clean={key:source_row.get(key) for key in SMART_FEATURES}
    value,presence,priority=authorized_observation(clean)
    return serial,value,presence,priority,(source_row["failure"]=="1")

def known_source_entity_slots(serials):
    """Stable identity map per 8-entity cohort; never outcome-dependent."""
    if len(serials)!=8 or len(set(serials))!=8 or any(
            not isinstance(x,str) or not x for x in serials):
        raise ValueError("eight distinct legitimate drive IDs required")
    return {name:i for i,name in enumerate(sorted(serials))}
