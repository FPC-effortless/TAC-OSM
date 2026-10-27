#!/usr/bin/env python3
"""Strict publisher-timestamp -> source-aligned Model01 inference request adapter.

Works ONLY with original UCI374 local-naive source time-grid and unchanged
ten-minute 145-observation cadence. No cross-building/live-data qualification,
timezone conversion, DST treatment or new-source validation is implied.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

from model01_reference_api import canonical, decode_request, InputError, predict, load_original

SOURCE_ORIGIN = datetime(2016, 1, 11, 17, 0, 0)
SOURCE_ID = "UCI374_ENERGYDATA_COMPLETE_ORIGINAL_PUBLISHER_CLOCK"
SOURCE_SCHEMA = "model01-original-source-timestamped-window-v1"
SOURCE_KEYS = {"schema","source_series_id","observations"}
OBS_KEYS = {"observed_at","energy_wh"}
STEP = timedelta(minutes=10)
FMT = "%Y-%m-%d %H:%M:%S"


def parse_source_time(text):
    if not isinstance(text,str) or len(text)!=19:
        raise InputError("source timestamp must use YYYY-MM-DD HH:MM:SS, no zone or offset")
    try:
        dt = datetime.strptime(text,FMT)
    except ValueError as exc:
        raise InputError("invalid original publisher timestamp") from exc
    if dt.strftime(FMT)!=text or dt.second!=0 or dt.microsecond!=0:
        raise InputError("noncanonical source timestamp")
    return dt


def adapt(value):
    if (not isinstance(value,dict) or set(value)!=SOURCE_KEYS or
        value["schema"]!=SOURCE_SCHEMA or value["source_series_id"]!=SOURCE_ID):
        raise InputError("timestamp adapter requires exact UCI374 source-owned schema")
    obs=value["observations"]
    if not isinstance(obs,list) or len(obs)!=145:
        raise InputError("exactly 145 source-time observations required")
    readings=[]
    first=None
    for idx, row in enumerate(obs):
        if not isinstance(row,dict) or set(row)!=OBS_KEYS:
            raise InputError("source record must contain only timestamp and energy")
        dt=parse_source_time(row["observed_at"])
        if first is None:
            first=dt
        if dt != first+STEP*idx:
            raise InputError("missing, repeated, unordered or irregular source timestamp")
        readings.append(row["energy_wh"])
    last=first+STEP*144
    elapsed=last-SOURCE_ORIGIN
    if elapsed<timedelta(0) or elapsed%STEP!=timedelta(0):
        raise InputError("time outside original grid")
    source_index=int(elapsed//STEP)
    output={"schema":"model01-energy-source-aligned-v1",
            "source_aligned_step_index":source_index,
            "last_145_energy_wh":readings}
    decode_request(output)  # type, bounds, NaN, exact schema
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window",type=Path,required=True)
    p.add_argument("--predict-offline",action="store_true",
                   help="optional original Git-checkpoint prediction without HTTP/auth")
    a=p.parse_args(argv)
    try:
        source=json.loads(a.window.read_text(encoding="utf-8"),
                          parse_constant=lambda _ : (_ for _ in ()).throw(InputError("nonfinite JSON denied")))
        request=adapt(source)
        if a.predict_offline:
            m,q=load_original()
            response=predict(m,q,request)
            print(canonical({"request":request,"prediction":response,
                             "source_clock":"PUBLISHER_NAIVE_TIME_NO_LIVE_ASOF_PROOF"}))
        else:
            print(canonical(request))
        return 0
    except (InputError,OSError,ValueError,TypeError) as e:
        print(canonical({"status":"BLOCKED_OR_FAILED","reason":str(e)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
