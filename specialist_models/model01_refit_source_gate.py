#!/usr/bin/env python3
"""REFIT appliance-channel watt-to-Wh conservative 10-minute source gate.

Original REFIT cleaned houses have 'Time,Unix,Aggregate,Appliance1...9'.
They contain publisher-imputed zeros and forward-filled gaps; original physical
observation masks are unavailable in cleaned version. Thus no output can be
promoted to independently physically observed 10-minute energy ground truth.
No inference, training, source acquisition or TEST access performed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

FIELDS=("Time","Unix","Aggregate")+tuple("Appliance"+str(i) for i in range(1,10))
MAX_ROWS=120000
MAX_SAMPLE_INTERVAL_SECONDS=16
WINDOW_SECONDS=600

class SourceError(ValueError):pass

def read_original_like(text, channel, max_rows=MAX_ROWS):
    """Read bounded source-shaped CSV. Exactly one IAM channel, never 'Aggregate'."""
    if channel not in FIELDS[3:]:raise SourceError("explicit REFIT appliance channel required")
    reader=csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ())!=FIELDS:
        raise SourceError("REFIT cleaned CSV schema changed / channel mapping missing")
    obs=[]
    for row in reader:
        if len(obs)>=max_rows:raise SourceError("bounded source fixture exceeded")
        if not isinstance(row,dict) or any(v is None for v in row.values()):
            raise SourceError("truncated source row")
        try:
            timestamp=int(row["Unix"])
            watts=float(row[channel])
        except (ValueError,TypeError) as exc:raise SourceError("invalid source numeric") from exc
        if not (0<timestamp<5000000000) or not math.isfinite(watts) or not 0<=watts<=4000:
            raise SourceError("unsafe source sample")
        if obs and timestamp<=obs[-1][0]:
            raise SourceError("non-monotonic original Unix timestamp")
        obs.append((timestamp,watts))
    if len(obs)<2:raise SourceError("insufficient source samples")
    return obs

def integrate_10minute_windows(obs):
    """No filling source-observed >16s holes. Integrate held watts only for <=16s."""
    if len(obs)<2:raise SourceError("source must have two real-shaped samples")
    by_window={}
    dropped_gap_seconds=0
    for (begin,watts),(end,_) in zip(obs,obs[1:]):
        if type(begin) is not int or type(end) is not int or end<=begin:
            raise SourceError("invalid time ordering")
        delta=end-begin
        if delta>MAX_SAMPLE_INTERVAL_SECONDS:
            dropped_gap_seconds+=delta
            continue
        cursor=begin
        while cursor<end:
            bucket=cursor//WINDOW_SECONDS*WINDOW_SECONDS
            stop=min(end,bucket+WINDOW_SECONDS)
            seconds=stop-cursor
            if bucket not in by_window:
                by_window[bucket]=[0.,0.]
            by_window[bucket][0]+=watts*seconds/3600
            by_window[bucket][1]+=seconds
            cursor=stop
    complete=[]
    partial=0
    for bucket,(watt_hours,covered_seconds) in sorted(by_window.items()):
        if covered_seconds != WINDOW_SECONDS:
            partial+=1
            continue
        complete.append({"unix_window_start":bucket,
                         "energy_wh":watt_hours,
                         "coverage_seconds":int(covered_seconds),
                         "physical_observation_status":"CLEANED_REFIT_IMPUTATIONS_UNOBSERVABLE"})
    return {
        "complete_publisher_cleaned_ten_minute_windows":len(complete),
        "incomplete_windows_excluded":partial,
        "known_timestamp_gap_seconds_excluded":dropped_gap_seconds,
        "original_unobserved_gap_imputation_identifiable":False,
        "cleaned_meter_imputation_qualification":"UNVERIFIED",
        "windows":complete}

def qualification(channel, house_id, source_csv_bytes):
    if type(house_id) is not int or not 1<=house_id<=21 or house_id==14:
        raise SourceError("REFIT house ID unavailable or invalid")
    if len(source_csv_bytes)>25_000_000:
        raise SourceError("intake fixture capped at 25 MB; full REFIT requires external stream audit")
    source=source_csv_bytes.decode("utf-8-sig")
    samples=read_original_like(source,channel)
    score=integrate_10minute_windows(samples)
    return {
      "schema":"model01-refit-appliance-source-prequalification-v1",
      "source_project":"REFIT Electrical Load Measurements (Cleaned)",
      "publisher_doi":"10.15129/9ab14b0e-19ac-4279-938f-27f643078cec",
      "publisher_license":"CC BY 4.0; cite original REFIT authors",
      "source_bytes_sha256":hashlib.sha256(source_csv_bytes).hexdigest(),
      "source_house_id":house_id,"selected_iam_column":channel,
      "target":"single_monitored_appliance_channel_integrated_Wh_per_10_minutes",
      "target_equivalent_to_UCI374_total_appliances":False,
      "operator_source_time_available_at_independently_verified":False,
      "publisher_cleaned_gap_fill_ambiguity":True,
      "original_model_invoked":False,
      "trained_model":False,
      "independent_source_predictive_evaluation":False,
      "scientific_qualification":"NOT_QUALIFIED",
      "integrated":score,
      "warnings":[
         "only IAM channel measurements, not whole-house or total appliance energy",
         "appliance identity can vary by house and channel over time; original metadata verification required",
         "REFIT CLEANED publisher zero/forward filled missing records; physical data gaps cannot be reconstructed",
         "sampled power is approximated as constant until next <=16-second reading; not direct original Wh metering",
         "published historical source time is not original operational availability time",
         "full REFIT archive not accessed by this bounded adapter"]}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv",type=Path,required=True)
    p.add_argument("--house",type=int,required=True)
    p.add_argument("--iam-channel",required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    try:
        if args.out.exists():raise SourceError("refuse overwritten evidence")
        result=qualification(args.iam_channel,args.house,args.source_csv.read_bytes())
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
        print(json.dumps({k:result[k] for k in ("schema","source_bytes_sha256",
               "selected_iam_column","target_equivalent_to_UCI374_total_appliances",
               "scientific_qualification")},sort_keys=True))
        return 0
    except (SourceError,OSError,ValueError,TypeError,UnicodeError) as e:
        print(json.dumps({"status":"BLOCKED_OR_FAILED","reason":str(e)},sort_keys=True),file=sys.stderr)
        return 2
if __name__=="__main__":sys.exit(main())
