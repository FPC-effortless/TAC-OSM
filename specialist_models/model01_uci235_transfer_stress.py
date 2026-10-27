#!/usr/bin/env python3
"""Cross-source UCI235 transfer stress-test of immutable UCI374 ridge predictor.

PROSPECTIVELY FIXED before any UCI235 source rows are read in this study.
Different target: UCI235 whole-house active Wh/10min, UCI374 appliances Wh.
Thus NOT independent same-task confirmation. No refitting/recalibration,
protected original UCI374 TEST access, automatic timezone mapping or customer claim.
"""
from __future__ import annotations
from datetime import datetime, timedelta
import hashlib
import io
import json
import math
from pathlib import Path,PurePosixPath
import sys
import urllib.request
import zipfile

import model01_reference_api as api

URL="https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
START=datetime(2006,12,17,0,0)
EVAL_START=datetime(2007,1,1,0,0)
END=datetime(2007,1,16,0,0)
MAX_COMPRESSED_BYTES=30_000_000
MAX_ORIGINAL_MEMBER_BYTES=150_000_000
MAX_WATTS=100_000
H=6
LOOKBACK=145

class SourceError(ValueError):
    pass

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False)

def original_zip():
    r=urllib.request.Request(URL,headers={"User-Agent":"TAC-OSM-external-source-qualification/1.0"})
    with urllib.request.urlopen(r,timeout=35) as response:
        if response.geturl()!=URL:
            raise SourceError("unexpected publisher redirect")
        data=bytearray()
        for chunk in iter(lambda:response.read(1024*1024),b""):
            data.extend(chunk)
            if len(data)>MAX_COMPRESSED_BYTES:
                raise SourceError("compressed source exceeds fixed budget")
    return bytes(data)

def ten_minute_energy(original_bytes):
    if not original_bytes or len(original_bytes)>MAX_COMPRESSED_BYTES:
        raise SourceError("source archive outside fixed bound")
    blocks={}
    source_header=False
    prior_minute=None
    accepted_minutes=missing_minutes=0
    with zipfile.ZipFile(io.BytesIO(original_bytes)) as archive:
        matches=[]
        for info in archive.infolist():
            path=PurePosixPath(info.filename)
            if not info.is_dir() and path.name=="household_power_consumption.txt":
                if ".." in path.parts or info.filename.startswith("/") or info.file_size>MAX_ORIGINAL_MEMBER_BYTES:
                    raise SourceError("unsafe oversized publisher source")
                matches.append(info)
        if len(matches)!=1:
            raise SourceError("expected exactly one original UCI235 publisher text member")
        with archive.open(matches[0]) as raw:
            stream=io.TextIOWrapper(raw,encoding="utf-8-sig")
            header=stream.readline().strip().split(";")
            if header[:3]!=["Date","Time","Global_active_power"]:
                raise SourceError("UCI235 source measurement schema changed")
            source_header=True
            for line in stream:
                parts=line.rstrip("\n\r").split(";")
                if len(parts)<3:
                    raise SourceError("source original row truncated")
                try:
                    stamp=datetime.strptime(parts[0]+" "+parts[1],"%d/%m/%Y %H:%M:%S")
                except ValueError as e:
                    raise SourceError("invalid publisher source timestamp") from e
                if stamp<START:continue
                if stamp>=END:break  # future publisher member rows left unparsed
                if prior_minute and stamp-prior_minute!=timedelta(minutes=1):
                    raise SourceError("publisher source skips or duplicates minute timestamps")
                prior_minute=stamp
                if stamp.second!=0:
                    raise SourceError("unaligned original minute timestamp")
                group=stamp.replace(minute=(stamp.minute//10)*10)
                if group not in blocks:
                    blocks[group]=[0.,0,False]
                record=blocks[group]
                record[1]+=1
                raw_power=parts[2].strip()
                if raw_power in ("?",""):
                    record[2]=True
                    missing_minutes+=1
                else:
                    try:
                        kw=float(raw_power)
                    except ValueError as e:
                        raise SourceError("non-numeric source power") from e
                    if not math.isfinite(kw) or not 0<=kw*1000<=MAX_WATTS:
                        raise SourceError("invalid original source power")
                    # Original minute-averaged kW * (1000 W/kW) * (1/60 h)
                    record[0]+=kw*1000/60
                accepted_minutes+=1
    if not source_header or accepted_minutes<20000:
        raise SourceError("source exposure insufficient")
    expected=[START+timedelta(minutes=10*i) for i in range(int((END-START).total_seconds()/600))]
    # Missing/short blocks preserved as None: never impute unknown source values.
    values=[]
    for stamp in expected:
        energy,count,missing=blocks.get(stamp,(0.,0,True))
        values.append(energy if count==10 and not missing else None)
    return values,{"accepted_minute_rows":accepted_minutes,
                   "missing_power_minute_rows":missing_minutes,
                   "complete_ten_minute_blocks":sum(x is not None for x in values),
                   "missing_or_incomplete_ten_minute_blocks":sum(x is None for x in values),
                   "raw_timestamp_semantics":"original source naive local wall time, timezone not verified"}

def compare(source_bytes):
    frozen,q=api.load_original()
    values,source_info=ten_minute_energy(source_bytes)
    days={}
    n=hits=0
    ridge_err=persistence_err=day_err=0.
    for i in range(LOOKBACK-1,len(values)-H):
        stamp=START+timedelta(minutes=10*i)
        if stamp<EVAL_START:
            continue
        window=values[i-(LOOKBACK-1):i+1]
        y=values[i+H]
        if y is None or any(x is None for x in window):
            continue
        # This index reuses the original learned 144-step phase on a different
        # household and origin; no transfer-time phase calibration/selection.
        request={"schema":"model01-energy-source-aligned-v1",
                 "source_aligned_step_index":i,
                 "last_145_energy_wh":window}
        p=api.predict(frozen,q,request)["prediction_wh"]
        ridge_err+=abs(p-y)
        persistence_err+=abs(values[i]-y)
        day_err+=abs(values[i-144]-y)
        hits+=int(p-q<=y<=p+q)
        n+=1
        daily=days.setdefault(stamp.date().isoformat(),{"n":0,"ridge_abs":0.,"persist_abs":0.})
        daily["n"]+=1
        daily["ridge_abs"]+=abs(p-y)
        daily["persist_abs"]+=abs(values[i]-y)
    if n<300 or len(days)<7:
        raise SourceError("too few complete independent-site exposure examples")
    return {
        "schema":"model01-uci235-independent-source-transfer-stress-v1",
        "study":"FROZEN_UCI374_MODEL_ON_DIFFERENT_HOUSEHOLD_AND_TARGET",
        "source_url":URL,
        "source_doi":"10.24432/C58K54",
        "source_license":"CC BY 4.0; original Hebrail & Berard attribution required",
        "source_zip_sha256":hashlib.sha256(source_bytes).hexdigest(),
        "original_model_sha256":api.ORIGINAL_MODEL_SHA256,
        "old_source":"UCI374 single-household appliances consumption Wh per 10min",
        "transfer_source":"UCI235 different-household total active energy Wh per 10min",
        "harmonization":"minute-average kilowatts integrated over ten minutes; NOT identical target",
        "source_index_phase":"relative external START, unaligned to original household clock",
        "external_source_rows":source_info,
        "eval_window_start":EVAL_START.isoformat(),"eval_window_end_exclusive":END.isoformat(),
        "horizon_minutes":60,"complete_scored_examples":n,
        "scored_days":len(days),
        "frozen_original_ridge_mae_wh":ridge_err/n,
        "external_persistence_mae_wh":persistence_err/n,
        "external_same_time_prev_day_mae_wh":day_err/n,
        "observed_original_calibrated_90_interval_coverage":hits/n,
        "original_calibrated_interval_halfwidth_wh":q,
        "per_day_means":[{"day":d,"n":v["n"],
                          "ridge_mae_wh":v["ridge_abs"]/v["n"],
                          "persistence_mae_wh":v["persist_abs"]/v["n"]}
                         for d,v in sorted(days.items())],
        "trained_or_calibrated_during_transfer":False,
        "target_equivalence":False,"independent_same_task_confirmation":False,
        "scientific_promotion":"NOT_QUALIFIED",
        "interpretation":"out-of-domain transfer stress only; new target semantics and source origin prevent same-task superiority claim",
        "warnings":["source is published retrospective data, operational available-at time unverified",
                    "all development choices frozen before opening source rows in this run; results now exposed",
                    "missing ten-minute blocks and any crossing 145-step input windows are dropped without imputation",
                    "naive local timestamps over December-January do not establish timezone mapping",
                    "new site, different target; no apples-to-apples model accuracy superiority",
                    "overlapping horizons, correlated minutes and 15 calendar days are not independent iid cases",
                    "original nominal-90 interval is not transfer-calibrated and no guarantee applies"]}

def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip-file",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    try:
        if args.out.exists():
            raise SourceError("refusing original evidence overwrite")
        payload=args.zip_file.read_bytes() if args.zip_file else original_zip()
        result=compare(payload)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
        print(canonical({k:result[k] for k in ("schema","source_zip_sha256",
             "complete_scored_examples","scored_days",
             "frozen_original_ridge_mae_wh","external_persistence_mae_wh",
             "external_same_time_prev_day_mae_wh","observed_original_calibrated_90_interval_coverage",
             "independent_same_task_confirmation")}))
        return 0
    except (OSError,ValueError,TypeError,UnicodeError,zipfile.BadZipFile,
            SourceError) as e:
        print(canonical({"status":"BLOCKED_OR_FAILED","reason":str(e)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
