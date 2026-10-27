#!/usr/bin/env python3
"""Original Backblaze 2013, frozen 46-day cohort feasibility — SOURCE ONLY.

Before any training, inspect 2013-10-01 .. 2013-11-15 original CSV days,
anchoring an enrollment cohort to day 1. November 16 onward is held sealed
(the compressed publisher ZIP includes it but its contents are NOT parsed).
Only aggregate sample sizes and censored outcomes are exported; no serials,
raw CSV, train labels, learned weights, TEST scores or ability claim.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import backblaze_2013_source_probe as original

SOURCE_SHA = "7f5a53e79b16e695b4b034955806bb3bb194534b169f6eca460dfd3dc48096fe"
FIRST = date(2013,10,1)
FIT_END = date(2013,11,1)
EXPOSED_END = date(2013,11,16)
HORIZON_DAYS = 7
ROW_CAP = 100_000
MEMBER_CAP = original.MAX_MEMBER

class CohortError(ValueError):
    pass

def days():
    return [FIRST+timedelta(days=i) for i in range((EXPOSED_END-FIRST).days)]

def count_source(archive_bytes):
    if original.sha(archive_bytes)!=SOURCE_SHA:
        raise CohortError("original publisher ZIP hash changed")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
        wanted={d.isoformat()+".csv":d for d in days()}
        members={}
        for m in z.infolist():
            if m.is_dir():
                continue
            path=PurePosixPath(m.filename)
            if ".." in path.parts or m.filename.startswith("/"):
                raise CohortError("unsafe original source ZIP member")
            if path.name in wanted:
                if path.name in members:
                    raise CohortError("duplicate original daily source path")
                if m.file_size > MEMBER_CAP:
                    raise CohortError("source daily file too large")
                members[path.name]=m
        if set(members)!=set(wanted):
            raise CohortError("missing source days in predeclared exposed period")
        enrolled=None
        timeline={}
        publisher_failures=0
        daily=[]
        missing={"smart_5_raw":0,"smart_9_raw":0,"smart_197_raw":0,"smart_198_raw":0}
        for day in days():
            name=day.isoformat()+".csv"
            with z.open(members[name]) as f:
                chunk=f.read(MEMBER_CAP+1)
            if len(chunk)>MEMBER_CAP:
                raise CohortError("decompression budget exceeded")
            # Future raw drive identifiers stay in-process only.
            reader=csv.DictReader(io.StringIO(chunk.decode("utf-8-sig")))
            fields=set(reader.fieldnames or [])
            if not {"date","serial_number","model","failure"}.issubset(fields):
                raise CohortError("source provenance schema changed")
            observed={}
            n=flag=0
            for row in reader:
                if n>=ROW_CAP:
                    raise CohortError("exceeded daily source row budget")
                if row["date"]!=day.isoformat():
                    raise CohortError("publisher source-date mismatch")
                key=(row["serial_number"],row["model"])
                if not key[0] or not key[1] or key in observed:
                    raise CohortError("missing/duplicate source-owned entity identity")
                if row["failure"] not in ("0","1"):
                    raise CohortError("invalid recorded failure")
                is_failure=(row["failure"]=="1")
                observed[key]=is_failure
                n+=1
                flag+=int(is_failure)
                for f in missing:
                    if f not in fields or not (row.get(f) or "").strip():
                        missing[f]+=1
            if enrolled is None:
                # Prospective-like cohort rule fixed BEFORE examining later days,
                # but this retrospective audit cannot prove a real-time freeze.
                enrolled={key for key,failed in observed.items() if not failed}
                if len(enrolled)<1:
                    raise CohortError("no functioning enrolled source entities")
                timeline={key:{} for key in enrolled}
            for key in enrolled:
                if key in observed:
                    timeline[key][day]=observed[key]
            publisher_failures+=flag
            daily.append({"date":day.isoformat(),"publisher_drive_day_rows":n,
                          "publisher_failure_flags":flag,
                          "original_source_csv_sha256":hashlib.sha256(chunk).hexdigest(),
                          "enrolled_rows_present":sum(key in observed for key in enrolled)})
        if enrolled is None:
            raise CohortError("empty source")
    # Audit machine-state constraints: once an enrolled drive is marked failed,
    # any subsequent day that it appears is anomalous, not a new healthy drive.
    first_fail={}
    for key,series in timeline.items():
        failures=[day for day,value in series.items() if value]
        if len(failures)>1:
            raise CohortError("duplicate physical failure after first")
        if failures:
            failure=failures[0]
            if any(day>failure for day in series):
                raise CohortError("source entity returned after recorded failure")
            first_fail[key]=failure

    def phase_outcomes(a,b):
        total=pos=neg=censored=0
        unique_positive_entities=set()
        for key,series in timeline.items():
            for day in sorted(series):
                if not a<=day<b or series[day]:
                    continue
                total+=1
                status="negative"
                for j in range(1,HORIZON_DAYS+1):
                    next_day=day+timedelta(days=j)
                    if next_day>=b or next_day not in series:
                        status="censored";break
                    if series[next_day]:
                        status="positive"
                        unique_positive_entities.add(key)
                        break
                if status=="censored":censored+=1
                elif status=="positive":pos+=1
                else:neg+=1
        if total!=pos+neg+censored:
            raise CohortError("invalid cohort outcome partition")
        return {"source_observation_windows":total,
                "provisional_positive_windows":pos,
                "complete_negative_windows":neg,
                "right_censored_windows":censored,
                "unique_drives_with_positive_windows":len(unique_positive_entities),
                "horizon_days":HORIZON_DAYS,
                "label_provenance":"PUBLISHER_RECORDED_FLAGS_NOT_INDEPENDENTLY_ADJUDICATED"}

    fit=phase_outcomes(FIRST,FIT_END)
    dev=phase_outcomes(FIT_END,EXPOSED_END)
    return {"schema":"model02-backblaze-retrospective-feasibility-v1",
            "source_url":original.ORIGINAL_ZIP,
            "source_zip_sha256":SOURCE_SHA,
            "publisher_original_2013_csv_days_parsed":len(daily),
            "exposure_window":[FIRST.isoformat(), EXPOSED_END.isoformat()],
            "heldout_begin_not_parsed":EXPOSED_END.isoformat(),
            "initial_functioning_enrolled_drives":len(enrolled),
            "recorded_failure_flags_all_46_source_days":publisher_failures,
            "enrolled_unique_failure_drives_46_days":len(first_fail),
            "publisher_drive_days_all":sum(d["publisher_drive_day_rows"] for d in daily),
            "missing_selected_smart_values_total":missing,
            "FIT_source_feasibility_exposed":fit,"DEV_source_feasibility_exposed":dev,
            "daily_source_digests":daily,
            "actual_source_time_availability_verified":False,
            "source_enrollment_precommitted_in_operational_reality":False,
            "heldout_evaluation_performed":False,"trained_model":False,
            "scientific_qualification":"NOT_QUALIFIED",
            "disclosures":[
                "source audit data are now exposed; no independent cohort remains inside audited days",
                "overlapping 7-day observation windows are statistically dependent",
                "the ZIP was downloaded in full, but member content at/after heldout boundary was not parsed",
                "retrospective publisher file release not equivalent to actual as-of real-time publication",
                "sample adequacy does not imply label/source licensing or causal predictive advantage"
            ]}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip-file",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    try:
        if args.out.exists():
            raise CohortError("refusing to overwrite receipt")
        payload=args.zip_file.read_bytes() if args.zip_file else original.fetch_zip()
        data=count_source(payload)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as f:
            json.dump(data,f,indent=2,sort_keys=True)
        print(json.dumps({k:data[k] for k in (
            "schema","initial_functioning_enrolled_drives",
            "recorded_failure_flags_all_46_source_days",
            "enrolled_unique_failure_drives_46_days",
            "FIT_source_feasibility_exposed","DEV_source_feasibility_exposed",
            "scientific_qualification")},sort_keys=True))
        return 0
    except (CohortError,OSError,ValueError,UnicodeError,zipfile.BadZipFile) as exc:
        print(json.dumps({"status":"BLOCKED_OR_FAILED","reason":str(exc)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
