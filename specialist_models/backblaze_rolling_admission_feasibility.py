#!/usr/bin/env python3
"""Post-exposure rolling entity-admission feasibility on original 2013 Backblaze.

Rule: new physical identity is eligible for prediction after SEVEN consecutive
observed nonfailure daily source records, with no hindsight on future results.
Only source-exposed 2013-10-01 through 2013-11-15. No model fit, raw identity
export, independent evaluation, publisher time provenance or heldout access.
This follow-up hypothesis was designed AFTER exposed 46-day first-day results;
therefore it is diagnostic, not an independent pre-registered source trial.
"""
from __future__ import annotations
import argparse
import csv
from datetime import date,timedelta
import io
import json
from pathlib import Path,PurePosixPath
import sys
import zipfile

import backblaze_2013_cohort_feasibility as source
import backblaze_2013_source_probe as original

WARMUP_DAYS = 7
HORIZON_DAYS = source.HORIZON_DAYS
MAX_ROWS = 100_000

class SourceError(ValueError):
    pass

def read_timeline(payload):
    if original.sha(payload)!=source.SOURCE_SHA:
        raise SourceError("publisher original ZIP SHA differs from 2013 pinned receipt")
    days=source.days()
    wanted={d.isoformat()+".csv":d for d in days}
    timeline={}
    observed_today_counts=[]
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        members={}
        for info in z.infolist():
            if info.is_dir():continue
            parts=PurePosixPath(info.filename)
            if ".." in parts.parts or info.filename.startswith("/"):
                raise SourceError("unsafe original ZIP entry")
            if parts.name in wanted:
                if parts.name in members or info.file_size>original.MAX_MEMBER:
                    raise SourceError("duplicate / over-budget original daily CSV")
                members[parts.name]=info
        if set(members)!=set(wanted):
            raise SourceError("source-selected 46-day span incomplete")
        for d in days:
            filename=d.isoformat()+".csv"
            with z.open(members[filename]) as f:
                raw=f.read(original.MAX_MEMBER+1)
            if len(raw)>original.MAX_MEMBER:
                raise SourceError("daily decompression budget exceeded")
            reader=csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
            if not {"date","serial_number","model","failure"}.issubset(reader.fieldnames or []):
                raise SourceError("invalid publisher source schema")
            seen=set()
            failures=0
            for row in reader:
                if len(seen)>=MAX_ROWS:
                    raise SourceError("source daily row budget exceeded")
                if row["date"]!=d.isoformat() or row["failure"] not in ("0","1"):
                    raise SourceError("publisher day/failure mismatch")
                key=row["serial_number"],row["model"]
                if not all(key) or key in seen:
                    raise SourceError("missing/duplicate physical source identity")
                seen.add(key)
                failed=row["failure"]=="1"
                timeline.setdefault(key,{})[d]=failed
                failures+=failed
            observed_today_counts.append({"day":d.isoformat(),"publisher_rows":len(seen),
                                          "publisher_failure_flags":failures})
    for key,seen in timeline.items():
        failures=[d for d,fail in seen.items() if fail]
        if len(failures)>1 or any(day>failures[0] for day in seen) if failures else False:
            raise SourceError("inconsistent source physical failure history")
    return timeline,observed_today_counts

def admitted_at(timeline,day):
    if day not in timeline or timeline[day]:
        return False
    # Observed evidence ONLY from t and its preceding six days.
    return all((day-timedelta(days=i) in timeline and
                not timeline[day-timedelta(days=i)])
               for i in range(WARMUP_DAYS))

def label_future(timeline,day,phase_end):
    for j in range(1,HORIZON_DAYS+1):
        t=day+timedelta(days=j)
        if t>=phase_end or t not in timeline:
            return "CENSORED"
        if timeline[t]:
            return "POSITIVE"
    return "NEGATIVE"

def audit(timeline,daily):
    def period(start,end):
        windows=positive=negative=censored=0
        unique_positives=set()
        first_admissions=set()
        for key,series in timeline.items():
            for d in sorted(series):
                if not start<=d<end or not admitted_at(series,d):
                    continue
                windows+=1
                first_admissions.add(key)
                label=label_future(series,d,end)
                if label=="POSITIVE":
                    positive+=1;unique_positives.add(key)
                elif label=="NEGATIVE":negative+=1
                else:censored+=1
        if positive+negative+censored!=windows:
            raise SourceError("impossible census count")
        return {"source_eligible_seven_day_warmup_windows":windows,
                "observed_future_failure_windows":positive,
                "complete_observed_nonfailure_windows":negative,
                "right_censored_windows":censored,
                "unique_drives_with_future_failure_windows":len(unique_positives),
                "unique_admitted_drives":len(first_admissions)}

    fit=period(source.FIRST,source.FIT_END)
    dev=period(source.FIT_END,source.EXPOSED_END)
    return {"schema":"backblaze-rolling-admission-exposed-feasibility-v1",
            "study":"POST_EXPOSURE_ROLLING_ENTITY_ADMISSION_DIAGNOSTIC",
            "original_zip_sha256":source.SOURCE_SHA,
            "exposed_dates_inclusive":[str(source.FIRST),str(source.EXPOSED_END-timedelta(days=1))],
            "future_members_at_or_after":str(source.EXPOSED_END),
            "enrollment_eligibility":"7 consecutive past-or-current observed nonfailure days",
            "publisher_source_rows":sum(d["publisher_rows"] for d in daily),
            "publisher_failure_flags":sum(d["publisher_failure_flags"] for d in daily),
            "distinct_publisher_entity_ids":len(timeline),
            "FIT_exposed":fit,"DEV_exposed":dev,
            "trained_model":False,"independent_evaluation":False,
            "actual_publisher_asof_verified":False,
            "protected_future_content_parsed":False,
            "scientific_qualification":"NOT_QUALIFIED",
            "warnings":[
                "rolling admission rule was proposed after first-day cohort counts already exposed",
                "at-risk windows share physical drives and future horizons, so events are NOT independent",
                "publisher recorded daily failure is not independently adjudicated disk failure",
                "full historic source ZIP is downloaded; future entries are not opened/parsed",
                "time-availability and source terms need independent authorization for deployment use"
            ]}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.out.exists():raise SourceError("refuse to overwrite original record")
        timeline,daily=read_timeline(original.fetch_zip())
        result=audit(timeline,daily)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2,sort_keys=True))
        print(json.dumps(result,sort_keys=True))
        return 0
    except (SourceError,OSError,UnicodeError,ValueError,zipfile.BadZipFile) as exc:
        print(json.dumps({"status":"BLOCKED_OR_FAILED","reason":str(exc)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    sys.exit(main())
