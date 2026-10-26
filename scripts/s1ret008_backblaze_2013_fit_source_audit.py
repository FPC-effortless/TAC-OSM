#!/usr/bin/env python3
"""S1RET008 Backblaze REAL drive FIT-only 2013 source authorization.

Source: https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
2013 annual archive source is 266 distinct original dated CSV snapshots.
2014 DEV and 2015 TEST live in DISTINCT ZIP archives; this script does not
construct their URLs and NEVER downloads/parses them.

2013 archive SHA and per-day member SHA are committed into source ledger
before anyone trains. Report aggregate statistics only; no raw drive serials.
Input failure field is always quarantined from prediction features; future
failure labels require prospective +1..+7 day windows with complete followup.
The dataset has no logged executed repair actions. Models receive explicit
absence of actions, not falsely observed interventions.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import date
import csv,hashlib,io,json,re,urllib.request,zipfile
from pathlib import Path
URL="https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_2013.zip"
MAX_BYTES=135_000_000
PATTERN=re.compile(r"(?:^|/)(20\d{2}-\d{2}-\d{2})\.csv$")
PREFFERED=("smart_5_raw","smart_9_raw","smart_187_raw","smart_188_raw",
 "smart_197_raw","smart_198_raw","smart_1_raw","smart_7_raw",
 "smart_193_raw","smart_194_raw","smart_3_raw","smart_4_raw",
 "smart_12_raw","smart_199_raw","smart_10_raw","smart_190_raw",
 "smart_192_raw","smart_196_raw","smart_200_raw")

def main():
    req=urllib.request.Request(URL,headers={
        "User-Agent":"S1RET008-original-2013-FIT-only-source-authority/1.0"})
    with urllib.request.urlopen(req,timeout=160) as stream:
        data=stream.read(MAX_BYTES+1)
    if len(data)>MAX_BYTES:
        raise RuntimeError("unexpected oversized original 2013 FIT source")
    sha=hashlib.sha256(data).hexdigest()
    archive=zipfile.ZipFile(io.BytesIO(data))
    selected=[]
    for member in archive.infolist():
        found=PATTERN.search(member.filename)
        if found and not member.is_dir():
            if found.group(1)[:4]!="2013":raise RuntimeError("non-FIT date file in original ZIP")
            selected.append((date.fromisoformat(found.group(1)),member))
    selected.sort(key=lambda p:p[0])
    if len(selected)<200 or len(selected)>366:
        raise RuntimeError("source daily census outside expected 2013 scope")
    if len(set(day for day,_ in selected))!=len(selected):
        raise RuntimeError("duplicate source chronology found")
    observations=0; failures=0;invalid=0;serials=set()
    serial_failure=set()
    per_drive_dates=defaultdict(set)
    source_members=[];fields=None;observed_feature_counts=Counter()
    for day,member in selected:
        digest=hashlib.sha256()
        rows=0;dayfails=0;features_for_day=set()
        with archive.open(member) as raw:
            header=csv.DictReader(io.TextIOWrapper(raw,encoding="utf-8-sig",newline=""))
            cols=header.fieldnames
            if not cols or not {"date","serial_number","failure"}.issubset(set(cols)):
                raise RuntimeError(f"invalid original real drive snapshot schema {day}")
            if fields is None:fields=cols
            for field in PREFFERED:
                if field in cols:features_for_day.add(field)
            # Per-member SHA is computed on the entire original bytes below.
            for record in header:
                if record["date"]!=day.isoformat() or record["failure"] not in ("0","1"):
                    invalid+=1;continue
                serial=record["serial_number"]
                if not serial:invalid+=1;continue
                observations+=1;rows+=1;serials.add(serial)
                per_drive_dates[serial].add(day)
                if record["failure"]=="1":
                    failures+=1;dayfails+=1;serial_failure.add(serial)
            if rows==0:raise RuntimeError("empty authenticated FIT snapshot")
        with archive.open(member) as datafile:
            for block in iter(lambda:datafile.read(1024*1024),b""):
                digest.update(block)
        source_members.append(dict(date=day.isoformat(),
            member=member.filename,uncompressed_bytes=member.file_size,
            sha256=digest.hexdigest(),total_observations=rows,
            observed_failures=dayfails))
        observed_feature_counts.update(features_for_day)
    if invalid:
        raise RuntimeError(f"{invalid} malformed original serial/date/failure source rows")
    # Features may be missing for entire model families; decide availability
    # on REAL FIT only, with a persistent missingness indicator contract.
    features=[f for f in PREFFERED if observed_feature_counts[f]>=int(len(selected)*.9)]
    if len(features)<14:
        raise RuntimeError(f"fewer than 14 broadly present real FIT SMART features: {features}")
    long_drives=sum(len(days)>=64 for days in per_drive_dates.values())
    report=dict(study="S1RET008",source="Backblaze Drive Stats",
        source_terms="cite Backblaze; free derivative research; do not resell raw data",
        original_FIT_archive_url=URL,original_FIT_archive_sha256=sha,
        archive_downloaded_bytes=len(data),FIT_year=2013,
        DEV_year=2014,TEST_year=2015,
        original_FIT_day_count=len(selected),
        member_source_date_receipts=source_members,
        original_actual_drive_day_observations=observations,
        original_real_failure_indicators=failures,
        source_unique_drive_count=len(serials),
        source_unique_failed_drives=len(serial_failure),
        source_drives_with_64_or_more_days=long_drives,
        original_FIT_schema_fields=fields,
        available_real_FIT_smart_features=features,
        modeling_feature_count=14,model_input_failure_field_allowed=False,
        complete_futuristic_plus_7_day_followup_required=True,
        source_identity_fields_were_used_only_for_aggregate_statistics=True,
        live_raw_serials_exported=False,
        DEV_archive_downloaded=False,TEST_archive_downloaded=False,
        original_models_trained=False,original_models_scored=False)
    out=Path("research/diagnostics/s1ret008-original-2013-FIT-source")
    out.mkdir(parents=True,exist_ok=True)
    (out/"FIT-ORIGINAL-AUTHORITY-LOCK.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("S1RET008_AUTHENTIC_FIT_SUMMARY",json.dumps({
        k:report[k] for k in ("original_FIT_archive_sha256",
             "original_FIT_day_count","original_actual_drive_day_observations",
             "original_real_failure_indicators","source_unique_drive_count",
             "source_unique_failed_drives","source_drives_with_64_or_more_days",
             "available_real_FIT_smart_features",
             "DEV_archive_downloaded","TEST_archive_downloaded")},
             sort_keys=True),flush=True)
    print("S1RET008_ORIGINAL_2013_FIT_ONLY_SOURCE_PASS",flush=True)

if __name__=="__main__":
    main()
