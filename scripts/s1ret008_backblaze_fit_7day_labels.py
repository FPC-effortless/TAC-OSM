#!/usr/bin/env python3
"""S1RET008 ORIGINAL real 2013 FIT-only seven-day outcome/censor audit.

Does not train, score or inspect the 2014 DEV/2015 TEST annual archives.
Never exports raw drive serials. Builds one bitmask of actual observed dates
per serial and another of actual last-operational failure days, rather than
caching all 5,091,501 raw observations in memory.

This is data-feasibility and source-authority evidence, not an external model
capability or calibration result. An observed zero-failure snapshot is NOT
necessarily a known negative for the coming seven days.
"""
from __future__ import annotations
from collections import Counter
from datetime import date
import csv,hashlib,io,json,urllib.request,zipfile
from pathlib import Path
from tac_osm.variants.s1ret008_backblaze_source import (
    PRIORITY_COUNTERS,_smart_val,validate_source_row)
LOCK=Path("research/parallel-six/S1RET008-BACKBLAZE-FIT-SOURCE-LOCK.json")
OUT=Path("research/diagnostics/s1ret008-first-FIT-7d-label-audit")
HORIZON=7
ARCHIVE_LIMIT=135_000_000

def download(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={
        "User-Agent":"S1RET008-verified-FIT-only-7d-labelling/1.0"}),
        timeout=170) as src:
        blob=src.read(ARCHIVE_LIMIT+1)
    if len(blob)>ARCHIVE_LIMIT:
        raise RuntimeError("source exceeds registered FIT archive limit")
    return blob

def run():
    authority=json.loads(LOCK.read_text())
    if (authority["DEV_archive_downloaded"] or
        authority["TEST_archive_downloaded"] or
        authority["FIT_year"]!=2013 or
        len(authority["member_source_date_receipts"])!=266):
        raise RuntimeError("source governance or original FIT day census changed")
    blob=download(authority["original_FIT_archive_url"])
    if hashlib.sha256(blob).hexdigest()!=authority["original_FIT_archive_sha256"]:
        raise RuntimeError("original publisher archive mutated after pre-FIT SHA freeze")
    archive=zipfile.ZipFile(io.BytesIO(blob))
    members=authority["member_source_date_receipts"]
    dates=[date.fromisoformat(r["date"]) for r in members]
    if any((dates[i+1]-dates[i]).days!=1 for i in range(len(dates)-1)):
        raise RuntimeError("source daily calendar has a gap: 7day censoring invalid")
    seen={};failed={};flagged={}
    prevalence_days=Counter()
    source_observations=0
    all_fails=0
    missing_or_invalid_priority=0
    for index,receipt in enumerate(members):
        with archive.open(receipt["member"]) as f:
            raw=f.read()
        if hashlib.sha256(raw).hexdigest()!=receipt["sha256"]:
            raise RuntimeError("source per-day checksum changed since FIT-only census")
        rows=0;failure_rows=0
        reader=csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        if reader.fieldnames is None:
            raise RuntimeError("no source feature schema")
        for row in reader:
            serial,x,p,priority,is_failed=validate_source_row(
                row,archive_day=dates[index])
            bit=1<<index
            if seen.get(serial,0)&bit:
                raise RuntimeError("duplicate same serial/day in real FIT source")
            seen[serial]=seen.get(serial,0)|bit
            if is_failed:
                failed[serial]=failed.get(serial,0)|bit
                failure_rows+=1
            if priority>=.75:
                flagged[serial]=flagged.get(serial,0)|bit
            rows+=1
        if rows!=receipt["total_observations"] or (
            failure_rows!=receipt["observed_failures"]):
            raise RuntimeError("original publisher member census mismatch")
        source_observations+=rows;all_fails+=failure_rows
    if source_observations!=authority["original_actual_drive_day_observations"] or (
        all_fails!=authority["original_real_failure_indicators"]):
        raise RuntimeError("FIT source rows or original failure prevalence changed")
    # Bound labels by calendar +7 days on THIS SAME drive, excluding
    # failure-day samples. Negative only if 7 later calendar days observed.
    positive=negative=censored=failed_today=postfailure_excluded=0
    relevance_positive=relevance_negative=0
    positive_unique=0
    eligible_long_drive=0
    for serial,presence in seen.items():
        fail=failed.get(serial,0);high=flagged.get(serial,0)
        if presence.bit_count()>=64:eligible_long_drive+=1
        had_positive=False
        first_failed_index=((fail&-fail).bit_length()-1) if fail else None
        while presence:
            low=presence & -presence
            day_index=low.bit_length()-1
            presence-=low
            if first_failed_index is not None and day_index>=first_failed_index:
                if day_index==first_failed_index:
                    failed_today+=1
                else:
                    postfailure_excluded+=1
                continue
            future_fail=(fail>>(day_index+1))&0x7f
            current_relevance=bool((high>>day_index)&1)
            if future_fail:
                positive+=1;had_positive=True
                relevance_positive+=int(current_relevance)
            elif ((seen[serial]>>(day_index+1))&0x7f)==0x7f:
                negative+=1
                relevance_negative+=int(current_relevance)
            else:
                censored+=1
        if had_positive:positive_unique+=1
    eligible=positive+negative
    if eligible==0 or positive<100:
        raise RuntimeError("original true +7day FIT event labels insufficient")
    result=dict(study="S1RET008",
        original_source_2013_FIT_archive_SHA256=authority["original_FIT_archive_sha256"],
        original_source_days=len(members),
        original_source_true_drive_days=source_observations,
        unique_real_drives=len(seen),
        unique_drives_with_at_least_64_calendar_observations=eligible_long_drive,
        original_true_failure_day_rows=all_fails,
        not_prediction_input_failure_day_rows=failed_today,
        records_after_first_failure_excluded=postfailure_excluded,
        known_future_7day_positive=positive,
        known_future_7day_healthy_negative=negative,
        unknown_or_source_censored=censored,
        classifiable_real_FIT_points=eligible,
        observed_7day_FIT_class_prevalence=positive/eligible,
        unique_drives_with_known_future_positive=positive_unique,
        observed_priority_high_fraction_among_known_positives=relevance_positive/positive,
        observed_priority_high_fraction_among_known_negatives=relevance_negative/negative
            if negative else None,
        source_7day_calendar_complete_followup_required=True,
        prediction_input_failure_flag_is_forbidden=True,
        source_daily_identity_unique=True,
        observed_maintenance_action_columns_available=False,
        models_trained=False,
        DEV_2014_archive_opened=False,
        TEST_2015_archive_opened=False,
        genuine_causal_replacement_decisions_measured=False)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"ORIGINAL-FIT-7DAY-FEASIBILITY.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("S1RET008_REAL_FIT_CENSORED_LABELS",json.dumps(result,sort_keys=True),flush=True)
    print("S1RET008_ORIGINAL_CENSORED_REAL_FIT_LABEL_AUDIT_PASS",flush=True)

if __name__=="__main__":
    run()
