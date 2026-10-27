#!/usr/bin/env python3
"""Bounded ORIGINAL Backblaze 2013 source-only audit (not an eligible fitted cohort).

Uses the Backblaze-published 2013 ZIP (about 81 MB) and reads ONLY 14
exposed-development CSV days. Outputs publisher archive/date hashes, schema
and failure denominators; NO serials, raw CSV or train/eval labels published.
CPU-only GitHub Actions. This is NOT evidence of historical public-as-of time.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import urllib.request
import zipfile


ORIGINAL_ZIP = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_2013.zip"
CITATION = "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data"
FIRST = date(2013,10,1)
DAYS = 14
MAX_ZIP = 110_000_000
MAX_MEMBER = 15_000_000
MAX_DAILY_ROWS = 100_000
SMART = ("smart_5_raw","smart_9_raw","smart_197_raw","smart_198_raw")


class SourceError(ValueError):
    pass


def sha(bytes_):
    return hashlib.sha256(bytes_).hexdigest()


def fetch_zip():
    req=urllib.request.Request(ORIGINAL_ZIP, headers={"User-Agent":"TAC-OSM-source-only-audit/1.0"})
    with urllib.request.urlopen(req,timeout=35) as resp:
        if resp.geturl()!=ORIGINAL_ZIP:
            raise SourceError("refusing unapproved source redirect")
        result=bytearray()
        for block in iter(lambda:resp.read(1024*1024),b""):
            result.extend(block)
            if len(result)>MAX_ZIP:
                raise SourceError("source ZIP over memory/network budget")
    return bytes(result)


def scan(zip_bytes):
    if not zip_bytes or len(zip_bytes)>MAX_ZIP:
        raise SourceError("source archive empty or outside budget")
    selected_days=[FIRST+timedelta(days=i) for i in range(DAYS)]
    wanted={d.isoformat()+".csv":d for d in selected_days}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        selected={}
        for info in archive.infolist():
            parts=PurePosixPath(info.filename)
            if info.is_dir() or ".." in parts.parts or info.filename.startswith("/"):
                raise SourceError("unsafe ZIP entry")
            if parts.name in wanted:
                if info.file_size>MAX_MEMBER:
                    raise SourceError("source CSV exceeds bounded daily size")
                if parts.name in selected:
                    raise SourceError("duplicate publisher daily file")
                selected[parts.name]=info
        if set(selected)!=set(wanted):
            raise SourceError("missing one or more original October 2013 day files")
        daily=[]
        for name,d in sorted(wanted.items()):
            with archive.open(selected[name]) as raw:
                content=raw.read(MAX_MEMBER+1)
            if len(content)>MAX_MEMBER:
                raise SourceError("overlarge decompressed CSV member")
            with io.StringIO(content.decode("utf-8-sig")) as stream:
                reader=csv.DictReader(stream)
                original_fields=set(reader.fieldnames or [])
                required={"date","serial_number","model","failure"}
                if not required.issubset(original_fields):
                    raise SourceError("publisher daily schema missing identity/failure fields")
                total=failures=0
                missing_smart={field:0 for field in SMART}
                available_smart={field:field in original_fields for field in SMART}
                unique=set()
                for row in reader:
                    if total>=MAX_DAILY_ROWS:
                        raise SourceError("daily source row count outside budget")
                    if row["date"]!=name[:-4]:
                        raise SourceError("actual CSV date differs from source member")
                    serial,model=row["serial_number"],row["model"]
                    if not serial or not model:
                        raise SourceError("missing original source identity")
                    identity=(serial,model)
                    if identity in unique:
                        raise SourceError("duplicate drive-day original identity")
                    unique.add(identity)
                    if row["failure"] not in ("0","1"):
                        raise SourceError("unknown publisher failure flag")
                    failures+=int(row["failure"])
                    for field in SMART:
                        if field not in original_fields or row[field] in ("",None):
                            missing_smart[field]+=1
                        elif not row[field].strip():
                            missing_smart[field]+=1
                        else:
                            try:
                                x=float(row[field])
                            except ValueError as exc:
                                raise SourceError("invalid publisher SMART numeric") from exc
                            if not math.isfinite(x) or x<0:
                                raise SourceError("invalid publisher SMART range")
                    total+=1
            if not total:
                raise SourceError("empty publisher daily file")
            daily.append({"date":name[:-4],"original_member_sha256":sha(content),
                          "rows":total,"recorded_failure_flags":failures,
                          "fields_present":available_smart,
                          "missing_fields_by_column":missing_smart})
    return {
        "study":"MODEL02-BACKBLAZE-2013-SOURCE-ONLY-EXPOSED",
        "evidence_tier":"PUBLISHER_ORIGINAL_SOURCE_SCHEMA_AUDIT_ONLY",
        "publisher_url":CITATION,"original_zip_url":ORIGINAL_ZIP,
        "original_zip_sha256":sha(zip_bytes),"compressed_bytes":len(zip_bytes),
        "date_start":FIRST.isoformat(),"date_count":DAYS,
        "total_daily_rows":sum(d["rows"] for d in daily),
        "total_recorded_failure_flags":sum(d["recorded_failure_flags"] for d in daily),
        "daily":daily,
        "source_license":"Publisher permits derivative products but not resale of original data; cite Backblaze",
        "source_publication_time_verified":False,
        "frozen_enrollment_precommitted":False,
        "fit_labels_created":False,"cohort_heldout":False,
        "trained_tac_model":False,"scientific_promotion":"NOT_QUALIFIED",
        "warnings":[
            "these dates are EXPOSED after source audit, not an untouched independent eval",
            "drive-day failure flags are not independently adjudicated physical outcomes",
            "single source archive does not prove originally available-at-time individual SMART readings",
            "no entity case-wise censoring or immutable prospective cohort qualification is established",
            "no original drive serial number or raw publisher CSV redistributed by receipt"
        ]
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip-file",type=Path,
                   help="operator-provided byte-identical original Backblaze 2013 ZIP (optional)")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.out.exists():
            raise SourceError("refusing to overwrite source receipt")
        bytes_=a.zip_file.read_bytes() if a.zip_file else fetch_zip()
        receipt=scan(bytes_)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open("x",encoding="utf-8") as output:
            json.dump(receipt,output,indent=2,sort_keys=True)
        print(json.dumps({key:receipt[key] for key in
              ("study","evidence_tier","original_zip_sha256","compressed_bytes",
               "total_daily_rows","total_recorded_failure_flags","scientific_promotion")},
              sort_keys=True))
        return 0
    except (OSError,SourceError,ValueError,UnicodeError,zipfile.BadZipFile,
            TimeoutError) as exc:
        print(json.dumps({"status":"BLOCKED_OR_FAILED","reason":str(exc)}),file=sys.stderr)
        return 2


if __name__=="__main__":
    sys.exit(main())
