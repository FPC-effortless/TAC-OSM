#!/usr/bin/env python3
"""ME-003: source-only real dynamic-state screening on official UCI MetroPT-3.
Reads the earliest 30 days of physical-sensor history only; NO models, no future TEST.
No extrapolated action efficacy or learned-architecture claim.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timedelta
import csv, hashlib, io, json, math, os, urllib.request, zipfile
from pathlib import Path

URL="https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip"
DOI="10.24432/C5VW3R"
ANALOG=("TP2","TP3","H1","DV_pressure","Reservoirs","Oil_temperature","Motor_current")
DIGITAL=("COMP","DV_eletric","Towers","MPG","LPS","Pressure_switch","Oil_level","Caudal_impulses")
# Names on UCI may have capitalization and punctuation variations; only
# actual source fields found will be measured. Do not silently invent columns.
MIN_ROWS=10000
MAX_ARCHIVE=500_000_000
MAX_MEMBER=2_000_000_000
INTERVAL_LIMIT=30.0

def acquire(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    file=root/"metropt3-official.zip"
    h=hashlib.sha256()
    n=0
    req=urllib.request.Request(URL,headers={"User-Agent":"TAC-OSM-ME003-real-source-audit/1.0"})
    with urllib.request.urlopen(req,timeout=120) as response, file.open("wb") as out:
        if getattr(response,"status",200)!=200:raise ValueError("publisher download not HTTP 200")
        while True:
            chunk=response.read(2*1024*1024)
            if not chunk:break
            n+=len(chunk)
            if n>MAX_ARCHIVE:raise ValueError("archive larger than bound")
            h.update(chunk);out.write(chunk)
    return file,{"official_source":URL,"doi":DOI,"archive_sha256":h.hexdigest(),"archive_bytes":n}

def audit(root):
    file,source=acquire(root)
    with zipfile.ZipFile(file) as z:
        names=[info for info in z.infolist()
               if Path(info.filename).name.lower()=="metropt3(aircompressor).csv"]
        if len(names)!=1:raise ValueError("expected single authentic MetroPT source CSV")
        info=names[0]
        if info.file_size>MAX_MEMBER:raise ValueError("uncompressed source too big")
        with z.open(info) as raw:
            # Read only first thirty calendar days of chronologically ordered rows.
            reader=csv.DictReader(io.TextIOWrapper(raw,encoding="utf-8-sig",newline=""))
            cols=reader.fieldnames or []
            if len(cols)!=len(set(cols)):raise ValueError("duplicate feature names")
            if "timestamp" not in cols:raise ValueError("missing publisher timestamp")
            numerical=[c for c in cols if c not in ("timestamp","Unnamed: 0","index")]
            if len(numerical)<12:raise ValueError(f"source features unexpected: {cols}")
            counts={c:Counter(pos=0,neg=0,stable=0,invalid=0) for c in numerical}
            start=stop=last_ts=None
            last=None
            rows=transitions=gaps=0
            observed_interval=Counter()
            for row in reader:
                t=datetime.fromisoformat(row["timestamp"].strip())
                if start is None:
                    start=t;stop=start+timedelta(days=30)
                if t >= stop:break
                if last_ts is not None and t<last_ts:
                    raise ValueError("source not chronological: no future peek/reorder allowed")
                x={}
                for c in numerical:
                    try:
                        v=float(row[c])
                        x[c]=v if math.isfinite(v) else None
                    except (ValueError,TypeError):
                        x[c]=None
                rows+=1
                if last_ts is not None:
                    dt=(t-last_ts).total_seconds()
                    observed_interval[str(min(int(dt),300))]+=1
                    if 0<dt<=INTERVAL_LIMIT:
                        transitions+=1
                        for c in numerical:
                            if x[c] is None or last[c] is None:
                                counts[c]["invalid"]+=1
                            else:
                                change=x[c]-last[c]
                                if change>1e-6:counts[c]["pos"]+=1
                                elif change<-1e-6:counts[c]["neg"]+=1
                                else:counts[c]["stable"]+=1
                    else:gaps+=1
                last=x;last_ts=t
    file.unlink(missing_ok=True)
    if rows<MIN_ROWS or transitions<MIN_ROWS//2:
        raise ValueError("insufficient authentic source rows/valid short-interval changes")
    measured={c:dict(counts[c]) for c in numerical}
    both=[c for c,v in counts.items() if v["pos"]>=100 and v["neg"]>=100]
    out={"study":"ME003","source":source,"evidence":"authentic TRAIN-ERA source suitability only",
         "first_timestamp":str(start),"cutoff_timestamp_exclusive":str(stop),
         "last_inspected_timestamp":str(last_ts),"rows_inspected":rows,
         "eligible_short_interval_pairs":transitions,"gaps_or_duplicate_timestamps":gaps,
         "source_csv_member":info.filename,"source_csv_full_uncompressed_size":info.file_size,
         "features":numerical,"bidirectional_100_plus_features":both,
         "bidirectional_count":len(both),"per_feature_transitions":measured,
         "observed_interval_seconds_buckets":dict(observed_interval),
         "future_reserved_labels_read":False,"model_fits":0,"promoted":False}
    Path(root,"source-audit.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"source_audit_complete","row_count":rows,
        "bidirectional_count":len(both),"features_with_both_directions":both,
        "future_read":False,"model_fits":0},sort_keys=True))
    return out

if __name__=="__main__":
    audit("results/me003")
