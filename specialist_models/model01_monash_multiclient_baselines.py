#!/usr/bin/env python3
"""Monash Electricity Hourly original v3: customer-disjoint source qualification.

The Monash hourly series are derived from UCI321, not the original UCI374
Appliances target. This module NEVER applies old TAC/GRU appliance checkpoints.
Sealed TEST clients' numeric values are NOT parsed/scored. Real publisher
hourly demand clients and no-fit forecast baselines on CAL/DEV ONLY.
No future predictions feed features. Publisher timestamps are NOT verified
actual data-availability times. Original ZIP MD5 pinned to Zenodo v3.
"""
from __future__ import annotations
import argparse,calendar,datetime as dt,hashlib,io,json,math,re,sys,urllib.request,zipfile
from collections import Counter
from pathlib import Path

URL="https://zenodo.org/records/4656140/files/electricity_hourly_dataset.zip?download=1"
PUBLISHER_RECORD="https://zenodo.org/records/4656140"
DOI="10.5281/zenodo.4656140"
UPSTREAM_DOI="10.24432/C58C86"
ZIP_MD5="18096614662b02640d265ad2a6a416bd"
MAX_COMPRESSED=16_000_000
MAX_SOURCE_BYTES=47_000_000
MIN_CLIENTS=300
HORIZONS=(1,24)
EVAL_FINAL_HOURS=336
HISTORY=168
SPLIT_SALT="MODEL01-MONASH321-CLIENT-DISJOINT-001-FROZEN-2026-10-27"

class SourceError(ValueError):pass

def source_url():
    return URL

def acquire():
    req=urllib.request.Request(source_url(),headers={"User-Agent":"TAC-OSM-model01-real-multiclient-source-audit/1.0"})
    with urllib.request.urlopen(req,timeout=90) as response:
        url=response.geturl()
        if not url.startswith("https://") or not any(
            host in url for host in ("zenodo.org/", "cern.ch/")):
            raise SourceError("unexpected original source download host")
        raw=response.read(MAX_COMPRESSED+1)
    if len(raw)>MAX_COMPRESSED or not raw:
        raise SourceError("original publisher archive size/budget changed")
    if hashlib.md5(raw).hexdigest()!=ZIP_MD5:
        raise SourceError("publisher original v3 MD5 differs; version not admitted")
    return raw

def split(name):
    v=hashlib.sha256((SPLIT_SALT+"|"+name).encode("utf-8")).digest()[0]%8
    return ("FIT" if v<4 else "CAL" if v==4 else "DEV" if v==5 else "SEALED_TEST")

def parse_header(lines):
    head={}
    for row in lines:
        line=row.strip()
        if not line or line.startswith("#"):continue
        if line=="@data":break
        if not line.startswith("@"):
            raise SourceError("unexpected Monash metadata record")
        k,sep,val=line.partition(" ")
        if not sep:raise SourceError("invalid Monash header field")
        if k in ("@frequency","@missing","@equallength","@horizon","@relation"):
            head[k]=val.strip().lower()
    if head.get("@frequency") not in ("hourly","hour"):
        raise SourceError("not original hourly frequency")
    if head.get("@missing")!="false":
        raise SourceError("source missingness semantics cannot be assumed")
    if head.get("@equallength")!="true":
        raise SourceError("unregistered nonuniform series")
    return head

def parse_line(line,allow_targets):
    # Critical: Do not call float conversion for sealed TEST rows.
    fields=line.strip().split(":",2)
    if len(fields)!=3:
        raise SourceError("invalid Monash two-attribute TSF source row")
    name,start,values=fields
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,70}",name):
        raise SourceError("unrecognized public source client identity")
    if not re.fullmatch(r"\d{4}-\d\d-\d\d \d\d-\d\d-\d\d",start):
        raise SourceError("unrecognized Monash timestamp")
    try: timestamp=dt.datetime.strptime(start,"%Y-%m-%d %H-%M-%S")
    except ValueError as e:raise SourceError("invalid Monash source clock") from e
    if not allow_targets:
        return name,timestamp,None
    # Never use sealed clients' historical targets, even to infer whether active.
    xs=[]
    for token in values.split(","):
        if token=="?" or not token:
            raise SourceError("publisher missing/raw imputed sample without observation flag")
        try:v=float(token)
        except ValueError as e:raise SourceError("non-numeric observed target") from e
        if not math.isfinite(v) or v<0 or v>1e7:
            raise SourceError("invalid original nonnegative load kW")
        xs.append(v)
    if len(xs)<HISTORY+EVAL_FINAL_HOURS+max(HORIZONS):
        raise SourceError("client has insufficient measured history")
    return name,timestamp,xs

def iter_source(archive):
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        ts=[i for i in z.infolist() if i.filename.lower().endswith((".tsf",".ts"))]
        if len(ts)!=1 or ts[0].file_size>MAX_SOURCE_BYTES:
            raise SourceError("only one bounded original Monash TSF member permitted")
        if sum(i.file_size for i in z.infolist())>MAX_SOURCE_BYTES*2:
            raise SourceError("unexpected zip expansion budget")
        with z.open(ts[0]) as f:
            lines=(line.decode("utf-8-sig").strip() for line in f)
            metadata=[]
            for line in lines:
                metadata.append(line)
                if line=="@data":
                    break
                if len(metadata)>55:raise SourceError("unbounded publisher header")
            header=parse_header(metadata)
            if not metadata or metadata[-1]!="@data":
                raise SourceError("missing source TSF data marker")
            yield "HEADER",header
            for line in lines:
                if not line or line.startswith("#"):continue
                # Raw source line is streamed through ZIP, but sealed values
                # never converted/used/scored. No protection for source bytes
                # from downloading the entire public archive is claimed.
                name=line.partition(":")[0]
                arm=split(name)
                n,ts,values=parse_line(line,allow_targets=(arm!="SEALED_TEST"))
                yield arm,(n,ts,values)

def last_sunday(year,month):
    d=dt.date(year,month,calendar.monthrange(year,month)[1])
    while d.weekday()!=6:d-=dt.timedelta(days=1)
    return d

def dst_ambiguous(timestamp):
    # These publisher-local timestamps are not evidence of physical as-of UTC.
    # Exclude 72h around source Portugal DST spring/fall encoding anomalies.
    for month in (3,10):
        center=last_sunday(timestamp.year,month)
        if abs((timestamp.date()-center).days)<=2:return True
    return False

def baseline_report(name,start,source,hours=EVAL_FINAL_HOURS):
    if len(source)<HISTORY+hours+max(HORIZONS):
        raise SourceError("source insufficient for fixed case audit")
    errors={str(h):{k:[] for k in ("last_observation","same_hour_previous_day","same_hour_previous_week","daily_weekly_blend")}
            for h in HORIZONS}
    eligibility=Counter()
    n=len(source)
    begin=n-max(HORIZONS)-hours
    if begin<HISTORY:raise SourceError("unsupported heldout window")
    for anchor in range(begin,begin+hours):
        # Past-only evidence of a genuinely active source point; publisher
        # reports padded zeros before some meters were activated.
        history=source[anchor-HISTORY+1:anchor+1]
        if sum(v>0 for v in history)<24:
            eligibility["inactive_or_not_yet_observed"]+=1
            continue
        now=start+dt.timedelta(hours=anchor)
        if dst_ambiguous(now):
            eligibility["source_dst_ambiguous"]+=1
            continue
        for h in HORIZONS:
            target_i=anchor+h
            target_time=start+dt.timedelta(hours=target_i)
            if dst_ambiguous(target_time):
                eligibility["source_dst_ambiguous_target"]+=1
                continue
            target=source[target_i]
            refs={
                "last_observation":source[anchor],
                "same_hour_previous_day":source[target_i-24],
                "same_hour_previous_week":source[target_i-168],
            }
            refs["daily_weekly_blend"]=.5*(
                refs["same_hour_previous_day"]+refs["same_hour_previous_week"])
            for k,v in refs.items():
                errors[str(h)][k].append(v-target)
            eligibility["scored_"+str(h)]+=1
    result={}
    for h,models in errors.items():
        result[h]={}
        for k,diff in models.items():
            if not diff:raise SourceError("zero genuine selected eligible cases "+h)
            result[h][k]={"n":len(diff),"mae_kw":sum(abs(v) for v in diff)/len(diff),
                          "rmse_kw":math.sqrt(sum(v*v for v in diff)/len(diff))}
    return {"client_public_hash":hashlib.sha256(name.encode()).hexdigest()[:16],
            "publisher_start_timestamp":start.isoformat(),
            "source_recorded_count":len(source),
            "eligible":dict(eligibility),
            "source_baselines":result}

def run(archive):
    seen=set()
    counts=Counter()
    samples={"CAL":[],"DEV":[]}
    header=None
    digest=hashlib.sha256(archive).hexdigest()
    for arm,payload in iter_source(archive):
        if arm=="HEADER":
            header=payload
            continue
        name,start,vals=payload
        if name in seen:raise SourceError("duplicate public physical client ID")
        seen.add(name)
        counts[arm]+=1
        if arm in samples:
            samples[arm].append(baseline_report(name,start,vals))
        # FIT not scored here; SEALED_TEST numeric targets unparsed.
    if len(seen)!=321 or any(counts[a]<15 for a in ("FIT","CAL","DEV","SEALED_TEST")):
        raise SourceError("actual publisher source/client split unexpectedly different")
    # CAL/DEV evaluated for no-fit source baseline sanity, and hence exposed
    # from this point. Reserve true physical-client holdout for only one future
    # separately registered final model, never baseline tuning.
    macro={}
    for arm in ("CAL","DEV"):
        macro[arm]={}
        for h in map(str,HORIZONS):
            macro[arm][h]={}
            for method in ("last_observation","same_hour_previous_day",
                           "same_hour_previous_week","daily_weekly_blend"):
                series=[client["source_baselines"][h][method]["mae_kw"]
                        for client in samples[arm]]
                macro[arm][h][method]={
                    "mean_client_mae_kw":sum(series)/len(series),
                    "evaluated_clients":len(series),
                    "total_hourly_decisions":sum(client["source_baselines"][h][method]["n"]
                                                for client in samples[arm])
                }
    return {"schema":"model01-monash-hourly-independent-client-source-baseline-v1",
            "source_title":"Monash Electricity Hourly version 3, derived from UCI321",
            "zenodo_publisher_record":PUBLISHER_RECORD,
            "doi":DOI,"upstream_uci321_doi":UPSTREAM_DOI,
            "source_zip_md5":ZIP_MD5,"source_zip_sha256":digest,
            "license":"CC BY 4.0, original Monash and UCI attribution required",
            "source_header":header,"source_arm_counts":dict(counts),
            "assignment":"SHA256(public_series_name + frozen salt) modulo 8, FIT 4/8 CAL1/8 DEV1/8 SEALED2/8",
            "assignment_salt":SPLIT_SALT,
            "prediction_target":"hourly average electricity load kW per customer, NOT UCI374 appliance Wh",
            "prediction_horizons_hours":list(HORIZONS),
            "last_hours_per_DEV_CAL_client":EVAL_FINAL_HOURS,
            "original_source_available_at_verified":False,
            "source_DST_repairs_and_pre_activation_zeros":True,
            "source_client_ids_are_real_observed_identities":True,
            "TEST_customer_numeric_targets_parsed":False,
            "TEST_customer_targets_scored":False,
            "model_trained":False,
            "neural_architecture_qualified":False,
            "independent_test_claim":False,
            "CAL_DEV_are_exposed_from_this_source_audit":True,
            "per_client_CAL_DEV":samples,
            "macro_references":macro,
            "warnings":[
                "UCI321 source customers are different task/target from old UCI374 single house; old neural weights cannot be compared or scored directly",
                "Monash hourly conversion has aggregation and source-specific DST handling, and local timestamps are not actual transmission available-at",
                "UCI reports zero padding before some clients joined; zero is not automatically a real nonconsumption observation",
                "CAL and DEV are independently identified customers but now exposed after baseline evaluation and never count as untouched confirmation",
                "SEALED_TEST series are streamed in publisher ZIP but their numeric values are never parsed, scored or used to select models",
                "Good naive baseline does not establish prospective value or generalizable TAC neural skill",
                "No protected customer holdout data were numerically loaded or scored; independent final test unperformed"
            ]}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    try:
        if args.out.exists():raise SourceError("never overwrite source baseline evidence")
        raw=acquire()
        result=run(raw)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
        print(json.dumps({
            "schema":result["schema"],"source_sha256":result["source_zip_sha256"],
            "arms":result["source_arm_counts"],"macro":result["macro_references"],
            "heldout_numeric_values_parsed":False,"old_TAConly_model_trained":False
        },sort_keys=True))
        return 0
    except (SourceError,OSError,ValueError,TypeError,zipfile.BadZipFile) as e:
        print(json.dumps({"status":"BLOCKED_OR_FAILED","reason":str(e)},sort_keys=True),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
