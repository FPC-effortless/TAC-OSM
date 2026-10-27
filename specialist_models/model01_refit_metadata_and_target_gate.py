#!/usr/bin/env python3
"""REFIT publisher metadata and partial-appliance target qualification only.

Queries publisher Zenodo record metadata (small), does NOT download ~490 MB
cleaned sensor archive, select test houses, train or score models. On trusted
8-second raw, exact 75 * 8s W -> Wh integration is definable but REFIT's
instrumented-appliance sum != total/all appliances (UCI374 'Appliances').
Cleaned REFIT interpolates/zeros some missing readings; original source
visibility/live as-of clock is unverified. No source-skill promotion.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import urllib.request

RECORD_URL="https://zenodo.org/api/records/5063428"
DOI="10.5281/zenodo.5063428"
SOURCE_PUBLISHER="University of Strathclyde"
SAMPLE_SECONDS=8
BUCKET_SAMPLES=75
MAX_METADATA=750000

class SourceError(ValueError):
    pass

def metadata_remote():
    request=urllib.request.Request(RECORD_URL,headers={"User-Agent":"TAC-OSM-metadata-receipt/1.0"})
    with urllib.request.urlopen(request,timeout=20) as r:
        if r.geturl()!=RECORD_URL:
            raise SourceError("unexpected REFIT metadata redirect")
        data=r.read(MAX_METADATA+1)
        if len(data)>MAX_METADATA:
            raise SourceError("metadata exceeds fixed download budget")
    return json.loads(data),hashlib.sha256(data).hexdigest()

def validate_record(data):
    if not isinstance(data,dict) or data.get("id")!=5063428:
        raise SourceError("unexpected publisher record")
    info=data.get("metadata")
    if not isinstance(info,dict):
        raise SourceError("metadata absent")
    doi=info.get("doi") or data.get("doi")
    if doi!=DOI:
        raise SourceError("unexpected dataset DOI")
    title=(info.get("title") or "").lower()
    if "refit" not in title or "cleaned" not in title:
        raise SourceError("dataset variant does not match cleaned REFIT")
    files=data.get("files")
    if not isinstance(files,list) or not files:
        raise SourceError("publisher file list absent")
    entries=[]
    for f in files:
        if not isinstance(f,dict):
            raise SourceError("invalid publisher file record")
        name=f.get("key") or f.get("filename")
        size=f.get("size") or f.get("filesize")
        if not isinstance(name,str) or "/" in name or ".." in name or type(size)!=int or not 0<size<10_000_000_000:
            raise SourceError("invalid source file metadata")
        check=f.get("checksum") or f.get("checksum_value")
        entries.append({"name":name,"size_bytes":size,
                        "checksum":check if isinstance(check,str) and len(check)<200 else None})
    if not any("CLEAN" in f["name"].upper() or "PROCESSED" in f["name"].upper() for f in entries):
        raise SourceError("no original cleaned REFIT package advertised")
    license_value=info.get("license")
    license_id=license_value.get("id") if isinstance(license_value,dict) else license_value
    # The original primary public record may encode the licence differently;
    # do not invent an exact license from missing JSON fields.
    legal_status=("METADATA_CC_BY_4_0" if isinstance(license_id,str) and
                  license_id.lower().replace("-","") in ("ccby40","ccby4.0") else
                  "LICENSE_FIELD_NOT_MACHINE_VERIFIED")
    return {
        "schema":"model01-refit-zenodo-original-record-audit-v1",
        "publisher_record":RECORD_URL,
        "publisher":SOURCE_PUBLISHER,
        "source_doi":DOI,
        "license_metadata_check":legal_status,
        "source_files":entries,
        "original_refit_raw_appliances_loaded":False,
        "source_reconstructed_targets":False,
        "refit_cleaned_missingness_native":False,
        "target_equivalence_to_uci374":False,
        "independent_qualification":False,
        "model_fitted_or_scored":False,
        "research_next_gate":"verify exact instrumented appliance coverage and original 8-second gaps/clock on source bytes, full publisher attribution, then preregister matched target and heldout houses",
        "warnings":[
            "publisher REFIT cleaned dataset uses zero/forward fill for missing samples",
            "only instrumented individual appliances are observed; not all household appliance energy",
            "not legitimate to transfer UCI374 full appliance target to a monitored subset silently",
            "original publication and device available-at timestamps not independently proven",
            "this is repository/source metadata verification, not a trained model or real evaluation"
        ]}

def checked_wh(samples):
    """Only original, explicit 8-second complete W samples for ONE appliance."""
    if not isinstance(samples,list) or len(samples)!=BUCKET_SAMPLES:
        raise SourceError("full original 75 eight-second samples mandatory")
    prior=None
    total=0.
    for r in samples:
        if not isinstance(r,dict) or set(r)!={"at_utc","watts","actually_observed"}:
            raise SourceError("missing sample provenance")
        if r["actually_observed"] is not True:
            raise SourceError("imputed/filled REFIT interval cannot claim observed")
        stamp=r["at_utc"]
        if not isinstance(stamp,str) or not stamp.endswith("Z"):
            raise SourceError("UTC source timestamp mandatory")
        try:
            now=datetime.fromisoformat(stamp.replace("Z","+00:00"))
        except ValueError as exc:
            raise SourceError("bad sensor timestamp") from exc
        if now.tzinfo is None or now.utcoffset()!=timedelta(0):
            raise SourceError("source UTC conversion not established")
        if prior is not None and (now-prior)!=timedelta(seconds=SAMPLE_SECONDS):
            raise SourceError("REFIT sample gap, duplicated clock or DST shift")
        prior=now
        val=r["watts"]
        if type(val) not in (float,int) or not math.isfinite(val) or val<0 or val>4000:
            raise SourceError("invalid individual appliance monitor power")
        total+=val*SAMPLE_SECONDS/3600
    return total

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.out.exists():
            raise SourceError("refusing evidence overwrite")
        metadata,original_digest=metadata_remote()
        receipt=validate_record(metadata)
        receipt["original_publisher_metadata_sha256"]=original_digest
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
        print(json.dumps({"schema":receipt["schema"],
                          "source_files_count":len(receipt["source_files"]),
                          "original_publisher_metadata_sha256":original_digest,
                          "license_metadata_check":receipt["license_metadata_check"],
                          "target_equivalence_to_uci374":False,
                          "model_fitted_or_scored":False},sort_keys=True))
        return 0
    except (SourceError,OSError,TypeError,ValueError,KeyError) as e:
        print(json.dumps({"status":"BLOCKED_OR_FAILED","reason":str(e)},sort_keys=True),file=sys.stderr)
        return 2

if __name__=="__main__":
    sys.exit(main())
