#!/usr/bin/env python3
"""S1TEMP008 publisher metadata ONLY -- DOES NOT download any source XES case.

Original data provider: Eindhoven University of Technology / 4TU,
BPI Challenge 2017, DOI10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b.
Use original Figshare/4TU original XES.gz, never third-party processed CSV.
The original archive is single-file, so source-case quarantine after trusted
partition must transparently disclose all raw bytes accessed by partitioner.
"""
from __future__ import annotations
import json,hashlib,urllib.request
from pathlib import Path
API="https://api.figshare.com/v2/articles/12696884"
DOI="10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b"
def original_source_metadata():
    req=urllib.request.Request(API,headers={
       "User-Agent":"S1TEMP008-original-BPIC2017-source-only-metadata/1.0",
       "Accept":"application/json"})
    with urllib.request.urlopen(req,timeout=45) as f:
        raw=f.read()
    x=json.loads(raw)
    if not isinstance(x,dict) or x.get("id")!=12696884:
        raise RuntimeError("publisher metadata article mismatch")
    name=str(x.get("title","")).lower()
    if "bpi challenge 2017" not in name or "offer log" in name:
        raise RuntimeError("original full multi-entity log not identified")
    identifier=str(x.get("doi","")).lower().removeprefix("https://doi.org/")
    if identifier!=DOI.lower():
        raise RuntimeError(f"publisher exact DOI mismatch: {identifier}")
    license_obj=x.get("license") or {}
    license_url=str(license_obj.get("url",""))
    license_name=str(license_obj.get("name",""))
    if "4tu" not in (license_url+" "+license_name).lower():
        raise RuntimeError("unverified original 4TU source license")
    files=x.get("files") or []
    xs=[file for file in files if str(file.get("name","")).lower().endswith(".xes.gz")]
    if len(xs)!=1:raise RuntimeError("require exactly one ORIGINAL complete XES gzip")
    file=xs[0];size=file.get("size")
    if (not isinstance(size,int) or not 1_000_000<size<200_000_000
        or not str(file.get("download_url","")).startswith("https://")
        or not isinstance(file.get("id"),int)):
        raise RuntimeError("original archive size/file identity missing")
    return dict(study="S1TEMP008",publisher_article=12696884,
         doi=identifier,article_title=x["title"],
         license_name=license_name,license_url=license_url,
         original_file_id=file["id"],original_file_name=file["name"],
         original_file_download_url=file["download_url"],
         original_file_bytes=size,
         publisher_MD5=file.get("computed_md5") or file.get("supplied_md5"),
         publisher_metadata_SHA256=hashlib.sha256(raw).hexdigest(),
         metadata_only_no_XES_download=True,
         no_training_or_DEV_scoring=True,
         single_archive_contains_ALL_CASES=True,
         source_training_quarantine_NOT_YET_VERIFIED=True,
         no_credit_decision_authority=True)

if __name__=="__main__":
    result=original_source_metadata()
    out=Path("research/diagnostics/s1temp008-original-metadata")
    out.mkdir(parents=True,exist_ok=True)
    (out/"original-4tu-bpi17-metadata.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("S1TEMP008_ORIGINAL_PUBLISHER_METADATA_PASS",
          json.dumps(result,sort_keys=True),flush=True)
