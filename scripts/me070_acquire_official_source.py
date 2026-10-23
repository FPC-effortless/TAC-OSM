#!/usr/bin/env python3
"""Fail-closed original BPIC2018 acquisition from Figshare/4TU publisher metadata.

This is source qualification only. Do not train, open model TEST scores, or
upload raw traces. Requires a real publisher file record, original checksum
and matching dataset DOI. A network failure is BLOCKED, not dataset failure.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.request import Request, urlopen

RECORD_ID = 12688355
DOI = "10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972"
API = f"https://api.figshare.com/v2/articles/{RECORD_ID}"

def checked_metadata(record):
    title = str(record.get("title", "")).lower().replace("_"," ")
    if "bpi" not in title or "2018" not in title or "challenge" not in title:
        raise ValueError(f"unexpected dataset title: {title[:120]}")
    doi = str(record.get("doi", "")).lower().replace("https://doi.org/", "")
    if doi and doi not in (DOI.lower(), DOI.lower().replace("uuid:", "")):
        raise ValueError("publisher metadata DOI differs from official original")
    entries = record.get("files") or []
    matches = [f for f in entries if f.get("name", "").lower().endswith((".xes.gz", ".xes"))]
    if len(matches)!=1:
        raise ValueError(f"expected exactly one original XES file, found {len(matches)}")
    f=matches[0]
    name=f["name"]; expected_md5=f.get("supplied_md5") or f.get("computed_md5") or f.get("md5")
    if not expected_md5 or not re.fullmatch("[0-9a-fA-F]{32}",expected_md5):
        raise ValueError("missing original publisher MD5; do not use unverified mirror")
    url = f.get("download_url")
    if not isinstance(url,str) or not url.startswith("https://"):
        raise ValueError("publisher file record missing official HTTPS download link")
    if not any(host in url.split("/",3)[2] for host in ("figshare.com","4tu.nl")):
        raise ValueError("file URL not controlled by recognized publisher storage")
    return {"file_name":name,"file_id":f.get("id"),"url":url,
            "publisher_md5":expected_md5.lower(),"declared_bytes":f.get("size"),
            "record_id":RECORD_ID,"publisher_doi":doi or DOI,"publisher_title":record["title"]}

def download():
    with urlopen(Request(API,headers={"User-Agent":"TAC-OSM-ME070-source-qualification/1.0"}),timeout=20) as response:
        return json.load(response)

def acquire(metadata,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".part")
    md5=hashlib.md5()
    sha=hashlib.sha256()
    length=0
    try:
        with urlopen(Request(metadata["url"],headers={"User-Agent":"TAC-OSM-ME070-source-qualification/1.0"}),timeout=50) as response, temp.open("wb") as out:
            while data := response.read(1<<20):
                out.write(data);md5.update(data);sha.update(data);length+=len(data)
        if md5.hexdigest()!=metadata["publisher_md5"]:
            raise ValueError("downloaded file differs from original publisher MD5")
        if metadata["declared_bytes"] is not None and length!=metadata["declared_bytes"]:
            raise ValueError("downloaded file size differs from publisher manifest")
        if path.name.endswith(".gz") and temp.open("rb").read(2)!=b"\x1f\x8b":
            raise ValueError("download claims gzip XES but magic is wrong")
        if path.name.endswith(".xes") and not temp.open("rb").read(80).lstrip().startswith((b"<?xml",b"<log")):
            raise ValueError("download claims XES but XML magic is wrong")
        os.replace(temp,path)
        return {"sha256":sha.hexdigest(),"md5":md5.hexdigest(),"bytes":length,
                "publisher_metadata":{k:v for k,v in metadata.items() if k!="url"}}
    finally:
        temp.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory",type=Path,default=Path("/tmp/me070-publisher"))
    args=parser.parse_args()
    meta=checked_metadata(download())
    target=args.directory/meta["file_name"]
    receipt=acquire(meta,target)
    print(json.dumps(receipt,indent=2,sort_keys=True))
    (args.directory/"ORIGINAL_SOURCE_PATH.txt").write_text(str(target))
    (args.directory/"PUBLISHER_ACQUISITION_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")

if __name__=="__main__":
    main()
