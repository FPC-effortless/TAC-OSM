#!/usr/bin/env python3
"""GET-range probe of official SCANIA train_tte; bounded first bytes only."""
import json
import urllib.request
import urllib.error
import urllib.parse

name="train_tte.csv"
urls=[]
for v in (3,2,1):
    b=f"https://api.researchdata.se/dataset/2024-34/{v}"
    urls += [b + "/file/data?" + urllib.parse.urlencode({"filePath":name}),
             b + "/file/zip"]
for u in urls:
    try:
        req=urllib.request.Request(u,headers={
            "User-Agent":"TAC-OSM-seq009-research/1.0",
            "Range":"bytes=0-127"})
        with urllib.request.urlopen(req,timeout=12) as rsp:
            prefix=rsp.read(128)
            print(json.dumps({"url":u,"status":rsp.status,
                              "type":rsp.headers.get("Content-Type"),
                              "size":rsp.headers.get("Content-Length"),
                              "range":rsp.headers.get("Content-Range"),
                              "prefix":prefix[:32].hex()}),flush=True)
    except urllib.error.HTTPError as e:
        print(json.dumps({"url":u,"status":e.code}),flush=True)
    except Exception as e:
        print(json.dumps({"url":u,"error":str(e)[:150]}),flush=True)
