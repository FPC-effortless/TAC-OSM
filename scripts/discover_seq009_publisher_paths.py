#!/usr/bin/env python3
"""Probe HEAD-only URLs for official SCANIA train_tte file.

No bytes of TEST, no modeling. Candidate endpoint discovery only.
"""
import json
import urllib.request
import urllib.parse
import urllib.error

BASE = "https://api.researchdata.se/dataset/2024-34/3"
name = "train_tte.csv"
queries = [name, "/" + name, "data/" + name, "Data/" + name,
           "./" + name]
candidates = [
    BASE + "/file/data?" + urllib.parse.urlencode({"filePath": path})
    for path in queries
]
candidates += [
    BASE + "/file?" + urllib.parse.urlencode({"filePath": name}),
    BASE + "/file/" + name,
    "https://doris.snd.se/api/file/2024-34/3/data?" +
    urllib.parse.urlencode({"filePath": name}),
    BASE + "/files",
    BASE,
    BASE + "/file/zip",
]
for url in candidates:
    try:
        req=urllib.request.Request(
            url, method="HEAD", headers={"User-Agent":"TAC-OSM-seq009-provenance/1.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            print(json.dumps({"url":url, "status":response.status,
                              "type":response.headers.get("Content-Type"),
                              "size":response.headers.get("Content-Length"),
                              "final_url":response.url}),flush=True)
    except urllib.error.HTTPError as e:
        print(json.dumps({"url":url, "status":e.code}), flush=True)
    except Exception as e:
        print(json.dumps({"url":url, "error":str(e)[:180]}), flush=True)
