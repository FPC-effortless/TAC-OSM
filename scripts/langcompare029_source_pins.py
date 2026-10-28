#!/usr/bin/env python3
"""No-fit authentic upstream-source identity audit, not Mamba training."""
from __future__ import annotations
import argparse, hashlib, json, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PINS=ROOT/"research/mathematical-engineering/langcompare029/UPSTREAM_SOURCE_PINS.json"

def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()

def audit():
    p=json.loads(PINS.read_text())
    result={}
    for arm in ("bdh","mamba3"):
        conf=p[arm]
        ownerrepo=conf["repo"];sha=conf["commit"]
        sources={}
        for path,expected in conf["files"].items():
            url=f"https://raw.githubusercontent.com/{ownerrepo}/{sha}/{path}"
            with urllib.request.urlopen(url,timeout=30) as response:
                content=response.read()
            observed=git_blob(content)
            if observed!=expected:
                raise ValueError(f"{arm} upstream source sha changed: {path}")
            if arm=="mamba3" and path.endswith("mamba3_siso_combined.py"):
                if "Q = Q.to(torch.bfloat16)" not in content.decode():
                    raise ValueError("Official Mamba3 BF16 path source audit cannot find published coercion")
            sources[path]={"git_blob_sha":observed,"bytes":len(content)}
        result[arm]={"source_commit":sha,"verified_files":sources}
    return {"study":"LANGCOMPARE029","status":"OFFICIAL_SOURCE_IDENTITIES_VERIFIED_NO_FIT",
            "sources":result,"mamba3_cuda_instantiation":"NOT_RUN",
            "actual_mamba3_parameters":"NOT_RUN","optimizer_steps":0,
            "DEV_NLL":"NOT_RUN","protected_TEST":"NOT_ACCESSED"}
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path)
    a=ap.parse_args()
    data=json.dumps(audit(),indent=2,sort_keys=True)+"\n"
    if a.output:
        if a.output.exists():raise ValueError("do not overwrite source receipt")
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(data)
    print(data,end="")
