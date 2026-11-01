"""Check actual source/math pins; metadata correspondence is not a proof."""
import hashlib
import json
from pathlib import Path
import re
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def verify():
    pin=json.loads((HERE/"PNDS_PIN.json").read_text())
    if pin["repository"]!="FPC-effortless/pnds-lean" or not re.fullmatch("[0-9a-f]{40}",pin["commit"]):
        raise ValueError("immutable PNDS pin required")
    def fetch(path):
        if not isinstance(path,str) or path.startswith("/") or ".." in path.split("/"):
            raise ValueError("repository-relative peer path required")
        url="https://raw.githubusercontent.com/FPC-effortless/pnds-lean/"+pin["commit"]+"/"+path
        with urllib.request.urlopen(url,timeout=30) as response:
            raw=response.read(2_000_001)
        if len(raw)>2_000_000:
            raise ValueError("peer file too large")
        return raw
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    remote=json.loads(fetch("foundation/v1/variant-math-links.json"))
    local=json.loads((ROOT/"research/pnds-variant-math-links.json").read_text())
    results=[]
    for item in pin["maps"]:
        raw=fetch(item["pnds_path"])
        if digest(raw)!=item["sha256"] or raw!=(HERE/item["local_path"]).read_bytes():
            raise ValueError("source map differs from mathematical authority")
        data=json.loads(raw)
        a=[r for r in local["prospective_candidates"] if r["id"]==data["variant_id"]]
        b=[r for r in remote["prospective_candidates"] if r["id"]==data["variant_id"]]
        if len(a)!=1 or a!=b or a[0]["tac_source_commit"]!=data["tac_commit"]:
            raise ValueError("candidate row/implementation pin mismatch")
        for path,expected in data["tac_source_sha256"].items():
            actual=(ROOT/path).resolve()
            if not actual.is_relative_to(ROOT) or digest(actual.read_bytes())!=expected:
                raise ValueError("actual source differs from original implementation: "+path)
        for path,expected in data["mathematical_files_sha256"].items():
            peer=fetch(path)
            if digest(peer)!=expected:
                raise ValueError("mathematical bytes differ from source map")
            if path==data["python_reference"] and peer!=(HERE/"math_reference.py").read_bytes():
                raise ValueError("independent reference differs from authority")
        results.append({"id":data["variant_id"],"source_map_equal":True,"source_files_verified":len(data["tac_source_sha256"]),
                        "whole_model_proved":False,"trained_capability_claim":False})
    print(json.dumps({"status":"SOURCE_PIN_BYTES_MATCH_NOT_PROOF","candidates":results}))


if __name__=="__main__":
    verify()
