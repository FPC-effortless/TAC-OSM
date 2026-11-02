"""Recover exact TRAIN/DEV only from the user's pinned TACOPT114 artifact.

The supplied code is parsed as literals, never imported or executed.
Usage: python -m collect011.restore_data /path/to/TACOPT114_Code_and_GPU_Evidence.zip
"""
import argparse,ast,base64,hashlib,io,json,zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument("archive",type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(a.archive) as outer:
        candidates=[n for n in outer.namelist() if n.endswith("tacround110_training/kaggle_profile_v3/TACROUND110_Profile.py")]
        if len(candidates)!=1:raise ValueError("exact frozen v3 launcher required")
        tree=ast.parse(outer.read(candidates[0]).decode())
    v={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ("PAYLOAD","PAYLOAD_SHA256")}
    blob=base64.b64decode(v["PAYLOAD"])
    if v["PAYLOAD_SHA256"]!="4be0fc390a016b0ca4cf0c1f3584e58d53d1859e0d22511b21be1aa16fd232d9" or hashlib.sha256(blob).hexdigest()!=v["PAYLOAD_SHA256"]:
        raise ValueError("frozen payload mismatch")
    expected=json.loads((root/"data/SPLIT_MANIFEST.json").read_text())
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for part in ("train","dev"):
            data=z.read("tacmatch108_final/base/data/"+part+".bin")
            if hashlib.sha256(data).hexdigest()!=expected["parts"][part]["sha256"] or len(data)!=expected["parts"][part]["bytes"]:
                raise ValueError("original split mismatch")
            (root/"data"/(part+".bin")).write_bytes(data)
            print(part,len(data),expected["parts"][part]["sha256"])

if __name__=="__main__":main()
