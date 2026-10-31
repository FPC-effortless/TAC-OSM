"""Verify all 18 owner-authorized FIT-only artifacts, no DEV or superiority."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def audit(root,protocol,out):
    p=json.loads(Path(protocol).read_text())
    expected={(scale,m["arm"],seed) for scale,group in p["models"].items()
              for m in group["configurations"] for seed in p["seeds"]}
    assert len(expected)==18
    results=[];actual=set()
    for path in sorted(Path(root).rglob("FIT_RECEIPT.json")):
        v=json.loads(path.read_text())
        key=(v["scale"],v["arm"],v["seed"])
        if key not in expected or key in actual:
            raise ValueError("Unplanned or duplicate training receipt: "+str(key))
        actual.add(key)
        if v.get("status")!="FIT_TRAINING_COMPLETED_UNQUALIFIED" or v["actual_steps"]!=p["fit"]["steps"] or v["smoke"] is not False:
            raise ValueError("Incomplete training: "+str(key))
        if v["dev_scored"] is not False or v["protected_test_access"] is not False or v["independently_qualified"] is not False:
            raise ValueError("Incorrect scientific label")
        model=path.parent/"TRAINED.pt"
        if not model.is_file() or sha(model)!=v["checkpoint_sha256"]:
            raise ValueError("Checkpoint bytes not retained or incorrect")
        if len(v["fit_loss_series"])!=p["fit"]["steps"]:
            raise ValueError("Incomplete per-step FIT loss curve")
        results.append({k:v[k] for k in ("scale","arm","seed","actual_steps","fit_loss_first","fit_loss_last","checkpoint_sha256")})
    if actual!=expected:
        raise ValueError("Incomplete 18-fit matrix: missing "+str(sorted(expected-actual)))
    out=Path(out)
    if out.exists():raise FileExistsError("No overwrite of previous full training audit")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"status":"18_FIT_COMPLETED_NOT_SCIENTIFICALLY_QUALIFIED","fits":results,"fit_count":18,
         "no_dev_scoring":True,"protected_test_access":False,"independent_validation":False,
         "architecture_superiority":False},indent=2)+"\n")
    print("PASS: 18 authentic-source FIT-only completed checkpoints; no DEV",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fitsdir",type=Path,required=True)
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();audit(a.fitsdir,a.protocol,a.output)
