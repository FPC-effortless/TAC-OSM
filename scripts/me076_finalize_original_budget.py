"""ME076 final honest runtime budget certification using already-fitted Git-owned models.

This performs NO optimizer step, does NOT alter original preregistration, trained
checkpoints, runtime receipts or source labels. All claims remain exposed DEV.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path("research/mathematical-engineering/me076")
ORIGINAL=ROOT/"FAIR_COMPARISON.json"
FINAL=ROOT/"FINAL_COMPARISON.json"
MODELS=Path("research/replay-bundles/ME076")
SEEDS=(76001,76002,76003,76004,76005)

def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def complete():
    from scripts.validate_research_parity import check_manifest
    original=json.loads(ORIGINAL.read_text())
    if check_manifest(original,"preflight")["status"]!="PASS_PREFIT_SINGLE_MODEL":
        raise RuntimeError("original immutable original model registration rejected")
    final=copy.deepcopy(original)
    final["prefit_registration_sha256"]=sha(ORIGINAL)
    final["original_registration_kept_immutable"]=True
    final["conclusion"]="trained real FIT, exposed DEV, no architecture winner or independent real-world gain"
    seen=[]
    for row in final["models"]:
        seed=row["seed"]
        if seed not in SEEDS or seed in seen:
            raise RuntimeError("missing or duplicate original model seed")
        seen.append(seed)
        folder=MODELS/f"seed-{seed}"
        checkpoint=folder/"final.pt"
        receipt=folder/"live-budget.json"
        original_scored=folder/"RESULTS.json"
        raw_forecast=folder/"ORIGINAL_REAL_PREDICTIONS.json"
        for f in (checkpoint,receipt,original_scored,raw_forecast):
            if not f.exists():raise RuntimeError("original fitted source evidence unavailable: "+str(f))
        score=json.loads(original_scored.read_text())
        live=json.loads(receipt.read_text())
        if live.get("manifest_sha256")!=final["prefit_registration_sha256"]:
            raise RuntimeError("original runtime receipt did not use frozen preregistration")
        if live.get("successful_optimizer_steps")!=160 or live.get("status")!="CHECKPOINT_VERIFIED":
            raise RuntimeError("original fit lacks qualifying verified optimizer/checkpoint")
        if score.get("original_checkpoint_sha256")!=sha(checkpoint):
            raise RuntimeError("original result/model SHA mismatch")
        row["measured"]={
            "trainable_parameters":100476,"deployed_parameters":100476,
            "actual_steps":160,"checkpoint_sha256":sha(checkpoint),
            "runtime_evidence":{
                "receipt_path":str(receipt),
                "receipt_sha256":sha(receipt),
                "checkpoint_path":str(checkpoint),
                "checkpoint_sha256":sha(checkpoint),
            },
        }
    if seen!=list(SEEDS):raise RuntimeError("expected all five original seeds")
    status=check_manifest(final,"final")
    if status["status"]!="PASS_SINGLE_MODEL":
        raise RuntimeError("final original runtime parity refused")
    FINAL.write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
    print("ME076_ALL_FIVE_REAL_ORIGINAL_RUNTIME_RECEIPTS_VERIFIED="+
          json.dumps({"models":len(SEEDS),"prefit_registration_sha256":sha(ORIGINAL),
                      "final_receipt_sha256":sha(FINAL),"status":status["status"]}),flush=True)

if __name__=="__main__":
    complete()
