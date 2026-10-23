"""ME066 independent clean-Git original trained checkpoint load audit; no original publisher data read."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch

from scripts.me065_prefit_original_fit import SEEDS,CONFIG,model_for
ROOT=Path("research/mathematical-engineering/me066")
OUT=Path("research/replay-bundles/ME066")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run():
    study=json.loads((ROOT/"FAIR_COMPARISON.json").read_text())
    task=json.loads((ROOT/"TASK_LEVEL_SPLIT.json").read_text())
    report=json.loads((OUT/"TRAINING_RESULT.json").read_text())
    if report["study"]!=study["study_id"] or report["target_trajectory_files_opened"]!=0:
        raise RuntimeError("original trained scientific source identity or unexposed target violated")
    rows=report["original_checkpoints"]
    if len(rows)!=20 or {(r["arm"],r["seed"]) for r in rows}!={(a,s) for a in CONFIG for s in SEEDS}:
        raise RuntimeError("original twenty 4x5 trained checkpoint identities missing")
    receipts=[]
    for row in rows:
        file=Path(row["path"])
        if not file.is_file() or sha(file)!=row["checkpoint_sha256"]:
            raise RuntimeError("original trained checkpoint byte missing or has modified SHA")
        original=torch.load(file,weights_only=False,map_location="cpu")
        if (original["study"]!=study["study_id"] or not original["training_complete"] or
            original["arm"]!=row["arm"] or original["seed"]!=row["seed"] or
            original["training_updates"]!=600 or
            original["scoring_status"]!="NO_SOURCE_CAL_OR_FUTURE_TARGET_OPENED" or
            original["FIT_source_sha256"]!=task["source_FIT_hash"] or
            original["heldout_task_metadata_hash"]!=task["heldout_member_digest"] or
            original["paired_order_sha256"]!=task["training_seed_sample_digests"][str(row["seed"])]):
            raise RuntimeError("original trained/checkpoint source identity changed")
        identified={k:np.asarray(v,np.float32) for k,v in original["identified"].items()}
        model=model_for(row["arm"],identified)
        model.load_state_dict(original["model"],strict=True)
        if not original.get("optimizer") or original.get("torch_rng_state") is None or (
                original.get("numpy_rng_state") is None):
            raise RuntimeError("optimizer and RNG recovery state must be stored")
        if sum(p.numel() for p in model.parameters())!=next(
                v["trainable_parameters"] for v in study["arms"] if
                v["architecture"]==row["arm"] and v["seed"]==row["seed"]):
            raise RuntimeError("original trained active parameter count changed")
        model.eval()
        receipts.append({"architecture":row["arm"],"seed":row["seed"],
                         "sha256":row["checkpoint_sha256"],"bytes":file.stat().st_size,
                         "model_state_strict_restore":True})
    final={
        "study":study["study_id"],
        "model_replay":"PASS",
        "original_git_trained_model_bytes_hashed_and_restored":20,
        "independent_checkout_required":True,
        "source_FIT_sha256":task["source_FIT_hash"],
        "untouched_new_task":task["heldout_task_name"],
        "source_CAL_trajectories_opened":0,
        "new_heldout_trajectories_opened":0,
        "original_me064_target_reused":False,
        "full_score_replay":"NOT_YET_RUN",
        "trained_models":receipts,
    }
    (OUT/"CLEAN_CHECKOUT_MODEL_REPLAY.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
    print("ME066_ORIGINAL_FOUR_ARM_CLEAN_GIT_CHECKPOINT_REPLAY_PASS "+json.dumps({
        "original_models":len(receipts),"untouched_target_trajectory_files_opened":0}),flush=True)

if __name__=="__main__":
    argparse.ArgumentParser().parse_args()
    run()
