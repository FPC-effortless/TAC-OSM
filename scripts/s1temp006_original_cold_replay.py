#!/usr/bin/env python3
"""Read-only cold replay S1TEMP006 ORIGINAL six SHA-frozen trained models.

Original DEV already opened only after train freeze. No new model fitting,
no epoch, no threshold tuning, no source TEST files accessed.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.s1temp006_real_long_history_fit import (
    FIT,DEV,SEEDS,ARMS,lock,get_episode,fit_norm,encode,model_pair,
    eval_context
)
LEDGER=ROOT/"research/parallel-six/S1TEMP006-ORIGINAL-LEDGER.json"

def replay(original:Path,cache:Path,out:Path):
    l=json.loads(LEDGER.read_text());source=lock()
    before=json.loads((original/"BEFORE-DEV-SHA-FREEZE.json").read_text())
    scores=json.loads((original/"FIRST-ORIGINAL-REAL-DEV-SCORES.json").read_text())
    if (not before["all_six_original_model_optimizer_rng_SHA_frozen_before_DEV"]
        or before["source_revision"]!=source["revision_sha"]
        or scores["complete_original_gate_pass"]
        or before["TEST_35_to_49_NEVER_DOWNLOADED"] is not True):
        raise RuntimeError("original frozen research integrity/gate mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit={}
    for ep in FIT:
        fit[ep],_ = get_episode(ep,source,cache)
    norm=fit_norm(fit)
    for k,v in norm.items():
        if not np.array_equal(np.asarray(v),np.asarray(
                before["normalizers_from_only_original_FIT"][k])):
            raise RuntimeError(f"original source FIT normalizer changed: {k}")
    dev_hashes={r["episode"]:r["sha256"]
                for r in scores["DEV_original_file_SHA_receipts"]}
    if set(dev_hashes)!=set(DEV):
        raise RuntimeError("original first DEV hash count inconsistent")
    dev={}
    for ep in DEV:
        rows,receipt=get_episode(ep,source,cache,released_dev=True)
        if receipt["sha256"]!=dev_hashes[ep]:
            raise RuntimeError("original DEV Parquet changed")
        dev[ep]=encode(rows,norm)
    delta=torch.tensor(norm["train_delta_scale_normalized"],dtype=torch.float32)
    verified=[]
    for seed in SEEDS:
        models=model_pair(seed,delta)
        first=next(x for x in scores["ORIGINAL_FIRST_DEV_context_test"]
                   if x["seed"]==seed)
        for arm in ARMS:
            filename=original/f"{arm}-{seed}.pt"
            real_digest=hashlib.sha256(filename.read_bytes()).hexdigest()
            original_sha=next(x for x in l["original_6_frozen_model_optimizer_rng_hashes"]
                 if x["seed"]==seed and x["arm"]==arm)["model_optimizer_rng_SHA256"]
            if real_digest!=original_sha:
                raise RuntimeError("original checkpoint tampering/digest mismatch")
            # Originally GitHub Actions-produced own model: SHA verified first.
            checkpoint=torch.load(filename,map_location="cpu",weights_only=False)
            if (checkpoint["immutable_publisher_sha"]!=source["revision_sha"]
                or checkpoint["FIT_original_hashes"]!=[
                      r["sha256"] for r in before["source_fit_Sha256"]]
                or checkpoint["arm"]!=arm or checkpoint["seed"]!=seed
                or "optimizer" not in checkpoint or "torch_rng" not in checkpoint):
                raise RuntimeError("original source/optimizer snapshot not qualified")
            model=models[arm]
            model.load_state_dict(checkpoint["model"],strict=True)
            model.eval()
            for H in (128,512):
                original_result=first["scores"][f"{arm}_H{H}"]
                actual=eval_context(model,dev,norm,H)
                if (not np.isclose(actual["mean_source_MSE"],
                       original_result["mean_source_MSE"],rtol=1e-5,atol=1e-11)
                    or (actual["motion_proxy_AUPRC"] is not None
                    and original_result["motion_proxy_AUPRC"] is not None
                    and abs(actual["motion_proxy_AUPRC"]-
                            original_result["motion_proxy_AUPRC"])>0.005)):
                    raise RuntimeError(f"original first DEV not reproducible {seed}/{arm}/H{H}")
                row=dict(seed=seed,arm=arm,H=H,original_SHA256=real_digest,
                    source_MSE=actual["mean_source_MSE"],
                    original_MSE=original_result["mean_source_MSE"],
                    original_AUPRC=original_result["motion_proxy_AUPRC"],
                    replay_AUPRC=actual["motion_proxy_AUPRC"])
                verified.append(row)
                print("S1TEMP006_FROZEN_COLD_REPLAY",json.dumps(row,sort_keys=True),flush=True)
    out.mkdir(parents=True,exist_ok=True)
    receipt=dict(study="S1TEMP006",original_fit_run=40291636780,
        original_6_model_SHA_verified=True,
        original_dev_source_SHA_verified=True,full_original_gate="FAIL",
        no_training_optimizer_calls=True,source_TEST_never_downloaded=True,
        independent_external_robot_site=False,
        original_12_arm_seed_horizon_scores_verified=verified)
    (out/"READONLY-ORIGINAL-REPLAY.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print("S1TEMP006_ORIGINAL_TWELVE_SCORES_COLD_REPLAY=PASS",flush=True)
    print("S1TEMP006_SCIENTIFIC_ORIGINAL_GATE=FAIL",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original-dir",required=True)
    p.add_argument("--cache",default="/tmp/s1temp006-original-READONLY")
    p.add_argument("--out",default="research/diagnostics/s1temp006-cold-replay")
    a=p.parse_args()
    replay(Path(a.original_dir),Path(a.cache),ROOT/a.out)
