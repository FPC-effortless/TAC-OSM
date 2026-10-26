#!/usr/bin/env python3
"""Read-only original real Coffee six-checkpoint velocity-head cold replay."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.s1motion006_real_coffee_fit import (
    FIT,DEV,SEEDS,ARMS,_load_lock,episode,make_models)
from scripts.s1motion004_real_towel_fit import prepare_fit_only,encode_source_episode
from scripts.s1motion003_real_tape_fit import score

def replay(original:Path,cache:Path,out:Path):
    src=_load_lock()
    pre=json.loads((original/"BEFORE-DEV-FREEZE.json").read_text())
    frozen=json.loads((original/"FIRST-REAL-DEV-SCORES.json").read_text())
    ledger=json.loads((ROOT/"research/parallel-six/S1MOTION006-ORIGINAL-LEDGER.json").read_text())
    if (not pre["all_SIX_model_optimizer_RNG_SHA_frozen_before_DEV"] or
        frozen["full_three_seed_acceptance"] or
        pre["locked_publisher_source"]!=src["revision_sha"]):
        raise RuntimeError("original source lock or failed gate changed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rawfit={}
    for ep in FIT:
        rawfit[ep],_ = episode(ep,src,cache)
    train,norm=prepare_fit_only(rawfit)
    for k,v in norm.items():
        if not np.array_equal(np.asarray(v),np.asarray(pre["normalizers_FIT_only"][k])):
            raise RuntimeError("unfrozen normalization dependency")
    dev_receipts={x["ep"]:x["sha256"]
                  for x in frozen["DEV_file_hashes_opened_after_FIT_model_freeze"]}
    if set(dev_receipts)!=set(DEV):
        raise RuntimeError("original DEV hash census incomplete")
    original_dev={}
    for ep in DEV:
        rows,receipt=episode(ep,src,cache,dev_released=True)
        if receipt["sha256"]!=dev_receipts[ep]:
            raise RuntimeError("original DEV source altered")
        original_dev[ep]=encode_source_episode(rows,norm)
    scale=torch.tensor(norm["train_delta_scale_normalized"],dtype=torch.float32)
    checked=[]
    for seed in SEEDS:
        models=make_models(seed,scale)
        prev=next(x for x in frozen["original_first_DEV_scores"] if x["seed"]==seed)
        for arm in ARMS:
            name=original/f"{arm}-{seed}.pt"
            digest=hashlib.sha256(name.read_bytes()).hexdigest()
            prior_digest=next(x for x in ledger["six_original_frozen_fit_checkpoint_SHAs"]
                              if x["seed"]==seed and x["arm"]==arm)["sha256"]
            if digest!=prior_digest:
                raise RuntimeError("source checkpoint SHA altered")
            ck=torch.load(name,map_location="cpu",weights_only=False)
            if (ck["source_revision"]!=src["revision_sha"] or ck["arm"]!=arm
                or ck["seed"]!=seed or "optimizer" not in ck
                or "torch_rng" not in ck or ck["FIT_source_sha256"]!=[
                  x["sha256"] for x in pre["FIT_verified_receipts"]]):
                raise RuntimeError("checkpoint source/optimizer chain invalid")
            m=models[arm];m.load_state_dict(ck["model"],strict=True);m.eval()
            with torch.no_grad():
                result=score(m,original_dev,norm)
            old=prev["scores"][arm]
            if (not np.isclose(result["raw_joint_MSE"]["tac"],
                                old["raw_joint_MSE"]["tac"],atol=1e-11,rtol=1e-5)
                or abs(result["proxy_AUPRC"]-old["proxy_AUPRC"])>0.005):
                raise RuntimeError("nonrepeatable original next-source MSE/AUPRC")
            receipt=dict(seed=seed,arm=arm,original_sha256=digest,
                         replay_MSE=result["raw_joint_MSE"]["tac"],
                         original_MSE=old["raw_joint_MSE"]["tac"],
                         replay_motion_AUPRC=result["proxy_AUPRC"],
                         original_motion_AUPRC=old["proxy_AUPRC"])
            checked.append(receipt)
            print("S1MOTION006_ORIGINAL_COLD_REPLAY",json.dumps(receipt,sort_keys=True),flush=True)
    out.mkdir(parents=True,exist_ok=True)
    summary=dict(study="S1MOTION006",original_run=40290076035,
        six_original_model_optimizer_rng_SHA_verified=True,
        source_original_FIT_and_DEV_verified=True,
        original_full_gate="FAIL_0_OF_3",
        replay_optimizer_steps=0,
        no_PROTECTED_TEST_downloads=True,
        original_first_scores=checked)
    (out/"NO_FIT-ORIGINAL-REPLAY.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print("S1MOTION006_ALL_6_FROZEN_MODEL_REPLAY=PASS",flush=True)
    print("S1MOTION006_SCIENTIFIC_GATE=FAIL_0_OF_3",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original-dir",required=True)
    p.add_argument("--cache",default="/tmp/s1motion006-cold-replay")
    p.add_argument("--out",default="research/diagnostics/s1motion006-cold-replay")
    a=p.parse_args();replay(Path(a.original_dir),Path(a.cache),ROOT/a.out)
