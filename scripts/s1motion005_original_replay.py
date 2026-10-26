#!/usr/bin/env python3
"""S1MOTION005 original 15 frozen-checkpoint read-only forensic replay.

Never calls optimizer.step; never uses protected TEST episode files.
Forensic measures of FIT gradient budget are source-only diagnostics,
not new research training or altered original source qualification.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.s1motion005_real_screwdriver_fit import (
    FIT,DEV,SEEDS,ARMS,LOCK,approved_source_metadata,
    episode_data,original_models
)
from scripts.s1motion004_real_towel_fit import prepare_fit_only,encode_source_episode
from scripts.s1motion003_real_tape_fit import score,windows,batch
from tac_osm.variants.s1motion005_gradient_budget import compute_gradients

LEDGER=ROOT/"research/parallel-six/S1MOTION005-ORIGINAL-LEDGER.json"

def replay(src,cache,out):
    lock=json.loads(LOCK.read_text());approved_source_metadata(lock)
    ledger=json.loads(LEDGER.read_text())
    prior=json.loads((src/"BEFORE-DEV-ALL-FROZEN.json").read_text())
    original=json.loads((src/"ORIGINAL-FIRST-DEV-SCORES.json").read_text())
    if (prior["source_locked_revision"]!=lock["revision_sha"]
        or not prior["all_15_model_optimizer_RNG_state_blobs_SHA_frozen_before_any_DEV_file"]
        or tuple(prior["seeds"])!=SEEDS
        or original["full_original_predictive_gate_pass"]):
        raise RuntimeError("pre-DEV frozen source/original scientific gate receipt changed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit_src={}
    for record in lock["fit_files"]:
        ep=int(record["episode"])
        fit_src[ep],current=episode_data(ep,lock,cache,expected_sha=record["sha256"])
    fit,norm=prepare_fit_only(fit_src)
    for key,value in norm.items():
        if not np.array_equal(np.asarray(value),np.asarray(prior["normalization_FIT_only"][key])):
            raise RuntimeError(f"FIT source normalization changed {key}")
    # Protected TEST file IDs are prohibited by the pinned source accessor.
    original_dev_receipts={
        int(x["episode"]):x["sha256"]
        for x in original["dev_downloaded_only_after_ALL_models_frozen"]}
    if set(original_dev_receipts)!=set(DEV):
        raise RuntimeError("DEV original hash census incomplete")
    devsrc={}
    for ep in DEV:
        devsrc[ep],r=episode_data(ep,lock,cache,expected_sha=original_dev_receipts[ep])
    dev={ep:encode_source_episode(devsrc[ep],norm) for ep in DEV}
    delta=torch.tensor(norm["train_delta_scale_normalized"],dtype=torch.float32)
    aud=[]
    for seed in SEEDS:
        arms=original_models(seed,delta)
        original_seed=next(x for x in original["original_first_dev_results"]
                           if x["seed"]==seed)
        for arm in ARMS:
            record=next(x for x in ledger["all_original_model_fit_receipts"]
                        if x["arm"]==arm and x["seed"]==seed)
            filename=src/f"{arm}-{seed}.pt"
            digest=hashlib.sha256(filename.read_bytes()).hexdigest()
            if digest!=record["model_SHA256"]:
                raise RuntimeError("original frozen checkpoint altered")
            # Artifact is our own original source model, hash checked before
            # trusted Torch state object deserialization.
            ck=torch.load(filename,map_location="cpu",weights_only=False)
            if (ck["source_revision"]!=lock["revision_sha"] or
                ck["source_FIT_checksums"]!=[x["sha256"] for x in lock["fit_files"]]
                or ck["seed"]!=seed or ck["arm"]!=arm or ck["updates"]!=192 or
                "optimizer" not in ck or "torch_rng" not in ck):
                raise RuntimeError("original checkpoint optimizer/source receipt mismatch")
            m=arms[arm]
            m.load_state_dict(ck["model"],strict=True)
            m.eval()
            scored=score(m,dev,norm)
            old=original_seed["score"][arm]
            old_mse=float(old["raw_joint_MSE"]["tac"])
            mse=float(scored["raw_joint_MSE"]["tac"])
            old_ap=float(old["proxy_AUPRC"])
            ap=float(scored["proxy_AUPRC"])
            if (not np.isclose(old_mse,mse,atol=1e-11,rtol=1e-5)
                or abs(ap-old_ap)>0.005
                or abs(scored["proxy_Brier"]-old["proxy_Brier"])>1e-4):
                raise RuntimeError(f"original score nonrepeatable {seed} {arm}: mse {mse}, AUPRC {ap}, originals {old_mse}, {old_ap}")
            outrow=dict(seed=seed,arm=arm,original_SHA256=digest,
                source_unit_MSE_original=old_mse,
                source_unit_MSE_replayed=mse,
                original_proxy_AUPRC=old_ap,
                replayed_proxy_AUPRC=ap,
                score_per_episode=old["per_episode"],
                model_state_bytes_unchanged=True)
            if arm.startswith("tac_"):
                selected,starts=windows(fit,seed)[0]
                sample=batch(fit,selected,starts)
                m.train()
                _,_,_,gmeta,_=compute_gradients(m,sample,arm=arm)
                m.eval()
                outrow["frozen_FIT_budget_gradient"]=gmeta
            aud.append(outrow)
            print("S1MOTION005_ORIGINAL_COLD_REPLAY",json.dumps(
                 {k:v for k,v in outrow.items() if k!="score_per_episode"},
                 sort_keys=True),flush=True)
    out.mkdir(parents=True,exist_ok=True)
    result=dict(study="S1MOTION005",original_run=40283616062,
        original_fit_artifact=12632762763,
        all_15_original_SHA256_states_verified=True,
        true_FIT_source_sha_and_scalers_repeated=True,
        original_DEV_source_SHA_repeated=True,
        source_PROTECTED_TEST_episode_files_never_downloaded=True,
        training_optimizer_steps_in_this_replay=0,
        actual_event_harm_labels_available=False,
        full_original_multitask_qualification_pass=False,
        all_original_model_records=aud)
    (out/"COLD-REPLAY-15-ORIGINAL.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("S1MOTION005_ALL_15_FROZEN_MODELS_REPLAY_PASS",flush=True)
    print("S1MOTION005_ORIGINAL_SCIENTIFIC_GATE=FAIL_2_OF_3_SEEDS",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--original-dir",required=True)
    parser.add_argument("--cache",default="/tmp/s1motion005-original-15-replay")
    parser.add_argument("--out",default="research/diagnostics/s1motion005-cold-replay")
    args=parser.parse_args()
    replay(Path(args.original_dir),Path(args.cache),ROOT/args.out)
