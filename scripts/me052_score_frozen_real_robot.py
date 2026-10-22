#!/usr/bin/env python3
"""ME052 original frozen checkpoint evaluation, no re-training.

Prerecorded training on immutable actual BodenAI robot histories completed in
Actions 39714369920; all TEN checkpoint SHA hashes are preserved. That run
failed after all checkpoint freeze on an EMPTY publisher DEV parquet.
This recovery only reads original frozen models, retries identical-revision
source files and excludes physically unavailable/corrupt ORIGINAL episode
parquets by SOURCE integrity, NEVER by model error or outcome label.
Any omission from originally registered 80 DEV/100 TEST is disclosed and
prevents strict preregistered capability superiority certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scripts.me052_train_real_robot import (
    ARMS,SEEDS,load_episode,normalize,network,score,sha_file,write_json,
    source_index)
from scripts.audit_real_hil_trajectory import TASK,download_file
from scripts.qualify_real_robot_cohort import REVISION

torch.set_num_threads(1)


def available_source(cache,idx,meta):
    name=f"{TASK}/data/chunk-{idx//1000:03d}/episode_{idx:06d}.parquet"
    last=None
    for attempt in range(3):
        cached=cache/name
        if cached.exists() and cached.stat().st_size==0:
            cached.unlink()
        try:
            row=load_episode(cache,idx,meta)
            if row["length"]<2:raise ValueError("original source insufficient")
            return row,None
        except Exception as ex:
            # A source file that cannot even be opened is NOT fabricated.
            last=type(ex).__name__+":"+str(ex)[:140]
            if attempt<2:time.sleep(1)
    return None,{"original_episode_id":idx,
                 "source_path":name,"source_integrity_failure":last,
                 "excluded_on_source_availability_not_on_scoring":True}


def load_original_cohort(cache,cohorts,meta,part):
    data={};bad=[]
    for k,(idx,n) in enumerate(cohorts,1):
        row,error=available_source(cache,idx,meta)
        if error:
            bad.append(error)
            continue
        if row["length"]!=n:
            bad.append({"original_episode_id":idx,"source_path":"metadata_length_mismatch",
                        "excluded_on_source_availability_not_on_scoring":True})
            continue
        data[idx]=row
        if k%25==0:
            print(json.dumps({"event":"ME052_ORIGINAL_SOURCE_EVAL_AUDIT",
                               "part":part,"checked":k,"available":len(data),
                               "missing":len(bad)}),flush=True)
    if len(data)<30:raise RuntimeError("fewer than 30 genuine original evaluation episodes")
    return data,bad


def group_bootstrap(scores):
    keys=set(scores[f"native_tac-s{SEEDS[0]}"]["episode_metrics"])
    for seed in SEEDS:
        keys &= set(scores[f"native_tac-s{seed}"]["episode_metrics"])
        keys &= set(scores[f"gru_world-s{seed}"]["episode_metrics"])
    keys=sorted(keys,key=int)
    diffs=[];weights=[]
    for eid in keys:
        diff=0.;weight=0
        for seed in SEEDS:
            tac=scores[f"native_tac-s{seed}"]["episode_metrics"][eid]
            gru=scores[f"gru_world-s{seed}"]["episode_metrics"][eid]
            if tac["observed_joint_coordinates"]!=gru["observed_joint_coordinates"]:
                raise RuntimeError("different real evaluation targets between architectures")
            diff+=gru["MAE_sum_real_units"]-tac["MAE_sum_real_units"]
            weight+=tac["observed_joint_coordinates"]
        diffs.append(diff);weights.append(weight)
    rng=np.random.default_rng(52052)
    draws=np.asarray(diffs,dtype=float);counts=np.asarray(weights,dtype=float)
    bootstrap=[]
    for _ in range(1000):
        ids=rng.integers(len(keys),size=len(keys))
        bootstrap.append(float(draws[ids].sum()/counts[ids].sum()))
    return {"complete_original_episodes_scored":len(keys),
       "paired_five_seed_gru_minus_tac_MAE_real_units_CI95":
           [float(x) for x in np.quantile(bootstrap,[.025,.975])],
       "episode_resamples":1000,"bootstrap_seed":52052}


def main(root):
    root=Path(root)
    original=json.loads((root/"results.json").read_text())
    records=original["actual_completed_training_models"]
    if len(records)!=10 or original["all_frozen_original_models_before_heldout_opening"] is not True:
        raise RuntimeError("original actual trained model freeze evidence missing")
    for r in records:
        if sha_file(root/r["original_checkpoint_path"])!=r["original_checkpoint_sha256"]:
            raise RuntimeError("real original trained weights SHA mismatch")
    cache=root/"private-EVAL-DO-NOT-UPLOAD"
    cache.mkdir(exist_ok=True)
    parts,meta,meta_hash=source_index(cache)
    if meta_hash!=original["original_source_metadata_sha256"]:
        raise RuntimeError("original source metadata changed after actual FIT")
    saved=torch.load(root/records[0]["original_checkpoint_path"],
                     map_location="cpu",weights_only=False)
    scaling=saved["fixed_fit_only_scaling"]
    results={"evaluation_stage":"ME052_original_frozen_models_authentic_robot_score",
      "original_training_run":"39714369920",
      "original_source_revision":REVISION,
      "trained_models":10,
      "model_updates_during_evaluation":0,
      "all_original_checkpoints_sha_verified":True,
      "same_heldout_available_source_episodes_for_both_architectures":True,
      "all_historical_negative_results_preserved":True,
      "original_registered_episode_count":{"DEV":80,"TEST":100},
      "deviation_from_original_registered_holodout":"Missing/unreadable original Parquet files disclosed; no postfit parameters or model-selection changes",
      "strict_original_test_superiority_gate_qualified":False,
      "original_source_missing_or_corrupt":{},"scores":{}}
    write_json(root/"original-source-recovered-evaluation.json",results)
    for part in ("DEV","TEST"):
        source,missing=load_original_cohort(cache,parts[part],meta,part)
        results["original_source_missing_or_corrupt"][part]=missing
        results[part+"_original_available_episodes"]=len(source)
        examples=normalize(source,scaling)
        results["scores"][part]={}
        for rec in records:
            model=network(rec["architecture"])
            cp=torch.load(root/rec["original_checkpoint_path"],
                          map_location="cpu",weights_only=False)
            model.load_state_dict(cp["state_dict"],strict=True)
            z=score(model,examples,meta,scaling,part)
            key=f'{rec["architecture"]}-s{rec["seed"]}'
            results["scores"][part][key]=z
            print(json.dumps({"event":"ME052_ACTUAL_HELDOUT_REAL_ROBOT_PREDICTION",
                    "part":part,"model":key,
                    "physical_MAE":z["physical_MAE_real_units"],
                    "real_persistence_MAE":z["persistence_MAE_real_units"],
                    "real_test_onset_positives":z["original_heldout_real_onset_positives"]}),flush=True)
            write_json(root/"original-source-recovered-evaluation.json",results)
    paired={}
    wins=0
    s=results["scores"]["TEST"]
    for seed in SEEDS:
        native=s[f"native_tac-s{seed}"]["physical_MAE_real_units"]
        gru=s[f"gru_world-s{seed}"]["physical_MAE_real_units"]
        wins+=int(native<gru)
        paired[str(seed)]={"native_Tac_MAE":native,"trained_GRU_MAE":gru,
               "native_beats_GRU":native<gru}
    results["original_Tac_seed_wins"]=wins
    results["five_seed_scores"]=paired
    results["paired_episode_bootstrap"]=group_bootstrap(s)
    results["strict_original_test_superiority_gate_qualified"]=(
        not results["original_source_missing_or_corrupt"]["DEV"]
        and not results["original_source_missing_or_corrupt"]["TEST"]
        and wins>=4
        and results["paired_episode_bootstrap"]["paired_five_seed_gru_minus_tac_MAE_real_units_CI95"][0]>0)
    results["evaluation_complete_on_original_available_heldout_episodes"]=True
    write_json(root/"original-source-recovered-evaluation.json",results)
    print(json.dumps({"event":"ME052_REAL_ROBOT_FROZEN_TEN_MODELS_HELDOUT_RESULTS",
         "test_original_episodes":results["TEST_original_available_episodes"],
         "missing_test_files":len(results["original_source_missing_or_corrupt"]["TEST"]),
         "five_seed_scores":paired,
         "confidence_interval":results["paired_episode_bootstrap"],
         "strict_qualification":results["strict_original_test_superiority_gate_qualified"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",default="results/me052")
    x=p.parse_args();main(x.root)
