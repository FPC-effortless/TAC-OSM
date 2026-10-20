#!/usr/bin/env python3
"""ME019F: original frozen 18 real-model checkpoints on FIRST unseen user4.

Requires immutable original ME019 manifest SHA256 pinned BEFORE source opening.
Never use this code to train/tune on future data or select best seed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from scripts.me014b_exact_action_memory import (
    END_CUT, DEV_CUT, REQUIRED, PIN_MD5, VOCAB,
    publisher_bytes, member_for_zip, batch, binary_scores, parameters,
)
from scripts.me019_authentic_history import (
    ARMS, LENGTHS, SEEDS, PARAMS, SOURCE_SHA,
    make_model, original_128_windows, view,
)

TRAIN_RUN = 39443325909
MANIFEST_SHA = "2a10ae3205e866ed792f5ad145e3c3c775e95c39de675f421184de44e56455b6"
FUTURE_MOD = 4
MAX_FUTURE = 50000
EVAL_BATCH = 384


def restore_models(root):
    path = Path(root) / "results.json"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != MANIFEST_SHA:
        raise ValueError("original once-frozen training manifest digest mismatch")
    m = json.loads(path.read_text())
    if (m["study"] != "ME019_GENUINE_128_VS_32_PAST_HISTORY_ORIGINAL_FIT_ONLY"
            or m["real_original_data_sha256"] != SOURCE_SHA
            or m["optimizer_updates_per_arm"] != 300
            or m["exact_all_arms_trainable_params"] != PARAMS
            or not m["same_real_training_targets_and_actions_all_arms"]
            or not m["same_last32_exact_inputs_in_both_history_lengths"]
            or m["user1_2_3_4_outcome_cohorts_opened"]
            or not m["dev_already_inspected_in_prior_research"]
            or len(m["models"]) != 18):
        raise ValueError("original trained model source/contract invalid")
    result = []
    for entry in m["models"]:
        p = Path(root) / entry["trained_weights"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != entry["checkpoint_sha256"]:
            raise ValueError("original trained weights invalid before new future data")
        ck = torch.load(p,map_location="cpu",weights_only=True)
        if (ck["arm"],ck["history"],ck["seed"],ck["updates"],ck["source_sha256"]) != (
                entry["arm"],entry["history"],entry["seed"],300,SOURCE_SHA):
            raise ValueError("original model checkpoint identity mismatch")
        net = make_model(entry["arm"],entry["history"],script_native=False)
        net.load_state_dict(ck["state_dict"], strict=True)
        if parameters(net) != PARAMS or entry["active_params"] != PARAMS:
            raise ValueError("strict parameter parity failed")
        net.eval()
        result.append((entry,net))
    if {(e["arm"],e["history"],e["seed"]) for e,_ in result} != {
        (a,h,s) for a in ARMS for h in LENGTHS for s in SEEDS
    }:
        raise ValueError("original 18-model matrix missing a seed/arm")
    return result


def read_original_user4():
    raw,meta = publisher_bytes()
    if meta["zip_sha256"] != SOURCE_SHA or meta["zip_md5"] != PIN_MD5:
        raise ValueError("genuine publisher ZIP hash mismatch")
    users={}
    future_records=0
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        member = member_for_zip(z)
        with z.open(member) as fd:
            reader=csv.DictReader(io.TextIOWrapper(fd,encoding="utf-8-sig",newline=""))
            if not set(REQUIRED).issubset(reader.fieldnames or []):
                raise ValueError("publisher schema changed")
            for r in reader:
                try:
                    uid = int(r["user_id"])
                    if uid<0 or uid%16!=FUTURE_MOD: continue
                    timestamp = float(r["timestamp"])
                except (ValueError,TypeError):
                    continue
                if not math.isfinite(timestamp) or timestamp<=1e9 or timestamp>=END_CUT:
                    continue
                try:
                    item = int(r["video_id"])
                    duration = float(r["video_duration"])
                    ratio = float(r["watch_ratio"])
                except (ValueError,TypeError):
                    continue
                if (not 0<=item<VOCAB or not math.isfinite(duration)
                        or duration<=0 or not math.isfinite(ratio) or ratio<0):
                    continue
                users.setdefault(uid,[]).append((
                    timestamp,item,math.log1p(duration)/12,int(ratio>=1.0)))
                future_records += int(timestamp>=DEV_CUT)
    for records in users.values():
        records.sort(key=lambda z:z[0])
    if future_records<1500:
        raise ValueError("too few genuine unopened future new-user outcomes")
    meta["genuine_user4_future_parsed_after_weights_frozen"]=True
    meta["genuine_future_response_records"]=future_records
    meta["source_member"]=member
    return users,meta


def model_predictions(net, data):
    p=[]
    start=time.perf_counter()
    with torch.inference_mode():
        for first in range(0,len(data["y"]),EVAL_BATCH):
            sl=slice(first,min(first+EVAL_BATCH,len(data["y"])))
            p.append(torch.sigmoid(net(*batch(data,sl))).flatten().cpu())
    return torch.cat(p).numpy(),round(time.perf_counter()-start,3)


def user_bootstrap_interval(delta, ids, seed=19019, repetitions=1500):
    delta=np.asarray(delta,dtype=np.float64).reshape(-1)
    users,inverse=np.unique(np.asarray(ids,dtype=np.int64),return_inverse=True)
    sums=np.bincount(inverse,weights=delta,minlength=len(users))
    counts=np.bincount(inverse,minlength=len(users))
    rng=np.random.default_rng(seed)
    samples=rng.integers(len(users),size=(repetitions,len(users)))
    resampled=(sums[samples].sum(axis=1)/counts[samples].sum(axis=1))
    return {"n_users":len(users),"observed_mean_delta":float(delta.mean()),
        "cluster_bootstrap_95_percent":list(map(float,np.quantile(resampled,[.025,.975]))),
        "replicates":repetitions,"bootstrap_seed":seed}


def run(original,output):
    torch.set_num_threads(1)
    # ALL 18 original checkpoints and original manifest verified BEFORE opening
    # even a single genuine user4 outcome.
    loaded=restore_models(original)
    users,source=read_original_user4()
    long=original_128_windows(users,DEV_CUT,END_CUT,limit=MAX_FUTURE)
    short=view(long,32)
    if not torch.equal(short["items"],long["items"][:,-32:]):
        raise ValueError("causal most-recent-32 history drift")
    if not torch.equal(short["metadata"],long["metadata"][:,-32:]):
        raise ValueError("causal observed feedback drift")
    if any(uid%16!=4 for uid in long["users"]):
        raise ValueError("new real users mixed with fitted cohort")
    if (min(long["timestamps"])<DEV_CUT
            or max(long["timestamps"])>=END_CUT
            or len(long["y"])<1500):
        raise ValueError("new user4 chronological/source minimum test failed")
    truth=long["y"].flatten().numpy()
    history_prior=long["metadata"][:,-32:,1].mean(-1).numpy()
    report={"study":"ME019F_FIRST_FROZEN_USER4_REAL_LONG_HISTORY",
        "train_original_run":TRAIN_RUN,
        "train_original_manifest_sha256":MANIFEST_SHA,
        "source_sha256":SOURCE_SHA,
        "source":source,
        "original_full_18_checkpoint_hashes_verified_before_user4_outcomes":True,
        "genuine_future_users":len(set(long["users"])),
        "genuine_future_test_windows":len(long["y"]),
        "history_exact_suffix_match":True,
        "frozen_original_train_user_remainder":0,
        "new_test_user_remainder":4,
        "genuine_new_test_period":["2020-08-14","2020-09-06"],
        "model_optimizer_updates_on_test":0,
        "all_trainable_params":PARAMS,
        "head_to_head_training_compute_fair":False,
        "causal_action_policy_value_claimed":False,
        "model_promoted":False,
        "true_prior32_baseline":binary_scores(history_prior,truth),
        "models":[],"paired_user_clustered_differences":[]}
    predictions={}
    for entry,model in loaded:
        history=entry["history"]
        tested=short if history==32 else long
        p,seconds=model_predictions(model,tested)
        key=(entry["arm"],history,entry["seed"])
        predictions[key]=p
        row={"arm":entry["arm"],"history":history,"seed":entry["seed"],
            "original_checkpoint_sha256":entry["checkpoint_sha256"],
            "original_training_seconds":entry["train_seconds"],
            "real_frozen_user4":binary_scores(p,truth),
            "inference_seconds":seconds}
        report["models"].append(row)
        print(json.dumps({"arm":entry["arm"],"history":history,
            "seed":entry["seed"],"real_user4_brier":row["real_frozen_user4"]["brier"]},
             sort_keys=True),flush=True)
    for seed in SEEDS:
        anchor=predictions[("native_chrono_anchor",128,seed)]
        for arm,h,name in (
                ("native_chrono_anchor",32,"native128_minus_native32"),
                ("gru_original",128,"native128_minus_original_GRU128"),
                ("gru_anchor",128,"native128_minus_anchored_GRU128")):
            comparator=predictions[(arm,h,seed)]
            delta=(anchor-truth)**2-(comparator-truth)**2
            ci=user_bootstrap_interval(delta,long["users"],seed=19019)
            report["paired_user_clustered_differences"].append({
                "seed":seed,"contrast":name,**ci})
    out=Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"ME019F_ORIGINAL_FROZEN_UNSEEN_USER4_REAL_RESPONSE_TEST",
        "full_18_frozen_models":len(report["models"]),
        "test_windows":len(long["y"]),
        "test_users":report["genuine_future_users"],
        "no_refits":True,"causal_effect_claimed":False},sort_keys=True))
    return report


def preflight():
    # Artificial numbers used ONLY for numeric/software contracts.
    torch.set_num_threads(1)
    torch.manual_seed(190190)
    ids=torch.randint(VOCAB,(3,128))
    meta=torch.rand(3,128,3)
    meta[:,:,1]=(meta[:,:,1]>.5).float()
    actions=torch.randint(VOCAB,(3,))
    duration=torch.rand(3,1)
    for arm in ARMS:
        for history in LENGTHS:
            net=make_model(arm,history,script_native=False)
            p=net(ids[:,-history:],meta[:,-history:],actions,duration)
            assert p.shape==(3,1) and torch.isfinite(p).all()
            assert parameters(net)==PARAMS
    sample=user_bootstrap_interval([.01,-.01,.02,-.02],[4,4,20,20])
    assert sample["n_users"]==2 and len(sample["cluster_bootstrap_95_percent"])==2
    print(json.dumps({"status":"ME019F_FROZEN_USER4_SOFTWARE_PREFLIGHT",
        "independent_data_read":False,"test_outcomes_used":False},sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",default="results/me019")
    p.add_argument("--out",default="results/me019f/unseen_user4_original.json")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    preflight() if a.self_test else run(a.original,a.out)
