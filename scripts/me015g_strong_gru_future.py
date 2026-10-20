#!/usr/bin/env python3
"""ME015G: first genuine USER%16==2 action-response test vs strongest original GRU.

Same authentic KuaiRec task. Frozen 135050-parameter model weights from ME014B
and ME015 verified BEFORE new disjoint user reaction records are parsed.
"""
from __future__ import annotations
import csv,hashlib,io,json,math,time,zipfile
from pathlib import Path
import numpy as np
import torch
from scripts.me015_native_optimization import EngineeredAgent,SEEDS,SOURCE_SHA
from scripts.me014b_exact_action_memory import (
    ActionMemory,publisher_bytes,member_for_zip,REQUIRED,VOCAB,
    FIT_CUT,DEV_CUT,END_CUT,build_examples,empirical_baselines,
    binary_scores,batch,parameters)

REGISTERED=("native_original","native_chrono_anchor","gru_anchor")
ALL_MODELS=REGISTERED+("gru_original_unanchored",)

def load_and_verify(me015,me014b):
    a=json.loads((Path(me015)/"results.json").read_text())
    b=json.loads((Path(me014b)/"results.json").read_text())
    if a["study"]!="ME015_NATIVE_TIMESCALE_CAUSAL_PRIOR_DIAGNOSIS":
        raise ValueError("wrong engineered original fit")
    if b["study"]!="ME014B_EXACT_PARAMETER_REAL_LOGGED_ACTION_MEMORY":
        raise ValueError("wrong original unanchored GRU fit")
    if a["future_Aug14_Sept06_reopened"] or b["future_test_parsed"]:
        raise ValueError("original fit contaminated")
    if a["source"]["zip_sha256"]!=SOURCE_SHA or b["source"]["zip_sha256"]!=SOURCE_SHA:
        raise ValueError("real publisher source checksum discrepancy")
    if len(a["models"])!=15 or len(b["fits"])!=6:raise ValueError("original archives incomplete")
    loaded=[]
    for fit in a["models"]:
        path=Path(me015)/fit["checkpoint"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=fit["checkpoint_sha256"]:
            raise ValueError("ME015 checkpoint digest invalid")
        ck=torch.load(path,weights_only=True,map_location="cpu")
        if ck["kind"]!=fit["model"] or ck["seed"]!=fit["seed"] or ck["steps"]!=300:
            raise ValueError("ME015 registered identity mismatch")
        if fit["trainable_params"]!=135050:raise ValueError("unmatched active weight count")
        if fit["model"] in REGISTERED:
            m=EngineeredAgent(fit["model"])
            m.load_state_dict(ck["state_dict"],strict=True);m.eval()
            loaded.append(({"arm":fit["model"],"seed":fit["seed"],
                            "sha256":fit["checkpoint_sha256"],
                            "trainable":fit["trainable_params"],
                            "train_seconds":fit["actual_training_seconds"]},m))
    for fit in b["fits"]:
        path=Path(me014b)/fit["checkpoint"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=fit["sha256"]:
            raise ValueError("original GRU control checkpoint digest invalid")
        ck=torch.load(path,weights_only=True,map_location="cpu")
        if ck["arm"]!=fit["arm"] or ck["seed"]!=fit["seed"] or ck["steps"]!=300:
            raise ValueError("original GRU control identity mismatch")
        if fit["parameters"]!=135050:raise ValueError("original GRU not exactly capacity matched")
        if fit["arm"]=="gru":
            m=ActionMemory("gru");m.load_state_dict(ck["state_dict"],strict=True);m.eval()
            loaded.append(({"arm":"gru_original_unanchored","seed":fit["seed"],
                            "sha256":fit["sha256"],"trainable":fit["parameters"],
                            "train_seconds":fit["train_wall_seconds"]},m))
    if {(x["arm"],x["seed"]) for x,m in loaded}!={(arm,seed)
        for arm in ALL_MODELS for seed in SEEDS}:
        raise ValueError("4 original three-seed arms not present")
    if any(parameters(m)!=135050 for _,m in loaded):
        raise ValueError("strict active parameter mismatch")
    return loaded

def read_disjoint_user2():
    raw,meta=publisher_bytes()
    if meta["zip_sha256"]!=SOURCE_SHA:raise ValueError("publisher ZIP changed")
    known={};novel={};n_future=0
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        member=member_for_zip(z)
        with z.open(member) as fd:
            reader=csv.DictReader(io.TextIOWrapper(fd,encoding="utf-8-sig",newline=""))
            if not set(REQUIRED).issubset(set(reader.fieldnames or [])):
                raise ValueError("publisher real exposure schema changed")
            for row in reader:
                try:uid=int(row["user_id"])
                except (TypeError,ValueError):continue
                if uid<0 or uid%16 not in (0,2):continue
                try:ts=float(row["timestamp"])
                except (TypeError,ValueError):continue
                if not math.isfinite(ts) or ts<=1e9 or ts>=END_CUT:continue
                if uid%16==0 and ts>=DEV_CUT:continue
                try:
                    item=int(row["video_id"])
                    duration=float(row["video_duration"])
                    ratio=float(row["watch_ratio"])
                except (TypeError,ValueError):continue
                if not (0<=item<VOCAB and math.isfinite(duration) and duration>0
                        and math.isfinite(ratio) and ratio>=0):continue
                record=(ts,item,math.log1p(duration)/12,int(ratio>=1))
                if uid%16==0:known.setdefault(uid,[]).append(record)
                else:
                    novel.setdefault(uid,[]).append(record)
                    n_future+=int(ts>=DEV_CUT)
    for group in (known,novel):
        for events in group.values():events.sort(key=lambda e:e[0])
    if set(known)&set(novel):raise ValueError("new cohort leakage")
    if n_future<500:raise ValueError("new user response cohort insufficient")
    meta.update({"source_member":member,"train_user_remainder":0,
                 "genuinely_unseen_user_remainder":2,
                 "future_user2_response_rows_first_opened":n_future,
                 "opened_after_all_frozen_checkpoint_hashes":True,
                 "parsed_no_outcomes_on_or_after_sep6":True})
    return known,novel,meta

def predictions(model,hold):
    model.eval();out=[];start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(hold["y"]),512):
            sl=slice(i,min(i+512,len(hold["y"])))
            out.append(torch.sigmoid(model(*batch(hold,sl))).reshape(-1))
    return torch.cat(out).numpy(),round(time.perf_counter()-start,4)

def run():
    checkpoints=load_and_verify("results/me015","results/me014b")
    known,novel,src=read_disjoint_user2()
    train=build_examples(known,0,FIT_CUT)
    dev=build_examples(known,FIT_CUT,DEV_CUT)
    if (len(train["y"]),len(dev["y"]))!=(68825,106245):
        raise ValueError("changed original FIT/DEV benchmark")
    test=build_examples(novel,DEV_CUT,END_CUT,max_windows=50000)
    if len(test["y"])<500 or any(uid%16!=2 for uid in test["users"]):
        raise ValueError("new user group invalid")
    baseline=empirical_baselines(train,test)
    baselines={key:binary_scores(baseline[field],test["y"].numpy())
       for key,field in (
          ("global_training_rate","global_predictions"),
          ("video_training_rate","item_predictions"),
          ("causal_real_prior32_user_rate","history_mean_predictions"))}
    result={"study":"ME015G_FIRST_FROZEN_STRONGEST_GRU_USER2",
      "source":src,"source_sha256":SOURCE_SHA,
      "user_group_never_fitted_by_models":2,
      "original_training_user_group":0,
      "future_period":["2020-08-14","2020-09-06"],
      "new_user_count":len(set(test["users"])),"new_actual_logged_reactions":len(test["y"]),
      "all_21_original_checkpoint_hashes_verified_before_user2":True,
      "original_me015_run":39398038257,"original_me014b_run":39394500641,
      "active_parameters_for_every_model":135050,
      "optimizer_steps_on_test":0,"trained_compute_equalized":False,
      "policy_propensities_known":False,"causal_action_value_estimated":False,
      "model_promoted":False,"real_trainonly_baselines":baselines,
      "models":[]}
    torch.set_num_threads(1)
    for record,net in checkpoints:
        p,elapsed=predictions(net,test)
        result["models"].append({**record,
             "future_scores":binary_scores(p,test["y"].numpy()),
             "test_inference_cpu_seconds":elapsed})
        print(json.dumps({"arm":record["arm"],"seed":record["seed"],
             "heldout_unseen_user_brier":result["models"][-1]["future_scores"]["brier"]},sort_keys=True),flush=True)
    folder=Path("results/me015g");folder.mkdir(parents=True,exist_ok=True)
    (folder/"strongest_gru_new_user_results.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"ME015G_ORIGINAL_FROZEN_STRONGEST_GRU_REAL_TEST_COMPLETE",
       "test_windows":len(test["y"]),"test_users":result["new_user_count"],
       "trained_refits":0},sort_keys=True))
    return result

if __name__=="__main__":
    run()
