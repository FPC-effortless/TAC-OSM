#!/usr/bin/env python3
"""ME015F: first genuine heldout disjoint-user action outcomes, SAME KuaiRec task.

Models FIT on user_id%16==0; TEST only user_id%16==1.
All 15 frozen weight hashes verified before ANY test-user row is parsed.
This is observational cross-user generalization, not randomized policy effect.
"""
from __future__ import annotations
import csv,hashlib,io,json,math,time,zipfile
from pathlib import Path
import numpy as np
import torch
from scripts.me015_native_optimization import EngineeredAgent,ARMS,SEEDS,SOURCE_SHA
from scripts.me014b_exact_action_memory import (
    publisher_bytes,member_for_zip,VOCAB,EMB,REQUIRED,FIT_CUT,DEV_CUT,END_CUT,
    HISTORY,batch,build_examples,empirical_baselines,binary_scores,parameters)

def verify_frozen(original):
    root=Path(original);ledger=json.loads((root/"results.json").read_text())
    if ledger["study"]!="ME015_NATIVE_TIMESCALE_CAUSAL_PRIOR_DIAGNOSIS":
        raise ValueError("not the preregistered genuine train study")
    if ledger["future_Aug14_Sept06_reopened"] or ledger["architecture_promoted"]:
        raise ValueError("training reported opened test or model promoted")
    if ledger["source"]["zip_sha256"]!=SOURCE_SHA:
        raise ValueError("source publisher checksum differs")
    if (ledger["real_train_examples"],ledger["historical_dev_examples"])!=(68825,106245):
        raise ValueError("historical registered source mismatch")
    if len(ledger["models"])!=len(ARMS)*len(SEEDS):raise ValueError("missing trained models")
    loaded=[]
    for entry in ledger["models"]:
        p=root/entry["checkpoint"]
        if hashlib.sha256(p.read_bytes()).hexdigest()!=entry["checkpoint_sha256"]:
            raise ValueError("frozen checkpoint hash failed BEFORE new source exposure")
        ck=torch.load(p,map_location="cpu",weights_only=True)
        if ck["kind"]!=entry["model"] or ck["seed"]!=entry["seed"] or ck["steps"]!=300:
            raise ValueError("wrong trained model identity")
        if ck["source_sha256"]!=SOURCE_SHA or entry["trainable_params"]!=135050:
            raise ValueError("source or strict parameter mismatch")
        m=EngineeredAgent(entry["model"])
        m.load_state_dict(ck["state_dict"],strict=True)
        if parameters(m)!=135050:raise ValueError("wrong active parameter count")
        m.eval()
        loaded.append((entry,m))
    if {(entry["model"],entry["seed"]) for entry,m in loaded}!={(a,s) for a in ARMS for s in SEEDS}:
        raise ValueError("original 15-model arm matrix incomplete")
    return ledger,loaded

def read_disjoint_users(original_sha):
    raw,source=publisher_bytes()
    if source["zip_sha256"]!=original_sha:
        raise ValueError("official KuaiRec source changed")
    known={};hold={};counts={"unseen_user_rows":0,"unseen_future_rows":0,
       "historical_known_rows":0,"unseen_invalid_rows":0}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        member=member_for_zip(z)
        with z.open(member) as file:
            reader=csv.DictReader(io.TextIOWrapper(file,encoding="utf-8-sig",newline=""))
            if not set(REQUIRED).issubset(set(reader.fieldnames or [])):
                raise ValueError("publisher big_matrix schema changed")
            for row in reader:
                try:uid=int(row["user_id"])
                except (ValueError,TypeError):continue
                if uid<0:continue
                residue=uid%16
                if residue not in (0,1):continue
                try:ts=float(row["timestamp"])
                except (ValueError,TypeError):continue
                if not math.isfinite(ts) or ts<=1e9 or ts>=END_CUT:continue
                if residue==0 and ts>=DEV_CUT:continue # DON'T reopen original user future.
                try:
                    item=int(row["video_id"])
                    duration=float(row["video_duration"])
                    ratio=float(row["watch_ratio"])
                except (ValueError,TypeError):
                    counts["unseen_invalid_rows"]+=int(residue==1);continue
                if not (0<=item<VOCAB and math.isfinite(duration) and duration>0 and
                        math.isfinite(ratio) and ratio>=0):
                    counts["unseen_invalid_rows"]+=int(residue==1);continue
                fact=(ts,item,math.log1p(duration)/12.,int(ratio>=1.))
                if residue==0:
                    known.setdefault(uid,[]).append(fact)
                    counts["historical_known_rows"]+=1
                else:
                    hold.setdefault(uid,[]).append(fact)
                    counts["unseen_user_rows"]+=1
                    counts["unseen_future_rows"]+=int(ts>=DEV_CUT)
    for a in (known,hold):
        for v in a.values():v.sort(key=lambda row:row[0])
    if set(known)&set(hold):raise AssertionError("cross-user cohort contamination")
    if counts["unseen_future_rows"]<500:raise ValueError("no genuine unseen future user outcomes")
    source.update(counts)
    source["fit_user_condition"]="user_id % 16 == 0"
    source["never_trained_user_condition"]="user_id % 16 == 1"
    source["no_post_test_end_rows_parsed"]=True
    source["unseen_source_opened_after_checkpoint_verification"]=True
    source["source_member"]=member
    return known,hold,source

def forward_model(model,records,ablate=False):
    model.eval();pred=[];start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(records["y"]),512):
            sl=slice(i,min(i+512,len(records["y"])))
            pred.append(torch.sigmoid(model(*batch(records,sl),ignore_history=ablate)).reshape(-1))
    return torch.cat(pred).numpy(),round(time.perf_counter()-start,4)

def evaluate(original,out):
    original,out=Path(original),Path(out);out.mkdir(parents=True,exist_ok=True)
    ledger,models=verify_frozen(original)
    old,new,source=read_disjoint_users(SOURCE_SHA)
    train=build_examples(old,0,FIT_CUT)
    dev=build_examples(old,FIT_CUT,DEV_CUT)
    if (len(train["y"]),len(dev["y"]))!=(68825,106245):
        raise ValueError("original cohort FIT/DEV reconstruction mismatch")
    test=build_examples(new,DEV_CUT,END_CUT,max_windows=50000)
    if len(test["y"])<500:raise ValueError("insufficient never-trained real user events")
    if any(uid%16!=1 for uid in test["users"]):raise ValueError("heldout identity leak")
    priors=empirical_baselines(train,test)
    baselines={name:binary_scores(priors[key],test["y"].numpy())
      for name,key in (("global_training_rate","global_predictions"),
        ("original_training_item_rate","item_predictions"),
        ("causal_unseen_user_prior32_rate","history_mean_predictions"))}
    data={"study":"ME015F_FIRST_NEVER_TRAINED_REAL_USER_OUTCOMES",
      "source":source,
      "original_model_fit_hash":SOURCE_SHA,
      "original_genuine_fit_windows":len(train["y"]),
      "disjoint_heldout_users":len(set(test["users"])),
      "disjoint_heldout_windows":len(test["y"]),
      "heldout_user_remainder":1,"training_user_remainder":0,
      "test_time_period":["2020-08-14","2020-09-06"],
      "all_original_models_verified_before_test_source":True,
      "test_optimizer_updates":0,"model_promoted":False,
      "behavior_policy_propensities_available":False,
      "causal_action_effect_claim":False,
      "training_compute_equalized":False,
      "historical_train_only_baselines":baselines,
      "models":[]}
    target=test["y"].numpy()
    torch.set_num_threads(1)
    for entry,model in models:
        pred,secs=forward_model(model,test)
        blank,blank_secs=forward_model(model,test,ablate=True)
        score=binary_scores(pred,target)
        gone=binary_scores(blank,target)
        result={"model":entry["model"],"seed":entry["seed"],
          "original_checkpoint_sha256":entry["checkpoint_sha256"],
          "trainable_parameters":entry["trainable_params"],
          "historical_train_wall_seconds":entry["actual_training_seconds"],
          "genuine_unseen_user_scores":score,
          "blanked_history_brier":gone["brier"],
          "history_ablation_delta_brier":gone["brier"]-score["brier"],
          "real_test_inference_cpu_seconds":secs}
        data["models"].append(result)
        print(json.dumps({"arm":entry["model"],"seed":entry["seed"],
               "unseen_user_brier":score["brier"],"logloss":score["binary_logloss"],
               "simple_history_brier":baselines["causal_unseen_user_prior32_rate"]["brier"]},sort_keys=True),flush=True)
    (out/"first_disjoint_user_results.json").write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"ME015F_FIRST_DISJOINT_REAL_USER_SOURCE_EVALUATED",
       "users":data["disjoint_heldout_users"],"windows":len(test["y"]),
       "frozen_model_count":len(data["models"]),"training_steps_at_test":0},sort_keys=True),flush=True)
    return data

if __name__=="__main__":
    evaluate("results/me015","results/me015f")
