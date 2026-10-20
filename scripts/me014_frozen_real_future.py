#!/usr/bin/env python3
"""ME014F frozen later-video-reaction evaluation; no optimizer or action-policy uplift.

All original six fitted model hashes are independently checked BEFORE any
2020-08-14+ user reactions are parsed. Current logged response is never input.
"""
from __future__ import annotations
import csv,hashlib,io,json,math,time,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from scripts.me014_real_logged_actions import (
  publisher_bytes,member_for_zip,FIT_CUT,DEV_CUT,END_CUT,REQUIRED,
  VOCAB,SEEDS,ARMS,ActionMemory,build_examples,empirical_baselines,
  binary_scores,batch,parameters)

def frozen_models(original):
    folder=Path(original)
    manifest=json.loads((folder/"results.json").read_text())
    if manifest["study"]!="ME014_REAL_LOGGED_ACTION_PERSISTENT_MEMORY" or len(manifest["fits"])!=6:
        raise ValueError("original real source model fit matrix unavailable")
    if manifest["source"]["zip_md5"]!="261550d472c48eff4990fb13c0e5bcf7":
        raise ValueError("expected pinned real source")
    if manifest["future_test_parsed"] or manifest["source"]["future_test_labels_parsed"]:
        raise ValueError("original trainer parsed future labels")
    models=[]
    for fit in manifest["fits"]:
        p=folder/fit["checkpoint"]
        if hashlib.sha256(p.read_bytes()).hexdigest()!=fit["sha256"]:
            raise ValueError("original model checkpoint HASH mismatch")
        ck=torch.load(p,weights_only=True,map_location="cpu")
        if ck["arm"]!=fit["arm"] or ck["seed"]!=fit["seed"] or ck["steps"]!=300:
            raise ValueError("wrong original model identity")
        if ck["data_sha256"]!=manifest["source"]["zip_sha256"]:
            raise ValueError("trained on different publisher source")
        model=ActionMemory(fit["arm"])
        model.load_state_dict(ck["state_dict"],strict=True)
        if parameters(model)!=fit["parameters"]:raise ValueError("parameter mismatch")
        model.eval()
        models.append((fit,model))
    if {(f["arm"],f["seed"]) for f,m in models}!={(arm,seed)
          for arm in ARMS for seed in SEEDS}:raise ValueError("missing original arm")
    return manifest,models

def read_original_later(archive_sha):
    raw,meta=publisher_bytes()
    if meta["zip_sha256"]!=archive_sha:raise ValueError("real training source archive changed")
    users={};counters={"cohort_all_valid_rows":0,"future_label_rows_opened":0,
                       "unparseable_timestamps":0,"invalid_values":0}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        member=member_for_zip(z)
        with z.open(member) as file:
            reader=csv.DictReader(io.TextIOWrapper(file,encoding="utf-8-sig",newline=""))
            if not set(REQUIRED).issubset(set(reader.fieldnames or [])):
                raise ValueError("publisher member columns changed")
            for r in reader:
                try:uid=int(r["user_id"])
                except (ValueError,TypeError):continue
                if uid<0 or uid%16!=0:continue
                try:ts=float(r["timestamp"])
                except (ValueError,TypeError):counters["unparseable_timestamps"]+=1;continue
                if not math.isfinite(ts) or ts<=1e9:
                    counters["unparseable_timestamps"]+=1;continue
                if ts>=END_CUT:continue
                try:
                    item=int(r["video_id"])
                    dur=float(r["video_duration"])
                    ratio=float(r["watch_ratio"])
                except (ValueError,TypeError):
                    counters["invalid_values"]+=1;continue
                if not (0<=item<VOCAB and math.isfinite(dur) and dur>0 and
                        math.isfinite(ratio) and ratio>=0):
                    counters["invalid_values"]+=1;continue
                users.setdefault(uid,[]).append((ts,item,math.log1p(dur)/12,int(ratio>=1.0)))
                counters["cohort_all_valid_rows"]+=1
                if ts>=DEV_CUT:counters["future_label_rows_opened"]+=1
    for record in users.values():record.sort(key=lambda x:x[0])
    meta.update(counters);meta["source_member"]=member
    meta["future_label_rows_opened_after_checkpoint_integrity"]=True
    meta["test_end_exclusive"]="2020-09-06T00:00:00Z"
    meta["future_rows_at_or_after_end_parsed"]=False
    return users,meta

def frozen_predict(model,hold,blank=False):
    torch.set_num_threads(1);predictions=[]
    model.eval()
    started=time.perf_counter()
    with torch.inference_mode():
        for first in range(0,len(hold["y"]),512):
            last=min(len(hold["y"]),first+512)
            ix=slice(first,last)
            logits=model(*batch(hold,ix),ignore_history=blank)
            predictions.append(torch.sigmoid(logits).flatten())
    return torch.cat(predictions).numpy(),time.perf_counter()-started

def evaluate(original,out):
    original,out=Path(original),Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest,models=frozen_models(original)
    users,source=read_original_later(manifest["source"]["zip_sha256"])
    train=build_examples(users,0,FIT_CUT)
    dev=build_examples(users,FIT_CUT,DEV_CUT)
    if len(train["y"])!=manifest["train_examples"] or len(dev["y"])!=manifest["dev_examples"]:
        raise ValueError("original trained-source history reconstruction differs")
    hold=build_examples(users,DEV_CUT,END_CUT,max_windows=50000)
    if len(hold["y"])<500:raise ValueError("not enough genuine future user reactions")
    baselines=empirical_baselines(train,hold)
    baseline_scores={
        name:binary_scores(baselines[field],hold["y"].numpy()) for name,field in
        (("global_training_rate","global_predictions"),
         ("video_training_rate","item_predictions"),
         ("causal_prior32_user_rate","history_mean_predictions"))}
    results={"study":"ME014F_FIRST_FROZEN_REAL_USER_ACTION_OUTCOME",
      "original_source":manifest["source"]["zip_sha256"],
      "source":source,
      "original_checkpoint_count":len(models),
      "historical_train_examples_reproduced":len(train["y"]),
      "historical_dev_examples_reproduced":len(dev["y"]),
      "test_window_count":len(hold["y"]),
      "test_period":["2020-08-14","2020-09-06"],
      "test_user_count":len(set(hold["users"])),
      "test_optimizer_updates":0,"train_data_unchanged":True,
      "evaluation_actions_observed_not_selected_by_model":True,
      "behavior_policy_propensities_missing":True,
      "causal_treatment_effects_estimated":False,
      "external_utility_verification_present":False,
      "baseline_scores":baseline_scores,
      "trained_models":[],"model_promoted":False}
    truth=hold["y"].numpy()
    for fit,model in models:
        probabilities,seconds=frozen_predict(model,hold)
        blank,blank_s=frozen_predict(model,hold,blank=True)
        scores=binary_scores(probabilities,truth)
        no_memory=binary_scores(blank,truth)
        # Observational memory sensitivity; blanking differs from factual counterfactual.
        group={}
        user_ids=np.asarray(hold["users"])
        for uid in np.unique(user_ids):
            ix=user_ids==uid
            if ix.sum()<10:continue
            group[str(uid)]={"n":int(ix.sum()),
                   "brier":binary_scores(probabilities[ix],truth[ix])["brier"]}
        result={"model":fit["arm"],"seed":fit["seed"],
           "sha256":fit["sha256"],"parameters":fit["parameters"],
           "test_brier":scores["brier"],"test_logloss":scores["binary_logloss"],
           "test_scores":scores,
           "blank_prior32_test_scores":no_memory,
           "history_ablation_brier_delta":no_memory["brier"]-scores["brier"],
           "per_user":group,
           "test_inference_cpu_seconds":round(seconds,4)}
        results["trained_models"].append(result)
        print(json.dumps({"model":fit["arm"],"seed":fit["seed"],
            "test_brier":scores["brier"],"blanked_history_brier":no_memory["brier"],
            "trainonly_item_rate_brier":baseline_scores["video_training_rate"]["brier"]},sort_keys=True),flush=True)
    (out/"future_scores.json").write_text(json.dumps(results,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"ME014F_FROZEN_AUTHENTIC_ACTION_HISTORY_COMPLETE",
      "test_exposures":len(hold["y"]),"user_count":results["test_user_count"],
      "model_fits":0},sort_keys=True),flush=True)
    return results

if __name__=="__main__":
    evaluate("results/me014","results/me014f")
