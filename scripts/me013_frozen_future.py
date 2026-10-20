#!/usr/bin/env python3
"""ME013F: original frozen real multi-site time AND unseen-station evaluation.

NO fitting or using future labels inside model inputs. Original nine checkpoint
digests and first training-source checksum are checked before parsing heldouts.
"""
from __future__ import annotations
import csv,hashlib,io,json,time,zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from scripts.me013_multisite import (
    source_zip,station_archive_members,parse_site,ALL_SITES,FIT_SITES,UNSEEN,
    FINAL_CUT,DEV_CUT,model_for,windows,D,HISTORY,SEEDS)

def checkpoints(original):
    folder=Path(original);manifest=json.loads((folder/"results.json").read_text())
    if manifest["study"]!="ME013_UCI_REAL_MULTI_STATION" or len(manifest["fits"])!=9:
        raise ValueError("wrong original model/run manifest")
    if manifest["future_test_read"] or manifest["architecture_promoted"]:
        raise ValueError("original future had already been opened")
    if set(manifest["source"]["parsed_stations"])!=set(FIT_SITES):
        raise ValueError("original training touched heldout site")
    if manifest["source"]["source_rows_at_or_after_cutoff_parsed"]:
        raise ValueError("training touched withheld time")
    models=[]
    for fit in manifest["fits"]:
        p=folder/fit["checkpoint"]
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=fit["checkpoint_sha256"]:raise ValueError("frozen checkpoint SHA mismatch")
        ck=torch.load(p,weights_only=True,map_location="cpu")
        if ck["seed"]!=fit["seed"] or ck["kind"]!=fit["model"] or ck["steps"]!=500:
            raise ValueError("trained checkpoint identity mismatch")
        if ck["source_sha256"]!=manifest["source"]["sha256"]:
            raise ValueError("checkpoint trained on a different dataset")
        m=model_for(fit["model"]);m.load_state_dict(ck["model_state"],strict=True);m.eval()
        models.append((fit,m))
    if {(fit["model"],fit["seed"]) for fit,m in models}!={(arm,seed)
        for arm in ("native_plain","native_routed","gru") for seed in SEEDS}:
        raise ValueError("incomplete original nine-model matrix")
    return manifest,models

def get_test_cohorts(manifest):
    # Only source hash and checkpoint integrity are checked BEFORE parsing time/site heldouts.
    archive,source=source_zip()
    if source["sha256"]!=manifest["source"]["sha256"]:
        raise ValueError("publisher archive SHA changed")
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        names=station_archive_members(zipped)
        real={site:parse_site(zipped,names[site],site,FINAL_CUT) for site in ALL_SITES}
    known={s:real[s] for s in FIT_SITES}
    novel={s:real[s] for s in UNSEEN}
    for label,group in (("known_future",known),("heldout_station_future",novel)):
        if not group:raise ValueError("missing genuine test station group")
    cohorts={"known_future":windows(known,manifest["stats"],DEV_CUT,FINAL_CUT),
             "heldout_station_future":windows(novel,manifest["stats"],DEV_CUT,FINAL_CUT)}
    source.update({"parsed_real_stations":list(ALL_SITES),
                   "test_source_cutoff_exclusive":str(FINAL_CUT),
                   "rows_after_source_cutoff_parsed":False,
                   "heldout_sites_opened_only_after_all_model_hashes":True})
    return cohorts,source

def physical_scores(pred,true,stats,names):
    true=true.reshape(-1);pred=pred.reshape(-1)
    scale=float(stats["train_only_sigma"][0])
    abs_err=(pred-true).abs()
    change=true.abs()>=1.
    up=true>=1.;down=true<=-1.
    record={"mean_absolute_error_ug_m3":float(abs_err.mean()*scale),
       "forecast_count":len(names),
       "large_change_mae_ug_m3":float(abs_err[change].mean()*scale) if bool(change.any()) else None,
       "quiet_mae_ug_m3":float(abs_err[~change].mean()*scale) if bool((~change).any()) else None,
       "sign_accuracy_conditional_on_changed":float(((torch.sign(pred)==torch.sign(true))&change).sum()/change.sum()) if bool(change.any()) else None,
       "large_change_count":int(change.sum()),"large_positive_count":int(up.sum()),"large_negative_count":int(down.sum()),
       "per_station_mae_ug_m3":{}}
    for s in sorted(set(names)):
        idx=torch.tensor([name==s for name in names],dtype=torch.bool)
        record["per_station_mae_ug_m3"][s]=float(abs_err[idx].mean()*scale)
    return record

def naive_predictions(hold):
    x,y,b,names=hold
    zero=torch.zeros_like(y)
    # Last seen PM2.5 is also "same-hour yesterday" for a +24h target;
    # report it ONCE as persistence, not an artificially distinct control.
    pm=x[:,:,0];mask=x[:,:,11]
    weighted=pm[:,-24:]*mask[:,-24:]
    windowmean=weighted.sum(1)/mask[:,-24:].sum(1).clamp_min(1)
    mean_delta=(windowmean-pm[:,-1]).unsqueeze(-1)
    previous_valid=(mask[:,-25]>0)
    trend=(pm[:,-1]-pm[:,-25]).where(previous_valid,0).unsqueeze(-1)
    return {"persistence":zero,"trailing_24h_observed_mean":mean_delta,
            "prior_day_motion_extrapolation":trend}

def run(original,out):
    original,out=Path(original),Path(out);out.mkdir(parents=True,exist_ok=True)
    ledger,models=checkpoints(original)
    cohorts,source=get_test_cohorts(ledger)
    report={"study":"ME013F_REAL_FIRST_FROZEN_TIME_AND_UNSEEN_SITE",
      "source":source,"original_archive_sha256":ledger["source"]["sha256"],
      "original_fit_checkpoint_count":len(models),"test_optimizer_updates":0,
      "historical_validation_not_used_to_choose_models":True,
      "trained_site_ids_not_supplied_at_inference":True,
      "model_promoted":False,
      "cohorts":{}}
    torch.set_num_threads(1)
    for label,hold in cohorts.items():
        x,y,b,names=hold
        group={"forecast_windows":len(x),"physical_baselines":{},
               "trained_models":[],"site_counts":{}}
        for station in sorted(set(names)):group["site_counts"][station]=names.count(station)
        for kind,pred in naive_predictions(hold).items():
            group["physical_baselines"][kind]=physical_scores(pred,y,ledger["stats"],names)
        for fit,model in models:
            model.eval();preds=[];routes=[];begin=time.perf_counter()
            with torch.inference_mode():
                for start in range(0,len(x),256):
                    pred,route=model(x[start:start+256],return_route=True)
                    preds.append(pred)
                    if route is not None:routes.append(route)
            pred=torch.cat(preds);stats=physical_scores(pred,y,ledger["stats"],names)
            result={"model":fit["model"],"seed":fit["seed"],
               "checkpoint_sha256":fit["checkpoint_sha256"],
               "params":fit["trainable_parameters"],
               "inference_cpu_seconds":round(time.perf_counter()-begin,4),
               "performance":stats}
            if routes:
                r=torch.cat(routes)
                result["route_entropy_nats"]=float(-(r*r.clamp_min(1e-9).log()).sum(-1).mean())
                result["mean_route_probs_by_station"]={
                  site:r[torch.tensor([s==site for s in names])].mean(0).tolist()
                  for site in sorted(set(names))}
            group["trained_models"].append(result)
            print(json.dumps({"cohort":label,"arm":fit["model"],"seed":fit["seed"],
              "MAE_ug_m3":stats["mean_absolute_error_ug_m3"],
              "persistence_ug_m3":group["physical_baselines"]["persistence"]["mean_absolute_error_ug_m3"]},sort_keys=True),flush=True)
        report["cohorts"][label]=group
    (out/"future_results.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"ME013F_FIRST_SOURCE_LOCKED_FROZEN_REAL_COMPLETE",
        "cohorts":{s:v["forecast_windows"] for s,v in report["cohorts"].items()},
        "optimizer_updates":0}),flush=True)
    return report

if __name__=="__main__":
    run("results/me013","results/me013f")
