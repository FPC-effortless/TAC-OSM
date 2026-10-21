#!/usr/bin/env python3
"""ME028 genuine KuaiRec fixed-300-updates native nonlinear readout experiment.

Sealed later outcomes remain unopened. Native recurrence is ME027 unchanged.
Synthetic preflight for math, NEVER model performance. Preserve negative gates.
"""
from __future__ import annotations
import argparse, hashlib, json, time, resource, platform
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.state_value import StateValueModel
from tac_osm.me028_nonlinear_readout import (
    NativeNonlinearObservableReadout, native_nonlinear_math_preflight,
    original_parameter_count
)
from scripts.me014b_exact_action_memory import (
    FIT_CUT, DEV_CUT, PIN_MD5, read_early, batch
)
from scripts.me019_authentic_history import (
    original_128_windows, FIT_MAX, DEV_MAX, SOURCE_SHA
)
from scripts.me025_real_transformer_parity import fit_only_popularity
from scripts.me022_native_readout import scores

SEEDS=(14001,14002,14003)
ARMS=("tac_linear","tac_nonlinear","mlp252")
COUNTS={"tac_linear":135067,"tac_nonlinear":135377,"mlp252":135057}
UPDATES=300
BATCH=128
TRACE_STEPS=(0,1,25,50,100,150,200,250,300)
WEIGHTS=(1/6,1/6,1/6,1/2)
BOOT=1000
BSEED=28028


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,payload):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".temp")
    tmp.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")
    tmp.replace(p)


def make_model(arm,mean,scale,pop):
    if arm=="tac_linear":return StateValueModel("tac",mean,scale,pop)
    if arm=="tac_nonlinear":return NativeNonlinearObservableReadout(mean,scale,pop)
    if arm=="mlp252":return StateValueModel("mlp",mean,scale,pop)
    raise ValueError("invalid experiment arm")


def load_real():
    torch.set_num_threads(1)
    people,source=read_early()
    if (source["zip_sha256"]!=SOURCE_SHA or source["zip_md5"]!=PIN_MD5 or
            source["future_test_labels_parsed"]):
        raise ValueError("authentic FIT-only source requirement failed")
    fit=original_128_windows(people,0,FIT_CUT,FIT_MAX)
    dev=original_128_windows(people,FIT_CUT,DEV_CUT,DEV_MAX)
    if (len(fit["y"]),len(dev["y"]))!=(10776,5952):
        raise ValueError("original source-derived FIT/DEV counts changed")
    if (max(fit["timestamps"])>=FIT_CUT or
            min(dev["timestamps"])<FIT_CUT or max(dev["timestamps"])>=DEV_CUT or
            any(u%16!=0 for u in fit["users"]+dev["users"])):
        raise ValueError("early history cohorts contaminated")
    mean=float(fit["duration"].mean())
    scale=max(float(fit["duration"].std(unbiased=False)),1e-5)
    pop=fit_only_popularity(fit)
    return fit,dev,mean,scale,pop


def evaluate(net,data):
    net.eval()
    preds=[]
    with torch.no_grad():
        for i in range(0,len(data["y"]),256):
            sl=slice(i,min(i+256,len(data["y"])))
            preds.append(torch.sigmoid(net(*batch(data,sl))).flatten().cpu().numpy())
    return np.concatenate(preds).astype(np.float64)


def frozen_latent_mean(net,fit):
    # Entirely FIT-only statistic computed after training; not a fitted new
    # forecasting head, and not a true causal intervention.
    latents=[]
    net.eval()
    with torch.no_grad():
        for i in range(0,min(1024,len(fit["y"])),128):
            sl=slice(i,min(i+128,1024))
            latents.append(net.latent_features(fit["items"][sl],
                                              fit["metadata"][sl]).cpu())
    stacked=torch.cat(latents)
    if stacked.shape != (1024,64):
        raise ValueError("FIT means must be computed on genuine 1024 windows")
    return stacked.mean(0)


def eval_mean_state(net,dev,mean):
    preds=[]
    net.eval()
    with torch.no_grad():
        for i in range(0,len(dev["y"]),256):
            sl=slice(i,min(i+256,len(dev["y"])))
            items,meta,action,duration=batch(dev,sl)
            common=net.observables(items,meta,action,duration,128)
            prior=(meta[:,-32:,1].sum(-1,keepdim=True)+1)/34
            logit=net.predict_features(mean.unsqueeze(0).expand(len(items),-1),common,prior)
            preds.append(torch.sigmoid(logit).flatten().cpu().numpy())
    return np.concatenate(preds).astype(np.float64)


def train_one(arm,seed,fit,dev,mean,scale,pop,root):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    net=torch.jit.script(make_model(arm,mean,scale,pop))
    if original_parameter_count(net)!=COUNTS[arm]:
        raise ValueError("active parameter mismatched")
    body=[];heads=[]
    for name,p in net.named_parameters():
        (heads if (name.startswith("head.") or
                   name.startswith("observable_hidden.") or
                   name.startswith("observable_output.")) else body).append(p)
    optimizer=torch.optim.AdamW([
        {"params":body,"lr":.001},{"params":heads,"lr":.006}
    ],weight_decay=1e-4)
    rng=torch.Generator().manual_seed(seed+300)
    prefix_weights=torch.tensor(WEIGHTS,dtype=torch.float32).reshape(1,4)
    trace=[]
    started=time.process_time()
    training_cpu=0.
    for step in range(UPDATES+1):
        if step in TRACE_STEPS:
            p=evaluate(net,dev)
            row={"step":step,"optimizer_cpu_seconds":training_cpu,
                 "dev":scores(p,dev["y"].numpy())}
            trace.append(row)
            print(json.dumps({"study":"ME028","arm":arm,"seed":seed,
                              "step":step,"Brier":row["dev"]["Brier"]}),flush=True)
        if step==UPDATES:break
        ids=torch.randint(len(fit["y"]),(BATCH,),generator=rng)
        targets=torch.cat((fit["metadata"][ids][:,[32,64,96],1],
                           fit["y"][ids]),dim=-1)
        start=time.process_time()
        net.train();optimizer.zero_grad(set_to_none=True)
        result=net.prefix_logits(*batch(fit,ids))
        if result.shape!=targets.shape:raise ValueError("real prefix target pairing")
        loss=(F.binary_cross_entropy_with_logits(
            result,targets,reduction="none")*prefix_weights).sum(-1).mean()
        if not bool(torch.isfinite(loss)):raise ArithmeticError("non-finite real BCE")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not bool(torch.isfinite(grad)):raise ArithmeticError("non-finite active grads")
        optimizer.step()
        training_cpu+=time.process_time()-start
    cp=root/f"{arm}-seed{seed}-300.pt"
    pp=root/f"{arm}-seed{seed}-predictions.npy"
    torch.save({"arm":arm,"seed":seed,"step":UPDATES,
                "source_sha256":SOURCE_SHA,"params":COUNTS[arm],
                "state_dict":net.state_dict(),
                "optimizer":optimizer.state_dict(),
                "model_rng":torch.get_rng_state(),
                "sample_rng":rng.get_state(),
                "training_cpu_seconds":training_cpu,
                "duration_mean":mean,"duration_scale":scale},cp)
    np.save(pp,p)
    state_probe=None
    if arm.startswith("tac_"):
        mu=frozen_latent_mean(net,fit)
        mean_predictions=eval_mean_state(net,dev,mu)
        state_probe={
            "frozen_FIT_state_mean_count":1024,
            "frozen_state_feature_mean":mu.numpy().tolist(),
            "mean_replaced_Brier":scores(mean_predictions,dev["y"].numpy())["Brier"],
            "mean_replaced_minus_native_Brier":
                scores(mean_predictions,dev["y"].numpy())["Brier"]-trace[-1]["dev"]["Brier"],
            "FIT_average_latent_logit_preserved":True,
            "causal_state_utility_proven":False,
        }
    return {"arm":arm,"seed":seed,"parameters":COUNTS[arm],
            "updates":UPDATES,"batch":BATCH,
            "drawn_fit_windows":UPDATES*BATCH,
            "supervised_real_prefix_outcomes":UPDATES*BATCH*4,
            "optimizer_cpu_seconds":training_cpu,
            "whole_fit_process_cpu_seconds_including_dev_eval":time.process_time()-started,
            "max_process_RSS_KiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "learning_curve":trace,"dev_final":trace[-1]["dev"],
            "state_mean_replacement":state_probe,
            "checkpoint":cp.name,"checkpoint_sha256":file_sha(cp),
            "predictions":pp.name,"predictions_sha256":file_sha(pp),
            "original_frozen_fit_source_only":True,
            "independent_future_outcomes_read":False}


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    start=time.process_time()
    fit,dev,mean,scale,pop=load_real()
    record={
        "study":"ME028_NONLINEAR_OBSERVABLE_NATIVE_READOUT_HISTORICAL_DEV",
        "source_SHA256":SOURCE_SHA,
        "original_FIT_windows":len(fit["y"]),
        "inspected_historical_DEV_windows":len(dev["y"]),
        "reserved_unseen_cohorts_opened":False,
        "model_parameters":COUNTS,
        "param_relative_gap_max":(max(COUNTS.values())-min(COUNTS.values()))/min(COUNTS.values()),
        "all_arms_same_real_sampled_training_windows":True,
        "all_arms_same_4_actual_causal_prefix_outcomes":True,
        "training_optimizer_steps_equal":True,
        "fully_compute_FLOP_matched":False,
        "independent_accuracy_claim_allowed":False,
        "architecture_promoted":False,
        "duration_FIT_mean":mean,"duration_FIT_scale":scale,
        "model_rows":[],
    }
    write_json(root/"results.json",record)
    for seed in SEEDS:
        for arm in ARMS:
            row=train_one(arm,seed,fit,dev,mean,scale,pop,root)
            record["model_rows"].append(row)
            write_json(root/"results.json",record)
    groups={arm:[r for r in record["model_rows"] if r["arm"]==arm] for arm in ARMS}
    delta=[b["dev_final"]["Brier"]-a["dev_final"]["Brier"]
           for a,b in zip(groups["tac_linear"],groups["tac_nonlinear"])]
    ece_delta=float(np.mean([b["dev_final"]["ECE10"]-a["dev_final"]["ECE10"]
                             for a,b in zip(groups["tac_linear"],groups["tac_nonlinear"])]))
    ablation=[x["state_mean_replacement"]["mean_replaced_minus_native_Brier"]
              for x in groups["tac_nonlinear"]]
    gate={
        "primary_paired_nonlinear_minus_linear_DEV_Brier_by_seed":delta,
        "mean_Brier_delta":float(np.mean(delta)),
        "calibration_mean_ECE10_delta":ece_delta,
        "mean_replaced_minus_nonlinear_Brier_by_seed":ablation,
        "passed_accuracy_gate":all(x<0 for x in delta) and float(np.mean(delta))<=-.0005,
        "passed_state_mean_replacement_gate":all(x>=.0002 for x in ablation),
        "passed_calibration_gate":ece_delta<=.003,
        "no_new_test_cohort":True
    }
    gate["passed_all_development_gates"]=all(
        gate[k] for k in ("passed_accuracy_gate","passed_state_mean_replacement_gate","passed_calibration_gate"))
    record["registered_gates"]=gate
    record["historical_mean_Brier"]={
        arm:float(np.mean([r["dev_final"]["Brier"] for r in rows]))
        for arm,rows in groups.items()}
    record["training_CPU_seconds_by_arm"]={
        arm:float(np.mean([r["optimizer_cpu_seconds"] for r in rows]))
        for arm,rows in groups.items()}
    record["all_nine_original_real_FIT_fits_completed"]=len(record["model_rows"])==9
    record["whole_study_process_CPU_seconds"]=time.process_time()-start
    record["all_future_outcomes_held_closed"]=True
    write_json(root/"results.json",record)
    print(json.dumps({"status":"ME028_AUTHENTIC_HISTORICAL_ONLY_COMPLETE",
        "mean_DEV_Brier":record["historical_mean_Brier"],
        "gates":gate,"promoted":False,"new_cohorts_opened":False},sort_keys=True),flush=True)


def audit(root):
    root=Path(root)
    r=json.loads((root/"results.json").read_text())
    assert r["source_SHA256"]==SOURCE_SHA
    assert not r["reserved_unseen_cohorts_opened"] and len(r["model_rows"])==9
    assert r["all_future_outcomes_held_closed"]
    fit,dev,mean,scale,pop=load_real()
    for row in r["model_rows"]:
        cp=root/row["checkpoint"]
        pp=root/row["predictions"]
        assert file_sha(cp)==row["checkpoint_sha256"]
        assert file_sha(pp)==row["predictions_sha256"]
        ck=torch.load(cp,map_location="cpu",weights_only=True)
        assert ck["source_sha256"]==SOURCE_SHA and ck["step"]==300
        net=torch.jit.script(make_model(row["arm"],mean,scale,pop))
        net.load_state_dict(ck["state_dict"],strict=True)
        actual=evaluate(net,dev)
        saved=np.load(pp)
        if actual.shape!=saved.shape or not np.allclose(actual,saved,rtol=0,atol=1e-7):
            raise ValueError("trained original checkpoint probability replay failed")
        if abs(scores(actual,dev["y"].numpy())["Brier"]-row["dev_final"]["Brier"])>1e-8:
            raise ValueError("recorded original Brier not reproducible")
    print(json.dumps({"status":"ME028_ALL9_FROZEN_REAL_DEV_FORECASTS_REPLAY",
                      "real_predictions_replayed":9*len(dev["y"]),
                      "no_future_labels_opened":True},sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/me028")
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--audit",action="store_true")
    a=p.parse_args()
    if a.self_test:
        print(json.dumps(native_nonlinear_math_preflight(),sort_keys=True))
    elif a.audit:
        audit(a.out)
    else:
        run(a.out)
