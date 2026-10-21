#!/usr/bin/env python3
"""ME025: genuine early KuaiRec feature parity for native TAC vs causal Transformer.

Development-only engineering research. Never parse later/future test responses.
All arms use exact same original 128 genuine history inputs and real prefix labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.me014b_exact_action_memory import (
    FIT_CUT, DEV_CUT, PIN_MD5, batch, read_early
)
from scripts.me019_authentic_history import (
    original_128_windows, FIT_MAX, DEV_MAX, SOURCE_SHA
)
from scripts.me021_primary_interaction import (
    features, train_logistic_or_mlp, predict_simple
)
from scripts.me022_native_readout import scores
from tac_osm.coupled_native import CoupledNative
from tac_osm.me025_feature_parity import (
    KernelNative, TinyCausalTransformer, parameter_count,
    software_preflight,
)

STUDY = "ME025_REAL_CAUSAL_FEATURE_PARITY_NATIVE_VS_TRANSFORMER"
SEEDS=(14001,14002,14003)
ARMS=("tac_rms","tac_kernel","tiny_transformer_kernel")
DEPTHS=(32,64,96)
WEIGHTS=(1/6,1/6,1/6,1/2)
UPDATES=300
BATCH=128
CPU_CEILING=600.
STEPS=(0,1,25,50,100,150,200,250,300)
OUT_PERIOD=384
PARAM_COUNTS={"tac_rms":135050,"tac_kernel":135056,
              "tiny_transformer_kernel":134896}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_only_popularity(train):
    ids=train["action"].numpy()
    target=train["y"].flatten().numpy()
    counts=np.bincount(ids,minlength=16000)
    positives=np.bincount(ids,weights=target,minlength=16000)
    global_rate=float(target.mean())
    return torch.from_numpy(((positives+16*global_rate)/(counts+16)).astype(np.float32))


def model_for(arm,mean,scale,item_pop):
    if arm=="tac_rms":
        return CoupledNative(0.,True,mean,scale)
    if arm=="tac_kernel":
        return KernelNative(mean,scale,item_pop)
    if arm=="tiny_transformer_kernel":
        return TinyCausalTransformer(mean,scale,item_pop)
    raise ValueError("unregistered architecture")


def prob_scores(net, dataset):
    net.eval()
    predictions=[]
    with torch.no_grad():
        for first in range(0,len(dataset["y"]),OUT_PERIOD):
            sl=slice(first,min(first+OUT_PERIOD,len(dataset["y"])))
            predictions.append(torch.sigmoid(net(*batch(dataset,sl))).flatten().cpu())
    return torch.cat(predictions).numpy()


def save_json(path,obj):
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
    tmp.replace(p)


def fit_arm(arm,seed,train,dev,mean,scale,rate,root):
    torch.manual_seed(seed)
    net=model_for(arm,mean,scale,rate)
    params=parameter_count(net)
    if params!=PARAM_COUNTS[arm]:
        raise AssertionError(f"active parameters changed: {arm} {params}")
    torch.set_num_threads(1)
    base=[];head=[]
    for n,p in net.named_parameters():
        (head if n.startswith("head.") else base).append(p)
    opt=torch.optim.AdamW([
        {"params":base,"lr":.001},
        {"params":head,"lr":.006},
    ],weight_decay=1e-4)
    gen=torch.Generator().manual_seed(seed+300)
    loss_w=torch.tensor(WEIGHTS,dtype=torch.float32).view(1,4)
    trace=[]
    cpu=wall=0.
    chk=root/f"{arm}-seed{seed}.pt"
    pred_path=root/f"{arm}-seed{seed}-dev-predictions.npy"
    name=f"{arm}-seed{seed}"

    def record(step,gradient=None):
        p=prob_scores(net,dev)
        t={"step":step,"dev":scores(p,dev["y"].numpy()),
           "charged_optimizer_CPU_seconds":cpu,
           "charged_optimizer_wall_seconds":wall,
           "gradient_norm_before_clip":gradient}
        trace.append(t)
        save_json(root/f"{name}-trace.json",trace)
        if step==UPDATES:
            np.save(pred_path,p)
        return p

    record(0)
    for step in range(1,UPDATES+1):
        ids=torch.randint(len(train["y"]),(BATCH,),generator=gen)
        args=batch(train,ids)
        truth=torch.cat((train["metadata"][ids][:,list(DEPTHS),1],
                         train["y"][ids]),dim=-1)
        start_cpu=time.process_time()
        start_wall=time.perf_counter()
        net.train()
        logits=net.prefix_logits(*args)
        if logits.shape!=truth.shape:
            raise ValueError(f"prefix mismatch {arm} {logits.shape} != {truth.shape}")
        loss=(F.binary_cross_entropy_with_logits(
            logits,truth,reduction="none")*loss_w).sum(-1).mean()
        if not bool(torch.isfinite(loss)):
            raise ArithmeticError("nonfinite genuine FIT loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad=nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not bool(torch.isfinite(grad)):
            raise ArithmeticError("nonfinite original trained gradient")
        opt.step()
        cpu+=time.process_time()-start_cpu
        wall+=time.perf_counter()-start_wall
        if step in STEPS or cpu>CPU_CEILING:
            final_predictions=record(step,float(grad))
            chk_tmp=chk.with_suffix(".partial")
            torch.save({
                "arm":arm,"seed":seed,"step":step,
                "state_dict":net.state_dict(),
                "optimizer":opt.state_dict(),
                "rng":torch.get_rng_state(),
                "sampler_rng":gen.get_state(),
                "source_sha256":SOURCE_SHA,
                "params":params,
                "duration_fit_mean":mean,
                "duration_fit_scale":scale,
                "fit_only_popularity_sha256":hashlib.sha256(
                    rate.numpy().tobytes()).hexdigest(),
                "charged_optimizer_CPU_seconds":cpu
            },chk_tmp)
            chk_tmp.replace(chk)
        if cpu>CPU_CEILING:
            raise RuntimeError("fixed 600-second training CPU limit exceeded")

    row={
        "arm":arm,"seed":seed,"parameters":params,"updates":UPDATES,
        "sampled_windows":UPDATES*BATCH,
        "genuine_event_history_presentations":UPDATES*BATCH*128,
        "supervised_true_next_responses":UPDATES*BATCH*4,
        "charged_optimizer_CPU_seconds":cpu,
        "charged_optimizer_wall_seconds":wall,
        "process_lifetime_peak_RSS_KiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "dev":trace[-1]["dev"],
        "trace":trace,
        "checkpoint":chk.name,"checkpoint_sha256":sha(chk),
        "prediction_file":pred_path.name,"prediction_sha256":sha(pred_path),
        "training_compute_FLOPs_fully_measured":False,
        "same_real_FIT_sample_draws_across_neural_arms":True,
        "independent_test_opened":False,
    }
    print(json.dumps({"arm":arm,"seed":seed,"dev_Brier":row["dev"]["Brier"],
                      "cpu_s":round(cpu,2),"params":params},
                     sort_keys=True),flush=True)
    return row


def fixed_reference(train,dev,item_pop):
    # Unmatched tiny practical feature baseline; FIT-only item popularity.
    trainfeat=features(train,item_pop.numpy())
    devfeat=features(dev,item_pop.numpy())
    output=[]
    for seed in SEEDS:
        fit=train_logistic_or_mlp(trainfeat,train["y"].numpy(),seed,True)
        p=predict_simple(fit,devfeat)
        output.append({"seed":seed,"parameters":fit["parameters"],
                       "updates":300,"dev":scores(p,dev["y"].numpy()),
                       "FIT_only":True,"same_real_DEV_targets":True,
                       "model_type":"unmatched_practical_tiny_MLP"})
    return output


def main(out):
    torch.set_num_threads(1)
    started=time.perf_counter()
    root=Path(out)
    root.mkdir(parents=True,exist_ok=False)
    save_json(root/"manifest.json",{
        "study":STUDY,"status":"AUTHENTIC_EARLY_SOURCE_FIT_PENDING",
        "source_sha256":SOURCE_SHA,
        "reserved_user_future_cohorts_opened":False,
        "parameters_requested":PARAM_COUNTS,
        "training_compute_fully_equal":False})
    users,source=read_early()
    if (source["zip_sha256"]!=SOURCE_SHA or source["zip_md5"]!=PIN_MD5
        or source["future_test_labels_parsed"]):
        raise ValueError("publisher source integrity or reserved-cohort violation")
    fit=original_128_windows(users,0,FIT_CUT,FIT_MAX)
    dev=original_128_windows(users,FIT_CUT,DEV_CUT,DEV_MAX)
    if (len(fit["y"])!=10776 or len(dev["y"])!=5952
        or any(u%16!=0 for u in fit["users"]+dev["users"])
        or max(fit["timestamps"])>=FIT_CUT
        or min(dev["timestamps"])<FIT_CUT
        or max(dev["timestamps"])>=DEV_CUT):
        raise ValueError("reused genuine FIT and historical DEV not reproducible")
    mean=float(fit["duration"].mean())
    scale=max(float(fit["duration"].std(unbiased=False)),1e-5)
    item_pop=fit_only_popularity(fit)
    result={
        "study":STUDY,
        "official_publisher_zip_sha256":SOURCE_SHA,
        "historical_DEV_already_exposed":True,
        "unseen_user_cohorts_opened":False,
        "real_FIT_windows":len(fit["y"]),"real_DEV_windows":len(dev["y"]),
        "same_genuine_training_inputs_across_neural_models":True,
        "same_four_causal_supervised_response_targets":True,
        "equal_neural_training_optimizer_steps":True,
        "actual_full_training_FLOPs_equal":False,
        "strict_architecture_superiority_claim_allowed":False,
        "full_neural_matched_parameter_tolerance_percent":1.,
        "source_only_rate_features_and_item_popularity":True,
        "duration_fit_mean":mean,"duration_fit_scale":scale,
        "param_counts":PARAM_COUNTS,
        "model_results":[],"reference_results":[],
        "independent_test_held_closed":True
    }
    save_json(root/"manifest.json",result)
    result["reference_results"]=fixed_reference(fit,dev,item_pop)
    save_json(root/"manifest.json",result)
    for seed in SEEDS:
        for arm in ARMS:
            result["model_results"].append(fit_arm(
                arm,seed,fit,dev,mean,scale,item_pop,root))
            save_json(root/"manifest.json",result)
    groups={arm:[r["dev"]["Brier"] for r in result["model_results"]
                 if r["arm"]==arm] for arm in ARMS}
    diff=[b-a for a,b in zip(groups["tac_rms"],groups["tac_kernel"])]
    ece_delta=float(np.mean([
        k["dev"]["ECE10"]-b["dev"]["ECE10"]
        for b,k in zip(
            [r for r in result["model_results"] if r["arm"]=="tac_rms"],
            [r for r in result["model_results"] if r["arm"]=="tac_kernel"]
        )
    ]))
    result["registered_primary"]={
        "contrast":"tac_kernel minus tac_rms, historical DEV",
        "per_seed_deltas":diff,
        "mean_delta":float(np.mean(diff)),
        "all_3_seed_improve":all(x<0 for x in diff),
        "ECE10_kernel_minus_native_mean":ece_delta,
        "passed_development_only_engineering_gate":
            all(x<0 for x in diff) and float(np.mean(diff))<=-.001
            and ece_delta<=.005,
        "no_external_efficacy_claim":True,
    }
    result["descriptive_means"]={
        arm:float(np.mean(group)) for arm,group in groups.items()}
    result["entire_study_wall_seconds"]=time.perf_counter()-started
    result["all_original_models_completed"]=len(result["model_results"])==9
    result["independent_test_held_closed"]=True
    save_json(root/"manifest.json",result)
    print(json.dumps({"status":"ME025_REAL_HISTORICAL_DEV_ONLY_COMPLETED",
                      "historical_Brier":result["descriptive_means"],
                      "primary":result["registered_primary"],
                      "future_labels_read":False},sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/me025")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        print(json.dumps(software_preflight(),sort_keys=True))
    else:
        main(a.out)
