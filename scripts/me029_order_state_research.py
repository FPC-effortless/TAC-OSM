#!/usr/bin/env python3
"""ME029 genuine-history native chronological-state causal order-value study.

Real recorded events only. Reuses already-opened historical DEV, never user4/5
or any unopened cohort. No Transformer superiority or commercial claims.
"""
from __future__ import annotations
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from scripts.me028_native_readout_fit import (
    load_real, evaluate, eval_mean_state, file_sha, write_json
)
from scripts.me014b_exact_action_memory import batch
from scripts.me022_native_readout import scores
from tac_osm.me028_nonlinear_readout import NativeNonlinearObservableReadout
from tac_osm.me029_order_state import (
    BlockSortedNative,OrderFreeNonlinearControl,block_order_sort,
    active_params,preflight
)

SEEDS=(14001,14002,14003)
ARMS=("chrono_native","block_sorted_native","order_free_pool")
COUNT={"chrono_native":135377,"block_sorted_native":135377,"order_free_pool":135913}
UPDATES=300;BATCH=128
STEPS=(0,1,25,50,100,150,200,250,300)
W=(1/6,1/6,1/6,1/2)
BOOT_SEED=29029
BOOT=1000
STUDY="ME029_AUTHENTIC_EARLY_CHRONOLOGICAL_LEARNED_STATE_INFORMATION"


def model_for(name,mean,scale,pop):
    if name=="chrono_native":
        return NativeNonlinearObservableReadout(mean,scale,pop)
    if name=="block_sorted_native":
        return BlockSortedNative(mean,scale,pop)
    if name=="order_free_pool":
        return OrderFreeNonlinearControl(mean,scale,pop)
    raise ValueError(name)


def extract_latent(net,arm,items,metadata):
    if arm=="block_sorted_native":
        items,metadata=block_order_sort(items,metadata)
    return net.latent_features(items,metadata)


def state_mean(net,arm,fit):
    net.eval()
    arr=[]
    with torch.no_grad():
        for i in range(0,1024,128):
            sl=slice(i,i+128)
            arr.append(extract_latent(net,arm,
                                      fit["items"][sl],
                                      fit["metadata"][sl]).cpu())
    z=torch.cat(arr)
    if z.shape!=(1024,64):
        raise ValueError("different genuine fit mean-state sample")
    return z.mean(0)


def train_one(arm,seed,fit,dev,mean,scale,pop,out):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    net=torch.jit.script(model_for(arm,mean,scale,pop))
    if active_params(net)!=COUNT[arm]:
        raise ValueError("registered active parameters changed")
    body,head=[],[]
    for name,p in net.named_parameters():
        (head if name.startswith(("head.","observable_hidden.",
                                   "observable_output.")) else body).append(p)
    opt=torch.optim.AdamW([
        {"params":body,"lr":.001},
        {"params":head,"lr":.006},
    ],weight_decay=1e-4)
    sampler=torch.Generator().manual_seed(seed+300)
    weights=torch.tensor(W).reshape(1,4)
    trace=[]
    cpu=0.
    start=time.process_time()
    for step in range(UPDATES+1):
        if step in STEPS:
            p=evaluate(net,dev)
            metric=scores(p,dev["y"].numpy())
            trace.append({"step":step,"dev":metric,"optimizer_CPU_seconds":cpu})
            print(json.dumps({"arm":arm,"seed":seed,"step":step,
                              "Brier":metric["Brier"]}),flush=True)
        if step==UPDATES:break
        idx=torch.randint(len(fit["y"]),(BATCH,),generator=sampler)
        y=torch.cat((fit["metadata"][idx][:,[32,64,96],1],
                     fit["y"][idx]),dim=-1)
        start_step=time.process_time()
        net.train()
        opt.zero_grad(set_to_none=True)
        logits=net.prefix_logits(*batch(fit,idx))
        if logits.shape!=y.shape:raise ValueError("prefix/true response misaligned")
        loss=(F.binary_cross_entropy_with_logits(logits,y,reduction="none")*
              weights).sum(-1).mean()
        if not bool(torch.isfinite(loss)):raise ArithmeticError("invalid FIT loss")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not bool(torch.isfinite(grad)):raise ArithmeticError("invalid gradient")
        opt.step()
        cpu+=time.process_time()-start_step
    checkpoint=out/f"{arm}-seed{seed}.pt"
    predictions=out/f"{arm}-seed{seed}-predictions.npy"
    torch.save({"arm":arm,"seed":seed,"step":UPDATES,
                "state_dict":net.state_dict(),
                "optimizer":opt.state_dict(),
                "sampler_rng":sampler.get_state(),
                "model_rng":torch.get_rng_state(),
                "parameter_count":COUNT[arm],
                "duration_mean":mean,"duration_scale":scale},checkpoint)
    np.save(predictions,p)
    mu=state_mean(net,arm,fit)
    substitute=eval_mean_state(net,dev,mu)
    old_brier=trace[-1]["dev"]["Brier"]
    state_loss=scores(substitute,dev["y"].numpy())["Brier"]-old_brier
    return {"arm":arm,"seed":seed,"parameters":COUNT[arm],
            "nonembedding_parameters":COUNT[arm]-128000,
            "updates":UPDATES,"batch":BATCH,
            "genuine_training_rows_drawn":UPDATES*BATCH,
            "true_supervised_response_occurrences":UPDATES*BATCH*4,
            "optimizer_CPU_seconds":cpu,
            "whole_fit_process_CPU_seconds":time.process_time()-start,
            "process_lifetime_peak_RSS_KiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "historical_dev_metrics":trace[-1]["dev"],
            "learning_curve":trace,
            "FIT_mean_state_count":1024,
            "FIT_mean_state_replacement_delta_Brier":float(state_loss),
            "state_substitution_proves_causal_value":False,
            "checkpoint":checkpoint.name,
            "checkpoint_sha256":file_sha(checkpoint),
            "predictions":predictions.name,
            "predictions_sha256":file_sha(predictions),
            "new_reserved_data_read":False,
            "FLOPs_fully_matched":False}


def user_cluster_interaction(root,rows,dev):
    group={arm:{r["seed"]:np.load(root/r["predictions"]).astype(np.float64)
                for r in rows if r["arm"]==arm} for arm in ARMS}
    truth=dev["y"].numpy().ravel()
    user=np.asarray(dev["users"],dtype=np.int64)
    unique,ix=np.unique(user,return_inverse=True)
    if len(unique)<20:raise ValueError("too few real historic users")
    rng=np.random.default_rng(BOOT_SEED)
    stats={}
    for arm in ("block_sorted_native","order_free_pool"):
        ds=[]
        for seed in SEEDS:
            old=(group[arm][seed]-truth)**2
            new=(group["chrono_native"][seed]-truth)**2
            ds.append(new-old)
        event=np.mean(ds,axis=0)
        sums=np.bincount(ix,weights=event,minlength=len(unique))
        count=np.bincount(ix,minlength=len(unique))
        samples=np.empty(BOOT)
        for i in range(BOOT):
            r=rng.integers(len(unique),size=len(unique))
            samples[i]=sums[r].sum()/count[r].sum()
        stats[f"chrono_minus_{arm}"]={
            "point":float(event.mean()),
            "per_fixed_seed":[float(d.mean()) for d in ds],
            "user_clustered_bootstrap_CI_95":[float(x) for x in
                np.quantile(samples,[.025,.975])],
            "bootstrap_seed":BOOT_SEED,"bootstrap_repeats":BOOT,
            "historic_unique_users":len(unique),
            "actual_hist_target_events":len(truth),
            "seeds_not_independent":True,
            "development_reused_not_independent_test":True
        }
    return stats


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    fit,dev,mean,scale,pop=load_real()
    if len(fit["y"])!=10776 or len(dev["y"])!=5952:
        raise ValueError("genuine FIT/DEV source changed")
    r={"study":STUDY,
       "source_SHA256":"c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c",
       "historical_development_already_exposed":True,
       "reserved_user_cohorts_opened":False,
       "training_FIT_windows":len(fit["y"]),
       "historical_DEV_windows":len(dev["y"]),
       "same_real_train_examples_and_supervised_targets":True,
       "all_neural_params_active":True,
       "trainable_parameter_counts":COUNT,
       "full_training_FLOPs_equal":False,
       "same_CPU_budget_and_train_examples_simultaneously_matched":False,
       "equal_300_optimizer_steps":True,
       "independent_competitiveness_claim_allowed":False,
       "all_models_complete":False,
       "model_results":[]}
    write_json(root/"results.json",r)
    for seed in SEEDS:
        for arm in ARMS:
            r["model_results"].append(train_one(arm,seed,fit,dev,mean,scale,pop,root))
            write_json(root/"results.json",r)
    groups={arm:[x for x in r["model_results"] if x["arm"]==arm] for arm in ARMS}
    means={arm:float(np.mean([x["historical_dev_metrics"]["Brier"] for x in rows]))
           for arm,rows in groups.items()}
    ce={arm:float(np.mean([x["historical_dev_metrics"]["ECE10"] for x in rows]))
        for arm,rows in groups.items()}
    chrono=groups["chrono_native"]
    ordered=[a["historical_dev_metrics"]["Brier"]-b["historical_dev_metrics"]["Brier"]
             for a,b in zip(chrono,groups["order_free_pool"])]
    sorted_delta=[a["historical_dev_metrics"]["Brier"]-b["historical_dev_metrics"]["Brier"]
                  for a,b in zip(chrono,groups["block_sorted_native"])]
    score={
        "mean_chrono_minus_order_free_Brier":float(np.mean(ordered)),
        "per_seed_chrono_minus_order_free_Brier":ordered,
        "mean_chrono_minus_block_sorted_Brier":float(np.mean(sorted_delta)),
        "per_seed_chrono_minus_block_sorted_Brier":sorted_delta,
        "chronological_better_than_order_free_all_seeds":
             all(x<0 for x in ordered) and np.mean(ordered)<=-.0005,
        "chronological_better_than_block_sorted_all_seeds":
             all(x<0 for x in sorted_delta) and np.mean(sorted_delta)<=-.0002,
        "chrono_minus_order_free_mean_ECE10":ce["chrono_native"]-ce["order_free_pool"],
        "calibration_guard":ce["chrono_native"]<=ce["order_free_pool"]+.003
    }
    score["all_three_gates_pass"]=all((score["chronological_better_than_order_free_all_seeds"],
                                      score["chronological_better_than_block_sorted_all_seeds"],
                                      score["calibration_guard"]))
    r["mean_Brier"]=means
    r["mean_ECE10"]=ce
    r["preregistered_descriptive_gates"]=score
    r["historical_user_clustered_contrasts"]=user_cluster_interaction(root,r["model_results"],dev)
    r["all_models_complete"]=True
    r["original_weights_and_probs_untouched"]=True
    r["reserved_cohorts_still_closed"]=True
    r["candidate_promoted"]=False
    write_json(root/"results.json",r)
    print(json.dumps({"study":STUDY,"historic_mean_Brier":means,
                      "prereg_gates":score,"new_cohort":False},sort_keys=True),flush=True)


def replay(root):
    root=Path(root)
    record=json.loads((root/"results.json").read_text())
    assert record["study"]==STUDY
    assert len(record["model_results"])==9 and record["all_models_complete"]
    assert record["reserved_cohorts_still_closed"]
    fit,dev,mean,scale,pop=load_real()
    for row in record["model_results"]:
        cp=root/row["checkpoint"];pp=root/row["predictions"]
        assert row["checkpoint_sha256"]==file_sha(cp)
        assert row["predictions_sha256"]==file_sha(pp)
        ck=torch.load(cp,weights_only=True,map_location="cpu")
        assert ck["arm"]==row["arm"] and ck["seed"]==row["seed"]
        assert ck["step"]==300 and ck["parameter_count"]==COUNT[ck["arm"]]
        net=torch.jit.script(model_for(ck["arm"],mean,scale,pop))
        net.load_state_dict(ck["state_dict"],strict=True)
        expected=np.load(pp)
        actual=evaluate(net,dev)
        if not np.allclose(expected,actual,rtol=0,atol=1e-7):
            raise ValueError("frozen true DEV replay failed "+row["arm"])
        if abs(scores(actual,dev["y"].numpy())["Brier"]-
               row["historical_dev_metrics"]["Brier"])>1e-8:
            raise ValueError("authentic reported Brier does not replay")
    print(json.dumps({"status":"ME029_ALL9_REAL_MODELS_REPLAY_PASS",
                      "replayed_authentic_probabilities":9*len(dev["y"]),
                      "future_outcomes_opened":False}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/me029")
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--replay",action="store_true")
    a=p.parse_args()
    if a.self_test:
        print(json.dumps(preflight(),sort_keys=True))
    elif a.replay:
        replay(a.out)
    else:run(a.out)
