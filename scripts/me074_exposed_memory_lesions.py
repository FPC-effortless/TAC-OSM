"""Frozen ME073 trained-memory readout lesions, genuine exposed BPI2012 only.

This is NOT a new matched architecture comparison or an independent capability
qualification. Without changing original weights or optimizer, zero either
native fast/slow/8x8 associative state AFTER factual observation and BEFORE
the next-event law. Verify full-state predictions against the exact Git-owned
original ME073 h1 marginals. Lesions are OOD manipulations of hidden states.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from tac_osm.workflow_me069 import EventBelief
from tac_osm.workflow_me073 import TrajectoryModel
from scripts.me073_autoregressive_world import (
    source, pack, cast_data, observed, ROOT as ORIGINAL_ROOT, BUNDLE as ORIGINAL_BUNDLE)
from scripts.me071_persistent_workflow import verify_blob
from scripts.me069_workflow_source import digest

BUNDLE=Path("research/replay-bundles/ME074")
AGES=(2,6,16)
VARIANTS=("full","fast_zero","slow_zero","association_zero","all_hidden_zero")
SEEDS=(73001,73002,73003,73004,73005)
EPS=1e-6

def lesion(belief,name):
    if name=="full":return belief
    z=lambda x:torch.zeros_like(x)
    return EventBelief(
        z(belief.fast) if name in ("fast_zero","all_hidden_zero") else belief.fast,
        z(belief.slow) if name in ("slow_zero","all_hidden_zero") else belief.slow,
        z(belief.memory) if name in ("association_zero","all_hidden_zero") else belief.memory)

def first_joint(model,belief,encoded,current,history):
    law=model.step(belief,encoded,current,history,0)
    surv=law.log_survival.exp()
    prob=torch.cat((surv[:,None]*law.mark_log_probability.exp(),
                    (1.-surv)[:,None]),-1)
    return ((prob+EPS)/(1+prob.shape[-1]*EPS)).detach().cpu().numpy()

def metrics(probs,targets,present):
    n=probs.shape[-1]-1
    indices=np.where(present,targets,n)
    chosen=np.maximum(np.take_along_axis(probs,indices[:,None],axis=1).ravel(),1e-12)
    ll=-np.log(chosen)
    cond=probs[:,:n]/np.maximum(probs[:,:n].sum(1,keepdims=True),1e-12)
    conditional_nll=-np.log(np.maximum(cond[np.arange(len(targets)),targets],1e-12))[present]
    return {"all_case_EOR_aware_NLL":float(ll.mean()),
            "all_case_Brier":float((np.square(probs).sum(1)-2*chosen+1).mean()),
            "EOR_Brier":float(np.square(probs[:,-1]-(~present).astype(float)).mean()),
            "future_activity_conditional_NLL":float(conditional_nll.mean()) if present.any() else None,
            "future_activity_accuracy":float((cond.argmax(1)[present]==targets[present]).mean()) if present.any() else None}

@torch.no_grad()
def main(cache):
    torch.set_num_threads(1)
    cases,pre,receipt=source(cache)
    inventory=json.loads((ORIGINAL_BUNDLE/"CHECKPOINTS.json").read_text())
    if len(inventory)!=5 or sorted(x["seed"] for x in inventory)!=list(SEEDS):
        raise ValueError("exact original five ME073 source-trained models required")
    test_original=json.loads((ORIGINAL_BUNDLE/"ORIGINAL_SCORES.json").read_text())
    scores={};deltas={}
    with np.load(ORIGINAL_BUNDLE/"ORIGINAL_FORECASTS.npz") as frozen:
        original={k:frozen[k].copy() for k in frozen if k.startswith("s") and "_paths" not in k and "_exact_suffix" not in k}
    for entry in inventory:
        verify_blob(entry["path"],entry["checkpoint_sha256"])
        saved=torch.load(entry["path"],weights_only=False,map_location="cpu")
        if saved["preprocessing"]!=pre:raise ValueError("original pre processing changed")
        model=TrajectoryModel(pre).double().eval()
        model.load_state_dict(saved["model"],strict=True)
        if sum(p.numel() for p in model.parameters())!=99781:
            raise ValueError("original ME073 model architecture / parameter count changed")
        for age in AGES:
            selected=[c for c in cases["REPORT"] if c["n"]>=age]
            if digest([c["id"] for c in selected])!=test_original["seeds"][str(entry["seed"])][str(age)]["IDs_sha256"]:
                raise ValueError("different real exposed cases selected for lesion comparison")
            collected={name:[] for name in VARIANTS}
            expected=original[f's{entry["seed"]}_age{age}'][:,0,:len(pre["activities"])+2]
            for start in range(0,len(selected),32):
                subset=selected[start:start+32]
                data=cast_data(pack(subset,list(range(len(subset))),length=age),torch.float64)
                (belief,encoded,current,history),_=observed(model,data,all_steps=False)
                for name in VARIANTS:
                    collected[name].append(first_joint(model,lesion(belief,name),encoded,current,history))
            predictions={k:np.concatenate(rows) for k,rows in collected.items()}
            np.testing.assert_allclose(predictions["full"],expected,atol=1e-5,rtol=1e-5)
            labels=np.asarray([int(c["activity"][age]) if c["n"]>age else 0 for c in selected])
            present=np.asarray([c["n"]>age for c in selected],bool)
            key=f"s{entry['seed']}_age{age}"
            scores[key]={name:metrics(pred,labels,present) for name,pred in predictions.items()}
            baseline=predictions["full"]
            target=np.where(present,labels,baseline.shape[-1]-1)
            base_logloss=-np.log(np.maximum(baseline[np.arange(len(target)),target],1e-12))
            deltas[key]={}
            for name in VARIANTS[1:]:
                p=predictions[name]
                ll=-np.log(np.maximum(p[np.arange(len(target)),target],1e-12))
                deltas[key][name]={
                    "mean_lesion_minus_full_all_case_NLL":float(np.mean(ll-base_logloss)),
                    "per_case_loss_difference_quantiles":np.quantile(ll-base_logloss,[.05,.5,.95]).tolist(),
                    "worse_case_fraction":float((ll>base_logloss).mean()),
                }
            print(json.dumps({"me074":"EXPOSED_GENUINE_MEMORY_LESIONS","seed":entry["seed"],
                "age":age,"full_NLL":scores[key]["full"]["all_case_EOR_aware_NLL"],
                "new_model_fits":0}),flush=True)
    summary={}
    for age in AGES:
        summary[str(age)]={}
        for name in VARIANTS:
            values=[scores[f"s{seed}_age{age}"][name]["all_case_EOR_aware_NLL"] for seed in SEEDS]
            summary[str(age)][name]={"mean_all_case_NLL":float(np.mean(values)),
                                     "five_original_seed_NLL":values}
        for name in VARIANTS[1:]:
            ds=[deltas[f"s{seed}_age{age}"][name]["mean_lesion_minus_full_all_case_NLL"] for seed in SEEDS]
            summary[str(age)][name]["delta_relative_to_full"]=float(np.mean(ds))
            summary[str(age)][name]["worse_trained_seeds"]=int(np.sum(np.asarray(ds)>0))
    result={
        "study":"ME074_POSTHOC_NATIVE_WORKFLOW_MEMORY_SENSITIVITY",
        "real_source_sha256":receipt["source_sha256"],
        "original_fitted_model_count":5,
        "new_model_fits":0,"new_optimizer_updates":0,"frozen_weights_unchanged":True,
        "original_REAL_REPORT_already_exposed":True,
        "original_TEST_opened":False,
        "horizon_tested":1,
        "ages":list(AGES),
        "complete_original_five_seed_score_rows":scores,
        "per_case_descriptive_lesion_deltas":deltas,
        "summary":summary,
        "evidence_limits":["Out-of-training-distribution zeroed belief readout is not an ablation retrained at matched compute",
                           "No original independent TEST; no forecasting architecture superiority",
                           "The original ME073 full readout exactly reproduces Git-owned original frozen h1 predictions",
                           "One-step readout sensitivity cannot establish 16/64/256 event memory or long-horizon counterfactual intelligence"],
    }
    BUNDLE.mkdir(parents=True,exist_ok=True)
    path=BUNDLE/"EXPOSED_MEMORY_LESIONS.json"
    if path.exists():raise ValueError("do not overwrite original diagnosis")
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("ME074_FROZEN_ME073_REAL_MEMORY_DIAG="+json.dumps(summary,sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--cache",type=Path,required=True)
    a=p.parse_args();main(a.cache)
