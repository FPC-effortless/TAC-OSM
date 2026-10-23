"""ESWAM002: real original BodenAI FIT development, 5 original models + Git replay.

No independently fresh TEST, no intervention/action-authority claim or architecture
superiority. Reuse identical original ME076 source, FIT/REPORT and query generator.
The target recorded a[t] is only an auxiliary *future label*, never model input.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.validate_research_parity import check_manifest
from scripts.me076_train_system1_real import (
    BATCH, HISTORY, SEEDS, UPDATES, TIME_STEP_SECONDS, MOTION_PHYSICAL_THRESHOLD,
    load_original, choose_training, build_batch, evaluation_origins, digest, sha, write,
)
from tac_osm.parameter_budget import (
    budgeted_optimizer, checked_optimizer_step, save_training_checkpoint,
)
from tac_osm.variants.exchange_world_action import ExchangeWorldActionModel

ROOT = Path("research/mathematical-engineering/eswam002")
MANIFEST = ROOT / "FAIR_COMPARISON.json"
OUT = Path("research/replay-bundles/ESWAM002")
OLD = Path("research/replay-bundles/ME076")
STUDY = "ESWAM002_EXCHANGE_REAL_ROBOT_DEVELOPMENT"
PARAMETERS = 100343


def ids_for(selection):
    # Stable externally observed EPISODE ids. One address per example; no
    # multiple-entity or 256-event learned memory claim follows.
    return torch.tensor([
        int.from_bytes(hashlib.sha256(f"ESWAM/episode/{task}/{ep}".encode()).digest()[:7], "big")
        for task, ep, _ in selection], dtype=torch.long)


def supervised_next_command(rows, selection):
    # Logged a[t] is a future *target* after the last known a[t-1].
    return torch.stack([rows[(task, ep)]["action"][t] for task, ep, t in selection])


def rollout(model, x, a, ids):
    if x.ndim != 3 or x.shape[1:] != (HISTORY, 14) or a.shape != x:
        raise ValueError("eight genuine real observed state/action frames required")
    if ids.shape != (len(x),) or ids.dtype != torch.long:
        raise ValueError("original episode identity required")
    state = None
    elapsed = x.new_full((len(x),), TIME_STEP_SECONDS)
    observed = torch.ones_like(x[:, 0], dtype=torch.bool)
    pred = None
    for j in range(HISTORY):
        pred = model(x[:,j], a[:,j], elapsed, ids, observed, state=state)
        state = pred.state
    return pred


def physical_loss(p, y, actual_next_command):
    if p.mean_next.shape != y.shape or actual_next_command.shape != p.proposed_action.shape:
        raise ValueError("registered physical and logged behavior targets required")
    scale = p.scale_next.clamp_min(1e-5)
    proper_nll = ((p.mean_next - y).abs()/scale + (2*scale).log()).mean()
    behavior_imitation = F.smooth_l1_loss(p.proposed_action, actual_next_command)
    return proper_nll + 0.05 * behavior_imitation, proper_nll, behavior_imitation


def score(model, rows, identified, queries):
    scale = torch.as_tensor(identified["state_scale"])
    gain = torch.as_tensor(identified["servo_gain"])
    limit = torch.as_tensor(identified["servo_step_limit"])
    model.eval()
    original=[]
    with torch.no_grad():
        for off in range(0,len(queries),24):
            selection=queries[off:off+24]
            x,a,y=build_batch(rows,selection)
            behavior=supervised_next_command(rows,selection)
            p=rollout(model,x,a,ids_for(selection))
            last=x[:,-1]; issued=a[:,-1]
            persistence=last
            servo=last+torch.clamp(gain*(issued-last),-limit,limit)
            err=((p.mean_next-y).abs()*scale).mean(-1)
            persist=((persistence-y).abs()*scale).mean(-1)
            servo_err=((servo-y).abs()*scale).mean(-1)
            motion=((y-last).abs()*scale).mean(-1)
            nll=(((p.mean_next-y).abs()/p.scale_next)+(2*p.scale_next).log()).mean(-1)
            command_l1=(p.proposed_action-behavior).abs().mean(-1)
            command_repeat=(issued-behavior).abs().mean(-1)
            for i,(task,ep,t) in enumerate(selection):
                original.append({
                    "task":task,"episode_index":ep,"frame_index":t,
                    "physical_movement_mean":float(motion[i]),
                    "motion_rich_threshold_0_025":bool(motion[i]>=MOTION_PHYSICAL_THRESHOLD),
                    "physical_mae_model":float(err[i]),
                    "physical_mae_persistence":float(persist[i]),
                    "physical_mae_fit_servo_lag1":float(servo_err[i]),
                    "normalized_laplace_nll":float(nll[i]),
                    "next_recorded_command_imitation_l1":float(command_l1[i]),
                    "repeat_last_recorded_command_l1":float(command_repeat[i]),
                    "measured_next_state_normalized":y[i].tolist(),
                    "forecast_mean_normalized":p.mean_next[i].tolist(),
                    "forecast_scale_normalized":p.scale_next[i].tolist(),
                })
    def summarize(kept):
        if not kept:return {"n":0,"ineligible":True}
        d={"n":len(kept),"episode_count":len(set((r["task"],r["episode_index"]) for r in kept))}
        for field in ("physical_mae_model","physical_mae_persistence",
                      "physical_mae_fit_servo_lag1","normalized_laplace_nll",
                      "next_recorded_command_imitation_l1","repeat_last_recorded_command_l1"):
            d[field]=sum(row[field] for row in kept)/len(kept)
        return d
    scores={
        "all":summarize(original),
        "movement_rich":summarize([r for r in original if r["motion_rich_threshold_0_025"]]),
        "near_stationary":summarize([r for r in original if not r["motion_rich_threshold_0_025"]]),
        "actual_intervention_benefit_measured":False,
        "actual_policy_success_label_available":False,
        "source_report_already_exposed_DEV":True,
    }
    return original, scores


def original_me076_same_case(seed, new_rows):
    # No refit of baseline weights; compare original Git-owned real predictions.
    path=OLD/f"seed-{seed}"/"ORIGINAL_REAL_PREDICTIONS.json"
    if not path.is_file():
        raise RuntimeError("frozen original ME076 raw case predictions absent")
    old=json.loads(path.read_text())
    if len(old)!=len(new_rows):
        raise RuntimeError("original ME076 case count differs")
    paired=[]
    for a,b in zip(old,new_rows):
        keys=("task","episode_index","frame_index")
        if tuple(a[k] for k in keys)!=tuple(b[k] for k in keys):
            raise RuntimeError("not identical measured robot episode/time case")
        for name in ("physical_mae_persistence","physical_mae_fit_servo_lag1"):
            if not math.isclose(a[name],b[name],rel_tol=1e-6,abs_tol=1e-6):
                raise RuntimeError("underlying original physical normalizer/query changed")
        paired.append(b["physical_mae_model"]-a["physical_mae_model"])
    return {
        "original_checkpoint_same_case":True,
        "original_checkpoint_source":"ME076 original Git-owned predictions, frozen unchanged",
        "total_cases":len(paired),
        "eswam_minus_me076_mean_physical_mae":sum(paired)/len(paired),
        "per_case_difference_sha256":digest(paired),
        "full_training_compute_parity_verified":False,
        "architecture_superiority_qualified":False,
    }


def preflight():
    live=ExchangeWorldActionModel()
    trainable=sum(p.numel() for p in live.parameters() if p.requires_grad)
    deployed=sum(p.numel() for p in live.parameters())
    if (trainable,deployed)!=(PARAMETERS,PARAMETERS):
        raise RuntimeError(f"unexpected whole-model size {trainable}/{deployed}")
    manifest=json.loads(MANIFEST.read_text())
    if check_manifest(manifest,"preflight")["status"]!="PASS_PREFIT_SINGLE_MODEL":
        raise RuntimeError("prospective registered real source/budget rejected")
    if {(r["seed"],r["architecture"],r["trainable_parameters"],r["deployed_parameters"])
            for r in manifest["models"]} != {
            (seed,live.architecture,PARAMETERS,PARAMETERS) for seed in SEEDS}:
        raise RuntimeError("original five model seed/count registration differs")
    return trainable


def train_one(fit, report, identified, source, seed):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    model=ExchangeWorldActionModel(14,14)
    order,fit_source=choose_training(fit,identified,seed)
    queries,dev_source=evaluation_origins(report,identified)
    folder=OUT/f"seed-{seed}"
    folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/"live-budget.json"
    optimizer=budgeted_optimizer(
        model,torch.optim.AdamW,model.parameters(),lr=.0007,weight_decay=.0001,
        budget_manifest=MANIFEST,budget_seed=seed,
        budget_architecture=model.architecture,budget_receipt=receipt)
    trace=[]
    began=time.process_time()
    for update,selection in enumerate(order,1):
        x,a,y=build_batch(fit,selection)
        behavior=supervised_next_command(fit,selection)
        optimizer.zero_grad(set_to_none=True)
        p=rollout(model,x,a,ids_for(selection))
        loss,nll,action_imitation=physical_loss(p,y,behavior)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite physical training loss")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):
            raise RuntimeError("nonfinite real-gradient norm")
        checked_optimizer_step(optimizer)
        if update in (1,40,80,120,160):
            trace.append({"step":update,"real_train_NLL":float(nll.detach()),
                          "recorded_action_imitation":float(action_imitation.detach())})
            print(json.dumps({"study":"ESWAM002","seed":seed,**trace[-1]}),flush=True)
    elapsed=time.process_time()-began
    final=folder/"final.pt"
    if final.exists():
        raise RuntimeError("original existing fitted checkpoint must not be overwritten")
    save_training_checkpoint({
        "study_id":STUDY,"architecture":model.architecture,"seed":seed,
        "original_live_training_steps":UPDATES,"weights":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "torch_rng_state":torch.get_rng_state(),
        "numpy_rng_state":np.random.get_state()[1].tolist(),
        "python_rng_state":random.getstate(),
        "original_source":source,"fit_order":fit_source,"train_trace":trace,
        "factual_input_contract":"state[t-7..t], last_known_previous_issued_qpos[t-8..t-1] and event id",
        "outcome_targets":"actual next state[t+1], next recorded issued qpos[t] imitation only",
    },final)
    real_rows,metrics=score(model,report,identified,queries)
    comparison=original_me076_same_case(seed,real_rows)
    write(folder/"ORIGINAL_REAL_PREDICTIONS.json",real_rows)
    summary={
        "study":STUDY,"seed":seed,"deployed_parameters":PARAMETERS,
        "real_optimizer_updates":UPDATES,"trained_on_synthetic_data":False,
        "source_dev_previously_exposed":True,"real_source":source,
        "original_checkpoint_sha256":sha(final),
        "original_prediction_sha256":sha(folder/"ORIGINAL_REAL_PREDICTIONS.json"),
        "training_cpu_seconds":elapsed,"fit_evidence":fit_source,
        "dev_evidence":dev_source,"real_dev_metrics":metrics,
        "frozen_original_me076_pair":comparison,
        "claim":"EXPOSED_DEV_ONLY; not causal action or fair architecture superiority",
    }
    write(folder/"RESULTS.json",summary)
    return summary


def fit(cache):
    preflight()
    if any(OUT.glob("seed-*/final.pt")):
        raise RuntimeError("cannot overwrite original source-trained models")
    torch.set_num_threads(1)
    _,fit_rows,report,identified,source=load_original(cache)
    write(ROOT/"SOURCE_QUALIFICATION.json",source)
    outputs=[]
    for seed in SEEDS:
        outputs.append(train_one(fit_rows,report,identified,source,seed))
        write(OUT/"ORIGINAL_SEED_INDEX.json",outputs)
    write(ROOT/"RESULTS.json",{
        "study":STUDY,"models":len(outputs),"all":{str(r["seed"]):r["real_dev_metrics"] for r in outputs},
        "paired_original_me076":{str(r["seed"]):r["frozen_original_me076_pair"] for r in outputs},
        "original_source":source,"claim":"EXPOSED_DEV_ONLY; no independent architecture intelligence gain"})
    print("ESWAM002_REAL_FIT_DONE="+json.dumps({
        "original_models":len(outputs),
        "all_case_mae":[o["real_dev_metrics"]["all"]["physical_mae_model"] for o in outputs],
        "motion_mae":[o["real_dev_metrics"]["movement_rich"]["physical_mae_model"] for o in outputs],
        "source_hash":source["original_160_episode_source_sha256"]}),flush=True)


def replay(cache):
    preflight()
    torch.set_num_threads(1)
    _,_,report,identified,source=load_original(cache)
    if source!=json.loads((ROOT/"SOURCE_QUALIFICATION.json").read_text()):
        raise RuntimeError("source fingerprint/source partition mismatch")
    queries,_=evaluation_origins(report,identified)
    records=[]
    for seed in SEEDS:
        folder=OUT/f"seed-{seed}"
        registered=json.loads((folder/"RESULTS.json").read_text())
        if sha(folder/"final.pt")!=registered["original_checkpoint_sha256"]:
            raise RuntimeError("original ESWAM weight bytes changed")
        original=json.loads((folder/"ORIGINAL_REAL_PREDICTIONS.json").read_text())
        saved=torch.load(folder/"final.pt",map_location="cpu",weights_only=False)
        model=ExchangeWorldActionModel(14,14)
        model.load_state_dict(saved["weights"],strict=True)
        if saved["seed"]!=seed or saved["original_live_training_steps"]!=UPDATES:
            raise RuntimeError("trained original model epoch/seed identity mismatch")
        observed,metrics=score(model,report,identified,queries)
        if len(original)!=len(observed):
            raise RuntimeError("original readout count changed")
        for a,b in zip(original,observed):
            for key in ("task","episode_index","frame_index"):
                if a[key]!=b[key]:
                    raise RuntimeError("original event/source ID does not match")
            for key in ("physical_mae_model","physical_mae_persistence",
                        "physical_mae_fit_servo_lag1","normalized_laplace_nll",
                        "next_recorded_command_imitation_l1"):
                if not math.isclose(a[key],b[key],rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError("original frozen real forecast changed: "+key)
        original_me076_same_case(seed,observed)
        records.append({"seed":seed,"real_cases":len(observed),
                        "checkpoint_sha256":registered["original_checkpoint_sha256"],
                        "full_original_real_score_replay":True})
    write(OUT/"INDEPENDENT_ORIGINAL_REAL_REPLAY.json",{
        "study":STUDY,"models":records,
        "claim":"original Git checkpoints replayed without any optimizer updates",
        "source_status":"previously exposed episode-disjoint authentic real DEV"})
    print("ESWAM002_REPLAY_VERIFIED="+json.dumps(records),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("preflight","fit","replay"))
    parser.add_argument("--cache",type=Path)
    args=parser.parse_args()
    if args.mode=="preflight":
        print("ESWAM002_MODEL_PREFLIGHT="+str(preflight()))
    else:
        if args.cache is None:
            raise ValueError("authentic source cache path required")
        if args.mode=="fit":
            fit(args.cache)
        else:
            replay(args.cache)
