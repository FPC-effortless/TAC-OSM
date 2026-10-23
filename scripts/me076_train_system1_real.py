"""ME076: real original FIT-only 14D one-step System1 development and full replay.

Source: BodenAI RW-RL-HIL publisher 5 task FIT episodes, pinned and SHA checked.
Not sealed TEST; not causal policy. All original fitted model+optimizer+RNG bytes and
original per-episode and query scores must be committed to Git in this branch.
No synthetic/simulated model fitting; no CPU time equality requirement.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, resource, time
from pathlib import Path
import numpy as np
import torch

from scripts.me064_mixed_real_fit import task_registration, get_meta, load_members
from scripts.me057_train_fused_anchor_robot import identify, normalize
from scripts.me060_train_trajectory import digest
from scripts.validate_research_parity import check_manifest
from tac_osm.parameter_budget import (
    budgeted_optimizer, checked_optimizer_step, save_training_checkpoint)
from tac_osm.variants.system1_native import System1FastModel, physical_one_step_nll

ROOT=Path("research/mathematical-engineering/me076")
MANIFEST=ROOT/"FAIR_COMPARISON.json"
OUT=Path("research/replay-bundles/ME076")
EXPECTED_ORIGINAL="4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776"
SEEDS=(76001,76002,76003,76004,76005)
HISTORY=8
UPDATES=160
BATCH=12
TIME_STEP_SECONDS=1.0/15.0 # publisher specified 15Hz retained timestamps
MOTION_PHYSICAL_THRESHOLD=0.025

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")

def classify_members(registration):
    """Original previously exposed FIT episodes partitioned episode-wide, not frame-wide."""
    fit=[];report=[]
    for task in registration["training_tasks"]:
        members=registration["task_evidence"][task]["FIT"]
        if len(members)!=32:
            raise RuntimeError("original frozen per-task FIT count changed")
        ranked=sorted(members,key=lambda member:hashlib.sha256(
            f"ME076/episode-split/{task}/{member[0]}".encode()).hexdigest())
        fit.extend((task,e,n) for e,n in ranked[:24])
        report.extend((task,e,n) for e,n in ranked[24:])
    if len(fit)!=120 or len(report)!=40 or set((t,e) for t,e,_ in fit)&set((t,e) for t,e,_ in report):
        raise RuntimeError("invalid independent original episode identities")
    return fit,report

def load_original(cache):
    reg=task_registration()
    fit_ids,report_ids=classify_members(reg)
    raw={}
    for task in reg["training_tasks"]:
        members=[(e,n) for t,e,n in fit_ids+report_ids if t==task]
        raw.update(load_members(cache,task,members,get_meta(cache,task,reg["task_evidence"][task])))
    if len(raw)!=160:
        raise RuntimeError("missing original authentic episode")
    original=digest({f"{task}:{episode}":row["sha256"]
                     for (task,episode),row in sorted(raw.items())})
    if original!=EXPECTED_ORIGINAL:
        raise RuntimeError("previously verified publisher source bytes changed")
    fit_raw={key:raw[key] for key in [(t,e) for t,e,n in fit_ids]}
    report_raw={key:raw[key] for key in [(t,e) for t,e,n in report_ids]}
    # CRITICAL: normalization/servo identification use FIT source exclusively.
    identified=identify(fit_raw)
    fit=normalize(fit_raw,identified)
    report=normalize(report_raw,identified)
    evidence={
       "original_160_episode_source_sha256":original,
       "FIT_120_episodes":len(fit),
       "REPORT_40_episodes":len(report),
       "original_FIT_previously_exposed":True,
       "sealed_original_TEST_opened":False,
       "source_CAL_opened":False,
       "sampled_original_episodes_hash":digest({"FIT":fit_ids,"REPORT":report_ids}),
       "FIT_rows_sha256":digest({f"{t}:{e}":raw[(t,e)]["sha256"] for t,e,n in fit_ids}),
       "REPORT_rows_sha256":digest({f"{t}:{e}":raw[(t,e)]["sha256"] for t,e,n in report_ids}),
       "publisher_license":"CC-BY-4.0",
       "publisher_attribution":"BodenAI/RW-RL-HIL-Dataset; RW-RL Dataset parent",
       "training_normalization_leakage":False,
       "action_signal":"lag1 previously recorded qpos; not established causal torque",
       "time_step_seconds":"15 FPS source grid; not a real deployment latency measurement",
       "evaluation_status":"historically exposed source FIT reused as episode-disjoint DEV",
    }
    return reg,fit,report,identified,evidence

def valid_origins(rows,identified):
    pools={}
    for (task,ep),r in sorted(rows.items()):
        x,a,mask=r["state"],r["action"],r["human"]
        candidates=[];movement=[]
        for t in range(HISTORY,len(x)-1):
            # Exclude all human-control history and future authority,
            # conditioned on verified autonomous future, NOT a success label.
            if mask[t-HISTORY:t+2].any():
                continue
            physical=(x[t+1]-x[t]).abs()*torch.as_tensor(identified["state_scale"])
            magnitude=float(physical.mean())
            candidates.append(t)
            if magnitude>=MOTION_PHYSICAL_THRESHOLD:
                movement.append(t)
        if candidates:
            pools[(task,ep)]=(candidates,movement)
    return pools

def choose_training(fit,identified,seed):
    pools=valid_origins(fit,identified)
    tasks=sorted({task for task,ep in pools})
    if len(tasks)!=5 or sum(len(v[0]) for v in pools.values())<1000:
        raise RuntimeError("source has insufficient genuine autonomous training windows")
    rng=np.random.default_rng(seed)
    flat={task:[] for task in tasks}
    moving={task:[] for task in tasks}
    for (task,ep),(all_times,motion_times) in pools.items():
        flat[task].extend((task,ep,t) for t in all_times)
        moving[task].extend((task,ep,t) for t in motion_times)
    selections=[]
    for update in range(UPDATES):
        sampled=[]
        for b in range(BATCH):
            task=tasks[(update*BATCH+b)%len(tasks)]
            pool=moving[task] if b%2==0 and moving[task] else flat[task]
            sampled.append(pool[int(rng.integers(len(pool)))])
        selections.append(sampled)
    return selections,{"real_train_episode_count":len(pools),
      "real_source_train_windows":sum(len(x[0]) for x in pools.values()),
      "real_movement_train_windows":sum(len(x[1]) for x in pools.values()),
      "training_sample_order_sha256":digest(selections)}

def build_batch(rows,selections):
    xs=[];acts=[];targets=[]
    for task,ep,t in selections:
        row=rows[(task,ep)]
        xs.append(row["state"][t-HISTORY+1:t+1])
        acts.append(row["action"][t-HISTORY:t]) # lag1 at each observed state
        targets.append(row["state"][t+1])
    return torch.stack(xs),torch.stack(acts),torch.stack(targets)

def rollout(model,x,a):
    if x.ndim!=3 or x.shape[1:]!=(HISTORY,14) or a.shape!=x.shape:
        raise ValueError("actual contiguous 8-frame state/action history required")
    h=None;prediction=None
    tick=x.new_full((len(x),),TIME_STEP_SECONDS)
    for j in range(HISTORY):
        prediction=model(x[:,j],a[:,j],tick,h)
        h=prediction.fast
    return prediction

def evaluation_origins(rows,identified):
    pools=valid_origins(rows,identified); selections=[]
    for key,(all_times,movement) in sorted(pools.items()):
        selected=sorted(all_times,key=lambda t:hashlib.sha256(
            f"ME076/DEV-query/{key[0]}/{key[1]}/{t}".encode()).hexdigest())[:24]
        selections.extend((key[0],key[1],t) for t in selected)
    if len(pools)<35 or len(selections)<500:
        raise RuntimeError("too few episode-disjoint already exposed real REPORT outcomes")
    return selections,{"REPORT_episode_count":len(pools),"REPORT_selected_real_queries":len(selections),
        "REPORT_selected_query_sha256":digest(selections),
        "original_REPORT_all_autonomous_windows":sum(len(p[0]) for p in pools.values())}

def score(model,report,identified,queries):
    scale=torch.as_tensor(identified["state_scale"])
    gain=torch.as_tensor(identified["servo_gain"])
    limit=torch.as_tensor(identified["servo_step_limit"])
    original=[]
    model.eval()
    with torch.no_grad():
        for i in range(0,len(queries),24):
            ids=queries[i:i+24];x,a,y=build_batch(report,ids)
            pred=rollout(model,x,a)
            current=x[:,-1]; prevaction=a[:,-1]  # lag1 known
            persistence=current
            servo=current+torch.clamp(gain*(prevaction-current),-limit,limit)
            model_error=((pred.mean_next-y).abs()*scale).mean(-1)
            persist_error=((persistence-y).abs()*scale).mean(-1)
            servo_error=((servo-y).abs()*scale).mean(-1)
            move=((y-current).abs()*scale).mean(-1)
            laplace=(((pred.mean_next-y).abs()/pred.scale_next)+
                      (2*pred.scale_next).log()).mean(-1)
            for j,(task,ep,t) in enumerate(ids):
                original.append({"task":task,"episode_index":ep,"frame_index":t,
                  "physical_movement_mean":float(move[j]),
                  "motion_rich_threshold_0_025":bool(move[j]>=MOTION_PHYSICAL_THRESHOLD),
                  "physical_mae_model":float(model_error[j]),
                  "physical_mae_persistence":float(persist_error[j]),
                  "physical_mae_fit_servo_lag1":float(servo_error[j]),
                  "normalized_laplace_nll":float(laplace[j]),
                  "measured_next_state_normalized":y[j].tolist(),
                  "forecast_mean_normalized":pred.mean_next[j].tolist(),
                  "forecast_scale_normalized":pred.scale_next[j].tolist()})
    def summarize(selected):
        if not selected:return {"n":0,"unqualified":True}
        d={"n":len(selected),"original_distinct_episodes":len(set((r["task"],r["episode_index"]) for r in selected))}
        for field in ("physical_mae_model","physical_mae_persistence","physical_mae_fit_servo_lag1",
                      "normalized_laplace_nll","physical_movement_mean"):
            d[field]=sum(row[field] for row in selected)/len(selected)
        d["model_minus_persistence_mae"]=d["physical_mae_model"]-d["physical_mae_persistence"]
        return d
    return original,{"all":summarize(original),
       "movement_rich":summarize([r for r in original if r["motion_rich_threshold_0_025"]]),
       "near_stationary":summarize([r for r in original if not r["motion_rich_threshold_0_025"]]),
       "source_end_is_success_label":False,"seeds_are_not_independent_episodes":True}

def train_one(fit,report,identified,evidence,seed):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    model=System1FastModel(dimension=14)
    n=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n!=100476:raise RuntimeError("100476 non-padded live System1 model altered")
    order,fit_stats=choose_training(fit,identified,seed)
    queries,dev_stats=evaluation_origins(report,identified)
    folder=OUT/f"seed-{seed}";folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/"live-budget.json"
    optimizer=budgeted_optimizer(
      model,torch.optim.AdamW,model.parameters(),
      lr=.0007,weight_decay=.0001,budget_seed=seed,
      budget_architecture="system1_native_fast",budget_manifest=MANIFEST,
      budget_receipt=receipt)
    trace=[];start=time.process_time()
    for update,picks in enumerate(order,1):
        x,a,y=build_batch(fit,picks)
        optimizer.zero_grad(set_to_none=True)
        prediction=rollout(model,x,a)
        loss=physical_one_step_nll(prediction,y,torch.ones(len(y),dtype=torch.bool))
        if not torch.isfinite(loss):raise RuntimeError("nonfinite real training loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError("nonfinite original real gradient")
        checked_optimizer_step(optimizer)
        if update in (1,40,80,120,160):
            trace.append({"update":update,"train_source_nll":float(loss.detach())})
            print(json.dumps({"study":"ME076","seed":seed,**trace[-1]}),flush=True)
    elapsed=time.process_time()-start
    checkpoint=folder/"final.pt"
    save_training_checkpoint({
      "study_id":"ME076_SYSTEM1_REAL_FIT_ONLY_ONE_STEP_CAPABILITY",
      "seed":seed,"architecture":"system1_native_fast",
      "weights":model.state_dict(),"optimizer":optimizer.state_dict(),
      "torch_rng_state":torch.get_rng_state(),"numpy_rng_state":{\n        "bit_generator":np.random.get_state()[0],\n        "state":np.random.get_state()[1].tolist(),\n        "pos":int(np.random.get_state()[2]),\n        "has_gauss":int(np.random.get_state()[3]),\n        "cached_gaussian":float(np.random.get_state()[4]),\n      },
      "updates":UPDATES,"original_source":evidence,"training_sampling":fit_stats,
      "observed_state_scale_physical":identified["state_scale"].tolist(),
      "FIT_only_identified":{k:v.tolist() for k,v in identified.items()},
      "model_input_contract":"state[t-7..t], action[t-8..t-1], verified autonomous source, next state[t+1]",
      "train_trace":trace,
    },checkpoint)
    raw,results=score(model,report,identified,queries)
    write(folder/"ORIGINAL_REAL_PREDICTIONS.json",raw)
    output={"seed":seed,"architecture":"system1_native_fast",
      "evidence_tier":"DEV_CAPABILITY_OBSERVED_ONLY_ON_PREVIOUSLY_EXPOSED_ORIGINAL_FIT",
      "trained_original_real_FIT_only":True,"untouched_external_TEST_qualified":False,
      "original_checkpoint_sha256":sha(checkpoint),
      "original_prediction_sha256":sha(folder/"ORIGINAL_REAL_PREDICTIONS.json"),
      "real_training_cpu_seconds":elapsed,"real_optimizer_updates":UPDATES,
      "parameter_count":n, "fit_source":fit_stats,"dev_source":dev_stats,
      "original_replay_pending":True,"metrics":results}
    write(folder/"RESULTS.json",output)
    return output

def fit(cache):
    if OUT.exists() and any(OUT.glob("seed-*/final.pt")):
        raise RuntimeError("original checkpoints already exist; NEVER overwrite or refit")
    manifest=json.loads(MANIFEST.read_text())
    if check_manifest(manifest,"preflight")["status"]!="PASS_PREFIT_SINGLE_MODEL":
        raise RuntimeError("authentic preregistration rejected")
    if set(item["seed"] for item in manifest["models"])!=set(SEEDS):
        raise RuntimeError("frozen original five seed registration mismatch")
    torch.set_num_threads(1)
    reg,train,report,identified,evidence=load_original(cache)
    write(ROOT/"SOURCE_QUALIFICATION.json",evidence)
    # Verification of source bytes and eligibility is complete BEFORE first optimizer update.
    records=[]
    for seed in SEEDS:
        records.append(train_one(train,report,identified,evidence,seed))
        write(OUT/"ORIGINAL_SEED_INDEX.json",records)
    summary={"study":"ME076","trained_models":len(records),"source":evidence,
             "real_origins_per_seed":records[0]["dev_source"],
             "seeds":{str(r["seed"]):r["metrics"] for r in records},
             "claim":"Previously exposed real FIT episode-disjoint DEV only; no independent skill upgrade"}
    write(ROOT/"RESULTS.json",summary)
    print("ME076_REAL_ORIGINAL_FIT_COMPLETE_5_MODELS="+json.dumps({
        "seeds":len(records),"source_sha256":evidence["original_160_episode_source_sha256"],
        "dev_case_count":records[0]["dev_source"]["REPORT_episode_count"],
        "measured_MAE_per_seed":[r["metrics"]["all"]["physical_mae_model"] for r in records]}),flush=True)

def replay(cache):
    torch.set_num_threads(1)
    reg,fit,report,identified,evidence=load_original(cache)
    original=json.loads((ROOT/"SOURCE_QUALIFICATION.json").read_text())
    if original!=evidence:raise RuntimeError("source identity or original per-task split changed")
    queries,_=evaluation_origins(report,identified)
    found=[]
    for seed in SEEDS:
        folder=OUT/f"seed-{seed}"
        registered=json.loads((folder/"RESULTS.json").read_text())
        if sha(folder/"final.pt")!=registered["original_checkpoint_sha256"]:
            raise RuntimeError("original checkpoint bytes changed")
        original_predictions=json.loads((folder/"ORIGINAL_REAL_PREDICTIONS.json").read_text())
        model=System1FastModel(dimension=14)
        state=torch.load(folder/"final.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(state["weights"],strict=True)
        if state["seed"]!=seed or state["updates"]!=UPDATES:
            raise RuntimeError("original fitted state counter altered")
        recomputed,metrics=score(model,report,identified,queries)
        if len(recomputed)!=len(original_predictions):raise RuntimeError("number of original results changed")
        for row,new in zip(original_predictions,recomputed):
            if (row["task"],row["episode_index"],row["frame_index"])!=(new["task"],new["episode_index"],new["frame_index"]):
                raise RuntimeError("original query source identity changed")
            for name in ("physical_mae_model","physical_mae_persistence","physical_mae_fit_servo_lag1","normalized_laplace_nll"):
                if not math.isclose(row[name],new[name],abs_tol=1e-6,rel_tol=1e-6):
                    raise RuntimeError("original raw real forecast score differs at "+name)
        for cohort in ("all","movement_rich","near_stationary"):
            for name in ("physical_mae_model","physical_mae_persistence","normalized_laplace_nll"):
                if not math.isclose(metrics[cohort][name],registered["metrics"][cohort][name],
                                    abs_tol=1e-6,rel_tol=1e-6):
                    raise RuntimeError("original cohort mean score changed")
        found.append({"seed":seed,"case_count":len(original_predictions),
                      "original_checkpoint_sha256":registered["original_checkpoint_sha256"],
                      "original_scores_exact_within_tolerance":True})
    write(OUT/"INDEPENDENT_ORIGINAL_REAL_REPLAY.json",{
      "original_models":len(found),"result":"PASS_REAL_FIT_ONLY_REPLAY_NOT_INDEPENDENT_TRANSFER",
      "test_exposure":"previously exposed original FIT episode holdout only",
      "models":found})
    print("ME076_ORIGINAL_CHECKPOINTS_INDEPENDENTLY_REPLAYED="+json.dumps(found),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["fit","replay"])
    parser.add_argument("--cache",type=Path,required=True)
    args=parser.parse_args()
    if args.mode=="fit":fit(args.cache)
    else:replay(args.cache)
