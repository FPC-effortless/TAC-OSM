"""Preregister, train and score native trajectory experts on authentic robots.

Development only: old ME057 TEST is prohibited. An outcome-independent subset
of FIT is used for interval calibration after every checkpoint is frozen.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from scripts.me057_train_fused_anchor_robot import (
    source_index, load_part, identify, normalize, sha_file, write_json)
from scripts.validate_research_parity import check_manifest
from tac_osm.real_robot_me060_trajectory import (
    WorldState, TrajectoryWorldModel, forecast_loss)

HISTORY, HORIZON, BATCH, UPDATES = 64, 16, 8, 600
SEEDS = (60001,60002,60003,60004,60005)
ROOT = Path("research/mathematical-engineering/me060")
CONFIG = {"native_tac": (32,64), "gru_world": (44,88)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def build(name, identified):
    config = json.loads((ROOT/"MODEL_CONFIG.json").read_text()) if (
        ROOT/"MODEL_CONFIG.json").exists() else CONFIG
    return TrajectoryWorldModel(identified,name,*config[name])


def cohorts(cache):
    split, metadata = source_index(cache)
    # Whole adjacent-episode blocks remain together. This is a calibration
    # split inside historically exposed FIT, not fresh independent evidence.
    blocks = sorted({e//4 for e,_ in split["FIT"]}, key=lambda b: digest(["me060-calibration",b]))
    calibration = set(blocks[:max(1,len(blocks)//10)])
    train = [(e,n) for e,n in split["FIT"] if e//4 not in calibration]
    cal = [(e,n) for e,n in split["FIT"] if e//4 in calibration]
    return {"FIT":train,"CAL":cal,"DEV":split["DEV"],"TEST":split["TEST"]},metadata


def schedules(source, seed):
    eligible = [(e,t) for e,row in sorted(source.items())
        for t in range(HISTORY,row["n"]-HORIZON)
        if not bool(row["human"][t])]
    if not eligible:
        raise ValueError("no real robot-controlled FIT origins")
    rng = np.random.default_rng(seed)
    return [[eligible[int(rng.integers(len(eligible)))] for _ in range(BATCH)]
            for _ in range(UPDATES)]


def batch(source, picks):
    return tuple(torch.stack([source[e][key][t-HISTORY:t+HORIZON+1]
                              for e,t in picks]) for key in ("state","action","human"))


def objective(model, data):
    states, actions, human = data
    belief = model.initial(len(states))
    with torch.no_grad():
        for t in range(HISTORY-16):
            belief = model.observe(belief,states[:,t],actions[:,t],states[:,t+1],human[:,t,None])
    snapshots=[]
    for t in range(HISTORY-16,HISTORY):
        belief = model.observe(belief,states[:,t],actions[:,t],states[:,t+1],human[:,t,None])
        # Snapshot at current time t+1, before its next physical outcome.
        snapshots.append(belief)
    forecast = model.forecast(belief,states[:,HISTORY],actions[:,HISTORY:HISTORY+HORIZON])
    alive=(~human[:,HISTORY:HISTORY+HORIZON]).cumprod(1).bool().unsqueeze(-1)
    primary=forecast_loss(forecast,states[:,HISTORY+1:HISTORY+HORIZON+1],alive)
    merged=WorldState(*(torch.stack([s[i] for s in snapshots],1).flatten(0,1) for i in range(3)))
    # Same 16 factual targets and chronological pre-feedback snapshots for both arms.
    start=HISTORY-15
    current=states[:,start:HISTORY+1].flatten(0,1)
    dense=model.forecast(merged,current,actions[:,start:HISTORY+1].flatten(0,1)[:,None])
    actual=states[:,start+1:HISTORY+2].flatten(0,1)[:,None]
    mask=(~human[:,start:HISTORY+1]).flatten()[:,None,None]
    return primary+.25*forecast_loss(dense,actual,mask)


def update_flops(model, data):
    model.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CPU],with_flops=True) as p:
        objective(model,data).backward()
    # PyTorch reports operator estimates, not all elementwise/optimizer FLOPs.
    return int(sum(event.flops for event in p.key_averages()))


def prepare(cache):
    split,metadata=cohorts(cache)
    raw=load_part(cache,split["FIT"],metadata,"ME060_FIT")
    identified=identify(raw)
    source=normalize(raw,identified)
    source_hash=digest({str(e):r["sha256"] for e,r in sorted(raw.items())})
    return split,metadata,identified,source,source_hash


def register(cache):
    torch.set_num_threads(1)
    split,metadata,identified,source,source_hash=prepare(cache)
    order={s:schedules(source,s) for s in SEEDS}
    data=batch(source,order[SEEDS[0]][0])
    models={name:build(name,identified) for name in CONFIG}
    counts={name:sum(p.numel() for p in m.parameters()) for name,m in models.items()}
    flops={name:update_flops(m,data) for name,m in models.items()}
    rows=[]
    for seed in SEEDS:
        for name in CONFIG:
            m=models[name]
            rows.append({"architecture":name,"seed":seed,
                "trainable_parameters":counts[name],"deployed_parameters":counts[name],
                "nontrainable_state_bytes":(128 if name=="native_tac" else m.width)*4,
                "dataset_sha256":source_hash,
                "train_split_sha256":digest(split["FIT"]),"dev_split_sha256":digest(split["CAL"]),
                # Generic parity schema calls the final scoring split TEST.
                # Here that role is explicitly historical source DEV.
                "test_split_sha256":digest(split["DEV"]),
                "training_sample_order_sha256":digest(order[seed]),
                **{key:digest(value) for key,value in {
                    "feature_contract_sha256":"64 factual state/action/authority transitions; current state and 16 logged future actions",
                    "target_contract_sha256":"16-step physical state plus 16 pre-feedback one-step transitions; intervention-censored",
                    "optimizer_schedule_sha256":"600 AdamW lr .001 weight_decay .0001 clip1 batch8",
                    "tuning_policy_sha256":"zero outcome-based tuning; resource-only width selection",
                    "selection_policy_sha256":"final update only; all five seeds; no DEV selection",
                    "evaluation_contract_sha256":"historically exposed DEV only; old TEST prohibited; one deterministic midpoint robot origin per episode"}.items()},
                "hardware_profile":platform.processor()+" one CPU process thread PyTorch "+torch.__version__,
                "budget":{"steps":UPDATES,"examples_seen":UPDATES*BATCH,
                    "tokens_seen":UPDATES*BATCH*(HISTORY+HORIZON),"batch_size":BATCH,
                    "history_events":HISTORY,"planned_training_flops":flops[name]*UPDATES,
                    "max_training_flops":int(max(flops.values())*UPDATES*1.01),
                    "tuning_trials":0,"tuning_flops":0,
                    "prefit_profile_method":"torch.profiler with_flops observation forward plus decoder forward/backward; optimizer and unsupported operators excluded identically",
                    "prefit_profile_training_flops":flops[name]}})
    plan={"schema_version":"tac-osm.parity.v1","study_id":"ME060_NATIVE_TRAJECTORY_EXPERTS_DEVELOPMENT",
        "study_type":"architecture_comparison","registered_before_first_fit":True,
        "real_world_source":{"name":"BodenAI RW-RL-HIL clean_stains_off_tabletop pinned 9290f5a96dc91ea58256e81d9ce2dd9621890b51",
          "license":"CC-BY-4.0 per publisher dataset card","observed_real_outcomes":True,"sha256":source_hash},
        "literature_gap":{"unresolved_TAConly_question":"Does native bounded fast/slow associative state with direct horizon-conditioned physical residual experts improve authentic factual trajectory accuracy and interval quality?",
          "what_is_not_being_rediscovered":"Direct multihorizon regression, recurrent burn-in, delta-rule associative memory and probabilistic regression are established methods; only this integrated TAC candidate is tested.",
          "established_work":[
            {"paper":"Recurrent Experience Replay in Distributed Reinforcement Learning","established_result":"recurrent replay and burn-in","url":"https://openreview.net/forum?id=r1lyTjAqYX"},
            {"paper":"Mastering Diverse Domains through World Models","established_result":"robust joint predictive dynamics training","url":"https://arxiv.org/abs/2301.04104"},
            {"paper":"Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles","established_result":"proper predictive uncertainty scoring","url":"https://arxiv.org/abs/1612.01474"}]},
        "qualification":"DEVELOPMENT ONLY: existing FIT/DEV previously studied; no independent superiority or decision claim",
        "acceptance":"development TAC MAE >=2% lower than matched GRU and both persistence/servo; episode paired interval upper bound <0; calibrated 90% interval coverage reported separately",
        "test_access":"prohibited; only historical TEST IDs hashed from metadata",
        "calibration_split_sha256":digest(split["CAL"]),
        "parity_split_roles":{"train":"source FIT minus CAL", "dev":"CAL", "test":"historically exposed source DEV"},
        "excluded_original_TEST_ids_sha256":digest(split["TEST"]),"arms":rows}
    ROOT.mkdir(parents=True,exist_ok=True)
    write_json(ROOT/"FAIR_COMPARISON.json",plan)
    write_json(ROOT/"PREFLIGHT.json",{"counts":counts,"profiled_flops":flops,
        "source_fit_sha256":source_hash,"FIT_episodes":len(source),
        "CAL_episodes":len(split["CAL"]),"DEV_episodes":len(split["DEV"]),
        "optimizer_steps":0,"test_trajectories_opened":0,
        "flop_convention":"PyTorch supported-operator estimate; not complete hardware FLOPs"})
    print(json.dumps({"counts":counts,"flops":flops}),flush=True)
    print(check_manifest(plan,"preflight"),flush=True)


def evaluate(model, source, identified):
    scale=torch.from_numpy(identified["state_scale"])
    results={}
    model.eval()
    with torch.no_grad():
        for e,row in sorted(source.items()):
            candidates=[t for t in range(HISTORY,row["n"]-HORIZON) if not bool(row["human"][t])]
            if not candidates:
                continue
            t=candidates[len(candidates)//2]; b=model.initial(1)
            for j in range(t-HISTORY,t):
                b=model.observe(b,row["state"][j:j+1],row["action"][j:j+1],
                    row["state"][j+1:j+2],row["human"][j:j+1,None])
            current=row["state"][t:t+1]
            p=model.forecast(b,current,row["action"][None,t:t+HORIZON])
            actual=row["state"][None,t+1:t+HORIZON+1]
            alive=(~row["human"][t:t+HORIZON]).cumprod(0).bool()
            error=(p.mean-actual).abs()[0]
            servo=current; base=[]
            for a in row["action"][t:t+HORIZON]:
                servo=servo+model.servo_step(servo,a[None]);base.append(servo[0])
            base=torch.stack(base)
            results[str(e)]={"origin":t,"scored_steps":int(alive.sum()),
                "MAE":float((error[alive]*scale).mean()),
                "persistence_MAE":float(((current-actual).abs()[0,alive]*scale).mean()),
                "servo_MAE":float(((base-actual[0]).abs()[alive]*scale).mean()),
                "scaled_errors":(error[alive]/p.scale[0,alive]).flatten().tolist(),
                "physical_error_by_horizon":(error[alive]*scale).mean(-1).tolist(),
                "scale_mean":float((p.scale[0,alive]*scale).mean())}
    return results


def train(cache,output):
    torch.set_num_threads(1)
    plan=json.loads((ROOT/"FAIR_COMPARISON.json").read_text()); check_manifest(plan,"preflight")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    split,metadata,identified,source,source_hash=prepare(cache)
    if source_hash!=plan["real_world_source"]["sha256"]:
        raise RuntimeError("original FIT hashes changed")
    records=[]
    for seed in SEEDS:
        picks=schedules(source,seed)
        for name in CONFIG:
            row=next(r for r in plan["arms"] if r["seed"]==seed and r["architecture"]==name)
            if digest(picks)!=row["training_sample_order_sha256"]:raise RuntimeError("schedule changed")
            torch.manual_seed(seed);model=build(name,identified)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            trace=[];start=time.process_time()
            for step,pick in enumerate(picks,1):
                opt.zero_grad(set_to_none=True);value=objective(model,batch(source,pick))
                if not torch.isfinite(value):raise RuntimeError("nonfinite physical training objective")
                value.backward()
                if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.)):
                    raise RuntimeError("nonfinite gradient")
                opt.step()
                if step%100==0:
                    trace.append({"update":step,"loss":float(value.detach())})
                    print(json.dumps({"arm":name,"seed":seed,**trace[-1]}),flush=True)
            cpu=time.process_time()-start
            measured_flops=update_flops(model,batch(source,picks[-1]))*UPDATES
            target=output/f"{name}-s{seed}.pt"
            torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),
                "identified":{k:v.tolist() for k,v in identified.items()},"seed":seed,
                "architecture":name,"trace":trace},target)
            measured={"actual_steps":UPDATES,"actual_examples":UPDATES*BATCH,
                "actual_tokens":UPDATES*BATCH*(HISTORY+HORIZON),"training_flops":measured_flops,
                "optimizer_cpu_seconds":cpu,"peak_memory_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                "flops_profiler":"same supported-operator profiler convention remeasured on final actual batch; excludes unsupported operators and optimizer",
                "checkpoint_sha256":sha_file(target),"actual_sample_order_sha256":digest(picks),
                "evaluation_split_sha256":row["test_split_sha256"]}
            row["measured"]=measured
            records.append({"arm":name,"seed":seed,"path":target.name,"measured":measured})
            write_json(output/"CHECKPOINTS_FROZEN.json",records)
    final=check_manifest(plan,"final");write_json(output/"FAIR_COMPARISON_FINAL.json",plan)
    # No heldout/source-calibration target is used before this freeze point.
    results={"registration":str(ROOT/"FAIR_COMPARISON.json"),"final_parity":final,
        "independent_qualification":False,"old_TEST_opened":False,"checkpoints":records,"evaluation":{}}
    for part in ("CAL","DEV"):
        raw=load_part(cache,split[part],metadata,"ME060_"+part);norm=normalize(raw,identified)
        results["evaluation"][part]={}
        for r in records:
            m=build(r["arm"],identified);state=torch.load(output/r["path"],weights_only=False)
            m.load_state_dict(state["model"])
            results["evaluation"][part][f'{r["arm"]}-s{r["seed"]}']=evaluate(m,norm,identified)
    write_json(output/"results.json",results)
    print("ME060_COMPLETE",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-cache",type=Path,required=True)
    p.add_argument("--register",action="store_true")
    p.add_argument("--output",default="results/me060")
    args=p.parse_args()
    if args.register:register(args.source_cache)
    else:train(args.source_cache,args.output)
