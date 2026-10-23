"""ME061 complete matched development fit after ME060's preserved failure."""
import argparse
import copy
import json
import platform
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.profiler import ProfilerActivity,profile

from scripts.me060_train_trajectory import (
    prepare,cohorts,schedules,batch,digest,evaluate,HISTORY,HORIZON,BATCH,UPDATES)
from scripts.me057_train_fused_anchor_robot import (
    load_part,normalize,sha_file,write_json)
from scripts.validate_research_parity import check_manifest
from tac_osm.real_robot_me061_reliable import (
    ReliableTrajectoryModel,VerifiedState,physical_forecast_loss)

ROOT=Path("research/mathematical-engineering/me061")
SEEDS=(61001,61002,61003,61004,61005)
CONFIG={"native_tac":(32,64),"gru_world":(44,82)}


from tac_osm.parameter_budget import budgeted_optimizer, checked_optimizer_step, save_training_checkpoint

def build(name,identified):
    config=json.loads((ROOT/"MODEL_CONFIG.json").read_text()) if (ROOT/"MODEL_CONFIG.json").exists() else CONFIG
    return ReliableTrajectoryModel(identified,name,*config[name])


def objective(model,data):
    states,actions,human=data;b=model.initial(len(states))
    with torch.no_grad():
        for t in range(HISTORY-16):
            b=model.observe(b,states[:,t],actions[:,t],states[:,t+1],human[:,t,None])
    snapshots=[]
    for t in range(HISTORY-16,HISTORY):
        b=model.observe(b,states[:,t],actions[:,t],states[:,t+1],human[:,t,None]);snapshots.append(b)
    p=model.forecast(b,states[:,HISTORY],actions[:,HISTORY:HISTORY+HORIZON])
    alive=(~human[:,HISTORY:HISTORY+HORIZON]).cumprod(1).bool().unsqueeze(-1)
    primary=physical_forecast_loss(model,p,states[:,HISTORY+1:HISTORY+HORIZON+1],alive)
    merged=VerifiedState(*(torch.stack([s[i] for s in snapshots],1).flatten(0,1) for i in range(5)))
    start=HISTORY-15
    dense=model.forecast(merged,states[:,start:HISTORY+1].flatten(0,1),
        actions[:,start:HISTORY+1].flatten(0,1)[:,None])
    target=states[:,start+1:HISTORY+2].flatten(0,1)[:,None]
    mask=(~human[:,start:HISTORY+1]).flatten()[:,None,None]
    return primary+.25*physical_forecast_loss(model,dense,target,mask)


def flops(model,data):
    model.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CPU],with_flops=True) as p:
        objective(model,data).backward()
    return int(sum(x.flops for x in p.key_averages()))


def register(cache):
    torch.set_num_threads(1)
    split,_,identified,source,source_hash=prepare(cache)
    orders={s:schedules(source,s) for s in SEEDS};data=batch(source,orders[SEEDS[0]][0])
    counts={};costs={}
    for name in CONFIG:
        m=build(name,identified);counts[name]=sum(p.numel() for p in m.parameters());costs[name]=flops(m,data)
    plan=json.loads(Path("research/mathematical-engineering/me060/FAIR_COMPARISON.json").read_text())
    plan["study_id"]="ME061_RELIABILITY_CONDITIONED_NATIVE_WORLD_MODEL_DEVELOPMENT"
    plan["literature_gap"]["unresolved_TAConly_question"]="Whether a persistence-exact TAC with causal physical-prior reliability, action-conditioned associative reads and physical-unit probabilistic training learns more useful factual dynamics."
    plan["literature_gap"]["what_is_not_being_rediscovered"]="Innovation-based observer reliability, zero-initialized residual forecasting, recurrent burn-in and proper uncertainty scoring are established methods; evaluate their integrated native TAC system."
    plan["parent_negative_result"]="ME060 source and different-task outcomes remain frozen; this is a second developmental design for both neural arms."
    template=plan["arms"][:2];rows=[]
    for seed in SEEDS:
        for name in CONFIG:
            row=copy.deepcopy(next(r for r in template if r["architecture"]==name));row["seed"]=seed
            row["trainable_parameters"]=row["deployed_parameters"]=counts[name]
            # Include the comparator's allocated zero slow/memory tensors,
            # not only its active GRU vector.
            row["nontrainable_state_bytes"]=(128+28 if name=="native_tac" else 2*CONFIG[name][0]+64+28)*4
            row["training_sample_order_sha256"]=digest(orders[seed])
            row["feature_contract_sha256"]=digest("ME061: same real64 history,16 action prefixes; 14 velocity and14 physical innovation statistics computed causally in each model")
            row["target_contract_sha256"]=digest("ME061: identical multihorizon and16 pre-feedback factual transitions; physical MAE+.01 FIT physical-step-scaled Laplace score")
            row["budget"]["planned_training_flops"]=costs[name]*UPDATES
            row["budget"]["max_training_flops"]=int(max(costs.values())*UPDATES*1.01)
            row["budget"]["prefit_profile_training_flops"]=costs[name]
            rows.append(row)
    plan["arms"]=rows;ROOT.mkdir(parents=True,exist_ok=True)
    write_json(ROOT/"FAIR_COMPARISON.json",plan)
    write_json(ROOT/"PREFLIGHT.json",{"counts":counts,"reported_update_flops":costs,"optimizer_steps":0,
        "selection":"count/FLOP-only width adjustment; no fit or new outcome used"})
    print(json.dumps({"counts":counts,"costs":costs}),flush=True);print(check_manifest(plan,"preflight"),flush=True)


def train(cache,folder):
    torch.set_num_threads(1)
    plan=json.loads((ROOT/"FAIR_COMPARISON.json").read_text());check_manifest(plan,"preflight")
    folder.mkdir(parents=True,exist_ok=False)
    _,_,identified,source,source_hash=prepare(cache)
    if source_hash!=plan["real_world_source"]["sha256"]:raise RuntimeError("source changed")
    records=[]
    for seed in SEEDS:
        picks=schedules(source,seed)
        for name in CONFIG:
            row=next(r for r in plan["arms"] if r["architecture"]==name and r["seed"]==seed)
            if digest(picks)!=row["training_sample_order_sha256"]:raise RuntimeError("schedule changed")
            torch.manual_seed(seed);model=build(name,identified)
            opt=budgeted_optimizer(model, torch.optim.AdamW, model.parameters(),lr=.001,weight_decay=.0001)
            trace=[];start=time.process_time()
            for step,pick in enumerate(picks,1):
                opt.zero_grad(set_to_none=True);value=objective(model,batch(source,pick))
                if not torch.isfinite(value):raise RuntimeError("nonfinite authentic objective")
                value.backward()
                if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.)):raise RuntimeError("nonfinite gradient")
                checked_optimizer_step(opt)
                if step%100==0:
                    trace.append({"update":step,"loss":float(value.detach())});print(json.dumps({"arm":name,"seed":seed,**trace[-1]}),flush=True)
            cpu=time.process_time()-start;cost=flops(model,batch(source,picks[-1]))*UPDATES
            path=folder/f"{name}-s{seed}.pt"
            save_training_checkpoint({"model":model.state_dict(),"optimizer":opt.state_dict(),"identified":{k:v.tolist() for k,v in identified.items()},
                "seed":seed,"architecture":name,"trace":trace},path)
            measured={"actual_steps":UPDATES,"actual_examples":UPDATES*BATCH,"actual_tokens":UPDATES*BATCH*(HISTORY+HORIZON),
                "training_flops":cost,"optimizer_cpu_seconds":cpu,"peak_memory_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                "flops_profiler":"common supported-operator estimate remeasured on final original batch; unsupported operations and optimizer excluded",
                "checkpoint_sha256":sha_file(path),"actual_sample_order_sha256":digest(picks),"evaluation_split_sha256":row["test_split_sha256"]}
            row["measured"]=measured;records.append({"arm":name,"seed":seed,"path":path.name,"measured":measured})
            write_json(folder/"CHECKPOINTS_FROZEN.json",records)
    check_manifest(plan,"final");write_json(folder/"FAIR_COMPARISON_FINAL.json",plan)
    # Freeze the entire model set before any new CAL/DEV scoring.
    split,metadata=cohorts(cache);evaluation={}
    for part in ("CAL","DEV"):
        raw=load_part(cache,split[part],metadata,"ME061_"+part);norm=normalize(raw,identified);evaluation[part]={}
        for r in records:
            m=build(r["arm"],identified);m.load_state_dict(torch.load(folder/r["path"],weights_only=False)["model"])
            evaluation[part][f'{r["arm"]}-s{r["seed"]}']=evaluate(m,norm,identified)
    write_json(folder/"results.json",{"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "checkpoints":records,"evaluation":evaluation,"old_TEST_opened":False,"independent_qualification":False})
    print("ME061_COMPLETE",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source-cache",type=Path,required=True)
    p.add_argument("--register",action="store_true");p.add_argument("--output",type=Path,default=Path("results/me061"))
    a=p.parse_args()
    if a.register:register(a.source_cache)
    else:train(a.source_cache,a.output)
