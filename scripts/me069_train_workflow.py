"""Matched original business-event fits, repository checkpoint freeze and scoring."""
from __future__ import annotations
import argparse
import json
import math
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.profiler import ProfilerActivity,profile

from scripts.me069_workflow_source import ROOT,HISTORY,load,digest,write_json
from scripts.validate_research_parity import check_manifest
from tac_osm.workflow_me069 import EventBelief,EventWorldModel,WorkflowForecast,workflow_loss,waiting_quantile,long_wait_probability

SEEDS=(69001,69002,69003,69004,69005)
UPDATES,BATCH=800,32
BUNDLE=Path("research/replay-bundles/ME069")
DEFAULT_CONFIG={"native_tac":[64,96],"gru_world":[80,143]}


from tac_osm.parameter_budget import budgeted_optimizer, checked_optimizer_step, save_training_checkpoint

def sha_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(arm,pre):
    config=json.loads((ROOT/"MODEL_CONFIG.json").read_text()) if (ROOT/"MODEL_CONFIG.json").exists() else DEFAULT_CONFIG
    return EventWorldModel(pre,arm,*config[arm])


def schedule(cases,seed):
    eligible=[(i,t) for i,c in enumerate(cases) for t in range(c["n"]-1)]
    rng=np.random.default_rng(seed)
    indices=rng.integers(len(eligible),size=(UPDATES,BATCH))
    return [[eligible[int(k)] for k in row] for row in indices]


def pack(cases,picks):
    n=len(picks)
    activity=np.zeros((n,HISTORY),np.int64);resource_ids=np.zeros_like(activity)
    features=np.zeros((n,HISTORY,8),np.float32);hours=np.zeros((n,HISTORY,1),np.float32)
    target=np.zeros_like(activity);delay=np.zeros((n,HISTORY),np.float32);mask=np.zeros((n,HISTORY),np.bool_)
    for i,(case,end) in enumerate(picks):
        c=cases[case];start=max(0,end-HISTORY+1);size=end-start+1;offset=HISTORY-size
        activity[i,offset:]=c["activity"][start:end+1]
        resource_ids[i,offset:]=c["resource"][start:end+1]
        features[i,offset:]=c["features"][start:end+1]
        hours[i,offset:,0]=c["gap"][start:end+1]/3600.
        target[i,offset:]=c["activity"][start+1:end+2]
        delay[i,offset:]=c["gap"][start+1:end+2]
        mask[i,offset:]=True
    return tuple(torch.from_numpy(x) for x in (activity,resource_ids,features,hours,target,delay,mask))


def predictions(model,data,dense=True):
    activity,resources,features,hours,target,delay,mask=data
    encoded=model.event_encoding(activity,resources,features)
    b=model.initial(len(activity));snapshots=[]
    for t in range(HISTORY):
        b=model.observe(b,encoded[:,t],hours[:,t],mask[:,t])
        if dense:snapshots.append(b)
    if dense:
        merged=EventBelief(*(torch.stack([b[i] for b in snapshots],1).flatten(0,1) for i in range(3)))
        p=model.forecast(merged,encoded.flatten(0,1),activity.flatten())
    else:p=model.forecast(b,encoded[:,-1],activity[:,-1])
    return p


def objective(model,data):
    p=predictions(model,data)
    return workflow_loss(p,data[4].flatten(),data[5].flatten(),data[6].flatten())


def reported_flops(model,data):
    model.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CPU],with_flops=True) as recording:
        objective(model,data).backward()
    return int(sum(x.flops for x in recording.key_averages()))


def register(cache):
    torch.set_num_threads(1)
    cohorts,pre,source=load(cache);orders={s:schedule(cohorts["FIT"],s) for s in SEEDS}
    first=pack(cohorts["FIT"],orders[SEEDS[0]][0])
    counts={};costs={};state_bytes={}
    for name in DEFAULT_CONFIG:
        m=build(name,pre);counts[name]=sum(p.numel() for p in m.parameters())
        costs[name]=reported_flops(m,first);state_bytes[name]=sum(x.numel()*x.element_size() for x in m.initial(1))
    arms=[]
    for seed in SEEDS:
        order=orders[seed]
        tokens=sum(min(t+1,HISTORY) for row in order for _,t in row)
        for name in DEFAULT_CONFIG:
            arms.append({"architecture":name,"seed":seed,"trainable_parameters":counts[name],"deployed_parameters":counts[name],
                "nontrainable_state_bytes":state_bytes[name],"dataset_sha256":source["source_sha256"],
                "train_split_sha256":source["cohorts"]["FIT"]["ids_sha256"],"dev_split_sha256":source["cohorts"]["DEV"]["ids_sha256"],
                "test_split_sha256":source["cohorts"]["TEST"]["ids_sha256"],"training_sample_order_sha256":digest(order),
                **{k:digest(v) for k,v in {
                    "feature_contract_sha256":"observed activity/lifecycle/resource, UTC temporal prefix features, previous gaps; no case ID or future actions",
                    "target_contract_sha256":"next recorded activity and zero/positive seconds until it; all valid chronological prefixes in each32 window",
                    "optimizer_schedule_sha256":"800 AdamW lr .001 wd .0001 clip1 batch32 fullBPTT32",
                    "tuning_policy_sha256":"zero outcome-based trials; model width selection by counts/operator FLOPs only",
                    "selection_policy_sha256":"all five final update checkpoints; no DEV weight selection",
                    "evaluation_contract_sha256":"one deterministic midpoint prefix per eligible case; case-paired bootstrap; sealed score access until repository checkpoint freeze"}.items()},
                "hardware_profile":"one process CPU thread PyTorch2.14.1+cpu; same local training process",
                "budget":{"steps":UPDATES,"examples_seen":UPDATES*BATCH,"tokens_seen":tokens,"batch_size":BATCH,"history_events":HISTORY,
                    "planned_training_flops":costs[name]*UPDATES,"max_training_flops":int(max(costs.values())*UPDATES*1.01),
                    "tuning_trials":0,"tuning_flops":0,"prefit_profile_training_flops":costs[name],
                    "prefit_profile_method":"torch.profiler with_flops supported-operator dense32 forward/backward on identical authentic padded windows; unsupported/optimizer ops omitted identically"}})
    plan={"schema_version":"tac-osm.parity.v1","study_id":"ME069_NATIVE_ENTERPRISE_WORKFLOW_WORLD_MODEL",
        "study_type":"architecture_comparison","registered_before_first_fit":True,
        "real_world_source":{"name":"Original BPI Challenge2012 loan application log,4TU article12689204/file24027287,version1",
            "license":"4TU General Terms of Use per original publisher metadata; raw records not redistributed",
            "sha256":source["source_sha256"],"observed_real_outcomes":True},
        "literature_gap":{"unresolved_TAConly_question":"Can elapsed-time native fast/slow associative predictive state improve authentic enterprise next-event and waiting-time forecasts over a matched GRU and empirical process priors?",
            "what_is_not_being_rediscovered":"LSTM predictive process monitoring, time-aware recurrence, categorical transition statistics and hurdle distributions are established; this tests their integration with native TAC persistent state, not a novelty claim for generic workflow prediction.",
            "established_work":[
                {"paper":"Predictive Business Process Monitoring with LSTM Neural Networks,Tax et al.,CAiSE2017","established_result":"next activity and event-time prediction from business-process prefixes","url":"https://arxiv.org/abs/1612.02130"},
                {"paper":"Patient Subtyping via Time-Aware LSTM Networks,Baytas et al.,KDD2017","established_result":"elapsed-time-aware recurrent representations","url":"https://doi.org/10.1145/3097983.3097997"},
                {"paper":"Recurrent Experience Replay in Distributed Reinforcement Learning,ICLR2019","established_result":"recurrent replay and temporal-state handling","url":"https://openreview.net/forum?id=r1lyTjAqYX"}]},
        "source_access_disclosure":source["heldout_access"],"arms":arms,
        "primary_gate":"TEST case-mean next-activity log loss at least2% below empirical transition prior and1% below matched GRU,paired case interval upper bound below0,>=4/5 seed wins; timing and risk separately reported",
        "limits":"next recorded event is not verified completion,timeout,SLA breach,causal action value or cost savings; no Woyengi integration deployed"}
    write_json(ROOT/"FAIR_COMPARISON.json",plan);write_json(ROOT/"FIT_PREPROCESSING.json",pre)
    write_json(ROOT/"SOURCE_QUALIFICATION.json",source)
    write_json(ROOT/"PREFLIGHT.json",{"counts":counts,"supported_operator_flops":costs,"optimizer_updates":0,"full_hardware_FLOPs_certified":False})
    print(json.dumps({"counts":counts,"flops":costs}),flush=True);print(check_manifest(plan,"preflight"),flush=True)


def train(cache):
    torch.set_num_threads(1)
    plan=json.loads((ROOT/"FAIR_COMPARISON.json").read_text());check_manifest(plan,"preflight")
    cohorts,pre,source=load(cache)
    if source["source_sha256"]!=plan["real_world_source"]["sha256"]:raise ValueError("original source changed")
    if digest(pre)!=source["FIT_preprocessing_sha256"]:raise ValueError("FIT preprocessing changed")
    folder=BUNDLE/"checkpoints";folder.mkdir(parents=True,exist_ok=True)
    if list(folder.glob("*.pt")):raise ValueError("never overwrite original checkpoints; use an explicit new registration")
    records=[]
    for seed in SEEDS:
        order=schedule(cohorts["FIT"],seed)
        for name in DEFAULT_CONFIG:
            row=next(r for r in plan["arms"] if r["architecture"]==name and r["seed"]==seed)
            if digest(order)!=row["training_sample_order_sha256"]:raise ValueError("FIT sampling schedule changed")
            torch.manual_seed(seed);model=build(name,pre)
            opt=budgeted_optimizer(model, torch.optim.AdamW, model.parameters(),lr=.001,weight_decay=.0001)
            trace=[];started=time.process_time();actual_tokens=0
            for step,picks in enumerate(order,1):
                data=pack(cohorts["FIT"],picks);actual_tokens+=int(data[6].sum())
                opt.zero_grad(set_to_none=True);loss=objective(model,data)
                if not torch.isfinite(loss):raise ValueError("nonfinite original workflow objective")
                loss.backward()
                if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.)):raise ValueError("nonfinite model gradient")
                checked_optimizer_step(opt)
                if step%100==0:
                    trace.append({"update":step,"loss":float(loss.detach())});print(json.dumps({"arm":name,"seed":seed,**trace[-1]}),flush=True)
            cpu=time.process_time()-started
            # Profile on the same original first window shape convention after fitting.
            flops=reported_flops(model,pack(cohorts["FIT"],order[0]))*UPDATES
            target=folder/f"{name}-s{seed}.pt"
            save_training_checkpoint({"model":model.state_dict(),"optimizer":opt.state_dict(),"torch_rng_state":torch.get_rng_state(),
                "preprocessing":pre,"seed":seed,"architecture":name,"trace":trace,
                "source_sha256":source["source_sha256"],"schedule_sha256":digest(order)},target)
            measured={"actual_steps":UPDATES,"actual_examples":UPDATES*BATCH,"actual_tokens":actual_tokens,
                "training_flops":flops,"optimizer_cpu_seconds":cpu,"peak_memory_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                "flops_profiler":"common supported-operator estimate repeated on registered first real window shape after training; optimizer/unsupported ops omitted",
                "checkpoint_sha256":sha_file(target),"actual_sample_order_sha256":digest(order),"evaluation_split_sha256":row["test_split_sha256"]}
            row["measured"]=measured;records.append({"architecture":name,"seed":seed,"path":str(target),"measured":measured})
            write_json(BUNDLE/"CHECKPOINT_MANIFEST.json",records)
    print(check_manifest(plan,"final"),flush=True);write_json(BUNDLE/"FAIR_COMPARISON_FINAL.json",plan)
    print("ME069_ALL_CHECKPOINTS_FROZEN_NO_DEV_OR_TEST_SCORING",flush=True)


def case_forecasts(model,cases,pre):
    values=[];picks=[(i,(c["n"]-2)//2) for i,c in enumerate(cases) if c["n"]>=2]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(picks),128):
            selected=picks[start:start+128];data=pack(cases,selected)
            p=predictions(model,data,dense=False);target=data[4][:,-1];delay=data[5][:,-1]
            prob=p.activity_logits.softmax(-1);loss=-prob.gather(1,target[:,None]).squeeze(1).clamp_min(1e-12).log()
            point=waiting_quantile(p,.5);lower=waiting_quantile(p,.05);upper=waiting_quantile(p,.95)
            risk=long_wait_probability(p,pre["long_wait_threshold_seconds"])
            for j,(index,t) in enumerate(selected):
                current=int(data[0][j,-1]);label=int(target[j]);seconds=float(delay[j])
                empirical_log=np.asarray(pre["transition_log_probability"][current])
                base=pre["time_prior"][current];base_p=WorkflowForecast(torch.tensor(empirical_log[None],dtype=torch.float32),
                    torch.tensor([base[0]]),torch.tensor([base[1]]),torch.tensor([base[2]]))
                empirical_time=float(waiting_quantile(base_p,.5)[0])
                empirical_risk=float(long_wait_probability(base_p,pre["long_wait_threshold_seconds"])[0])
                observed_risk=seconds>pre["long_wait_threshold_seconds"]
                values.append({"case_id":cases[index]["id"],"prefix_last_index":t,
                    "activity_NLL":float(loss[j]),"activity_correct":bool(prob[j].argmax()==target[j]),
                    "Markov_NLL":float(-empirical_log[label]),"Markov_correct":bool(empirical_log.argmax()==label),
                    "prevalence_NLL":float(-math.log(pre["global_next_activity_probability"][label])),
                    "repeat_activity_correct":current==label,"wait_MAE_seconds":float(abs(point[j]-delay[j])),
                    "Markov_wait_MAE_seconds":abs(empirical_time-seconds),"zero_wait_MAE_seconds":seconds,
                    "global_median_wait_MAE_seconds":abs(pre["global_wait_median_seconds"]-seconds),
                    "positive_wait_observed":seconds>0,"interval_90_covered":bool(lower[j]<=delay[j]<=upper[j]),
                    "interval_width_seconds":float(upper[j]-lower[j]),
                    "long_wait_Brier":float((risk[j]-int(observed_risk))**2),
                    "Markov_long_wait_Brier":(empirical_risk-int(observed_risk))**2})
    return values


def verify_git_owned(records):
    import hashlib
    for record in records:
        path=record["path"];expected=record["measured"]["checkpoint_sha256"]
        blob=subprocess.check_output(["git","show","HEAD:"+path])
        if hashlib.sha256(blob).hexdigest()!=expected or sha_file(path)!=expected:
            raise ValueError("original checkpoint must be committed and unmodified before scoring")


def score(cache):
    torch.set_num_threads(1)
    records=json.loads((BUNDLE/"CHECKPOINT_MANIFEST.json").read_text());verify_git_owned(records)
    plan=json.loads((BUNDLE/"FAIR_COMPARISON_FINAL.json").read_text());check_manifest(plan,"final")
    if len(records)!=10:raise ValueError("all ten checkpoints required before heldout scores")
    cohorts,pre,source=load(cache);out={}
    for part in ("DEV","TEST"):
        out[part]={}
        for record in records:
            checkpoint=torch.load(record["path"],weights_only=False,map_location="cpu")
            model=build(record["architecture"],pre);model.load_state_dict(checkpoint["model"],strict=True)
            out[part][f'{record["architecture"]}-s{record["seed"]}']=case_forecasts(model,cohorts[part],pre)
        print("ME069_SCORED_"+part,flush=True)
    write_json(BUNDLE/"REAL_EVALUATION.json",{"source":source,"scores":out,
        "scope":"authentic observational next-event and waiting time; not actual SLA breach,causal policy utility or paying-customer proof"})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--phase",choices=("register","train","score"),required=True);a=p.parse_args()
    {"register":register,"train":train,"score":score}[a.phase](a.cache)
