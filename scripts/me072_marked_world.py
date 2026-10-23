"""Original real-record multi-horizon native fits, score freeze and replay."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch

from scripts.me069_workflow_source import digest,write_json
from scripts.me071_persistent_workflow import source as inherited_source,schedules,verify_blob
from scripts.validate_research_parity import check_manifest
from tac_osm.workflow_me069 import EventBelief
from tac_osm.workflow_me071 import detach_belief
from tac_osm.workflow_me072 import (HORIZONS,MarkedWorldModel,record_log_probability,
                                  factor_scores,world_loss,time_quantiles)

ROOT=Path("research/mathematical-engineering/me072")
BUNDLE=Path("research/replay-bundles/ME072")
SEEDS=(72001,72002,72003,72004,72005)
EPOCHS,BATCH,CHUNK=6,32,32
AGES=(2,6,16,32,64)


from tac_osm.parameter_budget import budgeted_optimizer, checked_optimizer_step, save_training_checkpoint

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_preprocessing(cases,pre):
    pre=copy.deepcopy(pre);n=len(pre["activities"])+1
    counts=np.full((n,len(HORIZONS),n),.5);total=np.zeros(n);present=np.zeros((n,len(HORIZONS)))
    gaps=[[] for _ in range(n)]
    for c in cases:
        for t,current in enumerate(c["activity"]):
            current=int(current);total[current]+=1
            for j,h in enumerate(HORIZONS):
                if t+h<c["n"]:
                    present[current,j]+=1;counts[current,j,int(c["activity"][t+h])]+=1
            if t+1<c["n"]:gaps[current].append(float(c["gap"][t+1]))
    counts[0]=counts[1:].sum(0);present[0]=present[1:].sum(0);total[0]=total[1:].sum()
    marginal=(present+.5)/(total[:,None]+1)
    ratios=marginal/np.concatenate((np.ones((n,1)),marginal[:,:-1]),1)
    ratios=np.clip(ratios,1e-4,1-1e-4)
    all_gaps=[g for row in gaps for g in row];clock=[]
    for row in gaps:
        row=row or all_gaps;positive=np.log([g for g in row if g>0])
        if not len(positive):positive=np.log([g for g in all_gaps if g>0])
        sigma=float(np.clip(np.std(positive),.15,1.8));p=(sigma-.1)/1.9
        zero=(sum(g==0 for g in row)+.5)/(len(row)+1)
        clock.append([float(np.median(positive)),math.log(p/(1-p)),math.log(zero/(1-zero))])
    pre.update(future_mark_log_prior=np.log(counts/counts.sum(-1,keepdims=True)).tolist(),
        survival_increment_logit=np.log(ratios/(1-ratios)).tolist(),coherent_clock_prior=clock)
    return pre


def source(cache,frozen=True):
    cohorts,pre,receipt=inherited_source(cache)
    if frozen:
        plan=json.loads((ROOT/"FAIR_COMPARISON.json").read_text())
        original=json.loads((ROOT/"FIT_PREPROCESSING.json").read_text())
        if digest(original)!=plan["preprocessing_sha256"]:raise ValueError("changed frozen ME072 preprocessing")
        if original["activities"]!=pre["activities"] or original["resources"]!=pre["resources"]:raise ValueError("changed vocabulary")
        if receipt["source_sha256"]!=plan["real_world_source"]["sha256"]:raise ValueError("changed real source")
        pre=original
    else:pre=derive_preprocessing(cohorts["FIT"],pre)
    return cohorts,pre,receipt


def pack(cases,indices,start=0,length=CHUNK):
    b=len(indices);n=cases[0]["history_log_prior"].shape[-1];h=len(HORIZONS)
    activity=np.zeros((b,length),np.int64);resources=np.zeros_like(activity)
    features=np.zeros((b,length,8),np.float32);hours=np.zeros((b,length,1),np.float32)
    history=np.zeros((b,length,n),np.float32);target=np.zeros((b,length,h),np.int64)
    seconds=np.zeros((b,length,h),np.float32);present=np.zeros((b,length,h),bool);valid=np.zeros((b,length),bool)
    for row,index in enumerate(indices):
        c=cases[index];end=min(start+length,c["n"]);size=max(0,end-start)
        if not size:continue
        sl=slice(start,end)
        activity[row,:size]=c["activity"][sl];resources[row,:size]=c["resource"][sl]
        features[row,:size]=c["features"][sl];hours[row,:size,0]=c["gap"][sl]/3600.
        history[row,:size]=c["history_log_prior"][sl];valid[row,:size]=True
        for j,horizon in enumerate(HORIZONS):
            future=np.arange(start,end)+horizon;exists=future<c["n"]
            safe=np.minimum(future,c["n"]-1)
            target[row,:size,j]=np.where(exists,c["activity"][safe],0)
            seconds[row,:size,j]=np.where(exists,c["timestamps"][safe]-c["timestamps"][sl],0)
            present[row,:size,j]=exists
    return tuple(torch.from_numpy(x) for x in (activity,resources,features,hours,history,target,seconds,present,valid))


def forward(model,data,belief=None,all_steps=True):
    encoded=model.event_encoding(*data[:3]);belief=model.initial(len(encoded)) if belief is None else belief
    states=[]
    for t in range(encoded.shape[1]):
        belief=model.observe(belief,encoded[:,t],data[3][:,t],data[8][:,t])
        if all_steps:states.append(belief)
    if all_steps:
        merged=EventBelief(*(torch.stack([s[i] for s in states],1).flatten(0,1) for i in range(3)))
        p=model.forecast(merged,encoded.flatten(0,1),data[0].flatten(),data[4].flatten(0,1))
    else:p=model.forecast(belief,encoded[:,-1],data[0][:,-1],data[4][:,-1])
    return p,belief


def register(cache):
    torch.set_num_threads(1);cohorts,pre,receipt=source(cache,False)
    model=MarkedWorldModel(pre);count=sum(p.numel() for p in model.parameters())
    prefixes=sum(c["n"] for c in cohorts["FIT"])*EPOCHS
    plan={"schema_version":"tac-osm.parity.v1","study_id":"ME072_COHERENT_MULTIHORIZON_MARKED_NATIVE_WORLD_MODEL",
        "study_type":"capability_development","registered_before_first_fit":True,"architecture_count":1,
        "no_architecture_superiority_claim":True,"scientific_purpose":"Train one integrated native model for observed future activity, source-record boundary mass and coherent zero/positive waiting times at 1/2/4/8 events, carrying native state across full original cases. No attribution/comparator matrix or causal policy claim.",
        "real_world_source":{"name":"publisher verified BPI Challenge2012 actual financial institution event log","sha256":receipt["source_sha256"],
            "license":"4TU General Terms of Use; raw original traces external; exact publisher file hash verified","observed_real_outcomes":True},
        "split_sha256":{k:digest([c["id"] for c in cohorts[k]]) for k in ("FIT","TEMP","CQR","REPORT")},
        "preprocessing_sha256":digest(pre),"trainable_parameters":count,"seeds":list(SEEDS),
        "training_budget":{"epochs":EPOCHS,"batch_size":BATCH,"backprop_events":CHUNK,"expected_targets_per_seed":prefixes*len(HORIZONS),
            "observed_prefix_exposures":prefixes,"optimizer":"AdamW lr .001 wd .0001 clip1","selection":"all final models; no early stopping, calibration or outcome-based search"},
        "schedule_sha256":{str(s):digest(schedules(cohorts["FIT"],s)) for s in SEEDS},
        "repository_storage_plan":"Five original model/optimizer/RNG checkpoints under research/replay-bundles/ME072/checkpoints, approx12MB; per-query original distribution arrays, raw scores, source/config/environment/SHA manifest and independent GitHub clean-checkout restore and full factual scoring replay. Freeze all originals before REPORT scoring.",
        "horizons":list(HORIZONS),"query_ages":list(AGES),
        "development_support_before_fit":{str(a):sum(c["n"]>=a for c in cohorts["REPORT"]) for a in AGES},
        "roles":"inherit FIT and already exposed DEV partition; TEMP/CQR roles unused, no posthoc fitting; REPORT last925 DEV cases. Original TEST unscored; source file ingestion disclosed.",
        "target_contract":"For each reached prefix and futureh: actual activity and elapsed time if original record exists; otherwise explicit end-of-record mass, no timing target. Source boundary/censoring is NOT independently verified workflow completion, timeout, interruption or success.",
        "numerical_contract":{"primary_atol":1e-5,"primary_rtol":1e-5,"metric_atol":1e-4,"metric_rtol":1e-5},
        "interpretation":"Report all reached-prefix cases, present/boundary counts, conditional activity accuracy and time scores separately. Fewer than25 real cases at a query is sparse development only; no memory superiority or >128-event claim.",
        "prior_art":["Shchur et al. Intensity-Free Learning of Temporal Point Processes,ICLR2020,https://arxiv.org/abs/1909.12127",
            "Tax et al. Predictive Business Process Monitoring with LSTM Neural Networks,CAiSE2017,https://arxiv.org/abs/1612.02130"],
        "mathematical_contract":"Joint mass=P(EOR) or P(record)*P(activity|record)*[zero atom + positive lognormal mixture]; survival decreases, zero mass decreases, component locations increase and scales/weights shared over horizons, so conditional waiting quantiles cannot reverse. Observational futures, not intervention effects."}
    write_json(ROOT/"FAIR_COMPARISON.json",plan);write_json(ROOT/"FIT_PREPROCESSING.json",pre)
    write_json(ROOT/"SOURCE.json",receipt)
    print(check_manifest(plan),flush=True);print(json.dumps({"parameters":count,"prefix_exposures":prefixes,"horizon_targets":prefixes*4,"REPORT_support":plan["development_support_before_fit"]}),flush=True)


def train(cache):
    torch.set_num_threads(1);plan=json.loads((ROOT/"FAIR_COMPARISON.json").read_text());check_manifest(plan)
    verify_blob(ROOT/"FAIR_COMPARISON.json")
    cohorts,pre,receipt=source(cache);folder=BUNDLE/"checkpoints";folder.mkdir(parents=True,exist_ok=True)
    if list(folder.glob('*.pt')):raise ValueError("never overwrite fitted originals")
    records=[]
    for seed in SEEDS:
        torch.manual_seed(seed);model=MarkedWorldModel(pre);opt=budgeted_optimizer(model, torch.optim.AdamW, model.parameters(),lr=.001,weight_decay=.0001)
        order=schedules(cohorts["FIT"],seed)
        if digest(order)!=plan["schedule_sha256"][str(seed)]:raise ValueError("original schedule changed")
        steps=prefixes=0;trace=[];begin=time.process_time()
        for epoch,batches in enumerate(order,1):
            losses=[]
            for indices in batches:
                belief=model.initial(len(indices));longest=max(cohorts["FIT"][i]["n"] for i in indices)
                for start in range(0,longest,CHUNK):
                    data=pack(cohorts["FIT"],indices,start);opt.zero_grad(set_to_none=True)
                    prediction,belief=forward(model,data)
                    loss=world_loss(prediction,data[5].flatten(0,1),data[6].flatten(0,1),data[7].flatten(0,1),data[8].flatten())
                    if not torch.isfinite(loss):raise ValueError("nonfinite genuine multi-horizon loss")
                    loss.backward()
                    if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.)):raise ValueError("nonfinite gradient")
                    checked_optimizer_step(opt);belief=detach_belief(belief);prefixes+=int(data[8].sum());steps+=1;losses.append(float(loss.detach()))
            trace.append({"epoch":epoch,"loss":float(np.mean(losses)),"prefix_exposures":prefixes,"horizon_targets":prefixes*4,"updates":steps})
            print(json.dumps({"seed":seed,**trace[-1]}),flush=True)
            path=folder/f"native-s{seed}.pt"
            save_training_checkpoint({"model":model.state_dict(),"optimizer":opt.state_dict(),"torch_rng_state":torch.get_rng_state(),
                "preprocessing":pre,"seed":seed,"epoch":epoch,"trace":trace,"schedule_sha256":digest(order),"source_sha256":receipt["source_sha256"]},path)
        records.append({"seed":seed,"path":str(path),"checkpoint_sha256":sha(path),"epochs":EPOCHS,
            "actual_targets":prefixes*4,"actual_prefix_exposures":prefixes,"actual_updates":steps,"optimizer_cpu_seconds":time.process_time()-begin})
        write_json(BUNDLE/"CHECKPOINTS.json",records)
    plan["completed_model_records"]=records
    print(check_manifest(plan,"final"),flush=True);write_json(BUNDLE/"FINAL_REGISTRATION.json",plan)
    write_json(BUNDLE/"ENVIRONMENT.json",{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"platform":platform.platform(),"CPU_threads":1})


@torch.no_grad()
def metrics(p,data):
    target,seconds,present=data[5][:,-1],data[6][:,-1],data[7][:,-1]
    logs=record_log_probability(p).double();prob=logs.exp();classes=p.mark_log_probability.shape[-1]
    labels=torch.where(present,target,torch.full_like(target,classes))
    existence,mark,occurrence,time=factor_scores(p,target,seconds,present)
    q=time_quantiles(p);rows={}
    for j,h in enumerate(HORIZONS):
        yes=present[:,j];t=seconds[:,j].double();low,median,high=q[:,j].unbind(-1)
        n=int(yes.sum());width=high-low
        raw={"reached_query_cases":len(t),"future_record_cases":n,"record_boundary_cases":int((~yes).sum()),
            "activity_or_boundary_NLL":float(-logs[:,j].gather(-1,labels[:,j,None]).mean()),
            "activity_or_boundary_accuracy":float((logs[:,j].argmax(-1)==labels[:,j]).double().mean()),
            "joint_multiclass_Brier":float((prob[:,j].square().sum(-1)-2*prob[:,j].gather(-1,labels[:,j,None]).squeeze(-1)+1).mean()),
            "record_boundary_Brier":float(((1-p.log_survival[:,j].double().exp())-(~yes).double()).square().mean()),
            "conditional_activity_accuracy":float((p.mark_log_probability[:,j].argmax(-1)[yes]==target[:,j][yes]).double().mean()) if n else None,
            "conditional_activity_NLL":float(mark[:,j][yes].mean()) if n else None,
            "conditional_time_NLL_seconds":float((occurrence[:,j]+(seconds[:,j]>0)*time[:,j])[yes].mean()) if n else None,
            "conditional_zero_delay_Brier":float((p.log_zero[:,j].double().exp()-(t==0).double()).square()[yes].mean()) if n else None,
            "waiting_median_MAE_hours":float((median[yes]-t[yes]).abs().mean()/3600) if n else None,
            "conditional_time_90_coverage":float(((low<=t)&(t<=high))[yes].double().mean()) if n else None,
            "mean_interval_width_hours":float(width[yes].mean()/3600) if n else None,
            "mean_interval_score_hours":float((width+20*(low-t).clamp_min(0)+20*(t-high).clamp_min(0))[yes].mean()/3600) if n else None,
            "width_p50_p90_p99_hours":(torch.quantile(width[yes],torch.tensor([.5,.9,.99],dtype=torch.float64))/3600).tolist() if n else None}
        rows[str(h)]=raw
    return rows,q


def score(cache):
    torch.set_num_threads(1);cohorts,pre,_=source(cache);records=json.loads((BUNDLE/"CHECKPOINTS.json").read_text())
    if len(records)!=5:raise ValueError("all original seeds required")
    out={};arrays={}
    for row in records:
        verify_blob(row["path"],row["checkpoint_sha256"])
        saved=torch.load(row["path"],map_location="cpu",weights_only=False)
        if saved["preprocessing"]!=pre:raise ValueError("exact frozen preprocessing differs")
        model=MarkedWorldModel(pre);model.load_state_dict(saved["model"],strict=True);model.eval()
        opt=budgeted_optimizer(model, torch.optim.AdamW, model.parameters());opt.load_state_dict(saved["optimizer"])
        out[str(row["seed"]) ]={}
        with torch.no_grad():
            for age in AGES:
                cases=[c for c in cohorts["REPORT"] if c["n"]>=age]
                if not cases:
                    out[str(row["seed"])][str(age)]={"cases":0,"IDs_sha256":digest([]),"support":"unavailable: no recorded cases reach query", "horizons":{}}
                    continue
                data=pack(cases,list(range(len(cases))),length=age)
                p,_=forward(model,data,all_steps=False);rows,q=metrics(p,data)
                out[str(row["seed"])][str(age)]={"cases":len(cases),"IDs_sha256":digest([c["id"] for c in cases]),
                    "support":"sparse development, <25 cases" if len(cases)<25 else "development only", "horizons":rows}
                # Full marked log-mass, atom and mixture parameters, log1p quantiles.
                b,h=p.log_survival.shape
                array=torch.cat((record_log_probability(p).double(),p.log_zero.double()[...,None],
                    p.mixture_log_weight.double()[:,None,:].expand(-1,h,-1),p.log_time_location.double(),
                    p.log_time_scale.double()[:,None,:].expand(-1,h,-1),q.log1p()),-1).numpy()
                arrays[f's{row["seed"]}_age{age}']=array
    np.savez_compressed(BUNDLE/"ORIGINAL_FORECASTS.npz",**arrays)
    write_json(BUNDLE/"ORIGINAL_SCORES.json",{"scope":"exposed DEV record forecasting; EOR is source boundary, not verified completion or censor-free business outcome; conditional timing only when actual future record exists", "seeds":out})


def compare_scores(a,b,tol):
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError("metric identities differ")
        for k in a:compare_scores(a[k],b[k],tol)
    elif isinstance(a,(float,list)):
        np.testing.assert_allclose(a,b,atol=tol["metric_atol"],rtol=tol["metric_rtol"])
    elif a!=b:raise ValueError("score identity changed")


def replay(cache):
    paths=[BUNDLE/"ORIGINAL_FORECASTS.npz",BUNDLE/"ORIGINAL_SCORES.json"]
    for path in paths:verify_blob(path)
    original={p:p.read_bytes() for p in paths}
    with np.load(paths[0]) as z:arrays={k:v.copy() for k,v in z.items()}
    audit=ROOT/"REPLAY_AUDIT.json";audit.unlink(missing_ok=True)
    try:
        score(cache);tol=json.loads((ROOT/"FAIR_COMPARISON.json").read_text())["numerical_contract"]
        with np.load(paths[0]) as z:
            if set(z)!=set(arrays):raise ValueError("replay forecast keys changed")
            for k,v in z.items():np.testing.assert_allclose(v,arrays[k],atol=tol["primary_atol"],rtol=tol["primary_rtol"])
        compare_scores(json.loads(original[paths[1]]),json.loads(paths[1].read_text()),tol)
    except Exception as error:
        for p in paths:(p.parent/("FAILED_REPLAY_"+p.name)).write_bytes(p.read_bytes())
        write_json(audit,{"status":"FAIL","error":str(error),"original_files_restored":True});print(audit.read_text(),flush=True);raise
    finally:
        for p,raw in original.items():p.write_bytes(raw)
    write_json(audit,{"status":"PASS","original_models_restored":5,"original_distribution_arrays_recomputed":True,
        "all_real_REPORT_metrics_recomputed":True,"TEST_scored":False,"parameters_refit":False})
    print(audit.read_text(),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--phase",choices=("register","train","score","replay"),required=True);a=parser.parse_args()
    {"register":register,"train":train,"score":score,"replay":replay}[a.phase](a.cache)
