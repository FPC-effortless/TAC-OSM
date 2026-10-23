"""Single native architecture construction: real full-case TBPTT and calibration."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from scripts.me069_workflow_source import load,digest,write_json
from scripts.me070_ngram_real_workflow import STATE,predict
from scripts.validate_research_parity import check_manifest
from tac_osm.workflow_me069 import EventBelief
from tac_osm.workflow_me071 import (PersistentWorkflowModel,QuantileForecast,detach_belief,
    predictive_loss,fit_temperature,cqr_correction,calibrated_seconds)

ROOT=Path("research/mathematical-engineering/me071")
BUNDLE=Path("research/replay-bundles/ME071")
SEEDS=(71001,71002,71003,71004,71005)
EPOCHS,BATCH,CHUNK=6,32,32
PREFIX=5  # A fixed sixth observed event, independent of final trace length.


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_preprocessing(original,reconstructed,expected_digest):
    if digest(original)!=expected_digest:raise ValueError("frozen preprocessing checksum changed")
    for field in ("activities","resources"):
        if original[field]!=reconstructed[field]:raise ValueError("source vocabulary changed")
    return original  # Never replace recorded floats with regenerated statistics.


def source(cache,use_frozen=True):
    cohorts,pre,receipt=load(cache)
    state=json.loads(STATE.read_text())
    if state["real_source_sha256"]!=receipt["source_sha256"]:raise ValueError("history prior source differs")
    n=len(pre["activities"])+1
    delays=[[] for _ in range(n)]
    for c in cohorts["FIT"]:
        for a,d in zip(c["activity"][:-1],c["gap"][1:]):delays[int(a)].append(float(d))
    all_delays=[d for rows in delays for d in rows]
    if use_frozen:
        plan=json.loads((ROOT/"FAIR_COMPARISON.json").read_text())
        if (receipt["source_sha256"]!=plan["real_world_source"]["sha256"] or
                receipt["cohorts"]["FIT"]["ids_sha256"]!=plan["split_sha256"]["FIT"]):
            raise ValueError("original source/FIT identity changed")
        pre=frozen_preprocessing(json.loads((ROOT/"FIT_PREPROCESSING.json").read_text()),pre,plan["preprocessing_sha256"])
    else:
        pre["log_wait_quantile_prior"]=[np.quantile(np.log1p(rows or all_delays),[.05,.5,.95]).tolist() for rows in delays]
    cohorts.pop("TEST")
    for rows in cohorts.values():
        for c in rows:
            c["history_log_prior"]=np.asarray([np.log(np.clip(predict(state,c["activity"][:t+1].tolist(),5),1e-12,1))
                for t in range(c["n"])],dtype=np.float32)
    dev=cohorts.pop("DEV");a,b=len(dev)//4,len(dev)//2
    cohorts.update(TEMP=dev[:a],CQR=dev[a:b],REPORT=dev[b:])
    # No original TEST inference, calibration or distribution inspection.
    return cohorts,pre,receipt


def schedules(cases,seed):
    rng=np.random.default_rng(seed);out=[]
    for _ in range(EPOCHS):
        shuffled=rng.permutation(len(cases)).tolist();batches=[]
        for start in range(0,len(shuffled),256):
            pool=sorted(shuffled[start:start+256],key=lambda i:cases[i]["n"])
            batches.extend([pool[j:j+BATCH] for j in range(0,len(pool),BATCH)])
        rng.shuffle(batches);out.append(batches)
    return out


def pack(cases,indices,start=0,length=CHUNK):
    n=len(indices);classes=cases[0]["history_log_prior"].shape[-1]
    activity=np.zeros((n,length),np.int64);resource=np.zeros_like(activity)
    feature=np.zeros((n,length,8),np.float32);hours=np.zeros((n,length,1),np.float32)
    prior=np.zeros((n,length,classes),np.float32);targets=np.zeros_like(activity)
    wait=np.zeros((n,length),np.float32);valid=np.zeros((n,length),bool)
    for j,i in enumerate(indices):
        c=cases[i];end=min(start+length,c["n"]-1);size=max(0,end-start)
        if not size:continue
        activity[j,:size]=c["activity"][start:end];resource[j,:size]=c["resource"][start:end]
        feature[j,:size]=c["features"][start:end];hours[j,:size,0]=c["gap"][start:end]/3600.
        prior[j,:size]=c["history_log_prior"][start:end];targets[j,:size]=c["activity"][start+1:end+1]
        wait[j,:size]=c["gap"][start+1:end+1];valid[j,:size]=True
    return tuple(torch.from_numpy(x) for x in (activity,resource,feature,hours,prior,targets,wait,valid))


def forward_chunk(model,data,belief=None):
    activity,resource,features,hours,prior,_,_,valid=data
    encoded=model.event_encoding(activity,resource,features)
    belief=model.initial(len(activity)) if belief is None else belief
    states=[]
    for t in range(activity.shape[1]):
        belief=model.observe(belief,encoded[:,t],hours[:,t],valid[:,t]);states.append(belief)
    stacked=EventBelief(*(torch.stack([s[i] for s in states],1).flatten(0,1) for i in range(3)))
    forecast=model.forecast(stacked,encoded.flatten(0,1),activity.flatten(),prior.flatten(0,1))
    return forecast,belief


def registration(cache):
    torch.set_num_threads(1);cohorts,pre,receipt=source(cache,use_frozen=False)
    model=PersistentWorkflowModel(pre)
    targets=sum(c["n"]-1 for c in cohorts["FIT"])*EPOCHS
    manifest={"schema_version":"tac-osm.parity.v1","study_id":"ME071_PERSISTENT_CALIBRATED_NATIVE_EVENT_MODEL",
        "study_type":"capability_development","architecture_count":1,"registered_before_first_fit":True,
        "no_architecture_superiority_claim":True,"scientific_purpose":"Owner-requested construction and real-FIT training of one integrated native TAC event model: persistent chunk state, label smoothing, history log pooling, ordered log-time quantiles, disjoint DEV temperature and conformal calibration. Not an attribution or cross-architecture benchmark.",
        "real_world_source":{"name":"original BPI Challenge2012 loan event log","sha256":receipt["source_sha256"],
            "license":"4TU General Terms of Use; raw traces external, publisher-hash verified","observed_real_outcomes":True},
        "split_sha256":{k:digest([c["id"] for c in cohorts[k]]) for k in ("FIT","TEMP","CQR","REPORT")},
        "cohort_counts":{k:len(v) for k,v in cohorts.items()},"preprocessing_sha256":digest(pre),
        "history_prior_path":str(STATE),"history_prior_sha256":sha(STATE),"history_prior_bytes":STATE.stat().st_size,
        "trainable_parameters":sum(p.numel() for p in model.parameters()),"seeds":list(SEEDS),
        "training_budget":{"epochs":EPOCHS,"batch_size":BATCH,"backprop_events":CHUNK,"expected_targets_per_seed":targets,
            "label_smoothing":.05,"optimizer":"AdamW lr .001 weight_decay .0001 clip1","architecture_search_trials":0},
        "schedule_sha256":{str(s):digest(schedules(cohorts["FIT"],s)) for s in SEEDS},
        "repository_storage_plan":"Five original model/optimizer/Torch/NumPy RNG checkpoints at research/replay-bundles/ME071/checkpoints, approx8MB total; calibration JSON, original forecast NPZ and score JSON plus raw source legal acquisition, independent clean GitHub checkout full inference replay. Freeze originals before DEV calibration; freeze calibration before REPORT scoring.",
        "evaluation_contract":"DEV-only development readout: one target after exactly6 observed events per eligible case, separate TEMP first quarter, CQR second quarter, REPORT last half. Original TEST remains unscored. No early stopping or weight selection. Log loss, multiclass Brier, ECE diagnostic, quantile scores, MAE, interval score, coverage, width distribution; no superiority or fresh qualification.",
        "numerical_contract":{"forecast_absolute":1e-5,"forecast_relative":1e-5,"mean_metric_absolute":1e-4,"mean_metric_relative":1e-5,
            "units":"Store primary outputs as log probabilities/log1p waiting quantiles. Seconds derived in float64; compare primary outputs directly before seconds metrics. No bitwise portability promise."},
        "prior_art":["Guo et al. On Calibration of Modern Neural Networks,ICML2017,https://proceedings.mlr.press/v70/guo17a.html",
            "Romano et al. Conformalized Quantile Regression,NeurIPS2019,https://arxiv.org/abs/1905.03222",
            "Dai et al. Transformer-XL,ACL2019,https://aclanthology.org/P19-1285/"],
        "limits":"Previously studied institution and exposed DEV, not independent confirmation. CQR guarantee requires exchangeable cases, not guaranteed under chronological shift. No universal shrinkage, overconfidence diagnosis, GRU failure, >128 history or intelligence/autonomy claim."}
    write_json(ROOT/"FAIR_COMPARISON.json",manifest);write_json(ROOT/"FIT_PREPROCESSING.json",pre)
    write_json(ROOT/"SOURCE.json",receipt);print(check_manifest(manifest,"preflight"),flush=True)
    print(json.dumps({"parameters":manifest["trainable_parameters"],"targets_per_seed":targets,"cohorts":manifest["cohort_counts"]}),flush=True)


def verify_blob(path,expected=None):
    original=subprocess.check_output(["git","show","HEAD:"+str(path)])
    actual=hashlib.sha256(original).hexdigest()
    if actual!=sha(path) or (expected and actual!=expected):raise ValueError("uncommitted or changed original: "+str(path))


def train(cache):
    torch.set_num_threads(1)
    plan=json.loads((ROOT/"FAIR_COMPARISON.json").read_text());check_manifest(plan,"preflight")
    cohorts,pre,source_receipt=source(cache)
    if digest(pre)!=plan["preprocessing_sha256"]:raise ValueError("FIT preprocessing differs")
    verify_blob(ROOT/"FAIR_COMPARISON.json")
    folder=BUNDLE/"checkpoints";folder.mkdir(parents=True,exist_ok=True)
    if list(folder.glob("*.pt")):raise ValueError("never overwrite original fitted checkpoints")
    records=[]
    for seed in SEEDS:
        torch.manual_seed(seed);model=PersistentWorkflowModel(pre)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        schedule=schedules(cohorts["FIT"],seed)
        if digest(schedule)!=plan["schedule_sha256"][str(seed)]:raise ValueError("FIT order changed")
        steps=targets=0;trace=[];began=time.process_time()
        for epoch,batches in enumerate(schedule,1):
            losses=[]
            for indices in batches:
                belief=model.initial(len(indices));max_targets=max(cohorts["FIT"][i]["n"]-1 for i in indices)
                for start in range(0,max_targets,CHUNK):
                    data=pack(cohorts["FIT"],indices,start)
                    opt.zero_grad(set_to_none=True);p,belief=forward_chunk(model,data,belief)
                    loss=predictive_loss(p,data[5].flatten(),data[6].flatten(),data[7].flatten())
                    if not torch.isfinite(loss):raise ValueError("nonfinite original training loss")
                    loss.backward()
                    if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.)):raise ValueError("nonfinite gradient")
                    opt.step();belief=detach_belief(belief)
                    steps+=1;targets+=int(data[7].sum());losses.append(float(loss.detach()))
            trace.append({"epoch":epoch,"mean_batch_loss":float(np.mean(losses)),"updates":steps,"actual_targets":targets})
            print(json.dumps({"seed":seed,**trace[-1]}),flush=True)
            # Also preserve the last recoverable state if a subsequent epoch fails.
            path=folder/f"native-s{seed}.pt"
            torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"torch_rng_state":torch.get_rng_state(),
                "numpy_rng_state":np.random.get_state(),"preprocessing":pre,"seed":seed,"epoch":epoch,
                "schedule_sha256":digest(schedule),"trace":trace,"source_sha256":source_receipt["source_sha256"]},path)
        record={"seed":seed,"path":str(path),"checkpoint_sha256":sha(path),"actual_targets":targets,
            "epochs":EPOCHS,"actual_updates":steps,"optimizer_cpu_seconds":time.process_time()-began}
        records.append(record);write_json(BUNDLE/"CHECKPOINTS.json",records)
    plan["completed_model_records"]=records
    print(check_manifest(plan,"final"),flush=True);write_json(BUNDLE/"FINAL_REGISTRATION.json",plan)
    write_json(BUNDLE/"ENVIRONMENT.json",{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"platform":platform.platform()})


def original_models(pre):
    rows=json.loads((BUNDLE/"CHECKPOINTS.json").read_text())
    for r in rows:
        verify_blob(r["path"],r["checkpoint_sha256"])
        saved=torch.load(r["path"],weights_only=False,map_location="cpu")
        if saved["preprocessing"]!=pre:raise ValueError("original preprocessing changed")
        model=PersistentWorkflowModel(pre);model.load_state_dict(saved["model"],strict=True);model.eval()
        opt=torch.optim.AdamW(model.parameters());opt.load_state_dict(saved["optimizer"])
        yield r,model


@torch.no_grad()
def forecasts(model,cases):
    # The query prefix is known online; no dependency on eventual case length.
    selected=[c for c in cases if c["n"]>PREFIX+1]
    blocks=[];labels=[];wait=[]
    for start in range(0,len(selected),128):
        batch=selected[start:start+128]
        data=pack(batch,list(range(len(batch))),length=PREFIX+1)
        p,_=forward_chunk(model,data)
        count=PREFIX+1
        blocks.append(np.concatenate((p.activity_logits.reshape(len(batch),count,-1)[:,-1].numpy(),
            p.positive_wait_logit.reshape(len(batch),count)[:,-1,None].numpy(),
            p.log_wait_quantiles.reshape(len(batch),count,3)[:,-1].numpy()),axis=1))
        labels.extend(data[5][:,-1].tolist());wait.extend(data[6][:,-1].tolist())
    return np.concatenate(blocks),np.asarray(labels),np.asarray(wait),[c["id"] for c in selected]


def calibrate(cache):
    torch.set_num_threads(1);cohorts,pre,_=source(cache);records={}
    for row,model in original_models(pre):
        a,label,_,ids=forecasts(model,cohorts["TEMP"])
        t=fit_temperature(torch.tensor(a[:,:-4]),torch.tensor(label))
        q,_,wait,cids=forecasts(model,cohorts["CQR"])
        correction=cqr_correction(q[:,-3:],wait)
        records[str(row["seed"]) ]={"temperature":t,"log_time_cqr_correction":correction,
            "TEMP_ids_sha256":digest(ids),"CQR_ids_sha256":digest(cids),"TEMP_cases":len(ids),"CQR_cases":len(cids),
            "checkpoint_sha256":row["checkpoint_sha256"]}
    write_json(BUNDLE/"CALIBRATION.json",records)


def metrics(array,labels,wait,temperature=1.,correction=0.):
    logits=torch.tensor(array[:,:-4],dtype=torch.float64)/temperature
    p=logits.softmax(-1).numpy();labels=np.asarray(labels,dtype=np.int64)
    confidence=p.max(1);correct=p.argmax(1)==labels
    ece=sum(np.mean(hit)*abs(np.mean(confidence[hit])-np.mean(correct[hit])) for i in range(10)
        if np.any(hit:=((confidence>=i/10)&(confidence<(i+1)/10 if i<9 else confidence<=1))))
    q=array[:,-3:].astype(np.float64);y=np.log1p(wait)
    positive_probability=torch.tensor(array[:,-4],dtype=torch.float64).sigmoid().numpy()
    scored_q=q.copy()
    scored_q[:,0]=np.maximum(0,scored_q[:,0]-correction)
    scored_q[:,2]+=correction
    error=y[:,None]-scored_q;levels=np.asarray([.05,.5,.95])
    pinball=np.maximum(levels*error,(levels-1)*error).mean(0)
    seconds=calibrated_seconds(torch.tensor(q),correction).numpy();low,median,high=seconds.T
    width=high-low;covered=(wait>=low)&(wait<=high)
    interval=width+20*np.maximum(low-wait,0)+20*np.maximum(wait-high,0)
    loglow=np.maximum(0,q[:,0]-correction);loghigh=q[:,2]+correction
    loginterval=loghigh-loglow+20*np.maximum(loglow-y,0)+20*np.maximum(y-loghigh,0)
    return {"cases":len(wait),"activity_NLL":float(-np.log(p[np.arange(len(labels)),labels]).mean()),
        "activity_accuracy":float(correct.mean()),"multiclass_Brier":float(np.mean(np.sum(p*p,1)-2*p[np.arange(len(labels)),labels]+1)),
        "ECE_10_equal_width_bins":float(ece),"waiting_MAE_hours":float(np.mean(abs(median-wait))/3600),
        "positive_wait_Brier":float(np.mean((positive_probability-(wait>0))**2)),
        "positive_gap_MAE_hours":float(np.mean(abs(median[wait>0]-wait[wait>0]))/3600),
        "log1p_pinball_q05_q50_q95":pinball.tolist(),"log1p_interval_score":float(loginterval.mean()),
        "coverage_90":float(covered.mean()),"mean_width_hours":float(width.mean()/3600),
        "width_hours_p50_p90_p99_max":(np.quantile(width,[.5,.9,.99,1])/3600).tolist(),
        "mean_interval_score_hours":float(interval.mean()/3600)}


def score(cache):
    torch.set_num_threads(1);cohorts,pre,_=source(cache)
    verify_blob(BUNDLE/"CALIBRATION.json")
    calibration=json.loads((BUNDLE/"CALIBRATION.json").read_text());out={};arrays={}
    for row,model in original_models(pre):
        a,label,wait,ids=forecasts(model,cohorts["REPORT"]);c=calibration[str(row["seed"])]
        key=str(row["seed"]);arrays[key]=a
        out[key]={"raw":metrics(a,label,wait),"calibrated":metrics(a,label,wait,c["temperature"],c["log_time_cqr_correction"]),
            "ids_sha256":digest(ids),"checkpoint_sha256":row["checkpoint_sha256"]}
    # Original primary outputs, not merely aggregates. Public anonymized case
    # keys and target values remain external; this array contains predictions.
    np.savez_compressed(BUNDLE/"ORIGINAL_FORECASTS.npz",**arrays)
    write_json(BUNDLE/"ORIGINAL_SCORES.json",{"scope":"previously exposed BPI2012 DEV development, no architecture comparison or new TEST qualification", "seeds":out})


def replay(cache):
    paths=[BUNDLE/"ORIGINAL_SCORES.json",BUNDLE/"ORIGINAL_FORECASTS.npz"]
    for path in paths:verify_blob(path)
    original_bytes={p:p.read_bytes() for p in paths}
    original=json.loads(original_bytes[paths[0]])
    with np.load(paths[1]) as saved:arrays={k:v.copy() for k,v in saved.items()}
    audit=ROOT/"REPLAY_AUDIT.json";audit.unlink(missing_ok=True)
    try:
        score(cache)
        restored=json.loads(paths[0].read_text())
        plan=json.loads((ROOT/"FAIR_COMPARISON.json").read_text());tol=plan["numerical_contract"]
        with np.load(paths[1]) as current:
            if set(current)!=set(arrays):raise ValueError("replay seed arrays differ")
            for k,v in current.items():
                np.testing.assert_allclose(v,arrays[k],atol=tol["forecast_absolute"],rtol=tol["forecast_relative"])
        for seed,rows in original["seeds"].items():
            for field in ("checkpoint_sha256","ids_sha256"):
                if rows[field]!=restored["seeds"][seed][field]:raise ValueError("wrong model/case replay")
            for variant in ("raw","calibrated"):
                for field,value in rows[variant].items():
                    np.testing.assert_allclose(restored["seeds"][seed][variant][field],value,
                        atol=tol["mean_metric_absolute"],rtol=tol["mean_metric_relative"])
    except Exception as error:
        for p in paths:(p.parent/("FAILED_REPLAY_"+p.name)).write_bytes(p.read_bytes())
        write_json(audit,{"status":"FAIL","error":str(error),"original_files_restored":True})
        print(audit.read_text(),flush=True)
        raise
    finally:
        for p,value in original_bytes.items():p.write_bytes(value)
    write_json(audit,{"status":"PASS","original_models_restored":5,"original_forecast_arrays_recomputed":True,
        "real_REPORT_scores_recomputed":True,"parameters_refit":False,"TEST_scored":False})
    print((ROOT/"REPLAY_AUDIT.json").read_text(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--phase",choices=("register","train","calibrate","score","replay"),required=True);args=p.parse_args()
    {"register":registration,"train":train,"calibrate":calibrate,"score":score,"replay":replay}[args.phase](args.cache)
