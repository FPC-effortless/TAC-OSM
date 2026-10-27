"""LATFUSE006 genuine-event supervised READOUT DIAGNOSTIC ONLY, not a model fit.

Use original recorded UCI501 sensor-history PM2.5 at t as the historical
fact, ask whether it is decodable from frozen source-trained LATFUSE004
model state at t+{16,64,128,512}. Binary event >=+1.25 FIT z vs matched
ordinary |z|<=.25. External ridge heads fitted ONLY on original March-
April 2014 FIT labels; July original FIT label/target never enters fitting.

Four controls (all include station/time/current source-only nuisance):
  1. nuisance local actual current sensors/masks/age/last and calendar;
  2. nuisance + original fast/slow and source-owned entity;
  3. preceding + original forecast's TRUE learned calendar/belief-based
     associative readout (16D); NO oracle event key or event truth query;
  4. preceding + raw full original associative 256D matrix (external
     linear-decodability upper bound, NOT deployed 6800-byte parameter budget).

No TAC optimizer/weight mutation. The sidecar ridge DOES fit true source
historical labels and must NEVER be called the pretrained model's native
factual-retrieval capability or a new qualified compact architecture.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F

from scripts import latfuse002_real_fit as original
from scripts import latfuse004_factorial_training as trained
from scripts import latfuse005_source_event_delay as previous
from scripts.validate_research_parity import check_manifest
from tac_osm.latent_fusion_004_factorial import ARMS,FactorialFusion,head_weights

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse006/REGISTERED_REAL_EVENT_DECODER.json"
SEEDS=(40401,40402)
DELAYS=(16,64,128,512)
TRAIN_START=original.hour(dt.datetime(2014,3,1))
TRAIN_STOP=original.hour(dt.datetime(2014,4,18))
EVAL_START=original.hour(dt.datetime(2014,7,1))
EVAL_STOP=original.hour(dt.datetime(2014,8,1))
FEATURES=("current","nonassoc","learned_associative_read","entire_memory_matrix")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def event_cohort(data,start,stop,max_events):
    rows=[]
    for station in range(12):
        valid=data["valid"][station,:,0]
        pm=data["values"][station,:,0]
        spikes=[]
        normals=[]
        for t in range(start,stop):
            if bool(valid[t]) and float(pm[t])>=1.25 and (
                not spikes or t-spikes[-1]>=24):
                spikes.append(t)
                if len(spikes)==max_events:break
        for spike in spikes:
            ordinary=[i for i in range(start,stop)
                      if bool(valid[i]) and abs(float(pm[i]))<=.25
                      and i%24==spike%24 and abs(i-spike)>=24
                      and all(abs(i-u)>=24 for u in normals)]
            if not ordinary:continue
            normal=min(ordinary,key=lambda i:(abs(i-spike),i))
            normals.append(normal)
            rows.extend([(station,spike,1),(station,normal,0)])
    if len(rows)<24 or len(rows)%2:
        raise RuntimeError("at least twelve physical recorded source matched pairs required")
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2]
        if (a[0]!=b[0] or a[1]%24!=b[1]%24 or
           a[2]!=1 or b[2]!=0 or a[1]==b[1]):
            raise RuntimeError("source-real temporal pair integrity violated")
    if any(t<23 or t+max(DELAYS)+6>=original.hour(original.TRAIN_END)
           for _,t,_ in rows):
        raise RuntimeError("original observed source FIT boundary violated")
    return rows

def identity_source_and_parent_check(train,held):
    if train is held or not train or not held:return False
    if max(t for _,t,_ in train)+max(DELAYS)>=min(t for _,t,_ in held):
        raise RuntimeError("source lagged train/evaluation histories overlap")
    return True

def realtime_features(model,state,d,site,ts):
    """No event-time information passed to an event-specific decoder query."""
    n=len(site)
    x=d["values"][site,ts]
    mask=d["valid"][site,ts]
    age=d["age"][site,ts]
    last=d["last"][site,ts]
    calendar=original.calendar([int(v+6) for v in ts])
    src=F.one_hot(site,num_classes=12).to(x.dtype)
    nuisance=torch.cat([x,mask.to(x.dtype),age,last,calendar,src],dim=-1)
    idx=site.long().view(n,1,1).expand(-1,1,80)
    entity=state.entities.gather(1,idx).squeeze(1)
    entity=entity*state.active.gather(1,site.long().view(n,1)).to(entity.dtype)
    no_memory=torch.cat([nuisance,state.fast,state.slow,entity],dim=-1)
    joined=torch.cat((state.fast,state.slow),dim=-1)
    # Exact source-trained calendar+belief query used in model.forecast().
    key=(torch.tanh(model.action_map(calendar[:,None,:]))+
         model.belief_query(joined)[:,None,:])
    queries=F.normalize(key.reshape(n,model.head_count,model.key_width),
                        dim=-1,eps=1e-8)
    matrix=state.associative.reshape(n,16,model.head_count,model.key_width).permute(0,2,1,3)
    values=torch.einsum("bhvk,bhk->bhv",matrix,queries)
    read_probs,_=head_weights(model.read_router(joined),model.log_head_temperature,
                      head_count=model.head_count,soft=True)
    actual_read=(values*read_probs[:,:,None]).sum(1)
    # IMPORTANT: no source oracle anchor key, value or label supplied here.
    return dict(current=nuisance,
                nonassoc=no_memory,
                learned_associative_read=torch.cat([no_memory,actual_read],dim=-1),
                entire_memory_matrix=torch.cat([no_memory,state.associative.reshape(n,256)],dim=-1))

def collected(model,d,rows):
    n=len(rows)
    station=torch.tensor([r[0] for r in rows],dtype=torch.long)
    anchor=torch.tensor([r[1] for r in rows],dtype=torch.long)
    label=torch.tensor([r[2] for r in rows],dtype=torch.float32)
    state=model.initial(n)
    for j in range(24):
        t=anchor-23+j
        state=model.step(state,d["values"][station,t],d["valid"][station,t],
                         d["age"][station,t],d["last"][station,t],station)
    got={}
    for j in range(1,513):
        t=anchor+j
        state=model.step(state,d["values"][station,t],d["valid"][station,t],
                         d["age"][station,t],d["last"][station,t],station)
        if j in DELAYS:
            got[j]=realtime_features(model,state,d,station,t)
            for name,feat in got[j].items():
                if not bool(torch.isfinite(feat).all()):
                    raise RuntimeError(f"nonfinite original frozen feature {name} lag{j}")
    return got,label

def ridge_fit_and_score(train_x,train_y,test_x,test_y):
    """Fixed dual ridge, no validation tuning/July gradient and no model fit."""
    if (train_x.ndim!=2 or test_x.ndim!=2 or train_x.shape[1]!=test_x.shape[1] or
        train_x.shape[0]!=len(train_y) or test_x.shape[0]!=len(test_y)):
        raise ValueError("original source covariate/label parity")
    if (not torch.equal(train_y[::2],torch.ones_like(train_y[::2])) or
        not torch.equal(train_y[1::2],torch.zeros_like(train_y[1::2])) or
        not torch.equal(test_y[::2],torch.ones_like(test_y[::2])) or
        not torch.equal(test_y[1::2],torch.zeros_like(test_y[1::2]))):
        raise RuntimeError("source controls are not matched as registered")
    center=train_x.mean(0)
    spread=train_x.std(0,unbiased=False).clamp_min(1e-4)
    sx=((train_x-center)/spread).clamp(-6,6)/math.sqrt(train_x.shape[1])
    tx=((test_x-center)/spread).clamp(-6,6)/math.sqrt(train_x.shape[1])
    if not all(bool(torch.isfinite(v).all()) for v in (sx,tx)):
        raise RuntimeError("nonfinite authentic source train normalized decoder")
    mu=train_y.mean()
    kernel=sx@sx.T+torch.eye(len(train_y),dtype=torch.float32)
    fitted=torch.linalg.solve(kernel,(train_y-mu)[:,None])
    raw=(tx@sx.T@fitted).squeeze(-1)+mu
    if not bool(torch.isfinite(raw).all()):
        raise RuntimeError("nonfinite fit decoder outputs")
    p=raw.clamp(0,1)
    pred=(p>=.5).float()
    positive=test_y==1
    negative=test_y==0
    balanced=.5*(float(pred[positive].mean())+float((1-pred[negative]).mean()))
    scores_pair=p[::2]-p[1::2]
    pair=float(((scores_pair>0).float()+.5*(scores_pair==0).float()).mean())
    return dict(training_n=len(train_y),evaluation_n=len(test_y),
        feature_dim=train_x.shape[1],ridge_lambda=1.0,
        balanced_accuracy=balanced,
        real_event_Brier=float((p-test_y).square().mean()),
        physical_source_pair_ordering_accuracy=pair,
        mean_predicted_event_probability=float(p[positive].mean()),
        mean_predicted_ordinary_probability=float(p[negative].mean()),
        fraction_prob_clipped=float(((raw<0)|(raw>1)).float().mean()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(1)
    if args.output.exists():
        raise RuntimeError("cannot rewrite factual source-retrieval probe evidence")
    reg=json.loads(REG.read_text())
    check_manifest(reg,"preflight")
    if (reg["study_id"]!="LATFUSE006-REAL-LAGGED-EVENT-READOUT-001" or
        reg["original_source_sha256"]!=original.SOURCE_SHA or
        reg["delays_hours"]!=list(DELAYS)):
        raise RuntimeError("lag supervision protocol drift")
    if digest(args.source)!=original.SOURCE_SHA:
        raise RuntimeError("incorrect original physical source archive")
    d=original.source_frames(args.source,phase="fit")
    train=event_cohort(d,TRAIN_START,TRAIN_STOP,8)
    test=event_cohort(d,EVAL_START,EVAL_STOP,2)
    identity_source_and_parent_check(train,test)
    # Reuse exactly the original July matched cohort, not cherry-picked based
    # on trained model or this decoder's test outcomes.
    previous_cohort=previous.source_only_cohort(d)
    if [(s,t,1 if typ=="observed_pm25_high" else 0)
        for s,t,typ in previous_cohort]!=test:
        raise RuntimeError("prior source-anchored July cohort differs")
    original_report=json.loads((trained.OUT/"RESULTS.json").read_text())
    if (original_report["total_original_optimizer_updates"]!=1024 or
        len(original_report["models"])!=8):
        raise RuntimeError("pretrained true-source model receipts invalid")
    evid=[]
    with torch.no_grad():
        for seed in SEEDS:
            for arm in ARMS:
                r=next((z for z in original_report["models"] if z["seed"]==seed and z["arm"]==arm),None)
                if r is None:raise RuntimeError("frozen model record absent")
                checkpoint=trained.OUT/f"seed-{seed}"/arm/"final.pt"
                source_sha=digest(checkpoint)
                if r["original_checkpoint_sha256"]!=source_sha:
                    raise RuntimeError("source original checkpoint mutated")
                m=FactorialFusion(arm)
                original_weights=torch.load(checkpoint,map_location="cpu",weights_only=False)
                m.load_state_dict(original_weights["model"],strict=True)
                m.eval()
                old=[p.detach().clone() for p in m.parameters()]
                train_features,train_labels=collected(m,d,train)
                eval_features,eval_labels=collected(m,d,test)
                results={}
                for delay in DELAYS:
                    metrics={}
                    for key in FEATURES:
                        metrics[key]=ridge_fit_and_score(train_features[delay][key],
                            train_labels,eval_features[delay][key],eval_labels)
                    metrics["event_truth_chance_baseline"]=dict(
                         balanced_accuracy=.5,real_event_Brier=.25,
                         physical_source_pair_ordering_accuracy=.5)
                    metrics["delta_Brier_learned_query_minus_nonassoc"]=(
                         metrics["learned_associative_read"]["real_event_Brier"]-
                         metrics["nonassoc"]["real_event_Brier"])
                    metrics["delta_pair_order_learned_query_minus_nonassoc"]=(
                         metrics["learned_associative_read"]["physical_source_pair_ordering_accuracy"]-
                         metrics["nonassoc"]["physical_source_pair_ordering_accuracy"])
                    results[str(delay)]=metrics
                if source_sha!=digest(checkpoint) or any(
                    not torch.equal(p,z) for p,z in zip(old,m.parameters())):
                    raise RuntimeError("read-only feature extraction mutated original trained model")
                outcome=dict(seed=seed,arm=arm,checkpoint_sha256=source_sha,delays=results)
                evid.append(outcome)
                print("LATFUSE006_REAL_EVENT_ARM="+json.dumps({
                    "seed":seed,"arm":arm,
                    "learned_read_Brier":{str(k):v["learned_associative_read"]["real_event_Brier"] for k,v in results.items()},
                    "nonassoc_Brier":{str(k):v["nonassoc"]["real_event_Brier"] for k,v in results.items()}},
                    sort_keys=True),flush=True)
    result=dict(study_id="LATFUSE006-REAL-LAGGED-EVENT-READOUT-001",
        genuine_source_archive_sha256=original.SOURCE_SHA,
        observed_source_train_cohort_sha256=hashlib.sha256(json.dumps(train,separators=(",",":")).encode()).hexdigest(),
        observed_source_eval_cohort_sha256=hashlib.sha256(json.dumps(test,separators=(",",":")).encode()).hexdigest(),
        source_train_matched_pairs=len(train)//2,source_eval_matched_pairs=len(test)//2,
        source_train_period="2014-03-01 <= event <2014-04-18",
        source_eval_period="2014-07-01 <= event <2014-08-01",
        historical_anchors_source_authorized=True,
        observed_original_event_labels="trained FIT-normalized actual physical PM2.5 >=1.25 or abs<=.25",
        original_pretrained_model_count=len(evid),frozen_model_updated=False,
        external_sidecar_ridge_weights_trained_on_real_source=True,
        sidecar_probes_are_not_deployed_6800_B_architectures=True,
        historical_event_recall_is_NOT_original_model_native_capability=True,
        original_fit_source_starts_before_2014_and_trained_original_models_saw_ALL_FIT_periods=True,
        source_eval_is_not_independent_pretrained_model_holdout=True,
        no_2015_DEV_or_2016_17_protected_numeric_read=True,
        model_query_never_received_oracle_anchor_write_key=True,
        forecast_model_immutable=True,neural_model_optimization_steps=0,
        models=evid)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("LATFUSE006_REAL_LAGGED_EVENT_READOUT_PASS="+json.dumps(
       {"pretrained_original_models":len(evid),
        "source_train_pairs":len(train)//2,"source_eval_pairs":len(test)//2,
        "original_neural_optimizer_steps":0},sort_keys=True))
if __name__=="__main__":
    main()
