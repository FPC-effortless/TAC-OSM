"""PAR-MEM-007: real-source WRITE-key oracle and associative interference audit.

PRIVILEGED DIAGNOSTIC ONLY: obtaining the original writer key at an old
event requires storing that historical key (or the original measurement)
outside TAC's fixed 154-scalar state. It is NOT a deployable source-time
query capability. This study probes whether the already-written compressed
8x8 matrix still contains decodable information if supplied the correct
writer key rather than a timestamp-derived key.

All five original genuine-source TAC and PAR-MEM-006 timestamp-head
checkpoints are immutable and SHA checked; all source query sets are the
already frozen PAR-MEM-006 train and historically exposed DEV identities.
Identical ridge feature dimensions/alpha/cohorts for the four diagnostics.
No new neural training, no new protected data, and NO qualified fair
neural architecture superiority claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.partime002_real_source import SHA256
from scripts.partime005_continuous_real_training import (
    source_and_times, observed_source_step, SEEDS
)
from scripts.parmem006_source_history_query import (
    ROOT, SOURCE_ROOT, OUT as OLD_OUT,
    SourceHistoricQueryHead, original_frozen_core, source_query_metadata,
    fixed_plan as frozen_original_queries,
    checksum as sha,
)

PROTOCOL=ROOT/"research/parallel-six/PAR-MEM-007-PROTOCOL.json"
OUT=ROOT/"research/parallel-six/PAR-MEM-007-RESULTS.json"
LAGS=(16,64,128)
ARMS=("privileged_true_writer_key","privileged_adjacent_wrong_writer_key",
      "original_006_learned_timestamp_key","no_matrix_read")
REG=1.0
EXPECTED_DIM=31

def preflight(source:Path, *, replay:bool):
    reg=json.loads(PROTOCOL.read_text())
    if (reg["study"]!="PAR-MEM-007"
        or reg["phase"]!="PRIVILEGED_DIAGNOSTIC_REAL_SOURCE_NOT_DEPLOYABLE"
        or reg["source_sha256"]!=SHA256
        or reg["seeds"]!=list(SEEDS)
        or reg["arms"]!=list(ARMS)
        or reg["ridge_lambda"]!=REG
        or reg["expected_feature_dimension_including_intercept"]!=EXPECTED_DIM
        or reg["no_new_neural_model_fitting"] is not True):
        raise RuntimeError("original real-source diagnostic preregistration changed")
    if OUT.exists()!=replay:
        raise RuntimeError("source diagnostic results can only be created once, then replayed")
    p,(_,fit,dev,trainq,devq),source_receipt=frozen_original_queries(source)
    receipt=json.loads((OLD_OUT/"TRAIN_RECEIPT.json").read_text())
    if receipt["original_model_count"]!=10:
        raise RuntimeError("source query head family missing original checkpoint receipt")
    source_index_digest=hashlib.sha256(json.dumps(
        trainq,sort_keys=True,separators=(",",":")
    ).encode()).hexdigest()
    if p["fixed_train_source_query_clock_SHA256"]!=source_index_digest:
        raise RuntimeError("original genuine source query chronology changed")
    data,_,_=source_and_times(source)
    if len(trainq)<100 or len(devq)<100:
        raise RuntimeError("no genuine historical measured source events")
    return data,fit,dev,trainq,devq,source_receipt,receipt

def timestamp_head(seed,old_receipt):
    info=next((x for x in old_receipt["original_training_records"]
               if x["seed"]==seed and x["arm"]=="associative_read"),None)
    if info is None:raise RuntimeError("original trained timestamp query model absent")
    p=OLD_OUT/f"seed-{seed}-associative_read.pt"
    if sha(p)!=info["trained_query_head_SHA256"]:
        raise RuntimeError("previous source timestamp query head SHA changed")
    saved=torch.load(p,map_location="cpu",weights_only=False)
    head=SourceHistoricQueryHead()
    head.load_state_dict(saved["model_state"],strict=True)
    head.eval()
    for x in head.parameters():x.requires_grad_(False)
    return head

def assemble_features(seed,rows,queries,source_receipt,prior_receipt,data):
    core=original_frozen_core(seed,source_receipt)
    query_net=timestamp_head(seed,prior_receipt)
    groups={}
    for i,(site,anchor,lag,older,now) in enumerate(queries):
        if lag not in LAGS or anchor<lag+2:
            raise RuntimeError("cannot run source past writer-key audit on unknown event")
        groups.setdefault(anchor,[]).append((i,site,anchor,lag,older,now))
    values={name:[None]*len(queries) for name in ARMS}
    saved_keys=[]
    cosine_true_adj=[0.]*len(queries)
    cosine_true_timestamp=[0.]*len(queries)
    distance_true_adj=[0.]*len(queries)
    now_pm=[None]*len(queries)
    state=core.init_stream(12)
    before=None
    with torch.no_grad():
        for index in range(min(len(z) for z in rows)):
            clocks=[r[index] for r in rows]
            obs,covar,dt,admit=observed_source_step(data,clocks,before)
            _,state,trace=core.stream_step(state,obs,covar,
                         elapsed_dt=dt,write_mask=admit,return_trace=True)
            # Recompute EXACT key used at this source admission from frozen
            # model's actual write embedding; not measurement/target lookup.
            actual_key=F.normalize(
                core.core.address(trace["embedding"]),dim=-1,eps=1e-6)
            saved_keys.append(actual_key.detach().clone())
            before=clocks
            if index not in groups:continue
            if state.count!=index+1:raise RuntimeError("source persistent state reset")
            for i,site,anchor,lag,older,now in groups[index]:
                if rows[site][anchor]!=now or rows[site][anchor-lag]!=older:
                    raise RuntimeError("registered source clock identity mismatch")
                if len(saved_keys)<=anchor-lag+1:
                    raise RuntimeError("privileged source writer key not previously observed")
                orig=saved_keys[anchor-lag][site]
                wrong=saved_keys[anchor-lag+1][site]
                metadata=source_query_metadata(
                    [now],[older],[anchor-lag],[anchor],lag)[0]
                trained_key=F.normalize(
                    query_net.query_key(metadata),dim=-1,eps=1e-6)
                matrix=state.memory[site]
                # 8D read is determined WITHOUT returning the real answer.
                # All arms have EXACT same input feature width and source
                # metadata, no privileged source target numerical input.
                readouts={
                    ARMS[0]:matrix@orig,
                    ARMS[1]:matrix@wrong,
                    ARMS[2]:matrix@trained_key,
                    ARMS[3]:torch.zeros(8,dtype=obs.dtype),
                }
                for arm in ARMS:
                    z=torch.cat((obs[site],metadata,readouts[arm],
                                 obs.new_ones(1)),dim=-1)
                    if z.shape!=(EXPECTED_DIM,) or not torch.isfinite(z).all().item():
                        raise RuntimeError("misformed original-source query feature")
                    values[arm][i]=z.detach().clone()
                cosine_true_adj[i]=float((orig*wrong).sum())
                cosine_true_timestamp[i]=float((orig*trained_key).sum())
                distance_true_adj[i]=float((readouts[ARMS[0]]-readouts[ARMS[1]]).norm())
                now_pm[i]=obs[site,0].detach().clone()
    if any(any(z is None for z in values[arm]) for arm in ARMS):
        raise RuntimeError("source oracle rows not generated chronologically")
    if any(z is None for z in now_pm):
        raise RuntimeError("source current PM2.5 unavailable")
    # Read original real target past PM2.5 only AFTER the entire forward
    # feature record is fixed. Past writer observation was source-observed
    # in stream, but past label is never passed to the query model.
    target=[]
    for site,anchor,lag,older,now in queries:
        if older>=now or rows[site][anchor-lag]!=older:
            raise RuntimeError("target clock invalid or future-leaking")
        if not data["valid"][site,older,0].item():
            raise RuntimeError("target was not a real observed source reading")
        target.append(data["values"][site,older,0].detach().clone())
    return (dict((name,torch.stack(v).double()) for name,v in values.items()),
            torch.stack(target).double(),torch.stack(now_pm).double(),
            dict(mean_true_vs_wrong_writer_key_cosine=sum(cosine_true_adj)/len(queries),
                 mean_true_vs_learned_timestamp_key_cosine=sum(cosine_true_timestamp)/len(queries),
                 mean_true_vs_adjacent_memory_read_l2=sum(distance_true_adj)/len(queries),
                 recorded_real_source_queries=len(queries)))

def ridge_fit(train_x,train_y):
    if train_x.shape[1]!=EXPECTED_DIM or train_y.shape!=(train_x.shape[0],):
        raise RuntimeError("source ridge training contract changed")
    normal=train_x.T@train_x
    penalty=torch.eye(EXPECTED_DIM,dtype=torch.float64)*REG
    penalty[-1,-1]=0.0   # source train intercept is not shrunk
    weight=torch.linalg.solve(normal+penalty,train_x.T@train_y)
    if not torch.isfinite(weight).all().item():
        raise RuntimeError("true source linear diagnostic diverged")
    return weight

def run(source:Path,*,replay:bool):
    torch.set_num_threads(1)
    data,fit,dev,trainq,devq,source_receipt,prior_receipt=preflight(source,replay=replay)
    results=[]
    for seed in SEEDS:
        tx,ty,now_train,train_diag=assemble_features(
            seed,fit,trainq,source_receipt,prior_receipt,data)
        dx,dy,now_dev,dev_diag=assemble_features(
            seed,dev,devq,source_receipt,prior_receipt,data)
        xlag=torch.tensor([int(v[2]) for v in devq],dtype=torch.long)
        arms={}
        for arm in ARMS:
            w=ridge_fit(tx[arm],ty)
            pred=dx[arm]@w
            categories={}
            for lag in LAGS:
                choose=xlag==lag
                true=dy[choose]
                err=(pred[choose]-true).abs()
                categories[str(lag)]=dict(
                  recorded_source_lag_queries=int(choose.sum()),
                  dev_historical_normalized_PM25_MAE=float(err.mean()),
                  known_current_PM25_baseline_MAE=float(
                      (now_dev[choose]-true).abs().mean()),
                  actual_source_full_history_oracle_MAE=0.0,
                  mean_signed_prediction_bias=float((pred[choose]-true).mean()))
            arms[arm]=dict(original_fit_ridge_weight_L2=float(w.norm()),
                           per_genuine_source_lag=categories)
        row=dict(seed=seed,original_real_trained_pretrained_core_sha=next(
            x["original_checkpoint_SHA256"] for x in source_receipt["original_checkpoints"]
            if x["seed"]==seed),
          source_original_training_queries=train_diag["recorded_real_source_queries"],
          already_exposed_development_queries=dev_diag["recorded_real_source_queries"],
          original_real_source_DEV_writer_key_diagnostics=dev_diag,
          heads=arms)
        results.append(row)
        print("PAR_MEM_007_ORIGINAL_REAL_KEY_ORACLE_SEED="+json.dumps(
             dict(seed=seed,source_dev_queries=len(devq),
                  diagnostics=dev_diag,
                  per_arm_lag_64={k:v["per_genuine_source_lag"]["64"]["dev_historical_normalized_PM25_MAE"] for k,v in arms.items()}),
              sort_keys=True,allow_nan=False),flush=True)
    mean={}
    for lag in LAGS:
        mean[str(lag)]={}
        for arm in ARMS:
            scores=[r["heads"][arm]["per_genuine_source_lag"][str(lag)]["dev_historical_normalized_PM25_MAE"] for r in results]
            mean[str(lag)][arm]=sum(scores)/len(scores)
        mean[str(lag)]["current_PM25_baseline"]=results[0]["heads"][ARMS[0]]["per_genuine_source_lag"][str(lag)]["known_current_PM25_baseline_MAE"]
    outcome=dict(
        study_id="PAR-MEM-007-PRIVILEGED-WRITER-KEY",
        original_source_sha256=SHA256,
        original_trained_TAC_cores=5,
        ridge_lambda_original_fit=REG,
        real_past_observation_lags=list(LAGS),
        source_training_and_exposed_dev_query_timestamps_unchanged=True,
        source_DEV_previously_exposed=True,
        privileged_true_writer_key_requires_unbounded_per_event_sidecar=True,
        no_deployable_temporal_shared_addressing_demonstrated=True,
        full_training_FLOP_neural_model_parity_not_certified=True,
        genuine_source_diagnostic_NOT_independently_qualified=True,
        prior_all_source_original_checkpoints_unchanged=True,
        original_individual_seed_records=results,
        original_average_real_lag_results=mean
    )
    if replay:
        original=json.loads(OUT.read_text())
        def near(a,b,path="root"):
            if type(a)!=type(b):raise RuntimeError("source replay schema mismatch: "+path)
            if isinstance(a,dict):
                if set(a)!=set(b):raise RuntimeError("source replay keys changed: "+path)
                for k in a:near(a[k],b[k],path+"."+str(k))
            elif isinstance(a,list):
                if len(a)!=len(b):raise RuntimeError("source replay count changed: "+path)
                for i,(x,y) in enumerate(zip(a,b)):near(x,y,path+f".{i}")
            elif isinstance(a,float):
                if abs(a-b)>1e-5*max(1.,abs(a)):
                    raise RuntimeError("real-source numeric replay drift: "+path)
            elif a!=b:raise RuntimeError("source identity replay drift: "+path)
        near(original,outcome)
        print("PAR_MEM_007_INDEPENDENT_FROZEN_5_SEED_REAL_SOURCE_REPLAY_PASS",flush=True)
    else:
        if OUT.exists():raise RuntimeError("original source report appeared concurrently")
        OUT.write_text(json.dumps(outcome,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("PAR_MEM_007_REAL_SOURCE_SUMMARY="+json.dumps(
         dict(means=mean,key_cosine_mean=sum(r["original_real_source_DEV_writer_key_diagnostics"]["mean_true_vs_wrong_writer_key_cosine"] for r in results)/5,
              declared_privileged_not_deployable=True),sort_keys=True,allow_nan=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--mode",choices=("audit","replay"),required=True)
    args=p.parse_args()
    run(args.source,replay=(args.mode=="replay"))
