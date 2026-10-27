"""PAR-MEM-008 original real-source writer-key effective rank, NO FIT.

Never updates neural weights or uses source historical labels for query selection.
Record key written at each original real station event from the frozen
96,639-param TAC model, then report actual 8D key diversity in rolling
16/64/128/256-event authentic observation windows. Compare only to a
completely untrained, known-at-write/read source-ordinal Fourier address
reference; do NOT infer any prediction/retrieval advantage.

Source years 2015-2017 numeric fields never read. May-June 2014 was
previously exposed development; no independent performance claims.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from scripts.partime002_real_source import SHA256
from scripts.partime005_continuous_real_training import (
    source_and_times, observed_source_step, SEEDS
)
from scripts.parmem006_source_history_query import (
    original_frozen_core, fixed_plan as frozen_original_queries, ROOT
)

PROTO=ROOT/"research/parallel-six/PAR-MEM-008-PROTOCOL.json"
RESULTS=ROOT/"research/parallel-six/PAR-MEM-008-RESULTS.json"
ANCHORS=(256,384,512,640,767)
WINDOWS=(16,64,128,256)
DISTANCES=(1,16,64,128,256)
PERIODS=(7,13,29,59)
TYPES=("original_content_conditioned_trained_writer_key",
       "source_known_temporal_fourier_untrained_reference")

def contract(*,replay):
    p=json.loads(PROTO.read_text())
    if (p["study_id"]!="PAR-MEM-008-REAL-WRITER-ADDRESS-EFFECTIVE-RANK"
        or p["status"]!="ORIGINAL_REAL_SOURCE_FROZEN_MODEL_NO_FIT_DIAGNOSTIC"
        or p["source_original_zip_sha256"]!=SHA256
        or p["seeds"]!=list(SEEDS)
        or p["fixed_observed_event_anchors"]!=list(ANCHORS)
        or p["source_observed_history_windows"]!=list(WINDOWS)
        or p["source_event_distance_controls"]!=list(DISTANCES)
        or p["source_known_fourier_periods"]!=list(PERIODS)
        or p["source_address_types"]!=list(TYPES)
        or p["no_new_training"] is not True):
        raise RuntimeError("preregistered source address-rank study changed")
    if RESULTS.exists()!=replay:
        raise RuntimeError("immutable real-source rank receipt can be written once and then replayed")

def original_write_keys(seed, data, source_clock_rows, receipt):
    # This is PRIVILEGED original writer tracing during source replay;
    # these event-wise historical keys are NOT provided to a deployed
    # TAC model with only 154 persistent scalars.
    model=original_frozen_core(seed,receipt)
    state=model.init_stream(12)
    old=None
    keys=[]
    with torch.no_grad():
        for ordinal in range(min(len(z) for z in source_clock_rows)):
            current=[x[ordinal] for x in source_clock_rows]
            obs,covar,dt,admit=observed_source_step(data,current,old)
            _,state,trace=model.stream_step(
                state,obs,covar,elapsed_dt=dt,write_mask=admit,return_trace=True)
            addr=F.normalize(
                model.core.address(trace["embedding"]),dim=-1,eps=1e-6)
            if addr.shape!=(12,8) or not torch.isfinite(addr).all().item():
                raise RuntimeError("malformed genuine frozen writer key")
            keys.append(addr.detach().clone())
            old=current
    if state.count!=len(keys) or not state.write_seen.all().item():
        raise RuntimeError("genuine original source state unexpectedly reset")
    return torch.stack(keys,dim=0)

def time_keys(t,device,dtype):
    # Four genuine source-event ordinal frequencies known at both write
    # and read. No source numerical value or privileged original writer key.
    index=torch.arange(t,device=device,dtype=dtype).unsqueeze(-1)
    period=torch.tensor(PERIODS,device=device,dtype=dtype).unsqueeze(0)
    angle=2*math.pi*index/period
    features=torch.stack((angle.cos(),angle.sin()),dim=-1).reshape(t,8)
    return F.normalize(features,dim=-1,eps=1e-6)

def geometry(keys):
    t,b,d=keys.shape
    if b!=12 or d!=8 or t<=max(ANCHORS):
        raise RuntimeError("incomplete original source write-key observations")
    stats={str(win):{"rank_sum":0.0,"mean_abs_pair_cos_sum":0.0,
                    "windows":0} for win in WINDOWS}
    distances={str(dt):{"cos_sum":0.0,"abs_cos_sum":0.0,
                       "count":0} for dt in DISTANCES}
    with torch.no_grad():
        for anchor in ANCHORS:
            for site in range(12):
                for window in WINDOWS:
                    k=keys[anchor-window+1:anchor+1,site]
                    gram=k.T@k/k.shape[0]
                    eigen=torch.linalg.eigvalsh(gram.double()).clamp_min(0)
                    norm=eigen/(eigen.sum().clamp_min(1e-30))
                    active=norm[norm>0]
                    rank=float(torch.exp(-(active*active.log()).sum()))
                    if not (math.isfinite(rank) and 0.999<=rank<=8.001):
                        raise RuntimeError("genuine writer key rank out of bounds")
                    ck=(k@k.T).abs()
                    ck.fill_diagonal_(0)
                    collision=float(ck.sum()/(window*(window-1)))
                    rec=stats[str(window)]
                    rec["rank_sum"]+=rank
                    rec["mean_abs_pair_cos_sum"]+=collision
                    rec["windows"]+=1
                for dist in DISTANCES:
                    cs=float(keys[anchor,site]@keys[anchor-dist,site])
                    rec=distances[str(dist)]
                    rec["cos_sum"]+=cs
                    rec["abs_cos_sum"]+=abs(cs)
                    rec["count"]+=1
    return dict(
       per_source_window={
         w:dict(original_genuine_source_windows=n["windows"],
                effective_8D_writer_address_rank=n["rank_sum"]/n["windows"],
                mean_absolute_key_cosine_between_distinct_window_events=
                  n["mean_abs_pair_cos_sum"]/n["windows"])
         for w,n in stats.items()},
       per_true_source_event_lag={
         lag:dict(recorded_source_pairs=q["count"],
                  writer_key_cosine=q["cos_sum"]/q["count"],
                  writer_key_mean_absolute_cosine=q["abs_cos_sum"]/q["count"])
         for lag,q in distances.items()}
    )

def run(source:Path,*,replay:bool):
    torch.set_num_threads(1)
    contract(replay=replay)
    p,(_,_,_,_,_),receipt=frozen_original_queries(source)
    data,train,exposed=source_and_times(source)
    if (p["original_publisher_SHA256"]!=SHA256 or
        min(len(s) for s in train)<=max(ANCHORS) or
        min(len(s) for s in exposed)<=max(ANCHORS)):
        raise RuntimeError("original publisher source identity or length breached")
    result=[]
    for seed in SEEDS:
        entry=dict(seed=seed,phase={})
        for label,rows in (("original_REAL_fit",train),
                           ("historically_exposed_REAL_development",exposed)):
            writer=original_write_keys(seed,data,rows,receipt)
            ordinal=time_keys(writer.shape[0],writer.device,writer.dtype)
            known=ordinal[:,None,:].expand(-1,12,-1)
            entry["phase"][label]={
                TYPES[0]:geometry(writer),
                TYPES[1]:geometry(known)
            }
        result.append(entry)
        print("PAR_MEM_008_TRUE_SOURCE_KEY_RANK_SEED="+json.dumps(
            {"seed":seed,"DEV_lag_128_original_writer_cos":
                entry["phase"]["historically_exposed_REAL_development"][TYPES[0]]
                     ["per_true_source_event_lag"]["128"]["writer_key_cosine"],
             "DEV_window_256_original_effective_rank":
                entry["phase"]["historically_exposed_REAL_development"][TYPES[0]]
                     ["per_source_window"]["256"]["effective_8D_writer_address_rank"]
            },sort_keys=True),flush=True)
    means={}
    for label in ("original_REAL_fit","historically_exposed_REAL_development"):
        means[label]={}
        for typ in TYPES:
            sample=result[0]["phase"][label][typ]
            means[label][typ]={
                "window_effective_rank":{
                    win:sum(x["phase"][label][typ]
                          ["per_source_window"][win]["effective_8D_writer_address_rank"]
                            for x in result)/len(result)
                    for win in map(str,WINDOWS)},
                "distance_mean_cosine":{
                    lag:sum(x["phase"][label][typ]
                        ["per_true_source_event_lag"][lag]["writer_key_cosine"]
                        for x in result)/len(result)
                    for lag in map(str,DISTANCES)},
                "window_mean_absolute_key_cosine":{
                    win:sum(x["phase"][label][typ]
                       ["per_source_window"][win]
                       ["mean_absolute_key_cosine_between_distinct_window_events"]
                      for x in result)/len(result)
                   for win in map(str,WINDOWS)}
            }
    report=dict(
        study_id="PAR-MEM-008-REAL-WRITER-ADDRESS-EFFECTIVE-RANK",
        original_publisher_source_SHA256=SHA256,
        frozen_original_Tac_checkpoint_count=len(SEEDS),
        source_station_streams=12,
        true_source_ordinal_anchor_indices=list(ANCHORS),
        original_FIT_only_and_already_exposed_DEV=True,
        no_past_source_measurement_retrieval_claim=True,
        temporal_fourier_reference_NOT_trained_or_deployable_solved_memory=True,
        source_writer_keys_logged_in_privileged_diagnostic_sidecar_only=True,
        protected_2015_2017_numeric_never_read=True,
        active_nn_model_params_unchanged=True,
        original_real_source_records=result,
        original_five_seed_means=means)
    if replay:
        original=json.loads(RESULTS.read_text())
        def cmp(a,b,k="root"):
            if type(a)!=type(b):
                raise RuntimeError("key rank source replay schema drift "+k)
            if isinstance(a,dict):
                if set(a)!=set(b):raise RuntimeError("source rank keys drift "+k)
                for n in a:cmp(a[n],b[n],k+"."+str(n))
            elif isinstance(a,list):
                if len(a)!=len(b):raise RuntimeError("source rank seed count drift")
                for i,(x,y) in enumerate(zip(a,b)):cmp(x,y,k+"."+str(i))
            elif isinstance(a,float):
                if abs(a-b)>1e-5*max(1.,abs(a)):
                    raise RuntimeError("genuine source key rank numerical replay drift "+k)
            elif a!=b:raise RuntimeError("source rank source identity drift "+k)
        cmp(original,report)
        print("PAR_MEM_008_INDEPENDENT_REAL_SOURCE_WRITE_KEY_RANK_REPLAY_PASS",flush=True)
    else:
        RESULTS.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("PAR_MEM_008_REAL_SOURCE_FIVE_SEED_MEANS="+json.dumps(
        means,sort_keys=True,allow_nan=False),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--source",type=Path,required=True)
    a.add_argument("--mode",choices=("audit","replay"),required=True)
    cfg=a.parse_args()
    run(cfg.source,replay=cfg.mode=="replay")
