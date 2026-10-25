#!/usr/bin/env python3
"""EWA001 prequential delayed-utility vs neutral cache on authentic exposed DEV.

Uses frozen original ME097 checkpoint, original data SHA, original lagged-action
causality, FIT-only cache initialization and DEV outcome delivery AFTER forecasts.
NO neural training, real-world skill promotion, hardware action or TEST scoring.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from tac_osm.variants.evidence_retrieval_world import EvidenceRetrievalWorld
from tac_osm.variants.outcome_utility_memory import OnlineMemory

URL="https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet/resolve/7a752b39f7e69de7e38aee485a6bab07528a061a/data/chunk-000/file-000.parquet"
DATA_SHA="a3409d3282b2cf9a94d32b301756fbd15dca5862cc1c32973c7316dd8853c57a"
MODEL_INDEX=ROOT/"research/replay-bundles/ME097/FIT_INDEX.json"
SEEDS=(97001,97002,97003,97004,97005)
MAX_EVENTS=256
ARMS=("original","neutral","utility","persistence","velocity")

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def source_episodes(path):
    path=Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(urllib.request.urlopen(URL,timeout=90).read())
    if digest(path)!=DATA_SHA:
        raise RuntimeError("REAL_SOURCE_SHA_MISMATCH")
    # Read numeric episode IDs only to determine original untouched TEST boundary.
    all_ids=sorted(set(pq.read_table(path,columns=["episode_index"]).column(0).to_pylist()))
    n=len(all_ids)
    fit=all_ids[:int(.7*n)]
    dev=all_ids[int(.7*n):int(.85*n)]
    test=all_ids[int(.85*n):]
    if not(fit and dev and test and set(fit).isdisjoint(dev)):
        raise RuntimeError("invalid original train/dev/TEST episode split")
    # Filter numeric states/actions BEFORE materialization. No TEST numeric outcomes read.
    selected=set(fit+dev)
    table=pq.read_table(path,
        filters=[("episode_index","in",fit+dev)],
        columns=["episode_index","frame_index","timestamp","observation.state","action"])
    d=table.to_pydict()
    episodes={}
    for ep,frame,ts,state,action in zip(*(d[k] for k in [
        "episode_index","frame_index","timestamp","observation.state","action"])):
        if ep not in selected:
            raise RuntimeError("protected TEST leaked through filter")
        if len(state)!=14 or len(action)!=14 or not np.isfinite(state+action+[ts]).all():
            raise RuntimeError("invalid original source row")
        episodes.setdefault(ep,[]).append((int(frame),float(ts),state,action))
    for ep,rows in episodes.items():
        if len(rows)<4 or not all(b[0]==a[0]+1 and 0<b[1]-a[1]<.2 for a,b in zip(rows,rows[1:])):
            raise RuntimeError("nonconsecutive source episode")
    if set(episodes)!=selected:
        raise RuntimeError("episode mismatch / rows unavailable")
    return episodes,fit,dev,dict(original_episode_count=n,fit_episode_count=len(fit),
        exposed_dev_episode_count=len(dev),test_episode_count=len(test),
        numerical_test_rows_parsed=False,source_sha256=DATA_SHA)

def restored_model(seed,index):
    row=next((v for v in index["models"] if v["seed"]==seed),None)
    if row is None or row["trainable_parameters"]!=95874:
        raise RuntimeError("wrong original trained model registration")
    filename=ROOT/f"research/replay-bundles/ME097/seed-{seed}/final.pt"
    if digest(filename)!=row["checkpoint_sha256"]:
        raise RuntimeError("checkpoint original checksum mismatch")
    ck=torch.load(filename,weights_only=False,map_location="cpu")  # Trusted first-party Git-owned original.
    if ck["source_sha256"]!=DATA_SHA or ck["seed"]!=seed:
        raise RuntimeError("checkpoint original source identity mismatch")
    m=EvidenceRetrievalWorld()
    m.load_state_dict(ck["model"],strict=True)
    if sum(p.numel() for p in m.parameters())!=95874:
        raise RuntimeError("unexpected model parameter count")
    m.eval()
    prep={name:ck["preprocessing"][name].detach().cpu().numpy().astype(np.float32)
          for name in ["obs_mean","obs_scale","action_mean","action_scale"]}
    assert all(x.shape==(14,) for x in prep.values())
    return m,prep,row

@torch.no_grad()
def model_events(model,prep,recorded,ep):
    rows=recorded[ep]
    count=min(MAX_EVENTS,len(rows)-2)
    obs=np.asarray([r[2] for r in rows[1:1+count]],dtype=np.float32)
    prev=np.asarray([r[2] for r in rows[:count]],dtype=np.float32)
    lagged=np.asarray([r[3] for r in rows[:count]],dtype=np.float32)
    truth=np.asarray([r[2] for r in rows[2:2+count]],dtype=np.float32)
    mean,scale,am,asc=(prep[k] for k in ("obs_mean","obs_scale","action_mean","action_scale"))
    x=(obs-mean)/scale
    y=(truth-mean)/scale
    a=(lagged-am)/asc
    prev=(prev-mean)/scale
    state=None
    preds=[];keys=[]
    for start in range(0,count,64):
        end=min(start+64,count)
        p,state,trace=model(torch.from_numpy(x[start:end]).unsqueeze(0),
                            torch.from_numpy(a[start:end]).unsqueeze(0),
                            state=state,return_trace=True)
        preds.append(p.squeeze(0).numpy().copy())
        keys.append(trace["query"].squeeze(0).numpy().copy())
    prediction=np.concatenate(preds)
    query=np.concatenate(keys)
    if not(np.isfinite(prediction).all() and np.isfinite(query).all()):
        raise RuntimeError("nonfinite original model or learned query")
    if prediction.shape!=(count,14) or query.shape!=(count,8):
        raise RuntimeError("original model tensor contract mismatch")
    return dict(x=x,y=y,prev=prev,baseline=prediction,key=query,
                frames=np.asarray([r[0] for r in rows[1:1+count]],dtype=np.int64),
                episode_id=ep)

def create_arms():
    return dict(neutral=OnlineMemory("neutral"),utility=OnlineMemory("utility"))

def advance(events,banks,*,evaluate,threshold,counts,per_episode):
    for ev in events:
        x,y,prev,base,q=tuple(ev[k] for k in ("x","y","prev","baseline","key"))
        n=len(x)
        err={name:[] for name in ARMS}
        moving=[]
        for i in range(n):
            # All predictions issued from causal information known at this cut.
            neutral=banks["neutral"].predict(q[i],base[i],audit=evaluate)
            utility=banks["utility"].predict(q[i],base[i],audit=evaluate)
            preds=dict(original=base[i],neutral=neutral.forecast,utility=utility.forecast,
                       persistence=x[i],velocity=2*x[i]-prev[i])
            # Only after every arm forecast exists can the next real row arrive.
            actual=y[i]
            if evaluate:
                for name,pred in preds.items():
                    err[name].append(float(np.mean((pred-actual)**2)))
                moving.append(float(np.mean((actual-x[i])**2)))
            src=f"ALOHA-original:{ev['episode_id']}:{ev['frames'][i]}"
            banks["neutral"].observe(neutral,actual,q[i],src)
            banks["utility"].observe(utility,actual,q[i],src)
        counts["fit" if not evaluate else "exposed_dev"]+=n
        if evaluate:
            movemask=np.asarray(moving)>=threshold
            record=dict(episode=int(ev["episode_id"]),events=n,moving_events=int(movemask.sum()),
                       all_mse={name:float(np.mean(err[name])) for name in ARMS},
                       late_mse={name:float(np.mean(err[name][n//2:])) for name in ARMS},
                       moving_mse={name:float(np.mean(np.asarray(err[name])[movemask])) if movemask.any() else None for name in ARMS},
                       sum_squared_errors={name:float(sum(err[name])) for name in ARMS},
                       moving_sum_squared_errors={name:float(np.asarray(err[name])[movemask].sum()) for name in ARMS},
                       late_sum_squared_errors={name:float(sum(err[name][n//2:])) for name in ARMS})
            per_episode.append(record)

def summary(per_eps):
    counts=sum(e["events"] for e in per_eps)
    movecount=sum(e["moving_events"] for e in per_eps)
    nlate=sum(e["events"]-e["events"]//2 for e in per_eps)
    return dict(events=counts,moving_events=movecount,episodes=len(per_eps),
        mse={a:sum(e["sum_squared_errors"][a] for e in per_eps)/counts for a in ARMS},
        moving_mse={a:sum(e["moving_sum_squared_errors"][a] for e in per_eps)/movecount if movecount else None for a in ARMS},
        late_mse={a:sum(e["late_sum_squared_errors"][a] for e in per_eps)/nlate for a in ARMS})

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="/tmp/aloha-original.parquet")
    p.add_argument("--output",default="research/diagnostics/ewa001_original_exposed_dev.json")
    p.add_argument("--seeds",default="97001,97002,97003,97004,97005")
    a=p.parse_args()
    chosen=tuple(int(s) for s in a.seeds.split(","))
    if chosen!=SEEDS:
        raise RuntimeError("preregistered study requires all five original seeds")
    torch.set_num_threads(1)
    episodes,fit,dev,source=source_episodes(a.data)
    idx=json.loads(MODEL_INDEX.read_text())
    if idx["source_sha256"]!=DATA_SHA:
        raise RuntimeError("registered source mismatch")
    result=dict(status="EXPOSED_DEV_DIAGNOSTIC_ONLY_NOT_INDEPENDENT_REAL_QUALIFICATION",
                checkpoint_family="ME097",frozen_checkpoint_seeds=list(SEEDS),
                neural_trainable_parameters=95874,new_trainable_parameters=0,
                primary_scan_changed=False,external_cache_separate_from_608B=True,
                causal_candidate_action_policy=False,source=source,seeds=[])
    for seed in chosen:
        m,prep,row=restored_model(seed,idx)
        # Every real model FIT/DEV prediction is computed from original weights and source.
        fit_events=[model_events(m,prep,episodes,ep) for ep in fit]
        dev_events=[model_events(m,prep,episodes,ep) for ep in dev]
        fit_deltas=np.concatenate([np.mean((z["y"]-z["x"])**2,axis=-1) for z in fit_events])
        moving_threshold=float(np.quantile(fit_deltas,.9))
        banks=create_arms()
        counts=dict(fit=0,exposed_dev=0)
        for z in fit_events:
            advance([z],banks,evaluate=False,threshold=moving_threshold,counts=counts,per_episode=[])
        before={key:bank.stats() for key,bank in banks.items()}
        per_episode=[]
        for z in dev_events:
            advance([z],banks,evaluate=True,threshold=moving_threshold,counts=counts,per_episode=per_episode)
        sc=summary(per_episode)
        after={key:bank.stats() for key,bank in banks.items()}
        cache_stats={}
        for key in banks:
            end=after[key];begin=before[key]
            eval_queries=end["queries"]-begin["queries"]
            cache_stats[key]=dict(float_payload_bytes=end["float_payload_bytes"],
                live_cache_capacity=end["capacity"],
                avg_candidates_scored_dev=(end["matching_candidate_scores"]-begin["matching_candidate_scores"])/max(1,eval_queries),
                index_probes_dev=(end["queries"]-begin["queries"])*9,
                dev_query_hits=end["hits"]-begin["hits"],
                dev_queries=eval_queries,
                mean_exhaustive_top8_recall_dev=end["top8_exact_reference_recall"],
                extra_utility_updates=end["utility_updates"]-begin["utility_updates"],
                total_replacements=end["replacements"],
                lookup_cpu_total=end["measured_lookup_cpu_seconds"],
                write_cpu_total=end["measured_write_cpu_seconds"])
        pass_both=(sc["mse"]["utility"]<=.99*sc["mse"]["neutral"]
                   and sc["mse"]["utility"]<=.99*sc["mse"]["original"])
        strong_move=(sc["moving_mse"]["utility"] is not None
                     and sc["moving_mse"]["utility"]<=1.05*sc["moving_mse"]["original"])
        fast=all(cache_stats[k]["avg_candidates_scored_dev"]<32 for k in ("neutral","utility"))
        item=dict(seed=seed,checkpoint_sha256=row["checkpoint_sha256"],counts=counts,
                  fit_movement_threshold_mse=moving_threshold,score=sc,cache=cache_stats,
                  passed_predeclared_development_threshold=bool(pass_both and strong_move and fast),
                  per_original_dev_episode=per_episode)
        result["seeds"].append(item)
        print("EWA001_SEED",json.dumps({k:v for k,v in item.items() if k!="per_original_dev_episode"},sort_keys=True),flush=True)
    result["all_seeds_passed"]=all(x["passed_predeclared_development_threshold"] for x in result["seeds"])
    result["any_independent_real_qualification"]=False
    path=ROOT/a.output
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("EWA001_ALL_SEEDS_PASS="+str(result["all_seeds_passed"]))
    print("ORIGINAL_TRAINED_MODEL_WEIGHTS_MODIFIED=NO")
    print("TEST_NUMERICAL_OUTCOMES_OPENED=NO")
    print("PER_SOURCE_REAL_DIAGNOSTIC_COMPLETE=PASS")
if __name__=="__main__":main()
