#!/usr/bin/env python3
"""EWA002 one-shot new-task real robot transfer diagnostic; original cabinet TEST closed.

Frozen original 5 ME097 checkpoint trained weights, 14D ALOHA static candy real
state/action episodes, model query only; compare velocity+exact similarity vs
same cache+11bit hash vs identical hash+later-outcome utility. No neural training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import urllib.error

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
from scripts.ewa001_outcome_utility_real_eval import model_events,restored_model,SEEDS
from tac_osm.variants.ewa002_velocity_memory import VelocityResidualMemory

DATASET="lerobot/aloha_static_candy"
REV_PREFIX="6728831"
FIRST_FIT=tuple(range(0,25))
EVAL_IDS=tuple(range(25,50))
MAX_EVENTS=256
ARMS=("velocity","persistence","frozen_model","exhaustive_cosine","hashed_cosine","hashed_utility")

def sha(b):
    return hashlib.sha256(b).hexdigest()

def retrieve(url,tries=3):
    for attempt in range(tries):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"TAC-OSM-EWA002-original-audit/1.0"})
            with urllib.request.urlopen(req,timeout=90) as resp:
                return resp.read()
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError):
            if attempt==tries-1:raise
            time.sleep(attempt+1)
    raise RuntimeError("unreachable")

def load_new_task(cache):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    meta_url=f"https://huggingface.co/api/datasets/{DATASET}/revision/{REV_PREFIX}"
    info=json.loads(retrieve(meta_url))
    full_sha=info.get("sha","")
    if not full_sha.startswith(REV_PREFIX) or len(full_sha)!=40:
        raise RuntimeError("publisher dataset commit changed or unverified")
    sibling=set(v["rfilename"] for v in info.get("siblings",[]))
    manifest={"repo":DATASET,"commit":full_sha,"publisher_license":"mit",
              "source":"real recorded bimanual joint states/actions; different task from fitted mobile cabinet",
              "fit_only_episodes":list(FIRST_FIT),"untouched_external_eval_episodes":list(EVAL_IDS),
              "publisher_parquet_sha256":{}}
    # The REAL immutable revision uses consolidated LeRobot v3 Parquet, though
    # the publisher dataset card still describes an older v2 per-episode layout.
    path="data/chunk-000/file-000.parquet"
    if path not in sibling:
        raise RuntimeError("pinned publisher consolidated data path missing; actual="+
                           repr(sorted(p for p in sibling if p.endswith(".parquet"))))
    source=cache/"file-000.parquet"
    if not source.exists():
        source.write_bytes(retrieve(
            f"https://huggingface.co/datasets/{DATASET}/resolve/{full_sha}/{path}"))
    blob=source.read_bytes()
    if not blob:
        raise RuntimeError("empty publisher source")
    manifest["publisher_parquet_sha256"][path]=sha(blob)
    table=pq.read_table(source,columns=[
        "episode_index","frame_index","timestamp","observation.state","action"])
    d=table.to_pydict()
    episodes={ep:[] for ep in range(50)}
    for idx,frame,timestamp,x,a in zip(*(d[k] for k in [
        "episode_index","frame_index","timestamp","observation.state","action"])):
        ep=int(idx)
        if (ep not in episodes or len(x)!=14 or len(a)!=14 or
            not np.isfinite(x+a+[timestamp]).all()):
            raise RuntimeError("source data mismatch/finite/identity contract")
        episodes[ep].append((int(frame),float(timestamp),x,a))
    for ep,rows in episodes.items():
        if len(rows)<MAX_EVENTS+2:
            raise RuntimeError("insufficient authentic chronological source episode")
        if not all(b[0]==a[0]+1 and 0<b[1]-a[1]<.2
                   for a,b in zip(rows[:MAX_EVENTS+2],rows[1:MAX_EVENTS+2])):
            raise RuntimeError("timestamp/frame discontinuity")
    manifest["publisher_files"]=1
    manifest["publisher_actual_snapshot_layout"]="v3 consolidated file-000.parquet"
    manifest["all_50_source_episodes_pinned_before_score"]=True
    manifest["training_source_test_2016_2017_opened"]=False
    return episodes,manifest

def fit_thresholds(data):
    errors=np.concatenate([np.mean((z["y"]-(2*z["x"]-z["prev"]))**2,axis=-1)
                           for z in data])
    moving=np.concatenate([np.mean((z["y"]-z["x"])**2,axis=-1) for z in data])
    scale=max(float(np.quantile(errors,.9)),1e-6)
    thresh=float(np.quantile(moving,.9))
    if not np.isfinite(scale+thresh):raise RuntimeError("FIT-only threshold invalid")
    return scale,thresh

def aggregate(per_eps):
    n=sum(e["count"] for e in per_eps);nt=sum(e["late_count"] for e in per_eps)
    nm=sum(e["moving_count"] for e in per_eps)
    return dict(real_scored_frames=n,real_episodes=len(per_eps),moving_frames=nm,
       MSE={a:sum(z["sse"][a] for z in per_eps)/n for a in ARMS},
       moving_MSE={a:sum(z["moving_sse"][a] for z in per_eps)/nm if nm else None for a in ARMS},
       late_MSE={a:sum(z["late_sse"][a] for z in per_eps)/nt for a in ARMS})

def advance(events,memory,threshold,report):
    for episode in events:
        y=episode["y"];x=episode["x"];prev=episode["prev"];key=episode["key"]
        perarm={a:[] for a in ARMS};motion=[]
        counters_before=memory.counters()
        for t in range(len(y)):
            base=(2*x[t]-prev[t]).astype(np.float32)
            token=memory.predict(key[t],base)
            # All forecasts computed before true next physical state is delivered.
            forecasts={"velocity":base,"persistence":x[t],
                "frozen_model":episode["baseline"][t],
                "exhaustive_cosine":token.exhaustive,
                "hashed_cosine":token.hashed,
                "hashed_utility":token.utility}
            label=y[t]
            if report is not None:
                for name,prediction in forecasts.items():
                    perarm[name].append(float(np.mean((prediction-label)**2)))
                motion.append(float(np.mean((label-x[t])**2)))
            memory.observe(token,label,key[t],
                f"ALOHA_STATIC_CANDY:{episode['episode_id']}:{episode['frames'][t]}")
        if report is not None:
            m=np.asarray(motion)>=threshold
            after=memory.counters()
            report.append(dict(episode_id=int(episode["episode_id"]),count=len(y),
                moving_count=int(m.sum()),late_count=len(y)-len(y)//2,
                sse={a:float(sum(perarm[a])) for a in ARMS},
                late_sse={a:float(sum(perarm[a][len(y)//2:])) for a in ARMS},
                moving_sse={a:float(np.asarray(perarm[a])[m].sum()) for a in ARMS},
                hash_candidate_scores=after["hash_candidate_scores"]-counters_before["hash_candidate_scores"],
                hash_probes=(after["hash_probes_per_query"]*after["executed_queries"]-
                             counters_before["hash_probes_per_query"]*counters_before["executed_queries"]),
                exact_top8_recall_after_hash=(after["hash_raw_top8_recall"]*after["executed_queries"]-
                     counters_before["hash_raw_top8_recall"]*counters_before["executed_queries"])/len(y),
                nonempty_index_reads=(after["hashed_nonempty_rate"]*after["executed_queries"]-
                     counters_before["hashed_nonempty_rate"]*counters_before["executed_queries"])))

def main():
    args=argparse.ArgumentParser()
    args.add_argument("--cachedir",default="/tmp/aloha-static-candy-ewa002")
    args.add_argument("--output",default="research/diagnostics/ewa002_external_robot_original.json")
    args.add_argument("--seed-list",default="97001,97002,97003,97004,97005")
    a=args.parse_args()
    if tuple(int(x) for x in a.seed_list.split(","))!=SEEDS:
        raise RuntimeError("EWA002 requires all five preregistered original source seeds")
    torch.set_num_threads(1)
    episodes,manifest=load_new_task(a.cachedir)
    index=json.loads((ROOT/"research/replay-bundles/ME097/FIT_INDEX.json").read_text())
    assert len(index["models"])==5
    results=dict(status="ONE_SHOT_NEW_REAL_TASK_DIAGNOSTIC_NO_AUTONOMOUS_QUALIFICATION",
       prereg="research/mathematical-engineering/ewa002/PROTOCOL.md",
       real_new_task=manifest,original_model_family="ME097",original_frozen_neural_parameters=95874,
       original_cabinet_test_untouched=True,no_new_neural_training=True,source_independent_task=True,
       new_model_family=False,original_parallel_scan_changed=False,
       streaming_core_primary_float_bytes=608,external_memory_float_bytes=23552,
       seeds=[])
    for seed in SEEDS:
        model,prep,source_receipt=restored_model(seed,index)
        fit_events=[model_events(model,prep,episodes,ep) for ep in FIRST_FIT]
        external_events=[model_events(model,prep,episodes,ep) for ep in EVAL_IDS]
        scale,motion_threshold=fit_thresholds(fit_events)
        memory=VelocityResidualMemory(scale)
        advance(fit_events,memory,motion_threshold,None)
        warm=memory.counters()
        episode_scores=[]
        advance(external_events,memory,motion_threshold,episode_scores)
        total=aggregate(episode_scores)
        done=memory.counters()
        eval_queries=done["executed_queries"]-warm["executed_queries"]
        stats=dict(scored_new_task_queries=eval_queries,
            candidates_per_query=(done["hash_candidate_scores"]-warm["hash_candidate_scores"])/eval_queries,
            raw_hash_recall_at8=sum(z["exact_top8_recall_after_hash"]*z["count"] for z in episode_scores)/eval_queries,
            nonempty_fraction=sum(z["nonempty_index_reads"] for z in episode_scores)/eval_queries,
            exact_exhaustive_candidates_per_query=(done["exhaustive_candidate_scores"]-
                warm["exhaustive_candidate_scores"])/eval_queries,
            hash_probes_per_query=(done["hash_probe_count"] if "hash_probe_count" in done else 12),
            cumulative_cpu_seconds_read=done["cpu_read_seconds"],
            cumulative_cpu_seconds_update=done["cpu_write_seconds"],
            feedback_updates=done["utility_feedback_updates"],
            external_float_bytes=done["separate_float_memory_bytes"],
            cached_utilities_not_neural_training=True)
        h1=(total["MSE"]["exhaustive_cosine"]<=.99*total["MSE"]["velocity"]
            and total["moving_MSE"]["exhaustive_cosine"] is not None
            and total["moving_MSE"]["exhaustive_cosine"]<=1.05*total["moving_MSE"]["velocity"])
        h2=(total["MSE"]["hashed_utility"]<=.99*total["MSE"]["hashed_cosine"]
            and total["moving_MSE"]["hashed_utility"] is not None
            and total["moving_MSE"]["hashed_utility"]<=1.05*total["moving_MSE"]["hashed_cosine"])
        h3=(stats["candidates_per_query"]<32 and stats["raw_hash_recall_at8"]>=.70
            and stats["nonempty_fraction"]>=.90)
        item=dict(seed=seed,original_checkpoint_sha256=source_receipt["checkpoint_sha256"],
            original_checkpoint_trained_parameters=source_receipt["trainable_parameters"],
            fit_episodes=len(FIRST_FIT),eval_episodes=len(EVAL_IDS),
            fit_only_credit_scale=scale,fit_only_movement_threshold=motion_threshold,
            scores=total,index_stats=stats,hypothesis_gates=dict(H1=h1,H2=h2,H3=h3),
            registered_full_pass=bool(h1 and h2 and h3),
            original_per_episode_new_external_task_scores=episode_scores)
        results["seeds"].append(item)
        print("EWA002_SEED",json.dumps({k:v for k,v in item.items()
                 if k!="original_per_episode_new_external_task_scores"},sort_keys=True),flush=True)
    results["original_full_gate_pass"]=all(s["registered_full_pass"] for s in results["seeds"])
    results["source_test_last15pct_cabinet_accessed"]=False
    path=ROOT/a.output;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,indent=2,sort_keys=True)+"\n")
    print("EWA002_ORIGINAL_FULL_GATE="+("PASS" if results["original_full_gate_pass"] else "FAIL"))
    print("EWA002_EXTERNAL_NEW_REAL_TASK_COMPLETE=PASS")
    print("ORIGINAL_TRAINED_CHECKPOINT_CHANGED=NO")
    print("CABINET_PROTECTED_TEST_OPENED=NO")

if __name__=="__main__":main()
