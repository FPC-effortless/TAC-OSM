#!/usr/bin/env python3
"""Authenticated REAL robot pilot on public LeRobot ALOHA Mobile Cabinet data.

Main output is reproducible research evidence, NOT qualified model superiority.
Uses real timestamped 14D robot states, 14D recorded actions and next-state
outcomes, trains a separate continuous TAC world-model and GRU control.

Deliberately FAILS CLOSED when download, provenance, episode chronology,
dimension, train/test separation or minimum data assumptions fail.
No synthetic substitutions, no future observations as model inputs.
No image/video features, fabricated identity labels or intervention effects.
Matching: parameter count within 1%, same real samples, same minibatches,
optimizer, seeds and number of updates; FLOP budgets are NOT shown matched,
so do NOT claim a compulsory fair head-to-head model comparison.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/"src"))
from tac_osm.variants.parallel_continuous_world import ParallelTACWorld,RealGRUWorld

DATASET_ID="lerobot/aloha_mobile_cabinet"
DATA_PATH="data/chunk-000/file-000.parquet"
EXPECTED_STATE=14
EXPECTED_ACTION=14


def load_authentic_records():
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi,hf_hub_download
    info=HfApi().dataset_info(DATASET_ID)
    revision=info.sha
    if not isinstance(revision,str) or len(revision)!=40:
        raise RuntimeError("immutable 40-char data revision unavailable")
    path=hf_hub_download(
      repo_id=DATASET_ID,repo_type="dataset",revision=revision,
      filename=DATA_PATH)
    raw=Path(path).read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    table=pq.read_table(path,columns=[
      "episode_index","frame_index","timestamp",
      "observation.state","action"])
    required={"episode_index","frame_index","timestamp","observation.state","action"}
    if set(table.column_names)!=required:
        raise RuntimeError("data schema absent or unexpectedly modified")
    data=table.to_pydict()
    episodes={}
    for i,(ep,frame,ts,x,u) in enumerate(zip(
        data["episode_index"],data["frame_index"],data["timestamp"],
        data["observation.state"],data["action"])):
        if (ep is None or frame is None or ts is None or
            x is None or u is None or len(x)!=EXPECTED_STATE or
            len(u)!=EXPECTED_ACTION):
            raise RuntimeError("malformed real observation/action record")
        if not all(v is not None and abs(float(v))<1e10 for v in list(x)+list(u)):
            raise RuntimeError("nonfinite or excessive real source values")
        episodes.setdefault(int(ep),[]).append((
          int(frame),float(ts),tuple(map(float,x)),tuple(map(float,u))))
    if len(episodes)<10:
        raise RuntimeError("insufficient distinct authentic held-out episodes")
    for ep,rows in episodes.items():
        rows.sort(key=lambda row:row[0])
        for a,b in zip(rows,rows[1:]):
            if not b[0]>a[0] or not b[1]>a[1]:
                raise RuntimeError(f"nonmonotone episode {ep}, data leakage risk")
    return episodes,{"dataset_id":DATASET_ID,"relative_parquet":DATA_PATH,
         "dataset_revision_sha":revision,"parquet_sha256":digest,
         "real_episodes_available":len(episodes),
         "source_frame_records":table.num_rows}


def splits(episodes):
    ids=sorted(episodes)
    n=len(ids)
    # Entire episodes held out, disjoint exact sources, not overlapping frames.
    train=ids[:int(.70*n)]
    valid=ids[int(.70*n):int(.85*n)]
    test=ids[int(.85*n):]
    if not(train and valid and test):
        raise RuntimeError("held-out episode split missing")
    assert not(set(train)&set(valid)|set(valid)&set(test)|set(train)&set(test))
    return {"train":train,"validation":valid,"test":test}


def windows(episodes, ids, length=16, max_windows_per_episode=10):
    records=[]
    for episode in ids:
        rows=episodes[episode]
        possible=[]
        for s in range(len(rows)-length):
            chunk=rows[s:s+length+1]
            if not all(chunk[j+1][0]==chunk[j][0]+1 for j in range(length)):
                continue
            if not all(0<chunk[j+1][1]-chunk[j][1]<.2 for j in range(length)):
                continue
            possible.append(s)
        if not possible:continue
        stride=max(1,len(possible)//max_windows_per_episode)
        for start in possible[::stride][:max_windows_per_episode]:
            seq=rows[start:start+length+1]
            x=torch.tensor([r[2] for r in seq],dtype=torch.float32)
            u=torch.tensor([r[3] for r in seq[:-1]],dtype=torch.float32)
            records.append((x[:-1],u,x[1:],episode))
    if len(records)<5:
        raise RuntimeError("not enough real chronological next-state windows")
    return records


def normalization(train):
    observations=torch.cat([r[0].reshape(-1,14) for r in train],0)
    actions=torch.cat([r[1].reshape(-1,14) for r in train],0)
    return (observations.mean(0),observations.std(0).clamp_min(.0001),
            actions.mean(0),actions.std(0).clamp_min(.0001))


def pack(records,norm):
    xm,xs,am,asd=norm
    obs=torch.stack([(r[0]-xm)/xs for r in records])
    act=torch.stack([(r[1]-am)/asd for r in records])
    nxt=torch.stack([(r[2]-xm)/xs for r in records])
    return obs,act,nxt


def evaluate(model, tensors):
    o,a,y=tensors
    model.eval()
    with torch.no_grad():
        pred=model(o,a)
        mse=(pred-y).square().mean().item()
        mae=(pred-y).abs().mean().item()
    return {"normalized_next_state_mse":round(mse,8),
            "normalized_next_state_mae":round(mae,8)}


def pilot(steps=64,batch_size=8,seed=20261025):
    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    episodes,source=load_authentic_records()
    split_ids=splits(episodes)
    raw={k:windows(episodes,v) for k,v in split_ids.items()}
    norm=normalization(raw["train"])
    tensors={k:pack(v,norm) for k,v in raw.items()}
    # Same prechosen train indices and exposure to each real source window.
    gen=torch.Generator().manual_seed(seed)
    indices=torch.randint(0,len(raw["train"]),(steps,batch_size),generator=gen)
    models={"parallel_TAC_world":ParallelTACWorld(),
            "GRU_world":RealGRUWorld()}
    # Train-only measured state-change RMS; not a target/test-fitted gate.
    delta=(tensors["train"][2]-tensors["train"][0])
    scale=delta.square().mean(dim=(0,1)).sqrt().clamp_min(1e-5)
    for model in models.values():
        model.set_train_delta_scale(scale)
    counts={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()}
    gap=abs(counts["parallel_TAC_world"]-counts["GRU_world"])/counts["GRU_world"]
    if gap>.01:
        raise RuntimeError(f"trainable param mismatch {gap:.4f}>1%")
    output={}
    persist=((tensors["test"][0]-tensors["test"][2])**2).mean().item()
    for key,model in models.items():
        model.train()
        opt=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.001)
        initial=evaluate(model,tensors["validation"])
        for row in indices:
            obs,act,y=(part[row] for part in tensors["train"])
            opt.zero_grad(set_to_none=True)
            loss=(model(obs,act)-y).square().mean()
            if not torch.isfinite(loss).item():
                raise RuntimeError("nonfinite authentic world-model training")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        valid=evaluate(model,tensors["validation"])
        heldout=evaluate(model,tensors["test"])
        output[key]={"parameters":counts[key],"steps":steps,
                     "validation_untrained":initial,
                     "validation_after_steps":valid,
                     "heldout_real_test":heldout}
    result={"status":"REAL_ROBOT_TRAIN_ANCHORED_RESIDUAL_DEV_NO_COMPUTE_PARITY_CLAIM",
       "architecture_revision":"v2_state_anchored_residual",
       "test_cohort_status":"EXPOSED_IN_PREVIOUS_V1_PILOT; DESCRIPTIVE_ONLY",
       "delta_scale_fit_source":"TRAIN_EPISODES_ONLY",
       "train_only_residual_scale":scale.tolist(),
       "data":source,
       "split_episode_ids":split_ids,
       "real_windows_per_split":{k:len(v) for k,v in raw.items()},
       "training_source_exposure_matched":True,
       "optimizer_steps_matched":True,
       "parameters_gap_fraction":round(gap,6),
       "training_FLOPs_matched":False,
       "training_wallclock_equality_required":False,
       "full_fair_head_to_head_claim":False,
       "genuine_test_persistence_mse":round(persist,8),
       "metrics":output,
       "entity_aware_admission_evaluated":False,
       "robot_control_evaluated":False,
       "model_original_source_equivalence":False}
    print(json.dumps(result,sort_keys=True))
    return result


if __name__=="__main__":
    pilot(steps=int(os.environ.get("TAC_REAL_STEPS","64")))
