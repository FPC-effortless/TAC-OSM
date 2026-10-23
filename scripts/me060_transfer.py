"""Metadata-register and evaluate frozen ME060 models on a different robot task."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import numpy as np
import torch

from scripts.audit_real_hil_trajectory import download_file, audit_frame_table
from scripts.me055_reserve_independent_robot_task import CANDIDATES, REVISION, metadata_rows
from scripts.me060_train_trajectory import ROOT, HISTORY, HORIZON, digest, build, evaluate
from scripts.me057_train_fused_anchor_robot import sha_bytes, sha_file, normalize, write_json


def reserve(cache,root=ROOT,tag="me060",excluded=()):
    import hashlib
    available=[t for t in CANDIDATES if t!="clean_stains_off_tabletop" and t not in excluded]
    task=min(available,key=lambda t:hashlib.sha256(f"{REVISION}:{tag}-transfer:v1:{t}".encode()).hexdigest())
    info=download_file(REVISION,f"{task}/meta/info.json",cache/task/"meta/info.json",1_000_000)
    episodes=download_file(REVISION,f"{task}/meta/episodes.jsonl",cache/task/"meta/episodes.jsonl",2_000_000)
    metadata=json.loads(info);rows=metadata_rows(episodes)
    if metadata.get("robot_type")!="r1_lite" or metadata.get("fps")!=15:
        raise RuntimeError("different physical interface requires separate qualification")
    blocks=sorted({e//4 for e,n in rows if n>HISTORY+HORIZON},key=lambda b:digest([task,"me060-transfer-block",b]))[:20]
    cohort=[(e,n) for e,n in rows if e//4 in blocks and n>HISTORY+HORIZON]
    plan={"study":tag.upper()+"_FROZEN_DIFFERENT_TASK_TRANSFER","task":task,"revision":REVISION,
        "selection":f"minimum SHA256(revision:{tag}-transfer:v1:task), excluding source and prior targets; 20 hashed adjacent-episode blocks",
        "info_sha256":sha_bytes(info),"episodes_sha256":sha_bytes(episodes),"cohort":cohort,
        "cohort_sha256":digest(cohort),"trajectory_files_opened":0,"training_updates":0,
        "checkpoint_selection":"all ten source final checkpoints; no target normalization/calibration or model selection",
        "scoring":"same midpoint robot-controlled origin, 64 actual history transitions,16 supplied factual actions, pre-takeover physical MAE",
        "gate":"TAC >=2% lower MAE than GRU and both priors; episode paired TAC-GRU 95% interval upper bound below0",
        "scope":"different authentic task in this study, not a different robot/site/operator; causal policy value unsupported",
        "registration_before_target_trajectory_access":True}
    write_json(root/"TRANSFER_PROTOCOL.json",plan)
    print(json.dumps({"task":task,"episodes":len(cohort),"trajectory_files_opened":0}),flush=True)


def run(cache,folder,root=ROOT,model_factory=build):
    import pyarrow.parquet as pq
    torch.set_num_threads(1)
    plan=json.loads((root/"TRANSFER_PROTOCOL.json").read_text());task=plan["task"]
    report=json.loads((folder/"results.json").read_text());records=report["checkpoints"]
    if len(records)!=10:raise RuntimeError("all ten frozen source models required")
    for r in records:
        if sha_file(folder/r["path"])!=r["measured"]["checkpoint_sha256"]:
            raise RuntimeError("source checkpoint changed before transfer")
    meta_raw=download_file(REVISION,f"{task}/meta/episodes.jsonl",cache/task/"meta/episodes.jsonl",2_000_000)
    if sha_bytes(meta_raw)!=plan["episodes_sha256"]:raise RuntimeError("target metadata changed")
    meta={r["episode_index"]:r for r in map(json.loads,meta_raw.splitlines()) if r}
    raw={}
    for number,(e,n) in enumerate(plan["cohort"],1):
        relative=f"{task}/data/chunk-{e//1000:03d}/episode_{e:06d}.parquet"
        data=download_file(REVISION,relative,cache/relative,15_000_000)
        table=pq.read_table(io.BytesIO(data));audit_frame_table(table,meta[e])
        if len(table)!=n:raise RuntimeError("target length changed")
        raw[e]={"state":np.asarray(table["observations.state.qpos"].to_pylist(),dtype=np.float32),
            "action":np.asarray(table["action.qpos"].to_pylist(),dtype=np.float32),
            "human":np.asarray(table["intervention"].to_pylist(),dtype=np.bool_),
            "length":n,"sha256":sha_bytes(data)}
        if number%20==0:print(json.dumps({"target_episodes":number,"total":len(plan["cohort"])}),flush=True)
    predictions={}
    for r in records:
        checkpoint=torch.load(folder/r["path"],weights_only=False)
        identified={k:np.asarray(v,dtype=np.float32) for k,v in checkpoint["identified"].items()}
        model=model_factory(r["arm"],identified);model.load_state_dict(checkpoint["model"])
        predictions[f'{r["arm"]}-s{r["seed"]}']=evaluate(model,normalize(raw,identified),identified)
    write_json(folder/"TRANSFER_RESULTS.json",{"protocol":plan,"target_source_sha256":digest({str(e):r["sha256"] for e,r in sorted(raw.items())}),
        "source_checkpoint_hashes":[r["measured"]["checkpoint_sha256"] for r in records],
        "target_training_updates":0,"target_preprocessing_fitted":False,"evaluation":predictions})
    print("ME060_TRANSFER_COMPLETE",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source-cache",type=Path,required=True)
    p.add_argument("--register",action="store_true");p.add_argument("--output",type=Path,default=Path("results/me060"))
    a=p.parse_args()
    if a.register:reserve(a.source_cache)
    else:run(a.source_cache,a.output)
