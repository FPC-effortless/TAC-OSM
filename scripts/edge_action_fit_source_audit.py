"""Real ALOHA FIT-only action-time eligibility diagnostic; NEVER trains/scores TEST.

The original dataset exposes 'action', but source metadata alone does not prove
that this is an executed plant command in the interval [t,t+1]. This audit
verifies chronological observability, shape, provenance and alignment only.
It must NOT be used as an empirical capability win or causal identification.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import numpy as np
import pyarrow.parquet as pq

REV="7a752b39f7e69de7e38aee485a6bab07528a061a"
SHA256="a3409d3282b2cf9a94d32b301756fbd15dca5862cc1c32973c7316dd8853c57a"
URL=f"https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet/resolve/{REV}/data/chunk-000/file-000.parquet"
FIT_EPISODE_STOP=59   # Immutable ME099 TRAIN partition: source episodes [0,58].
FIELDS=("episode_index","frame_index","timestamp","observation.state","action")


def audit(path:Path):
    if not path.exists():
        with urllib.request.urlopen(URL,timeout=120) as stream:
            with path.open("wb") as out:
                while True:
                    part=stream.read(1<<20)
                    if not part:break
                    out.write(part)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA256:
        raise RuntimeError("source bytes mismatch: refuse action eligibility claim")
    # Predicate restricts all returned numerical rows to the OLD original FIT
    # episode partition. No DEV or TEST outcome, calibration or score is read.
    data=pq.read_table(path,columns=list(FIELDS),filters=[("episode_index","<",FIT_EPISODE_STOP)]).to_pydict()
    if not data["episode_index"]:raise RuntimeError("missing original FIT")
    groups={}
    for ep,frame,t,x,a in zip(*(data[k] for k in FIELDS)):
        if int(ep)<0 or int(ep)>=FIT_EPISODE_STOP:raise RuntimeError("non-FIT episode leaked")
        if len(x)!=14 or len(a)!=14 or not np.isfinite(x+a+[t]).all():
            raise RuntimeError("nonfinite/invalid source row")
        groups.setdefault(int(ep),[]).append((int(frame),float(t),np.asarray(x),np.asarray(a)))
    if len(groups)!=FIT_EPISODE_STOP or set(groups)!=set(range(FIT_EPISODE_STOP)):
        raise RuntimeError("original FIT partition differs; do not infer action alignment")
    dt=[];last_obs=[];next_obs=[];current_action=[];previous_action=[]
    for ep,rows in sorted(groups.items()):
        rows.sort(key=lambda row:row[0])
        for previous,current,nxt in zip(rows[:-2],rows[1:-1],rows[2:]):
            if current[0]!=previous[0]+1 or nxt[0]!=current[0]+1 or not(
                0<current[1]-previous[1]<.2 and 0<nxt[1]-current[1]<.2):
                raise RuntimeError(f"nonchronological published frames in FIT episode {ep}")
            dt.append(nxt[1]-current[1])
            last_obs.append(current[2]);next_obs.append(nxt[2])
            previous_action.append(previous[3]);current_action.append(current[3])
    x=np.stack(last_obs);y=np.stack(next_obs);prev=np.stack(previous_action);a=np.stack(current_action)
    result={
        "status":"REAL_FIT_SOURCE_ALIGNMENT_ONLY_NO_TRAIN_NO_PREDICTIVE_SKILL",
        "sha256":SHA256,
        "source_revision":REV,
        "fit_episodes":len(groups),
        "eligible_t_to_tplus1_transitions":len(dt),
        "median_seconds_between_frames":float(np.median(dt)),
        "same_event_state_dim":int(x.shape[1]),
        "candidate_action_dim":int(a.shape[1]),
        "all_returned_rows_fit_only":True,
        "candidate_action_index":"action[t] same original recorded frame as state[t]",
        "history_action_index":"action[t-1] from preceding observed source frame",
        "outcome_index":"observation.state[t+1] source frame",
        "mean_abs_next_delta":float(np.abs(y-x).mean()),
        "mean_abs_published_action_minus_state":float(np.abs(a-x).mean()),
        "source_command_executed_at_t_verified":False,
        "action_identifiability_verified":False,
        "protected_dev_test_outcomes_scored":False,
        "qualification":"BLOCKED_PENDING_DOCUMENTED_MEANING_OF_ACTION_AND_FROZEN_REAL_DATA_BUDGET"
    }
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",default="/tmp/aloha-action-fit-only.parquet")
    args=p.parse_args()
    print(json.dumps(audit(Path(args.data)),indent=2,sort_keys=True))
