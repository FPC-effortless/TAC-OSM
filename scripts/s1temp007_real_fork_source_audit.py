#!/usr/bin/env python3
"""S1TEMP007 prospective new real ALOHA coffee source-only authorization.

No optimizer, no model, no DEV/TEST rows accessed: only publisher metadata
and FIT episodes0..24. Source git revision is frozen as an OUTPUT receipt
and must be COMMITTED in Git-owned lock before ANY study optimizer step.
"""
from __future__ import annotations
import hashlib,io,json,math,urllib.request
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

SOURCE="lerobot/aloha_static_fork_pick_up"
FIT=tuple(range(50));DEV=tuple(range(50,70))
TEST=tuple(range(70,100))
COLUMNS=("episode_index","frame_index","timestamp","observation.state","action")

def get(url):
    with urllib.request.urlopen(urllib.request.Request(
        url,headers={"User-Agent":"S1TEMP007-TEMP-real-source-FIT-audit/1.0"}),
        timeout=120) as r:
        return r.read()

def audit():
    latest=json.loads(get(f"https://huggingface.co/api/datasets/{SOURCE}"))
    live_sha=latest.get("sha")
    if not isinstance(live_sha,str) or len(live_sha)!=40:
        raise RuntimeError("live publisher commit inaccessible")
    # Schema-only deterministic choice of NEWEST historical v2 release
    # according to publisher commit order, before any FIT data bytes,
    # labels, outcome scores or optimizer steps are ever opened.
    commits=json.loads(get(f"https://huggingface.co/api/datasets/{SOURCE}/commits/main"))
    historical=None;info=None;sha=None
    examined=[]
    for candidate in commits:
        rev=candidate.get("id")
        if not isinstance(rev,str) or len(rev)!=40:
            continue
        metadata=json.loads(get(
            f"https://huggingface.co/api/datasets/{SOURCE}/revision/{rev}"))
        if metadata.get("sha")!=rev:
            raise RuntimeError("publisher history integrity failure")
        try:
            meta=json.loads(get(
                f"https://huggingface.co/datasets/{SOURCE}/resolve/{rev}/meta/info.json"))
        except Exception:
            examined.append(dict(sha=rev,info_found=False))
            continue
        expected=(meta.get("codebase_version")=="v2.0"
            and meta.get("total_episodes")==100
            and meta.get("total_frames")==60000
            and meta.get("fps")==50
            and meta.get("data_path")==
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        examined.append(dict(sha=rev,codebase_version=meta.get("codebase_version"),
                             expected_schema=expected))
        if expected:
            historical,info,sha=metadata,meta,rev
            break
    if historical is None or info is None:
        raise RuntimeError("no publisher-verified v2 historical revision: source ineligible")
    historical_license=str((historical.get("cardData") or {}).get("license","")).lower()
    current_license=str((latest.get("cardData") or {}).get("license","")).lower()
    if (historical_license not in ("","apache-2.0") or current_license!="apache-2.0"):
        raise RuntimeError("license conflict; abort")
    license_id="apache-2.0"
    license_evidence=("historical_v2_AND_current" if historical_license=="mit"
                     else "current_publisher_card_only_historical_missing")
    known={x["rfilename"] for x in historical.get("siblings",[])}
    receipts=[]
    for ep in FIT:
        filename=f"data/chunk-000/episode_{ep:06d}.parquet"
        if filename not in known:raise RuntimeError(f"FIT path missing {filename}")
        blob=get(f"https://huggingface.co/datasets/{SOURCE}/resolve/{sha}/{filename}")
        table=pq.read_table(io.BytesIO(blob),columns=list(COLUMNS))
        if table.num_rows!=600:raise RuntimeError(f"FIT real episode length changed: {ep}:{table.num_rows}")
        cols=table.to_pydict();last=None
        for i,(episode,frame,clock,x,u) in enumerate(zip(
              *(cols[k] for k in COLUMNS))):
            x=np.asarray(x,dtype=np.float64);u=np.asarray(u,dtype=np.float64)
            if (int(episode)!=ep or int(frame)!=i
                or x.shape!=(14,) or u.shape!=(14,)
                or not np.isfinite(x).all() or not np.isfinite(u).all()
                or not math.isfinite(float(clock))):
                raise RuntimeError(f"wrong frame identity or numerical data ep={ep} frame={i}")
            if last is not None and not (0<float(clock)-last<.2):
                raise RuntimeError(f"out of clock sequence ep={ep} frame={i}")
            last=float(clock)
        receipts.append(dict(episode=ep,path=filename,
            sha256=hashlib.sha256(blob).hexdigest(),frames=600,
            state_dim=14,action_dim=14))
    report=dict(study="S1TEMP007",dataset=SOURCE,revision_sha=sha,
        historical_revision_license=historical_license or None,
        license_evidence_scope=license_evidence,license_current_publisher=license_id,
        source_live_head_sha=live_sha,source_revision_schema_audit=examined,
        codebase_version=info["codebase_version"],fps=50,
        source_publisher_episodes=50,source_publisher_total_frames=60000,
        FIT=list(FIT),DEV=list(DEV),TEST=list(TEST),
        expected_frames_each=600,FIT_receipts=receipts,
        FIT_episodes_downloaded=25,DEV_episode_files_downloaded=0,
        TEST_episode_files_downloaded=0,
        no_optimizer_steps=True,no_model_eval=True,
        source_revision_FROZEN_BEFORE_NEXT_STAGE=True)
    out=Path("research/diagnostics/s1temp007-source-original")
    out.mkdir(parents=True,exist_ok=True)
    (out/"source-lock.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("S1TEMP007_ORIGINAL_FIT_SOURCE_LOCK",json.dumps(report,sort_keys=True),flush=True)
    print("S1TEMP007_FIT_SOURCE_ONLY_PASS",flush=True)

if __name__=="__main__":audit()
