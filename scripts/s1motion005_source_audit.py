#!/usr/bin/env python3
"""S1MOTION005: immutable LeRobot v2 source-only FIT audit, NO training.

Downloads FIT episodes 0..24, not DEV 25..34 or TEST 35..49.
Publisher metadata alone describes withheld episode counts.
"""
from __future__ import annotations
import hashlib, io, json, sys, urllib.request
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

SOURCE="lerobot/aloha_static_screw_driver"
REV="0b873be59c9702f8d3de89022617121970db112f"
FIT=tuple(range(25)); DEV=tuple(range(25,35))
DATA_PATH="data/chunk-000/episode_{episode:06d}.parquet"
FIELDS=("episode_index","frame_index","timestamp","observation.state","action")

def download(url):
    req=urllib.request.Request(url,headers={"User-Agent":"S1MOTION005-source-only-audit/1.0"})
    with urllib.request.urlopen(req,timeout=120) as r:
        return r.read()

def audit(source_cache:Path):
    source_cache.mkdir(parents=True,exist_ok=True)
    api=json.loads(download(f"https://huggingface.co/api/datasets/{SOURCE}/revision/{REV}"))
    sha=api.get("sha","")
    if sha!=REV:
        raise ValueError("specific approved immutable v2 revision unavailable")
    if len(sha)!=40 or any(ch not in "0123456789abcdef" for ch in sha):
        raise ValueError("unresolved publisher immutable revision")
    historical_license=str((api.get("cardData") or {}).get("license","")).lower()
    current_publisher=json.loads(download(f"https://huggingface.co/api/datasets/{SOURCE}"))
    publisher_current_license=str(
        (current_publisher.get("cardData") or {}).get("license","")).lower()
    if (historical_license not in ("","mit") or publisher_current_license!="mit"):
        raise ValueError("historical license conflict or current publisher MIT card unavailable")
    # The v2 commit may lack a license card field: do not pretend it did.
    license_id="mit"
    license_evidence_scope=(
        "historical_revision_card_and_current_card" if historical_license=="mit"
        else "current_publisher_card_ONLY_historical_revision_metadata_missing")
    info=json.loads(download(f"https://huggingface.co/datasets/{SOURCE}/resolve/{sha}/meta/info.json"))
    if (info.get("total_episodes")!=50 or info.get("total_frames")!=20000
        or info.get("fps")!=50 or info.get("data_path")!=
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"):
        raise ValueError(f"unqualified recorded source info: {info.get('codebase_version')}")
    known_paths={x["rfilename"] for x in api.get("siblings",[])}
    if any(DATA_PATH.format(episode=ep) not in known_paths for ep in FIT):
        raise RuntimeError("required original FIT source episode missing")
    receipts=[]
    for ep in FIT:
        path=DATA_PATH.format(episode=ep)
        raw=download(f"https://huggingface.co/datasets/{SOURCE}/resolve/{sha}/{path}")
        digest=hashlib.sha256(raw).hexdigest()
        (source_cache / f"fit-{ep:02d}.parquet").write_bytes(raw)
        tbl=pq.read_table(io.BytesIO(raw),columns=list(FIELDS))
        if tbl.num_rows!=400:
            raise RuntimeError(f"unexpected episode {ep} length")
        d=tbl.to_pydict()
        last=None
        for i,(epid,frame,timestamp,x,u) in enumerate(zip(*(d[k] for k in FIELDS))):
            x=np.asarray(x,dtype=np.float64);u=np.asarray(u,dtype=np.float64)
            if (int(epid)!=ep or int(frame)!=i or x.shape!=(14,) or
                u.shape!=(14,) or not np.isfinite(x).all() or
                not np.isfinite(u).all() or not np.isfinite(float(timestamp))):
                raise RuntimeError(f"FIT source shape/time/identity mismatch ep={ep} frame={i}")
            if last is not None and not 0 < float(timestamp)-last < .2:
                raise RuntimeError(f"FIT timestamp monotonicity failed ep={ep}")
            last=float(timestamp)
        receipts.append(dict(episode=ep,path=path,sha256=digest,frames=400,
                             observed_state_dim=14,recorded_action_dim=14))
    manifest=dict(study="S1MOTION005",dataset=SOURCE,license=license_id,
        revision_sha=sha,source_codebase=info.get("codebase_version"),
        license_evidence_scope=license_evidence_scope,
        historical_revision_license_field=historical_license or None,
        historical_v2_release_explicitly_pinned=True,
        original_live_source_revision_rejected_as_unapproved_v3=True,
        publisher_frames=20000,publisher_episodes=50,
        publisher_fps=50,episode_frame_count=400,
        FIT=list(FIT),DEV=list(DEV),protected_TEST=list(range(35,50)),
        fit_files=receipts,fit_only_real_values_inspected=True,
        no_dev_or_test_episode_file_was_downloaded=True,
        no_optimizer_step=True,no_evaluation_or_model_score=True)
    print("S1MOTION005_ORIGINAL_SOURCE_LOCK",json.dumps(manifest,sort_keys=True),flush=True)
    return manifest

if __name__=="__main__":
    out=Path("research/diagnostics/s1motion005-source-original.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    receipt=audit(Path("/tmp/s1motion005-fit-screw-driver"))
    out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print("S1MOTION005_SOURCE_ONLY_AUDIT_PASS",flush=True)
