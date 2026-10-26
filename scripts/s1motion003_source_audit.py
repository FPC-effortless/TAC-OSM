#!/usr/bin/env python3
"""S1MOTION003 source eligibility only; does not fit or DEV/TEST score models."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pyarrow.parquet as pq

DATASET = "lerobot/aloha_static_tape"
DATA_PATH = "data/chunk-000/file-000.parquet"
FIT_EPISODES = frozenset(range(25))


def fetch(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "TAC-OSM-S1MOTION003-source-audit/1.0"})
    with urllib.request.urlopen(req, timeout=120) as stream:
        return stream.read()


def audit(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    info = json.loads(fetch(f"https://huggingface.co/api/datasets/{DATASET}"))
    sha = info.get("sha", "")
    license_id = str(info.get("cardData", {}).get("license", "")).lower()
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise RuntimeError("unresolved immutable publisher SHA")
    if license_id != "apache-2.0":
        raise RuntimeError(f"unverified dataset license {license_id}")
    revision = json.loads(fetch(
        f"https://huggingface.co/api/datasets/{DATASET}/revision/{sha}"))
    if revision.get("sha") != sha:
        raise RuntimeError("snapshot revision changed")
    siblings = {s["rfilename"] for s in revision.get("siblings", [])}
    if DATA_PATH not in siblings:
        raise RuntimeError("unsupported versioned parquet layout, abort before fit: "
                           + repr(sorted(v for v in siblings if v.endswith(".parquet"))))
    meta = json.loads(fetch(
        f"https://huggingface.co/datasets/{DATASET}/resolve/{sha}/meta/info.json"))
    if (meta.get("total_episodes") != 50 or meta.get("fps") != 50 or
        meta.get("data_path") !=
          "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"):
        raise RuntimeError("unexpected publisher source dataset contract")
    b = fetch(f"https://huggingface.co/datasets/{DATASET}/resolve/{sha}/{DATA_PATH}")
    digest = hashlib.sha256(b).hexdigest()
    if not b:
        raise RuntimeError("missing real parquet")
    parquet_file = cache / "tape-source.parquet"
    parquet_file.write_bytes(b)
    columns = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
    table = pq.read_table(parquet_file, columns=columns)
    counts = {}
    prev = {}
    # Metadata-only all-episode count: does not evaluate future outcomes.
    # FIT finite/time checks touch only actual source episodes 0..24.
    data = table.to_pydict()
    for ep, frame, ts, x, action in zip(*(data[k] for k in columns)):
        ep = int(ep)
        counts[ep] = counts.get(ep, 0) + 1
        if ep not in FIT_EPISODES:
            continue
        if not (len(x) == 14 and len(action) == 14):
            raise RuntimeError("bad fitting-source joint/action dimensions")
        if not np.isfinite(np.asarray(x + action + [ts], dtype=float)).all():
            raise RuntimeError("nonfinite fitting-source observations")
        if ep in prev:
            prior_frame, prior_time = prev[ep]
            if int(frame) != prior_frame + 1 or not 0 < ts-prior_time < .2:
                raise RuntimeError("FIT broken chronology")
        prev[ep] = int(frame), float(ts)
    if set(counts) != set(range(50)) or min(counts.values()) < 257:
        raise RuntimeError("publisher episode identity/length mismatch")
    report = dict(study="S1MOTION003",source=DATASET,publisher_sha=sha,
        publisher_license=license_id,publisher_data_path=DATA_PATH,
        publisher_parquet_sha256=digest,full_source_bytes=len(b),
        publisher_total_frames=table.num_rows,fit_episode_count=25,
        all_episode_frame_counts=counts,fit_minimum_frames=min(counts[k] for k in FIT_EPISODES),
        source_shape_state=14,source_shape_action=14,
        fit_frame_continuity_verified=True,
        dev_test_numeric_values_not_scored=True,
        entire_publisher_parquet_downloaded=True,
        no_model_fit_or_forecast=True,
        license_and_exact_revision_locked_before_model_fit=True)
    print("S1MOTION003_SOURCE_ELIGIBILITY",json.dumps(report,sort_keys=True),flush=True)
    return report


if __name__ == "__main__":
    dest = Path("research/diagnostics/s1motion003-source-audit.json")
    dest.parent.mkdir(parents=True,exist_ok=True)
    result=audit(Path("/tmp/s1motion003-source"))
    dest.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("S1MOTION003_SOURCE_PREFLIGHT_PASS",flush=True)
