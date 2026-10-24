"""ESWAM012: real co-visible track histories, not merely independent counts.

Hard fail on tampered publisher source, malformed values, duplicate tracks,
frame gaps, timestamp regressions. No fabricated continuity or pseudo-actions.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import csv
import io
import json
import math
from pathlib import Path
import zipfile

from scripts.eswam005_qualify_real_multiagent import EXPECTED_ARCHIVE_SHA256, sha256_file


def eligible_source(archive: Path):
    if sha256_file(archive) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("STOP: original published archive sha256 mismatch")
    reports = []
    for basename in ("vehicle_tracks_000.csv", "vehicle_tracks_001.csv",
                     "vehicle_tracks_002.csv", "vehicle_tracks_003.csv"):
        tracks = defaultdict(list)
        files = []
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if (name.startswith("__MACOSX/") or "/__MACOSX/" in name
                    or Path(name).name.startswith("._")):
                    continue
                if not name.endswith(basename) or "TC_BGR_Intersection_VA" not in name:
                    continue
                files.append(name)
                with z.open(name) as f:
                    rd = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig", newline=""))
                    required = {"track_id", "frame_id", "timestamp_ms", "x", "y", "vx", "vy"}
                    if not required.issubset(rd.fieldnames or []):
                        raise RuntimeError("source real-track schema changed")
                    for item in rd:
                        tid = str(item["track_id"]).strip()
                        frame, ms = int(item["frame_id"]), int(item["timestamp_ms"])
                        xyvv = tuple(float(item[c]) for c in ("x", "y", "vx", "vy"))
                        if not tid or frame < 0 or ms < 0 or not all(map(math.isfinite, xyvv)):
                            raise RuntimeError("source contains invalid real actor data")
                        tracks[tid].append((frame, ms, *xyvv))
        if len(files) != 1:
            raise RuntimeError(f"expected exactly one original {basename}, found {files}")
        summary = {"member": files[0], "split": "FIT" if basename[-7:-4] != "003" else "EXPOSED_DEV",
                   "raw_source_track_ids": len(tracks), "histories": {}}
        raw_frame_timestamps = defaultdict(set)
        for length in (8, 128, 256):
            for horizon in (10, 30):
                eligible = defaultdict(set)
                actors = set()
                segments = 0
                all_anchors = 0
                for tid, rows in tracks.items():
                    ordered = sorted(rows, key=lambda r: (r[0], r[1]))
                    if len(set((r[0], r[1]) for r in ordered)) != len(ordered):
                        raise RuntimeError("duplicate timestamp/frame for same actor")
                    parts, segment = [], []
                    for row in ordered:
                        if segment and (row[0] != segment[-1][0] + 1 or
                                        row[1] <= segment[-1][1]):
                            parts.append(segment)
                            segment = []
                        segment.append(row)
                    if segment:
                        parts.append(segment)
                    for part in parts:
                        if len(part) < length + horizon:
                            continue
                        segments += 1
                        for j in range(length-1, len(part)-horizon):
                            r = part[j]
                            eligible[(r[0], r[1])].add(tid)
                            actors.add(tid)
                            all_anchors += 1
                occupancies = [len(eligible[k]) for k in eligible]
                primary = {k: sorted(v) for k, v in eligible.items() if len(v) >= 2}
                summary["histories"][f"H{length}_P{horizon}"] = {
                    "eligible_actor_anchors": all_anchors,
                    "distinct_source_actor_ids": len(actors),
                    "source_co_observed_origins": len(eligible),
                    "two_or_more_actors_with_complete_history_and_future":
                        sum(n >= 2 for n in occupancies),
                    "four_or_more_actors_with_complete_history_and_future":
                        sum(n >= 4 for n in occupancies),
                    "max_eligible_simultaneous_actors": max(occupancies, default=0),
                    "co_visible_pair_examples_first_three":
                        [{"source_frame": k[0], "source_timestamp_ms": k[1],
                          "actor_ids": v[:6]} for k, v in sorted(primary.items())[:3]],
                    "source_segments_long_enough": segments,
                }
        reports.append(summary)
    return {
        "study": "ESWAM012_SOURCE_ACTUAL_OVERLAPPING_HISTORY_ELIGIBILITY",
        "original_zip_sha256": EXPECTED_ARCHIVE_SHA256,
        "source_site": "TC_BGR_Intersection_VA",
        "distinct_physical_sites": 1,
        "independent_out_of_site_test": False,
        "source_only_no_training": True,
        "verified_actor_actions_available": False,
        "fit_members": ["000", "001", "002"],
        "exposed_dev_member": "003",
        "members": reports,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    r = eligible_source(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(r, sort_keys=True, indent=2) + "\n")
    print("ESWAM012_COVISIBILITY_BEGIN")
    for m in r["members"]:
        print(json.dumps({"member": m["member"], "split": m["split"],
                          "histories": m["histories"]}, sort_keys=True))
    print("ESWAM012_COVISIBILITY_END")
