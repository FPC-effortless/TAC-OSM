#!/usr/bin/env python3
"""Qualify authentic long-case XES source BEFORE any ME070 architecture training.

This script never produces models or opens a research TEST score. It enumerates
real trace-length support and transparent temporal exclusions. Raw publisher
files must be supplied from a lawful source and are never committed to Git.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def attributes(element: ET.Element) -> dict:
    return {child.attrib.get("key"): child.attrib.get("value") for child in element
            if child.tag.rsplit("}", 1)[-1] in {"string", "date", "int", "float", "boolean"}}


def iter_traces(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as inp:
        for _, element in ET.iterparse(inp, events=("end",)):
            if element.tag.rsplit("}", 1)[-1] != "trace":
                continue
            tr = attributes(element)
            events = [attributes(ev) for ev in element
                      if ev.tag.rsplit("}", 1)[-1] == "event"]
            cid = tr.get("concept:name")
            if not cid or not events:
                raise ValueError("trace missing case ID or event sequence")
            times = []
            labels = []
            lifecycle = []
            for ev in events:
                timestamp = ev.get("time:timestamp")
                if not timestamp or not ev.get("concept:name"):
                    raise ValueError("event missing timestamp or activity")
                times.append(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp())
                labels.append(ev["concept:name"])
                lifecycle.append(ev.get("lifecycle:transition", "UNKNOWN").lower())
            if any(b < a for a, b in zip(times, times[1:])):
                raise ValueError("negative observed source event time; no silent reorder")
            yield {"id": cid, "length": len(events), "start": times[0], "end": times[-1],
                   "events_start": lifecycle.count("start"),
                   "events_complete": lifecycle.count("complete"),
                   "trivial_start_to_complete": sum(
                       lifecycle[i] == "start" and lifecycle[i+1] == "complete"
                       and labels[i] == labels[i+1] for i in range(len(events)-1)),
                   "event_pairs": max(0, len(events)-1)}
            element.clear()


def measure(path: Path, min_history: tuple[int, ...] = (64, 128, 256),
            fit_fraction: float = .6, dev_fraction: float = .2) -> dict:
    cases = list(iter_traces(path))
    if len(cases) < 3:
        raise ValueError("insufficient authentic cases to split")
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate original case IDs")
    ordered = sorted(cases, key=lambda c: (c["start"], c["id"]))
    a = int(len(ordered) * fit_fraction)
    b = int(len(ordered) * (fit_fraction + dev_fraction))
    if not (0 < a < b < len(ordered)):
        raise ValueError("empty chronological source cohort")
    fit_cut, test_cut = ordered[a]["start"], ordered[b]["start"]
    cohorts = {"FIT": [c for c in ordered[:a] if c["end"] < fit_cut],
               "DEV": [c for c in ordered[a:b] if c["end"] < test_cut],
               "TEST": ordered[b:]}
    def tally(c):
        return {"cases": len(c), "events": sum(x["length"] for x in c),
                "max_events_per_case": max((x["length"] for x in c), default=0),
                "history_support": {str(h): {
                    "distinct_cases_with_future_after_h": sum(x["length"] > h for x in c),
                    "authentic_next_event_targets_after_h": sum(max(0, x["length"] - h) for x in c)
                } for h in min_history}}
    lengths = sorted(x["length"] for x in cases)
    return {"status": "SOURCE_MEASUREMENT_ONLY_NOT_TRAINING",
            "original_source_sha256": sha256_file(path), "original_source_bytes": path.stat().st_size,
            "case_count": len(cases), "event_count": sum(lengths),
            "mean_trace_events": statistics.mean(lengths),
            "median_trace_events": statistics.median(lengths), "max_trace_events": lengths[-1],
            "source_lifecycle_start_events": sum(x["events_start"] for x in cases),
            "source_lifecycle_complete_events": sum(x["events_complete"] for x in cases),
            "start_immediately_followed_by_same_activity_complete": sum(
                x["trivial_start_to_complete"] for x in cases),
            "source_next_event_pairs": sum(x["event_pairs"] for x in cases),
            "fit_cross_cut_cases_excluded": a - len(cohorts["FIT"]),
            "dev_cross_cut_cases_excluded": b - a - len(cohorts["DEV"]),
            "cohorts": {key: tally(c) for key, c in cohorts.items()},
            "qualification_rule": "For a 256-event study require >=50 FIT and >=25 TEST distinct cases with observable next events after 256; otherwise STOP and select another original real source.",
            "long_256_support_gate": (sum(x["length"] > 256 for x in cohorts["FIT"]) >= 50
                                      and sum(x["length"] > 256 for x in cohorts["TEST"]) >= 25),
            "restrictions": ["No raw case IDs, customer records or event source file committed",
                             "A maximum event count alone never qualifies the cohort",
                             "This report cannot establish causal action effects or model skill"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path,
                        help="Legally obtained original .xes or .xes.gz file")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.input.exists():
        parser.error("actual original source file required; refusing synthetic stand-in")
    report = measure(args.input)
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
