#!/usr/bin/env python3
"""Retrospective ME069 exposed-case lifecycle-label audit on original verified BPI2012.

No model retraining, architecture tuning, or unexposed TEST claim. The original
source loader checks publisher MD5/SHA256 and reconstructs original source cuts.
Counts come solely from actual recorded events, never simulated labels.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from scripts.me069_workflow_source import load


def lifecycle(label: str) -> tuple[str, str]:
    if "|" not in label:
        return label, "missing"
    activity, phase = label.rsplit("|", 1)
    return activity, phase.lower().strip()


def audit(cases, vocab: dict) -> dict:
    back = {int(v): str(k) for k, v in vocab.items()}
    labels, phases, sequences = Counter(), Counter(), Counter()
    identical_base = starts_to_complete = fixed_current_label = 0
    lengths = []
    count = 0
    for c in cases:
        n = int(c["n"])
        if n < 2:
            continue
        i = (n - 2) // 2
        current = back.get(int(c["activity"][i]), "UNKNOWN")
        target = back.get(int(c["activity"][i + 1]), "UNKNOWN")
        ca, cl = lifecycle(current)
        ta, tl = lifecycle(target)
        count += 1
        lengths.append(n)
        labels[target] += 1
        phases[tl] += 1
        sequences[f"{cl}->{tl}"] += 1
        identical_base += ca == ta
        starts_to_complete += (cl == "start" and tl == "complete" and ca == ta)
        fixed_current_label += current == target
    if count == 0:
        raise ValueError("no eligible recorded next-event target")
    total = sum(labels.values())
    entropy = -sum((n / total) * math.log(n / total) for n in labels.values())
    return {
        "eligible_midpoint_cases": count,
        "mean_original_events_per_case": sum(lengths)/len(lengths),
        "original_trace_length_max": max(lengths),
        "next_label_unique_count": len(labels),
        "next_label_entropy_nats": entropy,
        "label_majority_accuracy": max(labels.values()) / total,
        "most_common_next_labels": labels.most_common(12),
        "target_lifecycle_counts": dict(phases),
        "preceding_to_target_lifecycle_counts": dict(sequences),
        "same_underlying_activity_fraction": identical_base / count,
        "start_to_same_activity_complete_fraction": starts_to_complete / count,
        "exactly_repeat_current_activity_label_fraction": fixed_current_label / count,
        "interpretation": "Exposed original ME069 target-composition diagnostic; labels split as concept:name|lifecycle; not a new architecture test."
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("/tmp/me070-source"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    cohorts, preprocessing, source = load(args.cache)
    report = {
        "scope": "POSTHOC_EXPOSED_ME069_TEST_NOT_NEW_PREDICTION_EVIDENCE",
        "authentic_original_source_sha256": source["source_sha256"],
        "cohorts": {name: audit(cases, preprocessing["activities"])
                    for name, cases in cohorts.items()},
    }
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content)


if __name__ == "__main__":
    main()
