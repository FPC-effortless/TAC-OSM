#!/usr/bin/env python3
"""Read-only ME-069 TEST/DEV case-level audit; does NOT retrain or requalify its exposed TEST.

Only original Git-tracked REAL_EVALUATION.json is read. Print measured accuracy
overlap and McNemar paired tests *per fitted seed*, plus distribution of forecast
interval widths. Never pool the five rescorings of the same cases as independent.
The stored ME069 scores lack label identities and lower/upper forecast quantiles:
those questions remain BLOCKED unless original raw XES is independently obtained.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path


def exact_mcnemar_p(tac_only: int, gru_only: int) -> float:
    """Exact two-sided binomial McNemar for discordant paired cases."""
    n = tac_only + gru_only
    if n == 0:
        return 1.0
    k = min(tac_only, gru_only)
    # Stable tail recurrence in floating point for all realistic cohort sizes.
    # Use binomial log terms to avoid integer/float overflow for large n.
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1)
            - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(logs)
    return min(1.0, 2.0 * math.exp(m) * sum(math.exp(x - m) for x in logs))


def audited_pair(native: list[dict], gru: list[dict]) -> dict:
    if len(native) != len(gru) or len(native) == 0:
        raise ValueError("missing or unequal scored case counts")
    a = {(str(x["case_id"]), int(x["prefix_last_index"])): x for x in native}
    b = {(str(x["case_id"]), int(x["prefix_last_index"])): x for x in gru}
    if len(a) != len(native) or len(b) != len(gru) or a.keys() != b.keys():
        raise ValueError("duplicates or unpaired case/prefix keys")
    keys = sorted(a)
    both_correct = both_wrong = tac_only = gru_only = 0
    for key in keys:
        x, y = bool(a[key]["activity_correct"]), bool(b[key]["activity_correct"])
        both_correct += x and y
        both_wrong += not x and not y
        tac_only += x and not y
        gru_only += not x and y
    if both_correct + both_wrong + tac_only + gru_only != len(keys):
        raise AssertionError("confusion counts do not exhaust cases")

    def widths(rows):
        val = [float(v["interval_width_seconds"]) / 3600 for v in rows]
        if not all(math.isfinite(v) and v >= 0 for v in val):
            raise ValueError("invalid stored interval widths")
        sorted_widths = sorted(val)
        n = len(val)
        q = lambda p: sorted_widths[max(0, math.ceil(p * n)-1)]
        return {
            "median_hours": statistics.median(val),
            "mean_hours": statistics.mean(val),
            "p90_hours": q(.9), "p99_hours": q(.99), "maximum_hours": sorted_widths[-1],
            "fraction_width_gt_168h": sum(v > 168 for v in val) / len(val),
            "fraction_width_le_24h": sum(v <= 24 for v in val) / len(val),
        }

    return {
        "case_count": len(keys),
        "both_correct": both_correct, "both_wrong": both_wrong,
        "native_only_correct": tac_only, "gru_only_correct": gru_only,
        "different_decisions_at_least": tac_only + gru_only,
        "accuracy_native": (both_correct + tac_only) / len(keys),
        "accuracy_gru": (both_correct + gru_only) / len(keys),
        "exact_mcnemar_p_two_sided": exact_mcnemar_p(tac_only, gru_only),
        "interval_widths_native": widths([a[k] for k in keys]),
        "interval_widths_gru": widths([b[k] for k in keys]),
        "positive_wait_rate": statistics.mean(bool(a[k]["positive_wait_observed"]) for k in keys),
        "coverage_positive_native": statistics.mean(
            bool(a[k]["interval_90_covered"]) for k in keys if a[k]["positive_wait_observed"]),
        "coverage_positive_gru": statistics.mean(
            bool(b[k]["interval_90_covered"]) for k in keys if b[k]["positive_wait_observed"]),
        "median_width_positive_native_hours": statistics.median(
            float(a[k]["interval_width_seconds"])/3600 for k in keys if a[k]["positive_wait_observed"]),
        "median_width_positive_gru_hours": statistics.median(
            float(b[k]["interval_width_seconds"])/3600 for k in keys if b[k]["positive_wait_observed"]),
        "coverage_native": statistics.mean(bool(a[k]["interval_90_covered"]) for k in keys),
        "coverage_gru": statistics.mean(bool(b[k]["interval_90_covered"]) for k in keys),
        "mean_logloss_native": statistics.mean(float(a[k]["activity_NLL"]) for k in keys),
        "mean_logloss_gru": statistics.mean(float(b[k]["activity_NLL"]) for k in keys),
    }


def audit(payload: dict) -> dict:
    if not isinstance(payload.get("scores"), dict):
        raise ValueError("expected immutable ME069 per-case score structure")
    report = {
        "scope": "POST-HOC EXPOSED ME069 score audit only; no new TEST promotion",
        "lifecycle_label_prevalence": "NOT AVAILABLE in committed case-score schema",
        "interval_score_and_pinball": "NOT AVAILABLE: true wait and endpoints not committed in case scores",
        "interpretation": "Five seeds rescore same cases; per-seed McNemar is descriptive, not a fivefold independent trial.",
        "cohorts": {}
    }
    for cohort in ("DEV", "TEST"):
        scores = payload["scores"].get(cohort, {})
        output = {}
        for seed in range(69001, 69006):
            a, b = f"native_tac-s{seed}", f"gru_world-s{seed}"
            if a not in scores or b not in scores:
                raise ValueError(f"missing original pair {cohort} / {seed}")
            output[str(seed)] = audited_pair(scores[a], scores[b])
        report["cohorts"][cohort] = output
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("research/replay-bundles/ME069/REAL_EVALUATION.json"))
    parser.add_argument("--output", type=Path, default=Path("research/mathematical-engineering/me070/EXPOSED_ME069_DIAGNOSTIC.json"))
    args = parser.parse_args()
    report = audit(json.loads(args.input.read_text(encoding="utf-8")))
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(encoded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
