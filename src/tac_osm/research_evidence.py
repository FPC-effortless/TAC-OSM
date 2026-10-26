"""Paired checkpoint / experiment evidence review without inventing scores.

Consumes independently generated, source-pinned run receipts. This does NOT
verify the receipts' truth, source rights, successful training, or a causal
policy claim. A passing match is compatibility only, never promotion.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any, Mapping

HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
SHARED = ("source_sha256", "fit_split_sha256", "dev_split_sha256",
          "test_split_sha256", "feature_contract_sha256",
          "target_contract_sha256", "evaluation_contract_sha256",
          "training_sample_order_sha256", "optimizer_schedule_sha256",
          "tuning_policy_sha256", "evaluation_metric", "direction",
          "seed", "examples_seen", "optimizer_steps")
HASHED = set(SHARED) - {
    "evaluation_metric", "direction", "seed", "examples_seen", "optimizer_steps"
}


def _number(value: Any, *, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("non-numeric or boolean experimental number")
    f = float(value)
    if not math.isfinite(f) or (strictly_positive and f <= 0):
        raise ValueError("nonfinite or nonpositive experimental number")
    return f


def _receipt(record: Mapping[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    bad = []
    for key in (*SHARED, "run_id", "checkpoint_sha256",
                "trainable_parameters", "deployed_parameters", "training_flops"):
        if key not in record:
            bad.append("missing:" + key)
    for key in (*HASHED, "checkpoint_sha256"):
        if key in record and (not isinstance(record[key], str)
                              or not HEX64.fullmatch(record[key])):
            bad.append("invalid_hash:" + key)
    for key in ("trainable_parameters", "deployed_parameters", "training_flops",
                "examples_seen", "optimizer_steps"):
        if key in record:
            try:
                _number(record[key], strictly_positive=True)
            except ValueError:
                bad.append("invalid:" + key)
    if record.get("direction") not in ("lower", "higher"):
        bad.append("invalid:direction")
    if not isinstance(record.get("run_id"), str) or not record.get("run_id"):
        bad.append("invalid:run_id")
    by_id: dict[str, dict[str, Any]] = {}
    rows = record.get("observations")
    if not isinstance(rows, list) or not rows:
        bad.append("missing:observations")
    else:
        for row in rows:
            if not isinstance(row, dict):
                bad.append("invalid:observation")
                continue
            ident = row.get("id_sha256")
            if not isinstance(ident, str) or not HEX64.fullmatch(ident):
                bad.append("invalid:observation_id")
                continue
            if ident in by_id:
                bad.append("duplicate:observation_id")
                continue
            try:
                _number(row.get("value"))
            except ValueError:
                bad.append("invalid:observation_value")
                continue
            if not isinstance(row.get("group"), str) or not isinstance(row.get("horizon"), str):
                bad.append("invalid:observation_stratum")
                continue
            by_id[ident] = row
    return sorted(set(bad)), by_id


def compare_evidence(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Prevents a false ranking from unmatched data, scale, run budgets or IDs.

    Strict exact equality for source, splits, sample order, seed, contract and
    steps/exposure. At most 1% relative gaps in live trainable parameters and
    training FLOPs. No missing sample or subgroup can silently disappear.
    """
    errors_left, a = _receipt(left)
    errors_right, b = _receipt(right)
    blockers = ["left:" + x for x in errors_left] + ["right:" + x for x in errors_right]
    if not blockers:
        for key in SHARED:
            if left[key] != right[key]:
                blockers.append("unmatched:" + key)
        if left["run_id"] == right["run_id"]:
            blockers.append("same_run_id")
        if left["checkpoint_sha256"] == right["checkpoint_sha256"]:
            blockers.append("same_checkpoint_sha256")
        for key in ("trainable_parameters", "training_flops"):
            x, y = _number(left[key], strictly_positive=True), _number(right[key], strictly_positive=True)
            if abs(x - y) / max(x, y) > .01:
                blockers.append("resource_gap_above_1pct:" + key)
        if a.keys() != b.keys():
            blockers.append("unmatched_evaluation_observation_ids")
        else:
            for ident, row in a.items():
                other = b[ident]
                if (row["group"], row["horizon"]) != (other["group"], other["horizon"]):
                    blockers.append("unmatched_group_or_horizon")
                    break
    report: dict[str, Any] = {
        "schema": "tac-osm-observatory/evidence-review-v1",
        "left_run": left.get("run_id"), "right_run": right.get("run_id"),
        "status": "BLOCKED" if blockers else "RECEIPT_COMPATIBLE",
        "blockers": sorted(set(blockers)),
        "evidence": "self-reported_receipts_not_independently_authenticated",
        "eligible_for_architecture_superiority": False,
        "reason": "single-seed paired receipts cannot prove fair training, source authenticity or generalization",
    }
    if blockers:
        return report
    by_group = defaultdict(list)
    changes = []
    for ident, row in a.items():
        other = b[ident]
        diff = _number(other["value"]) - _number(row["value"])
        by_group[(row["group"], row["horizon"])].append(diff)
        changes.append((ident, diff))
    beneficial = -1 if left["direction"] == "lower" else 1
    report["paired_sample_count"] = len(changes)
    report["metric"] = left["evaluation_metric"]
    report["right_minus_left"] = {
        "mean": sum(d for _, d in changes) / len(changes),
        "right_better_count": sum(beneficial * d > 0 for _, d in changes),
        "left_better_count": sum(beneficial * d < 0 for _, d in changes),
        "ties": sum(d == 0 for _, d in changes),
    }
    report["strata"] = [
        {"group": group, "horizon": horizon, "count": len(ds),
         "mean_right_minus_left": sum(ds) / len(ds)}
        for (group, horizon), ds in sorted(by_group.items())
    ]
    report["largest_absolute_paired_changes"] = [
        {"id_sha256": ident, "right_minus_left": d}
        for ident, d in sorted(changes, key=lambda x: (-abs(x[1]), x[0]))[:5]
    ]
    return report
