"""DEV-021 stdlib-only audit: independently regenerate public suffixes/labels.

The audit checks raw paired rows, exact transport tolerances and factorial
coverage, then recomputes summary statistics. It cannot independently replay
PyTorch checkpoints or prove that historical differences encode causal labels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.development.schema_maintenance import make_pair  # noqa: E402

ARMS = ("tac_joint", "tac_encoder")
SEEDS = (43, 59, 71)
REGIMES = ("short", "long")
DUE = (32, 128, 512)
DEADLINES = ("at", "after")
ROLES = ("pending_cancel", "reissued_cancel")
PREFIXES = ("32", "128", "512")
MODES = ("actual_l2", "transport_l2", "no_decay_l2", "no_erase_l2", "neither_l2")
CONTROLS = ("schema41_l2", "gru768_l2", "memoryless_l2")


def finite_nonnegative(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and v >= 0


def audit_one(d):
    if (d["study"] != "DEV-021" or d["status"] != "FROZEN_ADAPTIVE_DIAGNOSTIC_NOT_CONFIRMATORY"
            or d["smoke"] or d["dtype"] != "torch.float64" or d["source_run"] != 38802569971
            or d["domain"] != "dev021-transport-v1" or d["pairs_per_arm"] != 192
            or d["seed"] not in SEEDS or d["regime"] not in REGIMES):
        raise AssertionError("DEV-021 provenance/status mismatch")
    if d["atol"] != 2e-9 or d["rtol"] != 2e-5:
        raise AssertionError("transport tolerances changed")
    if tuple(d["checkpoint_sha256"]) != ARMS + ("schema41", "gru768", "memoryless"):
        raise AssertionError("checkpoint arm inventory changed")
    for sha in d["checkpoint_sha256"].values():
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise AssertionError("checkpoint SHA malformed")
    rows = d["rows"]
    if len(rows) != 384:
        raise AssertionError("incomplete paired observations")
    seen = set()
    reference = {}
    reconstructed = {}
    for row in rows:
        arm, due, deadline, role, rep = (row["arm"], row["due_delay"], row["deadline"],
                                         row["role"], row["repetition"])
        key = (arm, due, deadline, role, rep)
        if (arm not in ARMS or due not in DUE or deadline not in DEADLINES or
                role not in ROLES or rep not in range(16) or key in seen):
            raise AssertionError("mismatched/duplicated research grid")
        seen.add(key)
        condition = (due, deadline, role, rep)
        if condition not in reconstructed:
            pair = make_pair(d["seed"] * 1_000_000 + rep, 512, due, role, deadline,
                             domain="dev021-transport-v1")
            unequal = [i for i, (a, b) in enumerate(zip(pair[0].observations, pair[1].observations))
                       if a != b]
            if not unequal or not all(a == b for a, b in
                                      zip(pair[0].observations[unequal[-1]+1:],
                                          pair[1].observations[unequal[-1]+1:])):
                raise AssertionError("raw suffix not matched")
            if pair[0].wait_targets[pair[0].target_entity] == pair[1].wait_targets[pair[1].target_entity]:
                raise AssertionError("no distinct causal outcomes")
            reconstructed[condition] = (
                [pair[0].fingerprint, pair[1].fingerprint], unequal[-1]+1)
        fps, last_tick = reconstructed[condition]
        if (row["fingerprints"] != fps or row["divergence_tick"] != last_tick or
                row["suffix_events"] != 512-last_tick):
            raise AssertionError("public history, pivot or suffix mismatch")
        if not finite_nonnegative(row["initial_l2"]) or row["zero_initial"] != (row["initial_l2"] < 1e-12):
            raise AssertionError("bad initial distance")
        if tuple(row["prefixes"]) != PREFIXES:
            raise AssertionError("missing prefix")
        if not finite_nonnegative(row["max_transport_error"]):
            raise AssertionError("bad transport error")
        if not finite_nonnegative(row["read_input_l2"]):
            raise AssertionError("bad read distance")
        if not finite_nonnegative(row["original_probability_gap"]) or row["original_probability_gap"] > 1:
            raise AssertionError("bad native prediction gap")
        if row["original_class_difference"] not in (0, 1):
            raise AssertionError("bad class decision")
        if len(row["trained_decay_product"]) != 4 or any(
                not finite_nonnegative(v) or v > 1 for v in row["trained_decay_product"]):
            raise AssertionError("bad trained decay factors")
        for p, measures in row["prefixes"].items():
            if set(measures) != set(MODES + CONTROLS):
                raise AssertionError("missing factorial/observational arm")
            if any(not finite_nonnegative(v) for v in measures.values()):
                raise AssertionError("invalid observed norm")
            # Norm perturbation <= sqrt(768) * reported max component error.
            limit = math.sqrt(768) * row["max_transport_error"] + 1e-8
            if abs(measures["actual_l2"] - measures["transport_l2"]) > limit:
                raise AssertionError("independent transport distance mismatch")
            if measures["memoryless_l2"] > 1e-10:
                raise AssertionError("memoryless control gained a false past")
            control_key = (condition, p)
            observed = tuple(measures[x] for x in CONTROLS)
            if control_key in reference and any(abs(a-b) > 1e-8 for a, b in
                                                zip(reference[control_key], observed)):
                raise AssertionError("observational controls disagree across native arms")
            reference[control_key] = observed
    required = {(arm, due, deadline, role, rep)
                for arm in ARMS for due in DUE for deadline in DEADLINES for role in ROLES
                for rep in range(16)}
    if seen != required:
        raise AssertionError("missing factorial population")
    return d


def mean(rows, key):
    return statistics.mean(r[key] for r in rows)


def aggregate(raw):
    if len(raw) != 6:
        raise AssertionError("expected complete 3 seed x 2 regime evidence")
    reports = {}
    for d in raw:
        audit_one(d)
        key = (d["seed"], d["regime"])
        if key in reports:
            raise AssertionError("duplicate seed/regime report")
        reports[key] = d
    if set(reports) != {(s, r) for s in SEEDS for r in REGIMES}:
        raise AssertionError("absent seed-regime")
    output = {"study": "DEV-021", "status": "FROZEN_DIAGNOSTIC_NOT_CONFIRMATORY",
              "source_run": 38802569971, "analyses": 6,
              "paired_replays": 6 * 384, "TAC_memory_success": False,
              "per_regime": {}, "by_seed": []}
    for regime in REGIMES:
        per_arm = {}
        for arm in ARMS:
            summaries = []
            for seed in SEEDS:
                d = reports[(seed, regime)]
                rows = [r for r in d["rows"] if r["arm"] == arm]
                finals = [r["prefixes"]["512"] for r in rows]
                means = {m: statistics.mean(z[m] for z in finals) for m in MODES + CONTROLS}
                means.update(
                    initial_l2=mean(rows, "initial_l2"),
                    read_input_l2=mean(rows, "read_input_l2"),
                    original_probability_gap=mean(rows, "original_probability_gap"),
                    original_class_disagreement=mean(rows, "original_class_difference"),
                    zero_initial=sum(r["zero_initial"] for r in rows),
                    maximum_transport_error=max(r["max_transport_error"] for r in rows),
                    trained_decay_product=[statistics.mean(r["trained_decay_product"][k]
                                                           for r in rows) for k in range(4)]
                )
                # Norm ratios are explicitly descriptive and can be unstable at tiny baselines.
                means["aggregate_norm_no_decay_ratio"] = means["no_decay_l2"] / max(
                    means["actual_l2"], 1e-15)
                means["aggregate_norm_no_erase_ratio"] = means["no_erase_l2"] / max(
                    means["actual_l2"], 1e-15)
                means["aggregate_norm_neither_ratio"] = means["neither_l2"] / max(
                    means["actual_l2"], 1e-15)
                summaries.append(means)
                output["by_seed"].append({"seed": seed, "regime": regime,
                                           "arm": arm, "summary": means})
            per_arm[arm] = {k: statistics.mean(x[k] for x in summaries) if k != "trained_decay_product"
                            else [statistics.mean(x[k][i] for x in summaries) for i in range(4)]
                            for k in summaries[0]}
        output["per_regime"][regime] = per_arm
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = aggregate([json.loads(p.read_text()) for p in args.reports])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("DEV021_AUDITED " + json.dumps(result), flush=True)
