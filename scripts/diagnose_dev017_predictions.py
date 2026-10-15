"""DEV-019: read-only post-hoc forensic diagnostics for archived DEV-017 forecasts.

Uses held-out public fingerprints, labels, logits and physical receipts already in
the frozen reports. Does NOT retrain, tune, or certify a new TAC-OSM capability.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

ARMS = ("schema41", "schema41_encoder", "tac_reader", "tac_encoder", "tac_memory", "tac_joint")
CANCEL_ROLES = {"pending_cancel", "reissued_cancel"}
MATURE = {"at", "after"}


def p1(logits):
    a, b = logits
    if not (math.isfinite(a) and math.isfinite(b)):
        raise AssertionError("nonfinite archived logit")
    return 1. / (1. + math.exp(max(-700., min(700., a - b))))


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def accuracy(xs):
    return mean([e["pred"] == e["label"] for e in xs])


def critical(e):
    c = e["condition"]
    return (c["history"] == 512 and c["due_delay"] in (64, 128)
            and c["role"] in CANCEL_ROLES and c["deadline"] in MATURE)


def mature(e):
    return e["condition"]["deadline"] in MATURE


def extract(rows):
    out = []
    for row in rows:
        c = row["condition"]
        target = row["target_entity"]
        label = row["wait_targets"][target]
        logits = row["logits"][target][0]
        pred = row["predictions"][target][0]
        if label not in (0, 1) or pred not in (0, 1) or pred != int(logits[1] > logits[0]):
            raise AssertionError("target/argmax mismatch in archived row")
        if len(row["fingerprint"]) != 64:
            raise AssertionError("invalid public fingerprint")
        out.append({"condition": c, "target": target, "fingerprint": row["fingerprint"],
                    "label": label, "pred": pred, "logits": logits, "prob": p1(logits)})
    return out


def summarize(entries):
    if not entries:
        return None
    return {"n": len(entries), "accuracy": accuracy(entries),
            "actual_positive_rate": mean([e["label"] for e in entries]),
            "predicted_positive_rate": mean([e["pred"] for e in entries]),
            "mean_positive_probability": mean([e["prob"] for e in entries]),
            "brier": mean([(e["prob"] - e["label"]) ** 2 for e in entries]),
            "log_loss": mean([-math.log(max(1e-15, e["prob"] if e["label"] else 1-e["prob"]))
                              for e in entries]),
            "classification_hash": digest([e["pred"] for e in entries]),
            "logit_hash": digest([e["logits"] for e in entries])}


def pairs(entries):
    grouped = defaultdict(list)
    for e in entries:
        if critical(e):
            c = e["condition"]
            key = (c["history"], c["due_delay"], c["deadline"], c["role"], c["repetition"])
            grouped[key].append(e)
    out = []
    for key, part in sorted(grouped.items()):
        if len(part) != 2 or {p["condition"]["partner"] for p in part} != {0, 1}:
            raise AssertionError("counterfactual pair missing or repeated")
        x, y = sorted(part, key=lambda p: p["condition"]["partner"])
        if x["target"] != y["target"] or x["label"] == y["label"] or x["fingerprint"] == y["fingerprint"]:
            raise AssertionError("counterfactual pair truth or fingerprint invalid")
        out.append({"predictions_differ": x["pred"] != y["pred"],
                    "both_correct": x["pred"] == x["label"] and y["pred"] == y["label"],
                    "both_wrong": x["pred"] != x["label"] and y["pred"] != y["label"],
                    "absolute_probability_gap": abs(x["prob"] - y["prob"]),
                    "signed_true_direction_gap": (y["prob"] - x["prob"]) * (y["label"] - x["label"])})
    return out


def physical(rows):
    costs = defaultdict(list)
    for row in rows:
        for mode in ("none", "selected", "matched_random"):
            c = row["controls"][mode]
            expected = 4 * c["failures"] + len(c["services"]) + .1 * c["inspections"]
            if not math.isclose(c["total_cost"], expected, abs_tol=1e-8):
                raise AssertionError("physical cost arithmetic mismatch")
            if not math.isclose(c["cost_per_entity"], expected / 8, abs_tol=1e-8):
                raise AssertionError("normalized cost mismatch")
            if row["condition"]["history"] == 512:
                costs[mode].append(c["cost_per_entity"])
    return {mode + "_H512_cost_per_entity": mean(values) for mode, values in costs.items()}


def analyze(report):
    if report.get("study") != "DEV-017" or report.get("smoke") or report.get("status") != "DEVELOPMENT_ONLY_NOT_CONFIRMATORY":
        raise AssertionError("not complete frozen DEV-017 evidence")
    seed = report["seed"]
    if seed not in (43, 59, 71) or set(report["rows"]) != set(ARMS):
        raise AssertionError("unexpected seed or experimental arms")
    output, fingerprints = {}, None
    for arm in ARMS:
        rows = report["rows"][arm]
        entries = extract(rows)
        fps = [e["fingerprint"] for e in entries]
        if fingerprints is not None and fps != fingerprints:
            raise AssertionError("unmatched evaluation streams")
        fingerprints = fps
        if len(fps) != 1728 or len(set(fps)) != 1728:
            raise AssertionError("unexpected evaluation cardinality")
        key = [e for e in entries if critical(e)]
        mature_entries = [e for e in entries if mature(e)]
        comparison = pairs(entries)
        if len(key) != 128 or len(comparison) != 64:
            raise AssertionError("critical counterfactual cardinality changed")
        summary = report["summaries"][arm]
        if not math.isclose(accuracy(key), summary["H512_long_pending_accuracy"], abs_tol=1e-12):
            raise AssertionError("H512 archived summary mismatch")
        if not math.isclose(accuracy(mature_entries), summary["primary_future_accuracy"], abs_tol=1e-12):
            raise AssertionError("overall archived summary mismatch")
        by_horizon = {}
        for h in (32, 128, 512):
            group = [e for e in entries if e["condition"]["history"] == h and mature(e)
                     and e["condition"]["role"] in CANCEL_ROLES]
            by_horizon[str(h)] = summarize(group)
        pair_metrics = {"pair_n": len(comparison),
                        "predicted_pair_disagreement_rate": mean([e["predictions_differ"] for e in comparison]),
                        "both_correct_rate": mean([e["both_correct"] for e in comparison]),
                        "both_wrong_rate": mean([e["both_wrong"] for e in comparison]),
                        "mean_absolute_probability_gap": mean([e["absolute_probability_gap"] for e in comparison]),
                        "mean_signed_true_direction_gap": mean([e["signed_true_direction_gap"] for e in comparison])}
        output[arm] = {"critical_H512": summarize(key), "pair_metrics": pair_metrics,
                       "cancellation_by_history": by_horizon, "physical": physical(rows)}
    return {"seed": seed, "arms": output}


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(paths, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    results, fingerprints = [], []
    for path in paths:
        raw = path.read_bytes()
        fingerprints.append({"report_sha256": hashlib.sha256(raw).hexdigest(),
                             "report_path": str(path)})
        results.append(analyze(json.loads(raw)))
    if len({r["seed"] for r in results}) != len(results):
        raise AssertionError("duplicate seed report")
    if len(results) != 3:
        raise AssertionError("expected exactly three archived DEV-017 seed reports")
    report = {"study": "DEV-019", "status": "POSTHOC_FORENSIC_NOT_CONFIRMATORY",
              "description": "Read-only archival diagnosis, no re-training, no new experimental capability",
              "inputs": fingerprints, "results": sorted(results, key=lambda r: r["seed"])}
    rows, horizons = [], []
    for item in report["results"]:
        for arm, record in item["arms"].items():
            rows.append({"seed": item["seed"], "arm": arm,
                         **{k: v for k, v in record["critical_H512"].items() if not k.endswith("_hash")},
                         **record["pair_metrics"], **record["physical"]})
            for h, vals in record["cancellation_by_history"].items():
                horizons.append({"seed": item["seed"], "arm": arm, "history": h,
                                 **{k: v for k, v in vals.items() if not k.endswith("_hash")}})
    (output_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(output_dir / "critical-H512.csv", rows)
    write_csv(output_dir / "cancellation-by-history.csv", horizons)
    print("DEV-019 POSTHOC DIAGNOSTICS — NOT A CAPABILITY CLAIM")
    for row in rows:
        print(f"seed={row['seed']} {row['arm']} accuracy={row['accuracy']:.4f} "
              f"predicted_positive={row['predicted_positive_rate']:.4f} "
              f"pair_disagreement={row['predicted_pair_disagreement_rate']:.4f} "
              f"mean_gap={row['mean_absolute_probability_gap']:.5f} "
              f"selected_cost={row['selected_H512_cost_per_entity']:.4f}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.reports, args.output)
