"""Independent stdlib raw-row/ZIP audit and three-seed DEV-016 summary.

Does not load torch or claim to replay neural inference. The training runner
performs that checkpoint replay separately. This checks evidence derivation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.development.delayed_maintenance import make_pair  # noqa: E402

ARMS = ("frozen_writer", "joint_task", "joint_predictive", "joint_shuffled")
SEEDS = (43, 59, 71)


def classification(logits):
    if len(logits) != 2 or not all(math.isfinite(x) for x in logits):
        raise AssertionError("invalid saved logits")
    return max(range(2), key=lambda i: logits[i])


def read_report(path):
    if path.is_dir():
        report = json.loads((path / "report.json").read_text())
        content = lambda name: (path / name).read_bytes()
        return report, content
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        report_name = next(n for n in names if n == "report.json" or n.endswith("/report.json"))
        prefix = report_name[:-len("report.json")]
        contents = {name[len(prefix):]: archive.read(name) for name in names
                    if name.startswith(prefix) and not name.endswith("/")}
    return json.loads(contents["report.json"]), lambda name: contents[name]


def audit(path):
    report, content = read_report(path)
    seed = report["seed"]
    if (report["study"] != "DEV-016-LEARNED" or report["smoke"] or seed not in SEEDS or
            report["status"] != "DEVELOPMENT_ONLY_NOT_CONFIRMATORY"):
        raise AssertionError("wrong study, smoke or promotion")
    if set(report["rows"]) != set(ARMS) or set(report["models"]) != set(ARMS):
        raise AssertionError("incomplete arms")
    training = report["training"]
    for source, expected_hash in report["source_sha256"].items():
        if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != expected_hash:
            raise AssertionError("current source differs from trained evaluation source")
    if training["steps"] != 256 or len(set(training["train_fingerprints"])) != 2048:
        raise AssertionError("training budget mismatch")
    if len(set(training["initial_hashes"].values())) != 1:
        raise AssertionError("unequal initialization")
    if training["writer_hash_final"]["frozen_writer"] != training["writer_hash_initial"]:
        raise AssertionError("frozen writer changed")
    out = {}
    eval_fingerprints = None
    for arm in ARMS:
        digest = hashlib.sha256(content(f"{arm}-seed{seed}.pt")).hexdigest()
        if digest != report["models"][arm]["checkpoint_sha256"]:
            raise AssertionError("checkpoint bytes changed")
        rows = report["rows"][arm]
        fingerprints = [r["fingerprint"] for r in rows]
        if len(rows) != 1152 or len(set(fingerprints)) != 1152:
            raise AssertionError("missing or duplicate evaluation")
        if set(fingerprints) & set(training["train_fingerprints"]):
            raise AssertionError("train/eval overlap")
        if eval_fingerprints is not None and eval_fingerprints != fingerprints:
            raise AssertionError("unmatched evaluation arms")
        eval_fingerprints = fingerprints
        for row in rows:
            c = row["condition"]
            case = make_pair((seed + 16000) * 10000 + c["repetition"],
                             c["history"], c["delay"], domain="learned-eval",
                             role=c["role"])[row["partner"]]
            if (row["fingerprint"] != case.fingerprint or row["target"] != case.expected or
                    row["service_target"] != case.service_expected or row["memory_scalars"] != 768):
                raise AssertionError("generator, target or capacity drift")
            if row["action"] != classification(row["action_logits"]):
                raise AssertionError("action classification not derived from logits")
            if row["future"] != [classification(x) for x in row["future_logits"]]:
                raise AssertionError("future classification not derived from logits")
            failure = case.service_expected if row["action"] else case.expected
            if (row["post_action_failure"] != failure or
                    row["verification_accepted"] != (row["future"][row["action"]] == failure)):
                raise AssertionError("post-action or verification derivation changed")
        n = len(rows)
        summary = {
            "action_accuracy": sum(r["action"] == r["target"] for r in rows) / n,
            "wait_future_accuracy": sum(r["future"][0] == r["target"] for r in rows) / n,
            "service_future_accuracy": sum(r["future"][1] == r["service_target"] for r in rows) / n,
            "reset_action_accuracy": sum(r["reset_action"] == r["target"] for r in rows) / n,
            "reset_wait_future_accuracy": sum(r["reset_future"][0] == r["target"] for r in rows) / n,
            "post_action_failure_rate": sum(r["post_action_failure"] for r in rows) / n,
            "unnecessary_service_rate": sum(r["action"] == 1 and r["target"] == 0 for r in rows) / n,
            "service_rate": sum(r["action"] for r in rows) / n,
        }
        for h in (32, 128, 512):
            subset = [r for r in rows if r["condition"]["history"] == h]
            summary[f"H{h}_future_accuracy"] = sum(r["future"][0] == r["target"] for r in subset) / len(subset)
        for role in ("retention", "acquisition", "cancelled_service", "pending_cancel"):
            subset = [r for r in rows if r["condition"]["role"] == role]
            summary[role + "_future_accuracy"] = sum(r["future"][0] == r["target"] for r in subset) / len(subset)
        for key, value in summary.items():
            if key in report["summaries"][arm] and report["summaries"][arm][key] != value:
                raise AssertionError("saved summary not derived from raw rows")
        if summary["reset_action_accuracy"] != .5 or summary["reset_wait_future_accuracy"] != .5:
            raise AssertionError("balanced reset gate failed")
        out[arm] = summary
    return seed, out, report["source_sha256"]


def summarize(inputs):
    records = [audit(path) for path in inputs]
    if sorted(seed for seed, _, _ in records) != list(SEEDS):
        raise AssertionError("exactly three registered seeds required")
    if any(hashes != records[0][2] for _, _, hashes in records):
        raise AssertionError("different experiment sources across seeds")
    by_seed = {str(seed): metrics for seed, metrics, _ in records}
    means = {arm: {key: statistics.mean(metrics[arm][key] for _, metrics, _ in records)
                   for key in records[0][1][arm]} for arm in ARMS}
    contrasts = {}
    for control in ("frozen_writer", "joint_task", "joint_shuffled"):
        deltas = [metrics["joint_predictive"]["wait_future_accuracy"] -
                  metrics[control]["wait_future_accuracy"] for _, metrics, _ in records]
        mean = statistics.mean(deltas)
        half = 4.302652729749 * statistics.stdev(deltas) / math.sqrt(3)
        contrasts[control] = {"mean": mean, "exploratory_95_t_interval": [mean-half, mean+half],
                              "seed_deltas": deltas}
    primary = contrasts["frozen_writer"]
    passes = (primary["mean"] >= .05 and primary["exploratory_95_t_interval"][0] > 0 and
              means["joint_predictive"]["H512_future_accuracy"] >= .65 and
              contrasts["joint_task"]["mean"] > 0 and contrasts["joint_shuffled"]["mean"] > 0)
    return {"study": "DEV-016-LEARNED", "status": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
            "raw_rows_audited": 13824, "checkpoints_hashed": 12,
            "means": means, "by_seed": by_seed, "primary_future_contrasts": contrasts,
            "development_gate_passed": passes,
            "warning": "Raw-row audit, not neural replay or SGD replication; three-seed intervals are exploratory."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.inputs)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"means": result["means"], "contrasts": result["primary_future_contrasts"],
                      "gate_passed": result["development_gate_passed"]}, indent=2))
