"""Restore Git-owned originals, summarize paired cases, and independently replay ME069."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from scripts.me069_train_workflow import (
    BUNDLE, ROOT, SEEDS, build, score, sha_file, verify_git_owned,
)
from scripts.me069_workflow_source import load, write_json


def restore(cache):
    torch.set_num_threads(1)
    records = json.loads((BUNDLE / "CHECKPOINT_MANIFEST.json").read_text())
    verify_git_owned(records)
    _, pre, source = load(cache)
    receipts = []
    for record in records:
        saved = torch.load(record["path"], map_location="cpu", weights_only=False)
        model = build(record["architecture"], pre)
        model.load_state_dict(saved["model"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        optimizer.load_state_dict(saved["optimizer"])
        if saved["preprocessing"] != pre or saved["source_sha256"] != source["source_sha256"]:
            raise ValueError("checkpoint source/preprocessing identity mismatch")
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("nonfinite original checkpoint")
        receipts.append({"path": record["path"], "sha256": sha_file(record["path"]),
                         "strict_model_restore": True, "optimizer_restore": True,
                         "rng_bytes": saved["torch_rng_state"].numel()})
    write_json(ROOT / "RESTORATION.json", {
        "checkpoint_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "clean_tracked_checkout_before_receipt": not bool(subprocess.check_output(["git", "diff", "HEAD", "--name-only"], text=True).strip()),
        "heldout_inference_performed": False, "original_models": receipts})


def paired_interval(values):
    """Case bootstrap conditional on the five original fitted seed pairs."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(690069)
    replicates = []
    for _ in range(100):
        indices = rng.integers(len(values), size=(200, len(values)))
        replicates.extend(values[indices].mean(axis=1).tolist())
    return [float(x) for x in np.quantile(replicates, [.025, .975])]


def summarize(evaluation):
    out = {"scope": "one midpoint next-event target per case; five frozen seeds; no DEV selection",
           "uncertainty": "20,000 paired case bootstrap replicates, conditional on fitted seeds; not retraining uncertainty",
           "cohorts": {}}
    for cohort, models in evaluation["scores"].items():
        result = {"models": {}, "paired_primary": {}}
        ordered = {}
        for key, rows in models.items():
            ordered[key] = sorted(rows, key=lambda r: r["case_id"])
            fields = [k for k, v in rows[0].items() if isinstance(v, (int, float, bool)) and k != "prefix_last_index"]
            averages = {k: float(np.mean([r[k] for r in rows])) for k in fields}
            positive = [r["wait_MAE_seconds"] for r in rows if r["positive_wait_observed"]]
            averages["positive_wait_MAE_seconds"] = float(np.mean(positive))
            result["models"][key] = {"cases": len(rows), **averages}
        native = [ordered[f"native_tac-s{s}"] for s in SEEDS]
        gru = [ordered[f"gru_world-s{s}"] for s in SEEDS]
        ids = [(r["case_id"], r["prefix_last_index"]) for r in native[0]]
        if any([(r["case_id"], r["prefix_last_index"]) for r in rows] != ids for rows in native + gru):
            raise ValueError("unpaired evaluation cases")
        a = np.asarray([[r["activity_NLL"] for r in rows] for rows in native])
        b = np.asarray([[r["activity_NLL"] for r in rows] for rows in gru])
        markov = np.asarray([r["Markov_NLL"] for r in native[0]])
        for name, reference, margin in (("GRU", b, .01), ("Markov", markov[None, :], .02)):
            delta = a - reference
            ci = paired_interval(delta.mean(axis=0))
            wins = int((delta.mean(axis=1) < 0).sum())
            improvement = float(1 - a.mean() / reference.mean())
            result["paired_primary"][name] = {
                "native_minus_reference_NLL": float(delta.mean()), "case_bootstrap_95_CI": ci,
                "relative_native_improvement": improvement, "seed_wins": wins,
                "required_relative_improvement": margin,
                "gate_passed": improvement >= margin and ci[1] < 0 and wins >= 4}
        result["native_joint_primary_gate_passed"] = all(x["gate_passed"] for x in result["paired_primary"].values())
        out["cohorts"][cohort] = result
    return out


def compare(original, restored, path="root"):
    """Fixed pre-scoring contract: exact structure/identities, bounded score floats."""
    if isinstance(original, dict):
        if original.keys() != restored.keys():
            raise ValueError("replay keys differ: " + path)
        for k in original:
            compare(original[k], restored[k], path + "." + k)
    elif isinstance(original, list):
        if len(original) != len(restored):
            raise ValueError("replay length differs: " + path)
        for i, (a, b) in enumerate(zip(original, restored)):
            compare(a, b, path + f"[{i}]")
    elif isinstance(original, float):
        if not np.isfinite(restored) or abs(original - restored) > 1e-4 + 1e-5 * abs(original):
            raise ValueError(f"replay float differs: {path}; original={original!r}, restored={restored!r}, allowed={1e-4 + 1e-5 * abs(original)!r}")
    elif original != restored:
        raise ValueError("replay identity differs: " + path)


def replay(cache):
    target = BUNDLE / "REAL_EVALUATION.json"
    audit = ROOT / "REPLAY_AUDIT.json"
    audit.unlink(missing_ok=True)  # A prior committed receipt cannot attest this execution.
    original_bytes = subprocess.check_output(["git", "show", "HEAD:" + str(target)])
    original = json.loads(original_bytes)
    score(cache)
    replayed = json.loads(target.read_text())
    try:
        compare(original, replayed)
        before, after = summarize(original), summarize(replayed)
        for part in ("DEV", "TEST"):
            for key, row in before["cohorts"][part]["models"].items():
                delta = abs(row["activity_NLL"] - after["cohorts"][part]["models"][key]["activity_NLL"])
                if delta > 1e-5:
                    raise ValueError("primary replay tolerance exceeded")
    except ValueError as error:
        target.rename(BUNDLE / "FAILED_REPLAY_EVALUATION.json")
        target.write_bytes(original_bytes)
        write_json(audit, {"status": "FAIL", "test_metrics_recomputed": True,
                          "all_case_score_records_within_preregistered_tolerance": False,
                          "error": str(error), "original_score_bytes_preserved": True})
        raise
    # Keep the original committed score bytes, even when a replay differs within tolerance.
    exact = target.read_bytes() == original_bytes
    target.write_bytes(original_bytes)
    write_json(audit, {"status": "PASS", "original_checkpoint_count": 10,
        "cohorts_recomputed": ["DEV", "TEST"], "test_metrics_recomputed": True,
        "all_case_score_records_within_preregistered_tolerance": True,
        "bit_identical_evaluation_json": exact, "primary_absolute_tolerance": 1e-5,
        "record_absolute_tolerance": 1e-4, "record_relative_tolerance": 1e-5,
        "cases_per_seed": {p: len(next(iter(original["scores"][p].values()))) for p in ("DEV", "TEST")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("restore", "summarize", "replay"))
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "summarize":
        write_json(ROOT / "RESULTS.json", summarize(json.loads((BUNDLE / "REAL_EVALUATION.json").read_text())))
    else:
        {"restore": restore, "replay": replay}[args.phase](args.cache)
