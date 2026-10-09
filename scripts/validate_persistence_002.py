#!/usr/bin/env python3
"""Independent structural validator for PERSISTENCE-002 measurement artifacts.

Validates the *recorded* evidence against its pre-registered contract.
A valid record need not pass the scientific success criterion; that boolean
must be recomputed rather than assumed from the workflow exit status.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
ID = "TACOSM-PLM-PERSISTENCE-002"
DELAYS = (1, 2, 4, 8, 16)
SEEDS = (0, 1, 2, 3, 4)
FIELDS = (
    "persistent_accuracy",
    "reset_accuracy",
    "swapped_history_accuracy",
    "oracle_accuracy",
    "no_intervening_write_accuracy",
)


def _bootstrap(values: list[float]) -> list[float]:
    rng = random.Random(20261009)
    draws = 4000
    samples = sorted(
        statistics.fmean(rng.choices(values, k=len(values)))
        for _ in range(draws)
    )
    return [samples[int(draws * 0.025)], samples[int(draws * 0.975) - 1]]


def _close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-7


def validate(path: Path) -> dict:
    data = json.loads(path.read_text())
    contract_path = ROOT / "contracts" / (ID + ".json")
    contract = json.loads(contract_path.read_text())
    assert contract["status"] == "pre-registered"
    assert data["status"] == "measured"
    assert data["experiment_id"] == contract["experiment_id"] == ID
    assert data["provenance"]["contract_sha256"] == hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    for part, field in (
        ("src/tac_osm/persistence_002_benchmark.py", "benchmark_sha256"),
        ("src/tac_osm/persistence_002.py", "model_sha256"),
    ):
        assert data["provenance"][field] == hashlib.sha256(
            (ROOT / part).read_bytes()
        ).hexdigest()
    assert data["protocol"]["delay_levels"] == contract["h_levels"]
    assert data["protocol"]["seeds"] == contract["seeds"]
    assert data["protocol"]["train_steps"] == contract["steps"]
    assert data["protocol"]["evaluation_pairs_per_delay_per_seed"] == contract["eval_steps"]
    assert data["protocol"]["train_batch_size"] == contract["protocol"]["batch_size"]
    assert data["protocol"]["train_seed_namespace"] != data["protocol"]["eval_seed_namespace"]
    assert len(data["seed_results"]) == len(SEEDS)
    assert {r["seed"] for r in data["seed_results"]} == set(SEEDS)
    assert data["summary"]["full_registered_arm_count"] == len(contract["arms"]) == 5

    fingerprints = set()
    for seed_row in data["seed_results"]:
        train = seed_row["training"]
        assert train["parameter_changed"] and train["write_gradient_gate"]
        assert train["parameter_final_sha256"] != train["parameter_initial_sha256"]
        assert [r["delay"] for r in seed_row["evaluation"]] == list(DELAYS)
        for row in seed_row["evaluation"]:
            k = row["delay"]
            assert row["pair_count"] == contract["eval_steps"]
            assert row["episode_count"] == 2 * row["pair_count"]
            assert row["observed_write_invocations"] == 1 + k
            assert row["write_calls_per_episode"] == 1 + k
            assert row["representation_calls_per_episode"] == 2 + k
            assert row["mean_distractor_write_delta_norm"] > contract["thresholds"]["minimum_mean_distractor_state_change"]
            assert _close(row["reset_accuracy"], 0.5)
            assert _close(row["oracle_accuracy"], 1.0)
            for field in FIELDS:
                assert field in row
                assert 0 <= row[field] <= 1
            assert 0 <= row["opposite_past_prediction_disagreement"] <= 1
            assert len(row["pool_fingerprint"]) == 64
            assert row["pool_fingerprint"] not in fingerprints
            fingerprints.add(row["pool_fingerprint"])

    primary = [r["evaluation"][DELAYS.index(4)] for r in data["seed_results"]]
    accuracy = [r["persistent_accuracy"] for r in primary]
    gaps = [r["persistent_accuracy"] - r["reset_accuracy"] for r in primary]
    disagreements = [r["opposite_past_prediction_disagreement"] for r in primary]
    interval = _bootstrap(gaps)
    summary = data["summary"]
    assert _close(statistics.fmean(accuracy), summary["delay4_persistent_accuracy_mean"])
    assert _close(min(accuracy), summary["delay4_persistent_accuracy_min_seed"])
    assert _close(statistics.fmean(gaps), summary["delay4_mean_paired_gap"])
    assert _close(statistics.fmean(disagreements), summary["delay4_opposite_past_disagreement_mean"])
    assert all(_close(x, y) for x, y in zip(interval, summary["delay4_seed_bootstrap_gap_ci95"]))
    assert all(summary[key] for key in (
        "all_gradient_gates_pass", "all_pair_integrity_gates_pass",
        "all_real_write_gates_pass", "all_oracle_gates_pass",
        "all_reset_ceiling_gates_pass",
    ))
    t = contract["thresholds"]
    expected = (
        statistics.fmean(accuracy) >= t["delay4_mean"]
        and min(accuracy) >= t["delay4_min_seed"]
        and statistics.fmean(gaps) >= t["delay4_gap_mean"]
        and interval[0] > t["delay4_gap_ci_lower_strict"]
        and statistics.fmean(disagreements) >= t["delay4_pair_disagreement"]
        and all(_close(r["reset_accuracy"], t["reset_exact_accuracy"]) for r in primary)
    )
    assert data["registered_criterion_pass"] is expected
    return {
        "artifact_valid": True,
        "registered_criterion_pass": expected,
        "mean_delay4_accuracy": statistics.fmean(accuracy),
        "mean_delay4_gap": statistics.fmean(gaps),
        "evaluated_seed_delay_cells": len(SEEDS) * len(DELAYS),
    }


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "artifacts" / (ID + ".json")
    )
    print(json.dumps(validate(path), sort_keys=True, indent=2))
