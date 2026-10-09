#!/usr/bin/env python3
"""Independent recalculation of the frozen Address-003 success criterion.

Accepts a negative result if and only if artifact structure/integrity holds.
Does not import the measurement runner and does not reselect model checkpoints.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
ID = "TACOSM-PLM-LEARNED-ADDRESS-003"
MS = (3, 8, 16, 32, 64, 128, 256)
ISOS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)
STRUCTS = (0.0, 0.05, 0.1, 0.2, 0.4)
ARMS = ("structured_only", "mixed_unlabeled")
BASELINE = (
    ("src/tac_osm/learned_address_002_benchmark.py", "base_benchmark_sha256"),
    ("src/tac_osm/learned_address_002.py", "model_sha256"),
    ("scripts/run_plm_learned_address_003.py", "runner_sha256"),
)


def close(a: float, b: float) -> bool:
    return abs(a - b) < 1e-7


def bootstrap(vals: list[float]) -> list[float]:
    gen = random.Random(20261009)
    rounds = 4000
    v = sorted(statistics.fmean(gen.choices(vals, k=len(vals)))
               for _ in range(rounds))
    return [v[int(rounds * .025)], v[int(rounds * .975) - 1]]


def validate(path: Path) -> dict:
    result = json.loads(path.read_text())
    contract_path = ROOT / "contracts" / (ID + ".json")
    contract = json.loads(contract_path.read_text())
    assert contract["status"] == "pre-registered"
    assert result["status"] == "measured"
    assert result["experiment_id"] == contract["experiment_id"] == ID
    assert result["provenance"]["contract_sha256"] == hashlib.sha256(
        contract_path.read_bytes()).hexdigest()
    for file_path, field in BASELINE:
        assert result["provenance"][field] == hashlib.sha256(
            (ROOT / file_path).read_bytes()).hexdigest()
    assert result["protocol"]["seeds"] == contract["seeds"] == list(range(10))
    assert result["protocol"]["train_steps_each"] == contract["steps"] == 750
    assert result["protocol"]["train_batch"] == contract["protocol"]["train_batch"] == 512
    assert result["protocol"]["evaluation_trials_per_condition"] == 10000
    assert result["protocol"]["memory_sizes"] == list(MS)
    assert result["protocol"]["isotropic_sigmas"] == list(ISOS)
    assert result["protocol"]["structured_sigmas"] == list(STRUCTS)
    assert result["protocol"]["evaluation_namespace"] == 8_300_000
    assert result["protocol"]["channel_label_at_prediction"] is False
    assert result["protocol"]["evaluation_shared_between_learned_arms"] is True
    assert len(result["seed_results"]) == 10
    assert {r["seed"] for r in result["seed_results"]} == set(range(10))
    fingerprints = set()

    for seed_row in result["seed_results"]:
        assert set(seed_row["training"]) == set(ARMS)
        init_hashes = set()
        for arm in ARMS:
            train = seed_row["training"][arm]
            assert train["steps"] == 750 and train["train_batch"] == 512
            assert train["arm"] == arm and train["seed"] == seed_row["seed"]
            assert train["parameter_changed"] and train["training_gradient_pass"]
            assert train["initial_parameter_hash"] != train["final_parameter_hash"]
            init_hashes.add(train["initial_parameter_hash"])
            counts = train["condition_counts"]
            assert sum(counts.values()) == 750
            if arm == "structured_only":
                assert counts == {
                    "structured_0.2": 375, "structured_0.4": 375,
                    "isotropic_0.2": 0, "isotropic_0.4": 0,
                }
            else:
                assert counts == {
                    "structured_0.2": 188, "structured_0.4": 187,
                    "isotropic_0.2": 187, "isotropic_0.4": 188,
                }
        assert len(init_hashes) == 1, "initialization not shared"
        assert len(seed_row["isotropic"]) == len(MS) * len(ISOS)
        assert len(seed_row["structured"]) == len(MS) * len(STRUCTS)
        for family, sigmas in (("isotropic", ISOS), ("structured", STRUCTS)):
            expected = [(m, s) for m in MS for s in sigmas]
            actual = [(r["M"], r["sigma"]) for r in seed_row[family]]
            assert actual == expected
            for row in seed_row[family]:
                assert row["structured"] is (family == "structured")
                assert row["trials"] == 10000
                assert set(row["arms"]) == set(ARMS)
                assert 0 <= row["raw_dot_accuracy"] <= 1
                fingerprint = row["evaluation_fingerprint"]
                assert len(fingerprint) == 64 and fingerprint not in fingerprints
                fingerprints.add(fingerprint)
                for arm in ARMS:
                    metrics = row["arms"][arm]
                    for name in (
                        "accuracy", "oracle_query_accuracy",
                        "wrong_key_query_accuracy",
                        "permutation_accuracy",
                        "permutation_identity_accuracy",
                    ):
                        assert 0 <= metrics[name] <= 1
                    assert close(
                        metrics["learned_minus_raw"],
                        metrics["accuracy"] - row["raw_dot_accuracy"]
                    )
                    assert close(
                        metrics["permutation_identity_accuracy"], 1.0
                    )
    summary = result["summary"]
    assert len(fingerprints) == 10 * len(MS) * (len(ISOS) + len(STRUCTS))
    assert summary["unique_evaluation_pool_fingerprints"] == len(fingerprints)
    mixed_iso_seed = [
        statistics.fmean(row["arms"]["mixed_unlabeled"]["learned_minus_raw"]
                         for row in r["isotropic"])
        for r in result["seed_results"]
    ]
    mixed_struct_seed = [
        statistics.fmean(row["arms"]["mixed_unlabeled"]["learned_minus_raw"]
                         for row in r["structured"]
                         if row["M"] == 32 and row["sigma"] in (0.2, 0.4))
        for r in result["seed_results"]
    ]
    mean_iso, mean_struct = (
        statistics.fmean(mixed_iso_seed),
        statistics.fmean(mixed_struct_seed)
    )
    reference_iso = statistics.fmean(
        row["arms"]["structured_only"]["learned_minus_raw"]
        for r in result["seed_results"] for row in r["isotropic"]
    )
    reference_struct = statistics.fmean(
        row["arms"]["structured_only"]["learned_minus_raw"]
        for r in result["seed_results"] for row in r["structured"]
        if row["M"] == 32 and row["sigma"] in (0.2, 0.4)
    )
    assert all(close(x, y) for x, y in zip(mixed_iso_seed, summary["mixed_isotropic_seed_means"]))
    assert all(close(x, y) for x, y in zip(mixed_struct_seed, summary["mixed_structured_high_noise_seed_means"]))
    assert close(summary["mixed_isotropic_mean_gap"], mean_iso)
    assert close(summary["mixed_structured_high_noise_mean_gain"], mean_struct)
    assert all(close(x, y) for x, y in zip(
        bootstrap(mixed_struct_seed), summary["mixed_structured_high_noise_ci95"]
    ))
    assert close(summary["structured_only_isotropic_mean_gap"], reference_iso)
    assert close(summary["structured_only_structured_high_noise_gain"], reference_struct)
    t = contract["thresholds"]
    iso_pass = mean_iso >= t["mixed_isotropic_learned_minus_raw_mean_min"]
    structured_pass = mean_struct >= t["mixed_structured_M32_high_noise_gain_mean_min"]
    assert summary["mixed_isotropic_noninferiority_pass"] is iso_pass
    assert summary["mixed_structured_gain_pass"] is structured_pass
    assert summary["joint_registered_mechanism_pass"] is (iso_pass and structured_pass)
    assert summary["all_integrity_gates_pass"] is True
    return {
        "artifact_valid": True,
        "joint_registered_mechanism_pass": iso_pass and structured_pass,
        "isotropic_mean_gap": mean_iso,
        "structured_high_noise_gain": mean_struct,
        "unique_conditions": len(fingerprints),
    }


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "artifacts" / (ID + ".json")
    )
    print(json.dumps(validate(target), indent=2, sort_keys=True))
