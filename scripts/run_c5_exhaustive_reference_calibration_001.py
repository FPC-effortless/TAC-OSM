#!/usr/bin/env python3
"""Run TACOSM-C5-EXHAUSTIVE-REFERENCE-CALIBRATION-001."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tac_osm.contract import load_contract
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from scripts.run_c5_frontier_robustness_002 import (
    H_FIXED,
    M_LEVELS,
    NEGATIVE_COUNT,
    SEEDS,
    TARGET_CODES,
    TRAIN_CODES,
    build_scaled_task,
    normalized_embeddings,
    normalized_query,
)

TARGET_TEACHER_UPDATES = 24576
LATENT_DIM = 16
STATE_BITS = 10
K = 32
EVAL_STEPS = 100
ARMS = ("current_48", "matched_M_minus_16")
BOOTSTRAP_REPLICATES = 10_000
MATERIALITY_THRESHOLD = 0.10


def make_teacher(
    seed: int,
    train_codes: tuple[tuple[int, ...], ...],
) -> tuple[LearnedSemanticStateIndex, int, int]:
    teacher_epochs = max(1, round(TARGET_TEACHER_UPDATES / len(train_codes)))
    actual_updates = teacher_epochs * len(train_codes)
    if abs(actual_updates - TARGET_TEACHER_UPDATES) > TARGET_TEACHER_UPDATES * 0.01:
        raise AssertionError(
            f"teacher update budget drift: {actual_updates} vs target {TARGET_TEACHER_UPDATES}"
        )
    model = LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=STATE_BITS,
            latent_dim=LATENT_DIM,
            learning_rate=0.02,
            margin=0.25,
            epochs=teacher_epochs,
            bucket_bits=8,
            probe_radius=1,
            shortlist_k=1,
            seed=seed,
        )
    )
    updates = model.train_with_negative_coverage(
        train_codes,
        negative_count=NEGATIVE_COUNT,
        aggregation="mean",
        positive_views=1,
    )
    expected = actual_updates
    if updates != expected:
        raise AssertionError(
            f"teacher update count mismatch: {updates} != {expected}"
        )
    if updates != actual_updates:
        raise AssertionError(f"teacher update count mismatch: {updates} != {actual_updates}")
    return model, teacher_epochs, actual_updates


def matched_training_codes(first_task, m: int) -> tuple[tuple[int, ...], ...]:
    target_set = set(TARGET_CODES)
    values = {
        tuple(update.value)
        for update in first_task.state_updates
        if tuple(update.value) not in target_set
    }
    expected_size = m - len(TARGET_CODES)
    if len(values) != expected_size:
        raise AssertionError(
            f"matched training population mismatch: {len(values)} != {expected_size}"
        )
    if not set(TRAIN_CODES).issubset(values):
        raise AssertionError("matched training population dropped fixed training codes")
    if values & target_set:
        raise AssertionError("target-code leakage into matched training population")
    return tuple(sorted(values))


def evaluate_arm(
    seed: int,
    m: int,
    arm: str,
    teacher: LearnedSemanticStateIndex,
    teacher_epochs: int,
    teacher_updates: int,
    tasks,
    train_codes: tuple[tuple[int, ...], ...],
) -> dict:
    embeddings = normalized_embeddings(teacher, tasks[0])
    successes = 0
    for task in tasks:
        query = normalized_query(teacher, task)
        scored = [
            (sum(a * b for a, b in zip(query, embedding)), address)
            for address, embedding in embeddings
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        successes += int(scored[0][1] == task.target_address)
    return {
        "seed": seed,
        "M": m,
        "arm": arm,
        "training_population_size": len(train_codes),
        "teacher_epochs": teacher_epochs,
        "teacher_updates": teacher_updates,
        "target_teacher_updates": TARGET_TEACHER_UPDATES,
        "exhaustive_target_successes": successes,
        "evaluation_count": len(tasks),
        "exhaustive_target_recall": successes / len(tasks),
    }


def bootstrap_mean_ci(
    values: list[float],
    *,
    rng_seed: int,
) -> tuple[float, float]:
    if not values:
        raise AssertionError("bootstrap requires observations")
    rng = random.Random(rng_seed)
    n = len(values)
    samples = [
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    samples.sort()
    low = samples[int(0.025 * (len(samples) - 1))]
    high = samples[int(0.975 * (len(samples) - 1))]
    return low, high


def paired_bootstrap_ci(
    left: list[float],
    right: list[float],
    *,
    rng_seed: int,
) -> tuple[float, float, float]:
    if len(left) != len(right) or not left:
        raise AssertionError("paired bootstrap requires aligned seed observations")
    deltas = [a - b for a, b in zip(left, right)]
    rng = random.Random(rng_seed)
    n = len(deltas)
    samples = [
        statistics.fmean(deltas[rng.randrange(n)] for _ in range(n))
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    samples.sort()
    low = samples[int(0.025 * (len(samples) - 1))]
    high = samples[int(0.975 * (len(samples) - 1))]
    return statistics.fmean(deltas), low, high


def run(smoke: bool) -> dict:
    seeds = (10,) if smoke else SEEDS
    ms = (128,) if smoke else M_LEVELS
    steps = 5 if smoke else EVAL_STEPS

    cells = []
    for seed in seeds:
        for m in ms:
            first_task = build_scaled_task(seed, m, 0)
            matched_codes = matched_training_codes(first_task, m)
            train_sets = {
                "current_48": tuple(TRAIN_CODES),
                "matched_M_minus_16": matched_codes,
            }
            tasks = tuple(build_scaled_task(seed, m, step) for step in range(steps))
            for arm in ARMS:
                teacher, teacher_epochs, teacher_updates = make_teacher(seed, train_sets[arm])
                cells.append(
                    evaluate_arm(
                        seed,
                        m,
                        arm,
                        teacher,
                        teacher_epochs,
                        teacher_updates,
                        tasks,
                        train_sets[arm],
                    )
                )

    pooled = {}
    per_seed = {}
    for m in ms:
        for arm in ARMS:
            rows = [r for r in cells if r["M"] == m and r["arm"] == arm]
            total_successes = sum(r["exhaustive_target_successes"] for r in rows)
            total_tasks = sum(r["evaluation_count"] for r in rows)
            seed_values = [r["exhaustive_target_recall"] for r in rows]
            low, high = bootstrap_mean_ci(seed_values, rng_seed=20261001 + m)
            key = f"{m}:{arm}"
            pooled[key] = {
                "M": m,
                "arm": arm,
                "exhaustive_target_successes": total_successes,
                "evaluation_count": total_tasks,
                "exhaustive_target_recall": total_successes / total_tasks,
                "seed_bootstrap_95_ci_low": low,
                "seed_bootstrap_95_ci_high": high,
                "mean_training_population_size": statistics.fmean(
                    r["training_population_size"] for r in rows
                ),
                "mean_teacher_updates": statistics.fmean(
                    r["teacher_updates"] for r in rows
                ),
            }
            per_seed[key] = {
                "seed_ids": [r["seed"] for r in rows],
                "seed_values": seed_values,
            }

    paired = {}
    if not smoke:
        for arm in ARMS:
            by_seed = {}
            for row in cells:
                if row["arm"] != arm:
                    continue
                by_seed.setdefault(row["seed"], {})[row["M"]] = (
                    row["exhaustive_target_recall"]
                )
            values_128 = [by_seed[seed][128] for seed in SEEDS]
            values_512 = [by_seed[seed][512] for seed in SEEDS]
            mean_delta, low, high = paired_bootstrap_ci(
                values_512,
                values_128,
                rng_seed=20261002,
            )
            paired[arm] = {
                "mean_delta_512_minus_128": mean_delta,
                "seed_bootstrap_95_ci_low": low,
                "seed_bootstrap_95_ci_high": high,
                "seed_level_deltas": [
                    a - b for a, b in zip(values_512, values_128)
                ],
            }

        matched_512 = per_seed["512:matched_M_minus_16"]["seed_values"]
        current_512 = per_seed["512:current_48"]["seed_values"]
        mean_diff, low, high = paired_bootstrap_ci(
            matched_512,
            current_512,
            rng_seed=20261003,
        )
        paired["matched_minus_current_at_512"] = {
            "mean_difference": mean_diff,
            "pooled_difference": (
                pooled["512:matched_M_minus_16"]["exhaustive_target_recall"]
                - pooled["512:current_48"]["exhaustive_target_recall"]
            ),
            "seed_bootstrap_95_ci_low": low,
            "seed_bootstrap_95_ci_high": high,
            "materiality_threshold": MATERIALITY_THRESHOLD,
            "materiality_supported": low > MATERIALITY_THRESHOLD,
        }

    return {
        "protocol": {
            "name": "TACOSM-C5-EXHAUSTIVE-REFERENCE-CALIBRATION-001",
            "smoke": smoke,
            "seeds": list(seeds),
            "M_levels": list(ms),
            "H_fixed": H_FIXED,
            "K": K,
            "eval_steps": steps,
            "target_teacher_updates": TARGET_TEACHER_UPDATES,
            "negative_count": NEGATIVE_COUNT,
            "latent_dim": LATENT_DIM,
            "target_code_count": len(TARGET_CODES),
            "arms": list(ARMS),
            "materiality_threshold": MATERIALITY_THRESHOLD,
        },
        "pooled": pooled,
        "per_seed": per_seed,
        "paired_stability": paired,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not args.smoke:
        contract = load_contract(
            "TACOSM-C5-EXHAUSTIVE-REFERENCE-CALIBRATION-001"
        )
        contract.require_levels((H_FIXED,))
        contract.require_m_levels(M_LEVELS)
        contract.require_seeds(SEEDS)
        contract.require_k_levels([K])
        contract.require_eval_steps(EVAL_STEPS)
        if contract.primary_endpoint() != "exhaustive_target_recall":
            raise AssertionError("primary endpoint drift")

    result = run(args.smoke)
    output = Path(
        "artifacts/TACOSM-C5-EXHAUSTIVE-REFERENCE-CALIBRATION-001.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
