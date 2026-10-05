#!/usr/bin/env python3
"""PLM-018A: train and evaluate an amortized target-blind probe policy.

The exact teacher sees the full candidate evidence partition. The learned policy
sees only compact first/second-order statistics of the current six-bit trace
belief. Training and validation are completely separated from the held-out test
seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_graph_casm_structured_action_probe_015 as g15
from tac_osm.amortized_probe_policy import (
    AmortizedProbePolicy,
    budget_capped_utility,
    choose_greedy_budget_action,
    make_action_sketch_matrix,
)

EXPERIMENT_ID = "TACOSM-PLM-AMORTIZED-PROBE-POLICY-018A"
GENERATOR_COMMIT = g15.GENERATOR_COMMIT
TRAIN_SEEDS = tuple(range(20))
VAL_SEEDS = tuple(range(20, 25))
TEST_SEEDS = tuple(range(25, 30))
M_LEVELS = (32, 64, 128, 256, 512)
TASKS_PER_M = 32
TRAIN_SUBSETS_PER_M = 8
BUDGET = 8
HIDDEN = 64
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
TRAIN_SEED = 18018
POLICY_MACS_PER_ACTION = 28 * HIDDEN + HIDDEN * HIDDEN + HIDDEN
INPUT_ROWS = g15.INPUT_ROWS
TRACE_WIDTH = g15.TRACE_READ_UNITS
ACTION_ROWS = {i: tuple(int(x) for x in INPUT_ROWS[i]) for i in range(len(INPUT_ROWS))}
ACTION_COSTS = {i: statistics.fmean(
    g15.build_cache(g15.g010.generate(0, 1))[0].trace[(0, i)][1] + TRACE_WIDTH
) for i in range(len(INPUT_ROWS))}
# The action costs above are only a shape placeholder. Actual costs are recomputed
# from each library below and never used as training labels without the library.


@dataclass(frozen=True)
class LibraryBundle:
    library: tuple
    cache: object


def seed_task_index(seed: int, m: int, task_i: int) -> int:
    rng = random.Random(seed * 3000001 + m * 10007 + task_i * 1009 + 17)
    return rng.randrange(m)


def split_library(seed: int):
    training = g15.g010.generate(seed + 100, g15.TRAIN_PROGRAMS)
    train_structures = {g15.g010.structure_key(ep) for ep in training}
    train_truths = {g15.g010.truth_signature(ep) for ep in training}
    library = g15.g010.generate(
        seed + 5000,
        g15.LIBRARY_SIZE,
        exclude_structures=train_structures,
        exclude_truths=train_truths,
    )
    if len(library) != g15.LIBRARY_SIZE:
        raise RuntimeError("evaluation library construction failed")
    if train_structures & {g15.g010.structure_key(ep) for ep in library}:
        raise RuntimeError("train/evaluation structural leakage")
    if train_truths & {g15.g010.truth_signature(ep) for ep in library}:
        raise RuntimeError("train/evaluation truth-table leakage")
    cache, _ = g15.build_cache(library)
    return LibraryBundle(tuple(library), cache)


def action_stats(bundle: LibraryBundle, indices: Sequence[int], row: int):
    signatures = tuple(bundle.cache.trace[(idx, row)][0] for idx in indices)
    cost = statistics.fmean(
        bundle.cache.trace[(idx, row)][1] + TRACE_WIDTH for idx in indices
    )
    counts = {}
    for sig in signatures:
        counts[sig] = counts.get(sig, 0) + 1
    ub = budget_capped_utility(signatures, BUDGET)
    baseline = min(BUDGET, len(indices)) / len(indices)
    return signatures, cost, ub, baseline


def teacher_for_state(bundle: LibraryBundle, indices: Sequence[int], allowed_rows: Sequence[int]):
    evidence = {}
    costs = {}
    for row in allowed_rows:
        sig, cost, _, _ = action_stats(bundle, indices, row)
        evidence[row] = sig
        costs[row] = cost
    teacher = choose_greedy_budget_action(
        evidence,
        budget=BUDGET,
        acquisition_cost_by_action=costs,
    )
    teacher_rows = [row for row in allowed_rows if row == teacher.action]
    if not teacher_rows:
        raise RuntimeError("teacher chose an unavailable action")
    return teacher


def make_example(bundle: LibraryBundle, indices: Sequence[int]):
    allowed = tuple(range(len(INPUT_ROWS)))
    features, actions = make_action_sketch_matrix(
        {
            row: [bundle.cache.trace[(idx, row)][0] for idx in indices]
            for row in allowed
        },
        {row: ACTION_ROWS[row] for row in allowed},
        {
            row: statistics.fmean(
                bundle.cache.trace[(idx, row)][1] + TRACE_WIDTH for idx in indices
            )
            for row in allowed
        },
        budget=BUDGET,
    )
    teacher = teacher_for_state(bundle, indices, allowed)
    teacher_scores = []
    for row in actions:
        sig, cost, ub, baseline = action_stats(bundle, indices, int(row))
        teacher_scores.append((ub - baseline) / cost)
    return features, actions, teacher.action, teacher_scores


def random_training_subset(seed: int, m: int, subset_i: int):
    rng = random.Random(seed * 9176 + m * 131 + subset_i * 1009 + 11)
    size = max(2, int(round(m * (0.55 + 0.40 * rng.random()))))
    indices = sorted(rng.sample(range(m), size))
    return tuple(indices)


def build_dataset(seeds: Sequence[int], *, subsets_per_m: int):
    xs = []
    ys = []
    soft_scores = []
    metadata = []
    bundles = {}
    for seed in seeds:
        bundles[seed] = split_library(seed)
        for m in M_LEVELS:
            for subset_i in range(subsets_per_m):
                idx = random_training_subset(seed, m, subset_i)
                features, actions, teacher, scores = make_example(
                    bundles[seed], idx
                )
                xs.append(features)
                ys.append(actions.index(teacher))
                soft_scores.append(torch.tensor(scores, dtype=torch.float32))
                metadata.append((seed, m, subset_i, len(idx)))
    return xs, torch.tensor(ys, dtype=torch.long), torch.stack(soft_scores), metadata, bundles


def evaluate_policy(
    model: AmortizedProbePolicy,
    bundles: dict[int, LibraryBundle],
    seeds: Sequence[int],
):
    rows_by_seed = {}
    model.eval()
    for seed in seeds:
        bundle = bundles[seed]
        rows_by_seed[seed] = {}
        for m in M_LEVELS:
            task_rows = []
            for task_i in range(TASKS_PER_M):
                target = seed_task_index(seed, m, task_i)
                indices = tuple(range(m))
                features, actions, teacher, teacher_scores = make_example(
                    bundle, indices
                )
                with torch.no_grad():
                    predicted_pos = model.choose(features)
                predicted_action = int(actions[predicted_pos])
                teacher_pos = actions.index(teacher)

                selected_signature = bundle.cache.trace[(target, predicted_action)][0]
                evidence = tuple(
                    bundle.cache.trace[(idx, predicted_action)][0]
                    for idx in indices
                )
                bucket = tuple(
                    idx for idx, sig in enumerate(evidence)
                    if sig == selected_signature
                )
                shortlist = bucket[:BUDGET]
                verified_success = float(target in shortlist)

                chosen_sig, chosen_cost, chosen_ub, baseline = action_stats(
                    bundle, indices, predicted_action
                )
                teacher_ub = teacher.budget_utility
                teacher_score = teacher.utility_per_work
                learner_score = (chosen_ub - baseline) / chosen_cost
                max_score = max(float(x) for x in teacher_scores)
                normalized_regret = (
                    (max_score - learner_score) / max_score
                    if max_score > 1e-12 else 0.0
                )
                task_rows.append({
                    "task_i": task_i,
                    "target_index": target,
                    "teacher_row": int(teacher.action),
                    "predicted_row": predicted_action,
                    "teacher_agreement": float(predicted_action == int(teacher.action)),
                    "teacher_u8": float(teacher_ub),
                    "learner_u8": float(chosen_ub),
                    "teacher_utility_per_work": float(teacher_score),
                    "learner_utility_per_work": float(learner_score),
                    "normalized_regret": float(normalized_regret),
                    "verified_success": verified_success,
                    "target_bucket_size": len(bucket),
                    "shortlist_size": len(shortlist),
                    "policy_mac": POLICY_MACS_PER_ACTION * len(actions),
                })
            rows_by_seed[seed][m] = task_rows
    return rows_by_seed


def seed_bootstrap(values: Sequence[float], seed: int, rounds: int = 4000):
    if not values:
        return [None, None]
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(rounds)
    )
    return [
        float(draws[int(0.025 * (rounds - 1))]),
        float(draws[int(0.975 * (rounds - 1))]),
    ]


def train_model(features: Tensor, labels: Tensor, teacher_scores: Tensor, val_x, val_y, val_scores):
    torch.manual_seed(TRAIN_SEED)
    model = AmortizedProbePolicy(feature_dim=int(features.shape[-1]), hidden_dim=HIDDEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    dataset = TensorDataset(features, labels, teacher_scores)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(TRAIN_SEED))
    best_state = None
    best_metric = float("inf")
    history = []

    for epoch in range(EPOCHS):
        model.train()
        loss_values = []
        for xb, yb, scoreb in loader:
            logits = model(xb)
            # Hard teacher action label plus a soft ranking target derived from
            # the exact teacher score. The learner never receives these scores at
            # inference time.
            with torch.no_grad():
                centered = scoreb - scoreb.max(dim=1, keepdim=True).values
                soft_target = torch.softmax(centered * 20.0, dim=1)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = F.nll_loss(log_probs, yb) + 0.25 * F.kl_div(
                log_probs, soft_target, reduction="batchmean"
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_values.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_pred = torch.argmax(val_logits, dim=1)
            val_agreement = float((val_pred == val_y).float().mean())
            val_scores_np = val_scores.numpy()
            pred_score = val_scores_np[
                torch.arange(len(val_pred)).numpy(), val_pred.numpy()
            ]
            best_score = val_scores_np.max(axis=1)
            regret = float(((best_score - pred_score) / (best_score + 1e-12)).mean())
        history.append({
            "epoch": epoch,
            "train_loss": statistics.fmean(loss_values),
            "validation_teacher_agreement": val_agreement,
            "validation_normalized_regret": regret,
        })
        if regret < best_metric:
            best_metric = regret
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    return model, history


def main(smoke=False):
    from tac_osm.contract import load_contract

    contract = load_contract(EXPERIMENT_ID)
    if smoke:
        train_seeds = (0,)
        val_seeds = (20,)
        test_seeds = (25,)
        train_subsets = 1
        eval_tasks = 4
        levels = (32, 128)
    else:
        train_seeds = TRAIN_SEEDS
        val_seeds = VAL_SEEDS
        test_seeds = TEST_SEEDS
        train_subsets = TRAIN_SUBSETS_PER_M
        eval_tasks = TASKS_PER_M
        levels = M_LEVELS
        contract.require_levels(M_LEVELS)
        contract.require_seeds(TEST_SEEDS)
        contract.require_steps(1)
        contract.require_eval_steps(TASKS_PER_M)
        contract.require_arms([
            "greedy_budget_teacher",
            "greedy_information_teacher",
            "amortized_policy",
            "fixed_random",
        ])

    global M_LEVELS, TASKS_PER_M
    original_levels = M_LEVELS
    original_tasks = TASKS_PER_M
    M_LEVELS = levels
    TASKS_PER_M = eval_tasks

    train_x_list, train_y, train_scores, train_meta, train_bundles = build_dataset(
        train_seeds, subsets_per_m=train_subsets
    )
    val_x_list, val_y, val_scores, val_meta, val_bundles = build_dataset(
        val_seeds, subsets_per_m=1
    )
    train_x = torch.cat(train_x_list, dim=0)
    val_x = torch.cat(val_x_list, dim=0)
    model, history = train_model(
        train_x, train_y, train_scores, val_x, val_y, val_scores
    )
    test_bundles = {seed: split_library(seed) for seed in test_seeds}
    evaluated = evaluate_policy(model, test_bundles, test_seeds)

    seed_regrets = []
    seed_verified = []
    seed_agreement = []
    seed_u8 = []
    for seed in test_seeds:
        vals = [
            row for m in levels for row in evaluated[seed][m]
        ]
        primary_vals = evaluated[seed][512] if 512 in evaluated[seed] else vals
        seed_regrets.append(statistics.fmean(r["normalized_regret"] for r in primary_vals))
        seed_verified.append(statistics.fmean(r["verified_success"] for r in primary_vals))
        seed_agreement.append(statistics.fmean(r["teacher_agreement"] for r in primary_vals))
        seed_u8.append(statistics.fmean(r["learner_u8"] for r in primary_vals))

    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "provenance": {
            "tacosm_commit": os.environ.get("GITHUB_SHA", "local"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "generator_commit": GENERATOR_COMMIT,
            "python": sys.version,
            "torch": torch.__version__,
        },
        "protocol": {
            "train_seeds": list(train_seeds),
            "validation_seeds": list(val_seeds),
            "test_seeds": list(test_seeds),
            "M_levels": list(levels),
            "B": BUDGET,
            "tasks_per_test_seed_M": eval_tasks,
            "feature_dim": int(train_x.shape[-1]),
            "hidden_dim": HIDDEN,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "train_subsets_per_seed_M": train_subsets,
        },
        "split_integrity": {
            "train_eval_structure_disjoint_checked": True,
            "train_eval_truth_disjoint_checked": True,
            "test_teacher_outputs_generated_after_checkpoint_freeze": True,
            "test_teacher_outputs_used_for_training": False,
            "test_targets_used_for_action_selection": False,
        },
        "training_history": history,
        "summary": {
            "primary": {
                "M": 512 if 512 in levels else None,
                "seed_level_normalized_regret": seed_regrets,
                "mean_normalized_regret": statistics.fmean(seed_regrets) if seed_regrets else None,
                "seed_bootstrap_95ci": seed_bootstrap(seed_regrets, 18019) if seed_regrets else None,
            },
            "secondary": {
                "seed_verified_success": seed_verified,
                "mean_verified_success": statistics.fmean(seed_verified) if seed_verified else None,
                "seed_teacher_agreement": seed_agreement,
                "mean_teacher_agreement": statistics.fmean(seed_agreement) if seed_agreement else None,
                "seed_u8": seed_u8,
                "mean_u8": statistics.fmean(seed_u8) if seed_u8 else None,
                "policy_mac_per_action_set": POLICY_MACS_PER_ACTION * len(INPUT_ROWS),
            },
        },
        "results": evaluated,
        "scope": {
            "finite_domain_amortized_policy": True,
            "target_blind_action_selection": True,
            "learned_probe_policy_claim": True,
            "general_active_learning_claim": False,
            "submodularity_claim": False,
            "asymptotic_claim": False,
            "external_generalization_claim": False,
            "language_image_audio_claim": False,
            "hardware_speedup_claim": False,
        },
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts", f"{EXPERIMENT_ID}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    M_LEVELS = original_levels
    TASKS_PER_M = original_tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args().smoke)
