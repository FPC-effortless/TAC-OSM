#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.learned_physics_e2e import (
    LearnedPhysicsConfig,
    LearnedPhysicsE2E,
    physics_prior_loss,
)
from tac_osm.learned_physics_e2e_benchmark import (
    ACTION_COUNT,
    HISTORY,
    benchmark_manifest,
    episode_fingerprint,
    episode_key,
    generator_hash,
    sample_balanced_episodes,
    sample_history_pairs,
)

EXPERIMENT_ID = "TACOSM-PLM-LEARNED-PHYSICS-E2E-001"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
SEEDS = (0, 1, 2, 3, 4)
TRAIN_PER_ACTION = 100
TRAIN_STEPS = 400
BATCH_SIZE = 64
EVAL_PER_ACTION = 120
PAIR_COUNT = 100
HIDDEN = 64
DT = 0.08


def build_batch(episodes):
    text = torch.stack(
        [torch.stack([row[0] for row in episode["observations"]]) for episode in episodes]
    )
    image = torch.stack(
        [torch.stack([row[1] for row in episode["observations"]]) for episode in episodes]
    )
    audio = torch.stack(
        [torch.stack([row[2] for row in episode["observations"]]) for episode in episodes]
    )
    goal = torch.stack([episode["goal"] for episode in episodes])
    target = torch.tensor([episode["action"] for episode in episodes], dtype=torch.long)
    return text, image, audio, goal, target


def train_arm(episodes, seed: int, physics_weight: float, *, steps: int):
    torch.manual_seed(seed)
    model = LearnedPhysicsE2E(
        LearnedPhysicsConfig(
            hidden_dim=HIDDEN,
            action_count=ACTION_COUNT,
            physics_weight=physics_weight,
        )
    ).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.0001,
    )
    rng = random.Random(seed + 41001)
    for _ in range(steps):
        batch = [episodes[rng.randrange(len(episodes))] for _ in range(BATCH_SIZE)]
        text, image, audio, goal, target = build_batch(batch)
        logits, states = model(text, image, audio, goal)
        loss = F.cross_entropy(logits, target)
        if physics_weight > 0.0:
            loss = loss + physics_weight * physics_prior_loss(states, DT)["total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


@torch.no_grad()
def evaluate(model, episodes):
    model.eval()
    correct = 0
    margins = []
    for episode in episodes:
        text, image, audio, goal, target = build_batch([episode])
        logits, _ = model(text, image, audio, goal)
        probability = logits.softmax(dim=-1)
        predicted = int(probability.argmax(dim=-1).item())
        correct += int(predicted == int(target.item()))
        ordered = torch.sort(probability[0], descending=True).values
        margins.append(float(ordered[0] - ordered[1]))
    return {
        "accuracy": correct / len(episodes),
        "mean_action_margin": statistics.fmean(margins),
    }


@torch.no_grad()
def evaluate_pairs(model, pairs):
    model.eval()
    exact = 0
    action_gap = 0
    for left, right in pairs:
        left_batch = build_batch([left])
        right_batch = build_batch([right])
        left_action = int(model(*left_batch[:4])[0].argmax(dim=-1).item())
        right_action = int(model(*right_batch[:4])[0].argmax(dim=-1).item())
        exact += int(
            left_action == int(left["action"])
            and right_action == int(right["action"])
        )
        action_gap += int(left_action != right_action)
    return {
        "exact_pair_accuracy": exact / len(pairs),
        "history_conditioned_action_gap": action_gap / len(pairs),
    }


def paired_bootstrap(delta_by_seed, rounds=10000):
    rng = random.Random(20261007)
    seeds = sorted(delta_by_seed)
    values = []
    for _ in range(rounds):
        chosen = rng.choices(seeds, k=len(seeds))
        values.append(statistics.fmean(delta_by_seed[seed] for seed in chosen))
    values.sort()
    return [values[int(0.025 * rounds)], values[int(0.975 * rounds)]]


def run(smoke: bool = False):
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["status"] == "pre-registered"
    assert tuple(contract["seeds"]) == SEEDS
    assert contract["protocol"]["history_length"] == HISTORY
    assert contract["protocol"]["hidden_dim"] == HIDDEN
    assert contract["protocol"]["dt"] == DT

    seeds = (0,) if smoke else SEEDS
    train_per_action = 8 if smoke else TRAIN_PER_ACTION
    train_steps = 12 if smoke else TRAIN_STEPS
    eval_per_action = 4 if smoke else EVAL_PER_ACTION
    pair_count = 4 if smoke else PAIR_COUNT
    rows = {"learned_data_only": [], "learned_physics_prior": []}

    for seed in seeds:
        train_pool = sample_balanced_episodes(random.Random(seed + 40001), train_per_action)
        evaluation = sample_balanced_episodes(random.Random(seed + 40002), eval_per_action)
        training_keys = {episode_key(episode) for episode in train_pool}
        evaluation_keys = {episode_key(episode) for episode in evaluation}
        assert not (training_keys & evaluation_keys), f"train/eval overlap seed={seed}"
        fingerprint = episode_fingerprint(evaluation)
        pairs = sample_history_pairs(random.Random(seed + 40003), pair_count)

        for name, weight in (
            ("learned_data_only", 0.0),
            ("learned_physics_prior", 0.05),
        ):
            model = train_arm(train_pool, seed, weight, steps=train_steps)
            metrics = evaluate(model, evaluation)
            pair_metrics = evaluate_pairs(model, pairs)
            state_batch = build_batch(evaluation[:min(32, len(evaluation))])
            _, predicted_states = model(*state_batch[:4])
            prior = physics_prior_loss(predicted_states, DT)
            rows[name].append(
                {
                    "seed": seed,
                    "evaluation_episode_fingerprint": fingerprint,
                    "training_evaluation_overlap": len(training_keys & evaluation_keys),
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "evaluation_accuracy": metrics["accuracy"],
                    "mean_action_margin": metrics["mean_action_margin"],
                    "history_pair_exact_accuracy": pair_metrics["exact_pair_accuracy"],
                    "history_conditioned_action_gap": pair_metrics["history_conditioned_action_gap"],
                    "physics_loss_total": float(prior["total"]),
                }
            )

    assert len(rows["learned_data_only"]) == len(rows["learned_physics_prior"]) == len(seeds)
    assert [
        row["evaluation_episode_fingerprint"]
        for row in rows["learned_data_only"]
    ] == [
        row["evaluation_episode_fingerprint"]
        for row in rows["learned_physics_prior"]
    ]
    assert len({
        row["evaluation_episode_fingerprint"]
        for row in rows["learned_physics_prior"]
    }) == len(seeds)
    assert all(
        row["training_evaluation_overlap"] == 0
        for arm in rows.values()
        for row in arm
    )
    assert len({
        row["parameter_count"]
        for arm in rows.values()
        for row in arm
    }) == 1

    data_only_mean = statistics.fmean(
        row["evaluation_accuracy"]
        for row in rows["learned_data_only"]
    )
    physics_values = [
        row["evaluation_accuracy"]
        for row in rows["learned_physics_prior"]
    ]
    physics_mean = statistics.fmean(physics_values)
    delta_by_seed = {
        row["seed"]: (
            rows["learned_physics_prior"][index]["evaluation_accuracy"]
            - row["evaluation_accuracy"]
        )
        for index, row in enumerate(rows["learned_data_only"])
    }
    physics_delta = physics_mean - data_only_mean
    delta_ci = None if smoke else paired_bootstrap(delta_by_seed)
    pair_exact = statistics.fmean(
        row["history_pair_exact_accuracy"]
        for row in rows["learned_physics_prior"]
    )

    summary = {
        "data_only_mean_accuracy": data_only_mean,
        "physics_prior_mean_accuracy": physics_mean,
        "physics_prior_min_seed_accuracy": min(physics_values),
        "physics_prior_delta": physics_delta,
        "physics_prior_delta_ci95": delta_ci,
        "physics_prior_history_pair_exact_accuracy": pair_exact,
        "physics_prior_history_conditioned_action_gap": statistics.fmean(
            row["history_conditioned_action_gap"]
            for row in rows["learned_physics_prior"]
        ),
        "end_to_end_gate_pass": bool(
            not smoke
            and physics_mean >= 0.70
            and min(physics_values) >= 0.55
            and pair_exact >= 0.70
        ),
        "physics_prior_benefit_pass": bool(
            not smoke
            and physics_delta >= 0.05
            and delta_ci is not None
            and delta_ci[0] > 0.0
        ),
    }

    output = {
        "experiment_id": EXPERIMENT_ID,
        "status": "smoke" if smoke else "measured",
        "provenance": {
            "git_commit": os.environ.get("GITHUB_SHA", "unknown"),
            "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
            "benchmark_sha256": generator_hash(),
        },
        "protocol": {
            "seeds": list(seeds),
            "train_pool_per_action": train_per_action,
            "train_steps": train_steps,
            "batch_size": BATCH_SIZE,
            "evaluation_per_action": eval_per_action,
            "history_pair_count": pair_count,
            "history_length": HISTORY,
            "hidden_dim": HIDDEN,
            "parameter_counts_equal": True,
            "model_selection": "none",
        },
        "integrity": {
            "training_evaluation_semantic_overlap_zero": True,
            "same_evaluation_fingerprint_across_arms": True,
            "distinct_evaluation_fingerprints_across_seeds": True,
            "parameter_counts_equal": True,
            "no_physical_state_supervision": True,
            "no_entity_ids": True,
            "no_fixed_operator_table": True,
            "no_post_run_selection": True,
        },
        "seed_results": rows,
        "summary": summary,
        "benchmark_manifest": benchmark_manifest(),
        "claim_boundary": [
            "fully learned synthetic multimodal physical control only",
            "physics prior restricted to registered conservation and kinematic laws",
            "no general physical reasoning claim",
            "no real-world multimodal claim",
            "no learned semantic addressing claim",
            "no scaling or hardware claim",
        ],
    }

    output_path = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(smoke=parser.parse_args().smoke)
