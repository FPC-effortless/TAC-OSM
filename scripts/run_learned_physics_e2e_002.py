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

from tac_osm.learned_physics_e2e_002 import (
    LearnedPhysicsE2E002,
    LearnedPhysicsE2E002Config,
    hamiltonian_prior_loss,
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

EXPERIMENT_ID = "TACOSM-PLM-LEARNED-PHYSICS-E2E-002"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
SEEDS = (0, 1, 2, 3, 4)
TRAIN_PER_ACTION = 100
TRAIN_STEPS = 400
BATCH_SIZE = 64
EVAL_PER_ACTION = 120
PAIR_COUNT = 100
HIDDEN = 64
DT = 0.08


def pack_episodes(episodes):
    text = torch.stack([torch.stack([row[0] for row in e["observations"]]) for e in episodes])
    image = torch.stack([torch.stack([row[1] for row in e["observations"]]) for e in episodes])
    audio = torch.stack([torch.stack([row[2] for row in e["observations"]]) for e in episodes])
    goal = torch.stack([e["goal"] for e in episodes])
    target = torch.tensor([e["action"] for e in episodes], dtype=torch.long)
    return text, image, audio, goal, target


def train_arm(episodes, seed: int, physics_weight: float, *, steps: int):
    torch.manual_seed(seed)
    model = LearnedPhysicsE2E002(
        LearnedPhysicsE2E002Config(hidden_dim=HIDDEN, action_count=ACTION_COUNT)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    tensors = pack_episodes(episodes)
    rng = random.Random(seed + 41001)
    count = len(episodes)
    model.train()
    for _ in range(steps):
        indices = torch.tensor(
            [rng.randrange(count) for _ in range(BATCH_SIZE)],
            dtype=torch.long,
        )
        text, image, audio, goal, target = [tensor[indices] for tensor in tensors]
        logits, canonical, energy = model(text, image, audio, goal)
        loss = F.cross_entropy(logits, target)
        if physics_weight > 0.0:
            loss = loss + physics_weight * hamiltonian_prior_loss(
                canonical, energy, DT
            )["total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


@torch.no_grad()
def evaluate(model, tensors):
    model.eval()
    text, image, audio, goal, target = tensors
    logits, _, _ = model(text, image, audio, goal)
    probabilities = logits.softmax(dim=-1)
    predicted = probabilities.argmax(dim=-1)
    accuracy = float((predicted == target).float().mean())
    sorted_values = torch.sort(probabilities, dim=-1, descending=True).values
    margin = float((sorted_values[:, 0] - sorted_values[:, 1]).mean())
    return {"accuracy": accuracy, "mean_action_margin": margin}


@torch.no_grad()
def evaluate_pairs(model, pairs):
    model.eval()
    left_tensors = pack_episodes([left for left, _ in pairs])
    right_tensors = pack_episodes([right for _, right in pairs])
    left_actions = model(*left_tensors[:4])[0].argmax(dim=-1)
    right_actions = model(*right_tensors[:4])[0].argmax(dim=-1)
    left_targets = left_tensors[4]
    right_targets = right_tensors[4]
    exact = ((left_actions == left_targets) & (right_actions == right_targets)).float().mean()
    changed = (left_actions != right_actions).float().mean()
    return {
        "exact_pair_accuracy": float(exact),
        "history_conditioned_action_gap": float(changed),
    }


def paired_bootstrap(delta_by_seed, rounds=10000):
    rng = random.Random(20261007)
    seeds = sorted(delta_by_seed)
    values = []
    for _ in range(rounds):
        sample = [delta_by_seed[s] for s in rng.choices(seeds, k=len(seeds))]
        values.append(statistics.fmean(sample))
    values.sort()
    return [
        values[int(0.025 * rounds)],
        values[int(0.975 * rounds)],
    ]


def run(smoke=False):
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["status"] == "pre-registered"
    assert tuple(contract["seeds"]) == SEEDS
    assert contract["protocol"]["history_length"] == HISTORY
    assert contract["protocol"]["hidden_dim"] == HIDDEN
    assert contract["protocol"]["benchmark_generator_version"] == benchmark_manifest()["generator_version"]
    assert contract["physics_prior"]["allowed_information"] == benchmark_manifest()["physics_prior"]

    seeds = (0,) if smoke else SEEDS
    train_per_action = 8 if smoke else TRAIN_PER_ACTION
    train_steps = 12 if smoke else TRAIN_STEPS
    eval_per_action = 4 if smoke else EVAL_PER_ACTION
    pair_count = 4 if smoke else PAIR_COUNT

    rows = {"learned_data_only": [], "learned_physics_prior": []}

    for seed in seeds:
        train_pool = sample_balanced_episodes(
            random.Random(seed + 40001), train_per_action
        )
        evaluation = sample_balanced_episodes(
            random.Random(seed + 40002), eval_per_action
        )
        training_keys = {episode_key(e) for e in train_pool}
        evaluation_keys = {episode_key(e) for e in evaluation}
        assert not (training_keys & evaluation_keys), f"train/eval overlap seed={seed}"
        fingerprint = episode_fingerprint(evaluation)
        evaluation_tensors = pack_episodes(evaluation)
        pairs = sample_history_pairs(random.Random(seed + 40003), pair_count)

        for name, weight in (
            ("learned_data_only", 0.0),
            ("learned_physics_prior", 0.05),
        ):
            model = train_arm(train_pool, seed, weight, steps=train_steps)
            metrics = evaluate(model, evaluation_tensors)
            pair_metrics = evaluate_pairs(model, pairs)
            sample = tuple(t[: min(32, len(evaluation))] for t in evaluation_tensors[:4])
            with torch.enable_grad():
                _, canonical, energy = model(*sample, )
                prior = hamiltonian_prior_loss(canonical, energy, DT)
            rows[name].append(
                {
                    "seed": seed,
                    "evaluation_episode_fingerprint": fingerprint,
                    "training_evaluation_overlap": len(training_keys & evaluation_keys),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "evaluation_accuracy": metrics["accuracy"],
                    "mean_action_margin": metrics["mean_action_margin"],
                    "history_pair_exact_accuracy": pair_metrics["exact_pair_accuracy"],
                    "history_conditioned_action_gap": pair_metrics["history_conditioned_action_gap"],
                    "physics_loss_total": float(prior["total"]),
                }
            )

    assert len(rows["learned_data_only"]) == len(rows["learned_physics_prior"]) == len(seeds)
    assert [
        r["evaluation_episode_fingerprint"] for r in rows["learned_data_only"]
    ] == [
        r["evaluation_episode_fingerprint"] for r in rows["learned_physics_prior"]
    ]
    assert len(
        {r["evaluation_episode_fingerprint"] for r in rows["learned_physics_prior"]}
    ) == len(seeds)
    assert all(
        r["training_evaluation_overlap"] == 0
        for arm in rows.values()
        for r in arm
    )
    assert len({r["parameter_count"] for arm in rows.values() for r in arm}) == 1

    data_values = [r["evaluation_accuracy"] for r in rows["learned_data_only"]]
    physics_values = [r["evaluation_accuracy"] for r in rows["learned_physics_prior"]]
    data_mean = statistics.fmean(data_values)
    physics_mean = statistics.fmean(physics_values)
    delta_by_seed = {
        data_row["seed"]: physics_row["evaluation_accuracy"] - data_row["evaluation_accuracy"]
        for data_row, physics_row in zip(
            rows["learned_data_only"], rows["learned_physics_prior"]
        )
    }
    delta = physics_mean - data_mean
    ci = None if smoke else paired_bootstrap(delta_by_seed)
    data_pair_exact = statistics.fmean(
        r["history_pair_exact_accuracy"] for r in rows["learned_data_only"]
    )

    summary = {
        "data_only_mean_accuracy": data_mean,
        "data_only_min_seed_accuracy": min(data_values),
        "data_only_history_pair_exact_accuracy": data_pair_exact,
        "physics_prior_mean_accuracy": physics_mean,
        "physics_prior_min_seed_accuracy": min(physics_values),
        "physics_prior_history_pair_exact_accuracy": statistics.fmean(
            r["history_pair_exact_accuracy"] for r in rows["learned_physics_prior"]
        ),
        "physics_prior_history_conditioned_action_gap": statistics.fmean(
            r["history_conditioned_action_gap"] for r in rows["learned_physics_prior"]
        ),
        "physics_prior_delta": delta,
        "physics_prior_delta_ci95": ci,
        "end_to_end_gate_pass": bool(
            not smoke
            and data_mean >= 0.70
            and min(data_values) >= 0.55
            and data_pair_exact >= 0.70
        ),
        "physics_prior_benefit_pass": bool(
            not smoke
            and delta >= 0.05
            and ci is not None
            and ci[0] > 0.0
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
            "no_task_specific_physical_ontology": True,
            "no_post_run_selection": True,
            "physics_prior_manifest_matches_contract": True,
        },
        "seed_results": rows,
        "summary": summary,
        "benchmark_manifest": benchmark_manifest(),
        "claim_boundary": [
            "fully learned synthetic multimodal physical control only",
            "physics prior restricted to generic Hamiltonian equations and learned-H conservation",
            "no general physical reasoning claim",
            "no real-world multimodal claim",
            "no learned semantic addressing claim",
            "no scaling or hardware claim",
        ],
    }

    path = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(smoke=parser.parse_args().smoke)
