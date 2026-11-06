#!/usr/bin/env python3
"""Paired development benchmark for integrated PLM capacity.

This lane is for optimization/model selection only. It never evaluates the
sealed E2E-005 held-out compositions and cannot authorize a confirmatory claim.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import statistics
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.integrated_e2e_005 import FunctionalConfig, FunctionalMultimodalPLM
from tac_osm.integrated_e2e_005_benchmark import (
    ALL_COMBOS,
    HELDOUT,
    TRAIN_COMBOS,
    episode_fingerprint,
    episode_key,
    sample_episode,
)

EXPERIMENT_ID = "TACOSM-PLM-INTEGRATED-E2E-CAPACITY-DEV-001"
SEEDS = (0, 1, 2, 3, 4)
STEPS = 300
BATCH_SIZE = 96
EVAL_EPISODES = 200

DEV_COMBOS = (
    ("xor", 0, 5), ("xor", 0, 6), ("xor", 0, 7), ("xor", 0, 8),
    ("xor", 0, 9), ("xor", 0, 10), ("xor", 0, 11), ("xor", 1, 4),
    ("and", 0, 4), ("and", 0, 5), ("and", 0, 6), ("and", 0, 7),
    ("and", 0, 8), ("and", 0, 9), ("and", 0, 10), ("and", 0, 11),
    ("or", 0, 4), ("or", 0, 5), ("or", 0, 6), ("or", 0, 7),
    ("or", 0, 8), ("or", 0, 9), ("or", 0, 10), ("or", 0, 11),
    ("xnor", 0, 4), ("xnor", 0, 5), ("xnor", 0, 6), ("xnor", 0, 7),
    ("xnor", 0, 8), ("xnor", 0, 9), ("xnor", 0, 10), ("xnor", 0, 11),
)
DEV_SET = set(DEV_COMBOS)
SEALED_SET = set(HELDOUT)
TRAIN_DEV_COMBOS = tuple(
    combo
    for combo in ALL_COMBOS
    if combo not in DEV_SET and combo not in SEALED_SET
)

assert DEV_SET.isdisjoint(SEALED_SET)
assert DEV_SET <= set(TRAIN_COMBOS)
assert len(DEV_COMBOS) == 32
assert len(TRAIN_DEV_COMBOS) == 224
assert len(TRAIN_DEV_COMBOS) + len(DEV_COMBOS) + len(HELDOUT) == len(ALL_COMBOS)


def config_hash(config: FunctionalConfig) -> str:
    payload = {
        "hidden_dim": config.hidden_dim,
        "entity_count": config.entity_count,
        "latent_bits": config.latent_bits,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def sample_train_episode(rng: random.Random):
    combo1 = rng.choice(TRAIN_DEV_COMBOS)
    combo2 = rng.choice(TRAIN_DEV_COMBOS)
    return sample_episode(rng, combo1=combo1, combo2=combo2)


def sample_dev_episode(rng: random.Random, index: int):
    combo1 = DEV_COMBOS[index % len(DEV_COMBOS)]
    combo2 = DEV_COMBOS[(index * 7 + 3) % len(DEV_COMBOS)]
    if combo1 == combo2:
        combo2 = DEV_COMBOS[(index * 7 + 4) % len(DEV_COMBOS)]
    return sample_episode(rng, combo1=combo1, combo2=combo2)


def build_batch(episodes: list[tuple]) -> dict:
    from scripts.run_integrated_e2e_005 import build_batch as canonical_build_batch
    return canonical_build_batch(episodes)


def train_seed(seed: int, config: FunctionalConfig) -> tuple[nn.Module, set[tuple]]:
    torch.manual_seed(seed)
    rng = random.Random(seed + 700000)
    model = FunctionalMultimodalPLM(config=config).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.0001,
    )
    training_keys: set[tuple] = set()
    from scripts.run_integrated_e2e_005 import train_batch

    for _ in range(STEPS):
        episodes = [sample_train_episode(rng) for _ in range(BATCH_SIZE)]
        training_keys.update(episode_key(ep) for ep in episodes)
        batch = build_batch(episodes)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = train_batch(model, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model, training_keys


@torch.no_grad()
def evaluate(model: nn.Module, episodes: list[tuple]) -> dict:
    model.eval()
    correct = 0
    positives = 0
    mismatches = 0
    per_op = {op: [0, 0] for op in ("xor", "and", "or", "xnor")}

    for episode in episodes:
        memory = model.state.initial(1, torch.device("cpu"))
        for entity, text, image, audio in episode[0]:
            z = model.encode(
                text.unsqueeze(0),
                image.unsqueeze(0),
                audio.unsqueeze(0).unsqueeze(0),
            )
            memory, _ = model.state.write(
                memory,
                z,
                torch.tensor([entity], dtype=torch.long),
            )

        entity, i, j, op_name, target = episode[2]
        op_idx = ("xor", "and", "or", "xnor").index(op_name)
        out = model.query(
            memory,
            torch.tensor([entity], dtype=torch.long),
            torch.tensor([i], dtype=torch.long),
            torch.tensor([j], dtype=torch.long),
            torch.tensor([op_idx], dtype=torch.long),
        )
        pred = int(out["logits"].argmax(-1).item())
        threshold_pred = int((out["action"] >= 0.5).long().item())
        mismatches += int(pred != threshold_pred)
        positives += pred
        correct += int(pred == target)
        per_op[op_name][0] += int(pred == target)
        per_op[op_name][1] += 1

    n = len(episodes)
    return {
        "q2_accuracy": correct / n,
        "q2_positive_fraction": positives / n,
        "q2_decision_mismatches": mismatches,
        "per_op_accuracy": {
            op: per_op[op][0] / per_op[op][1] for op in per_op
        },
    }


def run() -> dict:
    results = []
    for seed in SEEDS:
        eval_rng = random.Random(seed + 800000)
        episodes = [sample_dev_episode(eval_rng, i) for i in range(EVAL_EPISODES)]
        eval_keys = {episode_key(ep) for ep in episodes}

        model40, keys40 = train_seed(seed, FunctionalConfig(hidden_dim=40))
        overlap40 = keys40 & eval_keys
        if overlap40:
            raise AssertionError(f"hidden40 train/dev overlap for seed {seed}")

        # Reuse exactly the same training episode stream for the candidate by
        # deriving a deterministic paired stream from the seed.
        # The model is trained separately; only the sampled episodes are shared.
        # This keeps the capacity comparison paired without sharing weights.
        torch.manual_seed(seed)
        rng = random.Random(seed + 700000)
        model64 = FunctionalMultimodalPLM(
            config=FunctionalConfig(hidden_dim=64)
        ).cpu()
        optimizer64 = torch.optim.AdamW(
            model64.parameters(),
            lr=0.002,
            weight_decay=0.0001,
        )
        keys64: set[tuple] = set()
        from scripts.run_integrated_e2e_005 import train_batch
        for _ in range(STEPS):
            episodes_batch = [sample_train_episode(rng) for _ in range(BATCH_SIZE)]
            keys64.update(episode_key(ep) for ep in episodes_batch)
            batch = build_batch(episodes_batch)
            model64.train()
            optimizer64.zero_grad(set_to_none=True)
            loss = train_batch(model64, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model64.parameters(), 1.0)
            optimizer64.step()
        overlap64 = keys64 & eval_keys
        if overlap64:
            raise AssertionError(f"hidden64 train/dev overlap for seed {seed}")

        # A fresh identical evaluation object stream is shared by both models.
        baseline = evaluate(model40, episodes)
        candidate = evaluate(model64, episodes)
        results.append({
            "seed": seed,
            "evaluation_fingerprint": episode_fingerprint(episodes),
            "training_dev_overlap_hidden40": len(overlap40),
            "training_dev_overlap_hidden64": len(overlap64),
            "hidden40": baseline,
            "hidden64": candidate,
        })

    baseline_means = [r["hidden40"]["q2_accuracy"] for r in results]
    candidate_means = [r["hidden64"]["q2_accuracy"] for r in results]
    summary = {
        "hidden40_mean_q2": statistics.fmean(baseline_means),
        "hidden64_mean_q2": statistics.fmean(candidate_means),
        "hidden40_min_q2": min(baseline_means),
        "hidden64_min_q2": min(candidate_means),
        "mean_gain": statistics.fmean(candidate_means) - statistics.fmean(baseline_means),
        "candidate_working_on_dev": bool(
            statistics.fmean(candidate_means) >= 0.80
            and min(candidate_means) >= 0.40
        ),
        "all_training_dev_overlaps_zero": all(
            r["training_dev_overlap_hidden40"] == 0
            and r["training_dev_overlap_hidden64"] == 0
            for r in results
        ),
        "all_classifier_decision_mismatches_zero": all(
            r["hidden40"]["q2_decision_mismatches"] == 0
            and r["hidden64"]["q2_decision_mismatches"] == 0
            for r in results
        ),
    }

    output = {
        "experiment_id": EXPERIMENT_ID,
        "status": "development",
        "provenance": {
            "git_commit": os.environ.get("GITHUB_SHA", "unknown"),
            "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "baseline_config_sha256": config_hash(FunctionalConfig(hidden_dim=40)),
            "candidate_config_sha256": config_hash(FunctionalConfig(hidden_dim=64)),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "train_steps": STEPS,
            "batch_size": BATCH_SIZE,
            "eval_episodes_per_seed": EVAL_EPISODES,
            "sealed_e2e005_excluded": [list(x) for x in HELDOUT],
            "dev_compositions": [list(x) for x in DEV_COMBOS],
            "training_compositions_count": len(TRAIN_DEV_COMBOS),
            "selection_is_dev_only": True,
            "e2e005_results_not_used": True,
        },
        "results": results,
        "summary": summary,
        "claim_boundary": [
            "development/model-selection evidence only",
            "does not establish a new confirmatory result",
            "does not license use of the sealed E2E-005 held-out set for tuning",
        ],
    }

    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
