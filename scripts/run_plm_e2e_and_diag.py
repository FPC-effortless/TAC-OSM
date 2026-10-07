#!/usr/bin/env python3
from __future__ import annotations

import argparse

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
    episode_fingerprint,
    episode_key,
    sample_episode,
)
from tac_osm.integrated_e2e_006_benchmark import HELDOUT as E2E006_HELDOUT
from tac_osm.integrated_e2e_007_benchmark import (
    DEV_COMBOS,
    HELDOUT as E2E007_HELDOUT,
    SEALED_E2E005,
)

EXPERIMENT_ID = "TACOSM-PLM-AND-DIAG-001"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
SEEDS = (0, 1, 2, 3, 4)
STEPS = 300
BATCH_SIZE = 96
EVAL_EPISODES = 200
OPS = ("xor", "and", "or", "xnor")

EXCLUDED = set(SEALED_E2E005) | set(E2E006_HELDOUT) | set(E2E007_HELDOUT) | set(DEV_COMBOS)
TRAIN_COMBOS = tuple(c for c in ALL_COMBOS if c not in EXCLUDED)

assert ("and", 4, 11) not in TRAIN_COMBOS
assert ("and", 5, 9) not in TRAIN_COMBOS
assert set(DEV_COMBOS).isdisjoint(set(TRAIN_COMBOS))
assert set(E2E006_HELDOUT).isdisjoint(set(TRAIN_COMBOS))
assert set(E2E007_HELDOUT).isdisjoint(set(TRAIN_COMBOS))
assert set(SEALED_E2E005).isdisjoint(set(TRAIN_COMBOS))


def build_batch(episodes):
    from scripts.run_integrated_e2e_005 import build_batch

    return build_batch(episodes)


def train_batch(model, batch):
    from scripts.run_integrated_e2e_005 import train_batch

    return train_batch(model, batch)


def train_seed(seed: int):
    torch.manual_seed(seed)
    rng = random.Random(seed + 130700)
    model = FunctionalMultimodalPLM(
        config=FunctionalConfig(hidden_dim=64, state_write_mode="residual_linear")
    ).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.0001,
    )
    keys = set()
    for _ in range(STEPS):
        episodes = [
            sample_episode(
                rng,
                combo1=rng.choice(TRAIN_COMBOS),
                combo2=rng.choice(TRAIN_COMBOS),
            )
            for _ in range(BATCH_SIZE)
        ]
        keys.update(episode_key(ep) for ep in episodes)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = train_batch(model, build_batch(episodes))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model, keys


def sample_dev(seed: int):
    rng = random.Random(seed + 130800)
    episodes = []
    for i in range(EVAL_EPISODES):
        combo1 = DEV_COMBOS[(i * 5) % len(DEV_COMBOS)]
        combo2 = DEV_COMBOS[(i * 11 + 3) % len(DEV_COMBOS)]
        if combo1 == combo2:
            combo2 = DEV_COMBOS[(i * 11 + 4) % len(DEV_COMBOS)]
        episodes.append(sample_episode(rng, combo1=combo1, combo2=combo2))
    return episodes


@torch.no_grad()
def evaluate(model, episodes):
    model.eval()
    total = 0
    correct = 0
    positives = 0
    decision_mismatches = 0
    dispatch_mismatches = 0
    bits_joint = 0
    state_errors = 0
    execution_errors_given_correct = 0
    correct_bit_cases = 0

    for ep in episodes:
        rows = ep[0]
        memory = model.state.initial(1, torch.device("cpu"))
        for entity, text, image, audio in rows:
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

        query = ep[2]
        entity, i, j, op_name, target = query
        assert op_name == "and"

        op_idx = OPS.index("and")
        out = model.query(
            memory,
            torch.tensor([entity]),
            torch.tensor([i]),
            torch.tensor([j]),
            torch.tensor([op_idx]),
        )

        pred = int(out["logits"].argmax(-1).item())
        threshold_pred = int((out["action"] >= 0.5).long().item())
        decision_mismatches += int(pred != threshold_pred)
        positives += pred
        total += 1
        correct += int(pred == target)

        bit_probs = out["bits"].squeeze(0)
        bit_preds = (bit_probs >= 0.5).long()
        true_i = ep[3][entity][i]
        true_j = ep[3][entity][j]
        both_correct = (
            bit_preds[i].item() == true_i
            and bit_preds[j].item() == true_j
        )
        bits_joint += int(both_correct)
        state_errors += int(not both_correct)

        # Fixed-dispatch integrity: the returned action for op=AND must equal
        # the implementation's AND combination of the two returned bit probs.
        expected_action = bit_probs[i] * bit_probs[j]
        dispatch_mismatches += int(
            not torch.allclose(out["action"].squeeze(0), expected_action, atol=1e-7, rtol=1e-6)
        )

        if both_correct:
            correct_bit_cases += 1
            execution_errors_given_correct += int(pred != target)

    return {
        "and_q2_accuracy": correct / max(total, 1),
        "and_positive_fraction": positives / max(total, 1),
        "queried_bit_joint_accuracy": bits_joint / max(total, 1),
        "and_state_error_rate": state_errors / max(total, 1),
        "and_execution_error_given_correct_bits": (
            execution_errors_given_correct / max(correct_bit_cases, 1)
        ),
        "classifier_decision_mismatches": decision_mismatches,
        "and_dispatch_mismatches": dispatch_mismatches,
        "and_correct_bit_cases": correct_bit_cases,
        "and_episode_count": total,
    }


def run(smoke=False):
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["status"] == "pre-registered"
    assert contract["seeds"] == list(SEEDS)
    assert contract["steps"] == STEPS
    assert contract["eval_steps"] == EVAL_EPISODES
    assert contract["protocol"]["hidden_dim"] == 64 if "protocol" in contract else True

    seeds = (0,) if smoke else SEEDS
    eval_n = 16 if smoke else EVAL_EPISODES
    rows = []

    for seed in seeds:
        model, training_keys = train_seed(seed)
        episodes = sample_dev(seed)[:eval_n]
        evaluation_keys = {episode_key(ep) for ep in episodes}
        overlap = training_keys & evaluation_keys
        if overlap:
            raise AssertionError(f"training/evaluation semantic overlap seed {seed}")

        metrics = evaluate(model, episodes)
        rows.append(
            {
                "seed": seed,
                "evaluation_fingerprint": episode_fingerprint(episodes),
                "training_evaluation_overlap": len(overlap),
                **metrics,
            }
        )

    def mean(key):
        return statistics.fmean(row[key] for row in rows)

    summary = {
        "mean_and_q2_accuracy": mean("and_q2_accuracy"),
        "mean_queried_bit_joint_accuracy": mean("queried_bit_joint_accuracy"),
        "mean_and_state_error_rate": mean("and_state_error_rate"),
        "mean_and_execution_error_given_correct_bits": mean(
            "and_execution_error_given_correct_bits"
        ),
        "mean_and_positive_fraction": mean("and_positive_fraction"),
        "all_training_evaluation_overlap_zero": all(
            row["training_evaluation_overlap"] == 0 for row in rows
        ),
        "all_classifier_decision_mismatches_zero": all(
            row["classifier_decision_mismatches"] == 0 for row in rows
        ),
        "all_and_dispatch_mismatches_zero": all(
            row["and_dispatch_mismatches"] == 0 for row in rows
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
            "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        },
        "protocol": {
            "seeds": list(seeds),
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "evaluation_episodes_per_seed": eval_n,
            "hidden_dim": 64,
            "state_write_mode": "residual_linear",
            "dev_compositions": [list(x) for x in DEV_COMBOS],
            "sealed_e2e005_excluded": True,
            "e2e006_heldout_excluded": True,
            "e2e007_heldout_excluded": True,
            "e2e007_test_outcomes_not_used": True,
        },
        "seed_results": rows,
        "summary": summary,
        "claim_boundary": [
            "development diagnosis only",
            "no confirmatory claim",
            "no E2E-007 heldout outcomes used for tuning or selection",
        ],
    }

    path = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.smoke)
