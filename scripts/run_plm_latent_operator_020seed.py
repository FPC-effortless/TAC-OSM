#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, statistics
from collections import Counter
from pathlib import Path
import torch
from torch import nn

from tac_osm.contract import load_contract
from tac_osm.latent_operator_001 import LatentOperatorPLM
from tac_osm.latent_operator_001_benchmark import (
    BASE_E2E008_GENERATOR_SHA256, BATCH_SIZE, EVAL_EPISODES,
    GENERATOR_VERSION, HELDOUT, OPS, STEPS, TRAIN_COMBOS,
    benchmark_manifest, episode_fingerprint, episode_key,
    sample_evaluation_episodes, sample_episode,
)
from run_plm_latent_operator_001 import (
    build_batch, query_target, support_tensor, write_observations,
    build_gradient_probe, parameter_count, write_observations_batch,
    query_target_batch,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "TACOSM-PLM-LATENT-OPERATOR-020SEED"
PARENT_ID = "TACOSM-PLM-LATENT-OPERATOR-001"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
SEEDS = tuple(range(20))
PARAM_COUNT = 77682
FROZEN_C19_BENCHMARK_GIT_BLOB_SHA = "a16f73a81f57fc58fae20fc643003ff225d4f181"

def contract_hash():
    return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()

def train_seed_with_counts(seed: int):
    torch.manual_seed(seed)
    rng = random.Random(seed + 185000)
    model = LatentOperatorPLM().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    training_keys = set()
    counts = Counter()
    for _ in range(STEPS):
        episodes = [
            sample_episode(rng, combo2=rng.choice(TRAIN_COMBOS))
            for _ in range(BATCH_SIZE)
        ]
        for ep in episodes:
            counts[ep[3]] += 1
        training_keys.update(episode_key(ep) for ep in episodes)
        batch = build_batch(episodes)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        memory = write_observations_batch(model, batch)
        out = query_target_batch(model, memory, batch)
        loss = torch.nn.functional.cross_entropy(out["logits"], batch["q_target"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model, training_keys, {op: counts[op] for op in OPS}

@torch.no_grad()
def evaluate_extension(model, episodes):
    model.eval()
    q2_correct = 0
    oracle_correct = 0
    no_memory_correct = 0
    selection_correct = 0
    confusion = {truth: {pred: 0 for pred in OPS} for truth in OPS}
    per_op = {
        truth: {"q2": 0, "selection": 0, "n": 0}
        for truth in OPS
    }
    for ep in episodes:
        observations, q2, support, hidden_op, _payload = ep
        memory = write_observations(model, observations)
        out = query_target(model, memory, q2, support)
        truth = hidden_op
        pred_op = OPS[int(out["operator_prediction"].item())]
        confusion[truth][pred_op] += 1
        selection_correct += int(pred_op == truth)
        per_op[truth]["selection"] += int(pred_op == truth)
        per_op[truth]["n"] += 1

        target = int(q2[3])
        pred = int(out["logits"].argmax(-1).item())
        q2_correct += int(pred == target)
        per_op[truth]["q2"] += int(pred == target)

        oracle = query_target(
            model, memory, q2, support,
            operator_override=torch.tensor([OPS.index(truth)], dtype=torch.long),
        )
        oracle_correct += int(int(oracle["logits"].argmax(-1).item()) == target)

        no_memory = query_target(model, torch.zeros_like(memory), q2, support)
        no_memory_correct += int(int(no_memory["logits"].argmax(-1).item()) == target)

    n = len(episodes)
    return {
        "q2_accuracy": q2_correct / n,
        "operator_selection_accuracy": selection_correct / n,
        "oracle_operator_q2_accuracy": oracle_correct / n,
        "no_memory_q2_accuracy": no_memory_correct / n,
        "operator_confusion_matrix": confusion,
        "per_operator": {
            op: {
                "q2_accuracy": per_op[op]["q2"] / per_op[op]["n"],
                "operator_selection_accuracy": per_op[op]["selection"] / per_op[op]["n"],
                "episodes": per_op[op]["n"],
            }
            for op in OPS
        },
    }

def run():
    contract = load_contract(EXPERIMENT_ID)
    assert contract.check_consistency() == []
    manifest = benchmark_manifest()
    assert manifest["generator_version"] == GENERATOR_VERSION
    assert manifest["base_e2e008_generator_sha256"] == BASE_E2E008_GENERATOR_SHA256
    assert parameter_count(LatentOperatorPLM()) == PARAM_COUNT
    gate = build_gradient_probe()
    assert gate["pass"], gate

    rows = []
    aggregate = {truth: {pred: 0 for pred in OPS} for truth in OPS}
    for seed in SEEDS:
        model, training_keys, training_counts = train_seed_with_counts(seed)
        evaluation = sample_evaluation_episodes(random.Random(seed + 186000), EVAL_EPISODES)
        overlap = training_keys & {episode_key(ep) for ep in evaluation}
        assert not overlap, f"training/evaluation overlap for seed {seed}"
        metrics = evaluate_extension(model, evaluation)
        for truth in OPS:
            for pred in OPS:
                aggregate[truth][pred] += metrics["operator_confusion_matrix"][truth][pred]
        rows.append({
            "seed": seed,
            "evaluation_episode_fingerprint": episode_fingerprint(evaluation),
            "training_evaluation_semantic_overlap": len(overlap),
            "training_operator_counts_combo2": training_counts,
            **metrics,
            "q2_lift_over_0.625": metrics["q2_accuracy"] - 0.625,
        })

    assert len({r["evaluation_episode_fingerprint"] for r in rows}) == len(rows)
    assert all(r["training_evaluation_semantic_overlap"] == 0 for r in rows)
    totals = {truth: sum(aggregate[truth].values()) for truth in OPS}

    output = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "provenance": {
            "git_commit": os.environ.get("GITHUB_SHA", "unknown"),
            "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "contract_sha256": contract_hash(),
            "benchmark_sha256": hashlib.sha256(
                (ROOT / "src" / "tac_osm" / "latent_operator_001_benchmark.py").read_bytes()
            ).hexdigest(),
            "c19_benchmark_git_blob_sha": FROZEN_C19_BENCHMARK_GIT_BLOB_SHA,
        },
        "protocol": {
            "seeds": list(SEEDS),
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "evaluation_episodes_per_seed": EVAL_EPISODES,
            "evaluation_episodes_per_operator": 150,
            "heldout": [list(x) for x in HELDOUT],
            "train_combos_count": len(TRAIN_COMBOS),
            "parent_experiment_id": PARENT_ID,
            "model_selection": "none",
        },
        "seed_results": rows,
        "summary": {
            "operator_selection_mean": statistics.fmean(r["operator_selection_accuracy"] for r in rows),
            "operator_selection_min": min(r["operator_selection_accuracy"] for r in rows),
            "q2_mean": statistics.fmean(r["q2_accuracy"] for r in rows),
            "q2_min": min(r["q2_accuracy"] for r in rows),
            "q2_lift_over_0.625_mean": statistics.fmean(r["q2_lift_over_0.625"] for r in rows),
            "oracle_operator_q2_mean": statistics.fmean(r["oracle_operator_q2_accuracy"] for r in rows),
            "oracle_minus_q2_mean": statistics.fmean(
                r["oracle_operator_q2_accuracy"] - r["q2_accuracy"] for r in rows
            ),
            "no_memory_q2_mean": statistics.fmean(r["no_memory_q2_accuracy"] for r in rows),
            "aggregate_operator_confusion_matrix": aggregate,
            "aggregate_selection_accuracy_by_truth": {
                truth: aggregate[truth][truth] / totals[truth] for truth in OPS
            },
            "operator_selection_failure_rate": 1.0 - statistics.fmean(
                r["operator_selection_accuracy"] for r in rows
            ),
        },
        "claim_boundary": [
            "post-C19 robustness extension only",
            "does not alter C19 preregistered status",
            "fixed Boolean primitive library",
            "explicit entity addressing",
            "no operator synthesis",
            "no real-world multimodal claim",
            "no scaling claim",
        ],
    }
    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))

if __name__ == "__main__":
    run()
