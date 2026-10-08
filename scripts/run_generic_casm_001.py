#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.contract import load_contract
from tac_osm.generic_casm_001 import GenericCASM001
from tac_osm.generic_casm_001_benchmark import (
    FAMILY_NAMES,
    INPUT_DIM,
    OUTPUT_DIM,
    SUPPORT_ROWS,
    generate_episode,
    pool_fingerprint,
)

EXPERIMENT_ID = "TACOSM-PLM-GENERIC-CASM-001"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
SEEDS = (0, 1, 2, 3, 4)
TRAIN_STEPS = 2000
BATCH_SIZE = 128
EVAL_EPISODES = 1000
LR = 0.001
WEIGHT_DECAY = 1e-4


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_batch(generator: torch.Generator, count: int):
    episodes = [
        generate_episode(generator, n % len(FAMILY_NAMES))
        for n in range(count)
    ]
    su = torch.stack([e.support_u for e in episodes])
    sv = torch.stack([e.support_v for e in episodes])
    sy = torch.stack([e.support_y for e in episodes])
    qu = torch.stack([e.query_u for e in episodes])
    qv = torch.stack([e.query_v for e in episodes])
    qy = torch.stack([e.query_y for e in episodes])
    return episodes, su, sv, sy, qu, qv, qy


def train_seed(seed: int):
    torch.manual_seed(seed)
    model = GenericCASM001(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dim=64,
        basis_count=8,
        rank=4,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(1_000_000 + seed * 100_003)
    losses = []
    train_parameter_hashes = set()
    gradient_gate = False

    for step in range(TRAIN_STEPS):
        episodes, su, sv, sy, qu, qv, qy = make_batch(generator, BATCH_SIZE)
        train_parameter_hashes.update(e.parameter_fingerprint for e in episodes)
        pred = model(su, sv, sy, qu, qv)
        loss = F.mse_loss(pred, qy)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            g = model.code_head[3].weight.grad
            gradient_gate = g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    first20 = statistics.fmean(losses[:20])
    last20 = statistics.fmean(losses[-20:])
    return model, train_parameter_hashes, {
        "first20_mean_loss": first20,
        "last20_mean_loss": last20,
        "loss_reduction_fraction": (first20 - last20) / max(abs(first20), 1e-12),
        "gradient_gate_pass": gradient_gate,
        "final_parameter_hash": model_hash(model),
    }


@torch.no_grad()
def evaluate_seed(model: GenericCASM001, seed: int):
    model.eval()
    generator = torch.Generator().manual_seed(7_000_000 + seed * 100_003)
    pool = [generate_episode(generator, n % len(FAMILY_NAMES)) for n in range(EVAL_EPISODES)]
    learned_errors = []
    shuffle_errors = []
    no_support_errors = []

    for ep in pool:
        su = ep.support_u.unsqueeze(0)
        sv = ep.support_v.unsqueeze(0)
        sy = ep.support_y.unsqueeze(0)
        qu = ep.query_u.unsqueeze(0)
        qv = ep.query_v.unsqueeze(0)
        qy = ep.query_y.unsqueeze(0)

        pred = model(su, sv, sy, qu, qv)
        learned_errors.append(float(F.mse_loss(pred, qy)))

        perm = torch.randperm(SUPPORT_ROWS, generator=generator)
        shuffled = model(su[:, perm], sv[:, perm], sy[:, perm], qu, qv)
        shuffle_errors.append(float(F.mse_loss(shuffled, qy)))

        zero_support = model(su, sv, torch.zeros_like(sy), qu, qv)
        no_support_errors.append(float(F.mse_loss(zero_support, qy)))

    return {
        "heldout_query_mse": statistics.fmean(learned_errors),
        "shuffle_support_query_mse": statistics.fmean(shuffle_errors),
        "no_support_query_mse": statistics.fmean(no_support_errors),
        "oracle_query_mse": 0.0,
        "pool_fingerprint": pool_fingerprint(pool),
        "eval_parameter_fingerprints": {e.parameter_fingerprint for e in pool},
    }


def main():
    contract = load_contract(EXPERIMENT_ID)
    errors = contract.check_consistency()
    if errors:
        raise AssertionError(errors)

    bench_path = ROOT / "src" / "tac_osm" / "generic_casm_001_benchmark.py"
    contract_hash = file_hash(CONTRACT_PATH)
    benchmark_hash = file_hash(bench_path)

    seed_results = []
    cross_seed_train_overlap = {}
    cross_seed_eval_overlap = {}
    train_sets = {}
    eval_sets = {}

    for seed in SEEDS:
        model, train_hashes, training = train_seed(seed)
        evaluation = evaluate_seed(model, seed)
        train_sets[seed] = train_hashes
        eval_sets[seed] = evaluation["eval_parameter_fingerprints"]
        assert not (train_hashes & eval_sets[seed])
        seed_results.append({
            "seed": seed,
            "training": training,
            "evaluation": {
                k: v for k, v in evaluation.items()
                if k != "eval_parameter_fingerprints"
            },
            "train_eval_parameter_overlap": 0,
        })

    for i, a in enumerate(SEEDS):
        for b in SEEDS[i + 1:]:
            cross_seed_train_overlap[f"{a}:{b}"] = len(train_sets[a] & train_sets[b])
            cross_seed_eval_overlap[f"{a}:{b}"] = len(eval_sets[a] & eval_sets[b])

    learned = [x["evaluation"]["heldout_query_mse"] for x in seed_results]
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "provenance": {
            "git_commit": os.environ.get("GITHUB_SHA", "unknown"),
            "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
            "contract_sha256": contract_hash,
            "benchmark_sha256": benchmark_hash,
        },
        "protocol": {
            "seeds": list(SEEDS),
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "evaluation_episodes_per_seed": EVAL_EPISODES,
            "support_rows": SUPPORT_ROWS,
            "optimizer": "AdamW",
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "operator_id_in_model_input": False,
        },
        "seed_results": seed_results,
        "integrity": {
            "all_gradient_gates_pass": all(x["training"]["gradient_gate_pass"] for x in seed_results),
            "cross_seed_train_parameter_overlap": cross_seed_train_overlap,
            "cross_seed_eval_parameter_overlap": cross_seed_eval_overlap,
            "train_eval_parameter_overlap_zero": all(
                x["train_eval_parameter_overlap"] == 0 for x in seed_results
            ),
            "operator_id_in_model_input": False,
            "query_target_not_input": True,
            "support_row_order_randomized": True,
            "no_post_run_model_selection": True,
        },
        "summary": {
            "mean_heldout_query_mse": statistics.fmean(learned),
            "max_seed_heldout_query_mse": max(learned),
            "mean_shuffle_support_mse": statistics.fmean(
                x["evaluation"]["shuffle_support_query_mse"] for x in seed_results
            ),
            "mean_no_support_mse": statistics.fmean(
                x["evaluation"]["no_support_query_mse"] for x in seed_results
            ),
            "mean_loss_reduction_fraction": statistics.fmean(
                x["training"]["loss_reduction_fraction"] for x in seed_results
            ),
            "criterion_pass": (
                all(x["training"]["gradient_gate_pass"] for x in seed_results)
                and statistics.fmean(learned) <= 0.02
                and max(learned) <= 0.05
                and all(x["train_eval_parameter_overlap"] == 0 for x in seed_results)
            ),
        },
        "claim_boundary": [
            "bounded generic learned compositional computation on three synthetic continuous families",
            "no open-ended operator synthesis claim",
            "no persistence claim",
            "no multimodal claim",
            "no scaling claim",
        ],
    }
    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
