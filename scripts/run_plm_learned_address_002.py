#!/usr/bin/env python3
"""TACOSM-PLM-LEARNED-ADDRESS-002 confirmatory measurement."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics

import torch
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.contract import load_contract
from tac_osm.learned_address_002 import LearnedAddressMetric002
from tac_osm.learned_address_002_benchmark import (
    ADDRESS_DIM,
    ISOTROPIC_SIGMAS,
    MEMORY_SIZES,
    STRUCTURED_SIGMAS,
    STRUCTURED_HIGH_NOISE_M,
    STRUCTURED_HIGH_NOISE_SIGMAS,
    TRIALS,
    benchmark_manifest,
    fingerprint_batch,
    make_trial_batch,
    generator_sha256,
)

EXPERIMENT_ID = "TACOSM-PLM-LEARNED-ADDRESS-002"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"

SEEDS = tuple(range(10))
TRAIN_MEMORY = 32
TRAIN_STEPS = 750
TRAIN_BATCH = 512
TRAIN_SIGMAS = STRUCTURED_HIGH_NOISE_SIGMAS
EVAL_BATCH = 2_000
LR = 0.01
WEIGHT_DECAY = 1e-4
RAW_ISOTROPIC_TOL = 0.02
STRUCTURED_GAIN = 0.03


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def bootstrap_mean(values: list[float], rounds: int = 5000, seed: int = 20261008) -> list[float]:
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(rng.choices(values, k=len(values)))
        for _ in range(rounds)
    )
    return [
        samples[int(0.025 * rounds)],
        samples[int(0.975 * rounds)],
    ]


def train_seed(seed: int) -> tuple[LearnedAddressMetric002, dict]:
    torch.manual_seed(seed)
    model = LearnedAddressMetric002(address_dim=ADDRESS_DIM)
    initial_hash = parameter_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    gen = torch.Generator().manual_seed(1_900_000 + seed * 10_007)
    losses: list[float] = []

    for step in range(TRAIN_STEPS):
        sigma = TRAIN_SIGMAS[step % len(TRAIN_SIGMAS)]
        keys, query, target, _payload = make_trial_batch(
            gen,
            batch=TRAIN_BATCH,
            memory=TRAIN_MEMORY,
            sigma=sigma,
            structured=True,
        )
        loss = F.cross_entropy(model(query, keys), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    first20 = statistics.fmean(losses[:20])
    last20 = statistics.fmean(losses[-20:])
    return model, {
        "initial_parameter_hash": initial_hash,
        "final_parameter_hash": parameter_hash(model),
        "parameter_changed": initial_hash != parameter_hash(model),
        "first20_mean_loss": first20,
        "last20_mean_loss": last20,
        "loss_reduction_fraction": (first20 - last20) / max(abs(first20), 1e-12),
        "train_steps": TRAIN_STEPS,
        "train_memory": TRAIN_MEMORY,
        "train_batch": TRAIN_BATCH,
        "train_sigmas": list(TRAIN_SIGMAS),
    }


def eval_condition(
    model: LearnedAddressMetric002,
    *,
    seed: int,
    memory: int,
    sigma: float,
    structured: bool,
) -> dict:
    gen = torch.Generator().manual_seed(
        3_100_000
        + seed * 1_000_003
        + memory * 10_007
        + int(round(sigma * 1000.0)) * 101
        + (500_000 if structured else 0)
    )

    learned_correct = raw_correct = oracle_correct = 0
    corrupt_correct = shuffled_correct = total = 0
    fingerprint = hashlib.sha256()

    while total < TRIALS:
        batch_size = min(EVAL_BATCH, TRIALS - total)
        keys, query, target, payload = make_trial_batch(
            gen,
            batch=batch_size,
            memory=memory,
            sigma=sigma,
            structured=structured,
        )
        fingerprint.update(
            fingerprint_batch(keys, query, target, payload).encode("ascii")
        )

        with torch.no_grad():
            learned_logits = model(query, keys)
            raw_logits = torch.einsum("bd,bmd->bm", query, keys)

            learned_correct += int(
                (learned_logits.argmax(dim=1) == target).sum()
            )
            raw_correct += int(
                (raw_logits.argmax(dim=1) == target).sum()
            )

            oracle_query = keys[torch.arange(batch_size), target]
            oracle_correct += int(
                (model(oracle_query, keys).argmax(dim=1) == target).sum()
            )

            wrong = (target + 1) % memory
            corrupt_query = keys[torch.arange(batch_size), wrong]
            corrupt_correct += int(
                (model(corrupt_query, keys).argmax(dim=1) == target).sum()
            )

            # One random slot permutation per batch. Accuracy is measured
            # against the target's new position, not its pre-permutation index.
            perm = torch.randperm(memory, generator=gen)
            shuffled_keys = keys[:, perm, :]
            inverse = torch.empty_like(perm)
            inverse[perm] = torch.arange(memory)
            shuffled_target = inverse[target]
            shuffled_correct += int(
                (model(query, shuffled_keys).argmax(dim=1) == shuffled_target).sum()
            )

        total += batch_size

    n = float(TRIALS)
    return {
        "M": memory,
        "sigma": sigma,
        "structured": structured,
        "trials": TRIALS,
        "evaluation_fingerprint": fingerprint.hexdigest(),
        "learned_accuracy": learned_correct / n,
        "raw_dot_accuracy": raw_correct / n,
        "learned_minus_raw": (learned_correct - raw_correct) / n,
        "oracle_accuracy": oracle_correct / n,
        "corrupted_query_accuracy": corrupt_correct / n,
        "shuffle_invariant_accuracy": shuffled_correct / n,
    }


def main() -> None:
    contract = load_contract(EXPERIMENT_ID)
    errors = contract.check_consistency()
    if errors:
        raise AssertionError(errors)
    assert contract.steps == TRAIN_STEPS
    assert contract.eval_steps == TRIALS
    assert tuple(contract.seeds) == SEEDS

    manifest = benchmark_manifest()
    contract_hash = sha256_file(CONTRACT_PATH)
    benchmark_hash = sha256_file(
        ROOT / "src" / "tac_osm" / "learned_address_002_benchmark.py"
    )
    contract_spec = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert benchmark_hash == contract_spec["benchmark_generator_sha256"]

    seed_results: list[dict] = []
    training_meta: dict[str, dict] = {}

    for seed in SEEDS:
        model, train_meta = train_seed(seed)
        training_meta[str(seed)] = train_meta

        isotropic = [
            eval_condition(
                model,
                seed=seed,
                memory=M,
                sigma=sigma,
                structured=False,
            )
            for M in MEMORY_SIZES
            for sigma in ISOTROPIC_SIGMAS
        ]
        structured = [
            eval_condition(
                model,
                seed=seed,
                memory=M,
                sigma=sigma,
                structured=True,
            )
            for M in MEMORY_SIZES
            for sigma in STRUCTURED_SIGMAS
        ]
        seed_results.append({
            "seed": seed,
            "isotropic": isotropic,
            "structured": structured,
        })

    # The integrity identity is over the actual evaluation stream, not the
    # resulting accuracies. Within a seed, condition fingerprints are unique;
    # across seeds the aggregate set must also be unique.
    condition_fingerprints = [
        row["evaluation_fingerprint"]
        for seed_row in seed_results
        for family in ("isotropic", "structured")
        for row in seed_row[family]
    ]
    assert len(condition_fingerprints) == len(set(condition_fingerprints))

    isotropic_rows = [
        row
        for seed_row in seed_results
        for row in seed_row["isotropic"]
    ]
    high_noise_rows = [
        row
        for seed_row in seed_results
        for row in seed_row["structured"]
        if row["M"] == STRUCTURED_HIGH_NOISE_M
        and row["sigma"] in STRUCTURED_HIGH_NOISE_SIGMAS
    ]

    iso_gaps = [row["learned_minus_raw"] for row in isotropic_rows]
    isotropic_mean_gap = statistics.fmean(iso_gaps)
    isotropic_min_gap = min(iso_gaps)

    seed_high_noise_gains: list[float] = []
    for seed_row in seed_results:
        vals = [
            row["learned_minus_raw"]
            for row in seed_row["structured"]
            if row["M"] == STRUCTURED_HIGH_NOISE_M
            and row["sigma"] in STRUCTURED_HIGH_NOISE_SIGMAS
        ]
        seed_high_noise_gains.append(statistics.fmean(vals))

    structured_mean_gain = statistics.fmean(seed_high_noise_gains)

    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "provenance": {
            "git_commit": os.environ.get("GITHUB_SHA", "unknown"),
            "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "contract_sha256": contract_hash,
            "benchmark_sha256": benchmark_hash,
            "benchmark_manifest": manifest,
        },
        "protocol": {
            "seeds": list(SEEDS),
            "address_dim": ADDRESS_DIM,
            "memory_sizes": list(MEMORY_SIZES),
            "isotropic_sigmas": list(ISOTROPIC_SIGMAS),
            "structured_sigmas": list(STRUCTURED_SIGMAS),
            "structured_high_noise_subset": {
                "M": STRUCTURED_HIGH_NOISE_M,
                "sigmas": list(STRUCTURED_HIGH_NOISE_SIGMAS),
            },
            "training_memory": TRAIN_MEMORY,
            "training_steps": TRAIN_STEPS,
            "training_batch": TRAIN_BATCH,
            "evaluation_trials_per_condition": TRIALS,
            "evaluation_batch": EVAL_BATCH,
            "optimizer": "AdamW",
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "initialization": "identity query/key metric; zero candidate-bias output",
            "model_selection": "none",
        },
        "training": training_meta,
        "seed_results": seed_results,
        "summary": {
            "isotropic_mean_learned_minus_raw": isotropic_mean_gap,
            "isotropic_min_condition_learned_minus_raw": isotropic_min_gap,
            "isotropic_noninferiority_pass": isotropic_mean_gap >= -RAW_ISOTROPIC_TOL,
            "structured_high_noise_seed_means_M32": seed_high_noise_gains,
            "structured_high_noise_mean_gain_M32": structured_mean_gain,
            "structured_high_noise_bootstrap_ci95_over_seeds": bootstrap_mean(seed_high_noise_gains),
            "structured_high_noise_gain_pass": structured_mean_gain >= STRUCTURED_GAIN,
            "overall_registered_mechanism_gate": (
                isotropic_mean_gap >= -RAW_ISOTROPIC_TOL
                and structured_mean_gain >= STRUCTURED_GAIN
            ),
        },
        "integrity": {
            "evaluation_condition_fingerprints_unique": (
                len(condition_fingerprints) == len(set(condition_fingerprints))
            ),
            "same_generator_for_all_conditions": True,
            "no_post_run_selection": True,
            "payload_not_in_scorer": True,
            "raw_dot_reference_on_every_condition": True,
            "oracle_control_present": True,
            "corrupted_query_control_present": True,
            "shuffle_control_present": True,
        },
        "claim_boundary": [
            "bounded learned associative address-metric evidence only",
            "no semantic entity-resolution claim",
            "no content-derived addressing claim",
            "no multimodal capability claim",
            "no long-horizon persistence claim",
            "no C5 scaling claim",
        ],
    }

    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
