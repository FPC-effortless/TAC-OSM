#!/usr/bin/env python3
"""Address-003: fixed-budget structured-only vs unlabeled mixed-channel training.

The original 002 model and benchmark are imported unmodified. Two learned arms
share each exact held-out evaluation batch and are compared with raw-dot.
The experiment's joint success rule is read from the frozen JSON contract.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.contract import load_contract  # noqa: E402
from tac_osm.learned_address_002 import LearnedAddressMetric002  # noqa: E402
from tac_osm.learned_address_002_benchmark import (  # noqa: E402
    ADDRESS_DIM,
    ISOTROPIC_SIGMAS,
    MEMORY_SIZES,
    STRUCTURED_HIGH_NOISE_M,
    STRUCTURED_HIGH_NOISE_SIGMAS,
    STRUCTURED_SIGMAS,
    TRIALS,
    fingerprint_batch,
    make_trial_batch,
)

ID = "TACOSM-PLM-LEARNED-ADDRESS-003"
CONTRACT_PATH = ROOT / "contracts" / (ID + ".json")
BENCHMARK_PATH = ROOT / "src/tac_osm/learned_address_002_benchmark.py"
MODEL_PATH = ROOT / "src/tac_osm/learned_address_002.py"
ARTIFACT_PATH = ROOT / "artifacts" / (ID + ".json")

SEEDS = tuple(range(10))
STEPS = 750
TRAIN_BATCH = 512
TRAIN_M = 32
EVAL_BATCH = 2000
LR = 0.01
WEIGHT_DECAY = 0.0001
TRAIN_NAMESPACE = 1_900_000
EVAL_NAMESPACE = 8_300_000
EVAL_STRIDE = 1_000_003
TRAIN_STRIDE = 10_007


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, param in model.state_dict().items():
        h.update(name.encode("ascii"))
        h.update(param.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate_contract() -> dict:
    registered = json.loads(CONTRACT_PATH.read_text())
    c = load_contract(ID)
    assert not c.check_consistency(), c.check_consistency()
    assert c.status == "pre-registered"
    assert c.steps == STEPS and c.eval_steps == TRIALS
    assert c.seeds == SEEDS and c.h_levels == MEMORY_SIZES
    protocol = registered["protocol"]
    assert protocol["train_steps_per_learned_arm"] == STEPS
    assert protocol["train_batch"] == TRAIN_BATCH
    assert protocol["train_memory"] == TRAIN_M
    assert protocol["address_dim"] == ADDRESS_DIM
    assert protocol["memory_sizes"] == list(MEMORY_SIZES)
    assert protocol["isotropic_sigmas"] == list(ISOTROPIC_SIGMAS)
    assert protocol["structured_sigmas"] == list(STRUCTURED_SIGMAS)
    assert protocol["high_noise_M"] == STRUCTURED_HIGH_NOISE_M
    assert protocol["high_noise_sigmas"] == list(STRUCTURED_HIGH_NOISE_SIGMAS)
    assert protocol["evaluation_batch"] == EVAL_BATCH
    assert protocol["evaluation_trials_per_condition"] == TRIALS
    assert protocol["lr"] == LR and protocol["weight_decay"] == WEIGHT_DECAY
    assert protocol["training_rng_namespace"] == TRAIN_NAMESPACE
    assert protocol["evaluation_rng_namespace"] == EVAL_NAMESPACE
    assert protocol["eval_seed_stride"] == EVAL_STRIDE
    assert protocol["seed_stride"] == TRAIN_STRIDE
    assert len(c.arms) == 6
    assert [x.name for x in c.arms] == [
        "structured_only", "mixed_unlabeled", "raw_dot",
        "oracle_target_key", "wrong_key_query", "slot_permutation"
    ]
    assert registered["thresholds"]["mixed_isotropic_learned_minus_raw_mean_min"] == -0.02
    assert registered["thresholds"]["mixed_structured_M32_high_noise_gain_mean_min"] == 0.03
    assert registered["thresholds"]["bootstrap_rounds"] == 4000
    return registered


def training_condition(step: int, arm: str) -> tuple[bool, float]:
    if arm == "structured_only":
        return True, (0.2, 0.4)[step % 2]
    if arm == "mixed_unlabeled":
        return step % 2 == 0, (0.2, 0.4)[(step % 4) // 2]
    raise ValueError("unregistered training arm")


def train(seed: int, arm: str, steps: int = STEPS):
    torch.manual_seed(seed)
    model = LearnedAddressMetric002(address_dim=ADDRESS_DIM)
    initial_hash = parameter_hash(model)
    g = torch.Generator().manual_seed(TRAIN_NAMESPACE + seed * TRAIN_STRIDE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    losses = []
    gradient_pass = False
    condition_counts = {
        "structured_0.2": 0, "structured_0.4": 0,
        "isotropic_0.2": 0, "isotropic_0.4": 0,
    }
    for step in range(steps):
        structured, sigma = training_condition(step, arm)
        condition_counts[
            ("structured_" if structured else "isotropic_") + str(sigma)
        ] += 1
        keys, query, labels, _unused_payload = make_trial_batch(
            g, batch=TRAIN_BATCH, memory=TRAIN_M,
            sigma=sigma, structured=structured,
        )
        loss = F.cross_entropy(model(query, keys), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            gradient_pass = (
                model.query.weight.grad is not None
                and torch.isfinite(model.query.weight.grad).all().item()
                and float(model.query.weight.grad.abs().sum()) > 0
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    result = {
        "seed": seed, "arm": arm, "steps": steps,
        "train_batch": TRAIN_BATCH, "condition_counts": condition_counts,
        "initial_parameter_hash": initial_hash,
        "final_parameter_hash": parameter_hash(model),
        "parameter_changed": initial_hash != parameter_hash(model),
        "training_gradient_pass": bool(gradient_pass),
        "first20_loss": statistics.fmean(losses[:min(20, len(losses))]),
        "last20_loss": statistics.fmean(losses[-min(20, len(losses)):]),
    }
    return model, result


@torch.no_grad()
def evaluate_condition(
    learned: dict[str, LearnedAddressMetric002],
    *, seed: int, memory: int, sigma: float, structured: bool,
) -> dict:
    g = torch.Generator().manual_seed(
        EVAL_NAMESPACE
        + seed * EVAL_STRIDE
        + memory * 10_007
        + int(round(sigma * 1000)) * 101
        + (500_000 if structured else 0)
    )
    counts = {
        arm: {"correct": 0, "oracle_correct": 0,
              "wrong_key_correct": 0, "permutation_correct": 0,
              "permutation_identity_correct": 0}
        for arm in learned
    }
    raw_correct = 0
    hash_ = hashlib.sha256()
    samples = 0
    while samples < TRIALS:
        n = min(EVAL_BATCH, TRIALS - samples)
        keys, query, target, payload = make_trial_batch(
            g, batch=n, memory=memory, sigma=sigma,
            structured=structured,
        )
        hash_.update(fingerprint_batch(keys, query, target, payload).encode("ascii"))
        raw_logits = torch.einsum("bd,bmd->bm", query, keys)
        raw_correct += int((raw_logits.argmax(dim=1) == target).sum())
        oracle_query = keys[torch.arange(n), target]
        wrong_query = keys[torch.arange(n), (target + 1) % memory]
        perm = torch.randperm(memory, generator=g)
        permuted = keys[:, perm, :]
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(memory)
        target_perm = inverse[target]

        for arm, model in learned.items():
            logits = model(query, keys)
            prediction = logits.argmax(dim=1)
            permuted_prediction = model(query, permuted).argmax(dim=1)
            counters = counts[arm]
            counters["correct"] += int((prediction == target).sum())
            counters["oracle_correct"] += int(
                (model(oracle_query, keys).argmax(dim=1) == target).sum()
            )
            counters["wrong_key_correct"] += int(
                (model(wrong_query, keys).argmax(dim=1) == target).sum()
            )
            counters["permutation_correct"] += int(
                (permuted_prediction == target_perm).sum()
            )
            counters["permutation_identity_correct"] += int(
                (permuted_prediction == inverse[prediction]).sum()
            )
        samples += n

    raw_acc = raw_correct / TRIALS
    result = {
        "M": memory, "sigma": sigma, "structured": structured,
        "trials": TRIALS, "evaluation_fingerprint": hash_.hexdigest(),
        "raw_dot_accuracy": raw_acc, "arms": {},
    }
    for arm, count in counts.items():
        result["arms"][arm] = {
            "accuracy": count["correct"] / TRIALS,
            "learned_minus_raw": (count["correct"] - raw_correct) / TRIALS,
            "oracle_query_accuracy": count["oracle_correct"] / TRIALS,
            "wrong_key_query_accuracy": count["wrong_key_correct"] / TRIALS,
            "permutation_accuracy": count["permutation_correct"] / TRIALS,
            "permutation_identity_accuracy": (
                count["permutation_identity_correct"] / TRIALS
            ),
        }
        if count["permutation_identity_correct"] != TRIALS:
            raise AssertionError("slot permutation identity control failed")
    return result


def bootstrap(values: list[float]) -> list[float]:
    g = random.Random(20261009)
    draws = 4000
    s = sorted(statistics.fmean(g.choices(values, k=len(values)))
               for _ in range(draws))
    return [s[int(draws * 0.025)], s[int(draws * 0.975)-1]]


def measure() -> dict:
    torch.set_num_threads(2)
    contract = validate_contract()
    results = []
    fingerprints = set()
    for seed in SEEDS:
        models = {}
        train_rows = {}
        initial_hashes = []
        for arm in ("structured_only", "mixed_unlabeled"):
            model, training = train(seed, arm)
            if not training["training_gradient_pass"] or not training["parameter_changed"]:
                raise AssertionError("training failed integrity")
            models[arm] = model.eval()
            train_rows[arm] = training
            initial_hashes.append(training["initial_parameter_hash"])
        if initial_hashes[0] != initial_hashes[1]:
            raise AssertionError("learned arms must share initial parameters")
        isotropic = [
            evaluate_condition(models, seed=seed, memory=m,
                               sigma=sigma, structured=False)
            for m in MEMORY_SIZES for sigma in ISOTROPIC_SIGMAS
        ]
        structured = [
            evaluate_condition(models, seed=seed, memory=m,
                               sigma=sigma, structured=True)
            for m in MEMORY_SIZES for sigma in STRUCTURED_SIGMAS
        ]
        for row in (*isotropic, *structured):
            if row["evaluation_fingerprint"] in fingerprints:
                raise AssertionError("duplicate held-out evaluation pool")
            fingerprints.add(row["evaluation_fingerprint"])
        results.append({
            "seed": seed, "training": train_rows,
            "isotropic": isotropic, "structured": structured
        })
    iso_seed_means = [
        statistics.fmean(x["arms"]["mixed_unlabeled"]["learned_minus_raw"]
                         for x in seed_row["isotropic"])
        for seed_row in results
    ]
    structured_seed_means = [
        statistics.fmean(x["arms"]["mixed_unlabeled"]["learned_minus_raw"]
                         for x in seed_row["structured"]
                         if x["M"] == STRUCTURED_HIGH_NOISE_M
                         and x["sigma"] in STRUCTURED_HIGH_NOISE_SIGMAS)
        for seed_row in results
    ]
    ref_iso = statistics.fmean(
        x["arms"]["structured_only"]["learned_minus_raw"]
        for seed_row in results for x in seed_row["isotropic"]
    )
    ref_struct = statistics.fmean(
        x["arms"]["structured_only"]["learned_minus_raw"]
        for seed_row in results for x in seed_row["structured"]
        if x["M"] == STRUCTURED_HIGH_NOISE_M
        and x["sigma"] in STRUCTURED_HIGH_NOISE_SIGMAS
    )
    iso = statistics.fmean(iso_seed_means)
    struct = statistics.fmean(structured_seed_means)
    thresholds = contract["thresholds"]
    noninferior = iso >= thresholds["mixed_isotropic_learned_minus_raw_mean_min"]
    improved = struct >= thresholds["mixed_structured_M32_high_noise_gain_mean_min"]
    output = {
        "experiment_id": ID, "status": "measured",
        "provenance": {
            "git_commit": os.environ.get("GITHUB_SHA", "local-unverified"),
            "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "local-unverified"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "contract_sha256": file_hash(CONTRACT_PATH),
            "base_benchmark_sha256": file_hash(BENCHMARK_PATH),
            "model_sha256": file_hash(MODEL_PATH),
            "runner_sha256": file_hash(Path(__file__)),
        },
        "protocol": {
            "seeds": list(SEEDS), "train_steps_each": STEPS,
            "train_batch": TRAIN_BATCH, "evaluation_trials_per_condition": TRIALS,
            "memory_sizes": list(MEMORY_SIZES),
            "isotropic_sigmas": list(ISOTROPIC_SIGMAS),
            "structured_sigmas": list(STRUCTURED_SIGMAS),
            "evaluation_namespace": EVAL_NAMESPACE,
            "model_selection": "none", "channel_label_at_prediction": False,
            "evaluation_shared_between_learned_arms": True,
        },
        "seed_results": results,
        "summary": {
            "mixed_isotropic_seed_means": iso_seed_means,
            "mixed_isotropic_mean_gap": iso,
            "mixed_isotropic_noninferiority_pass": noninferior,
            "mixed_structured_high_noise_seed_means": structured_seed_means,
            "mixed_structured_high_noise_mean_gain": struct,
            "mixed_structured_high_noise_ci95": bootstrap(structured_seed_means),
            "mixed_structured_gain_pass": improved,
            "structured_only_isotropic_mean_gap": ref_iso,
            "structured_only_structured_high_noise_gain": ref_struct,
            "joint_registered_mechanism_pass": noninferior and improved,
            "unique_evaluation_pool_fingerprints": len(fingerprints),
            "all_integrity_gates_pass": True,
        },
        "claim_boundary": [
            "Synthetic 16-dimensional opaque associative addressing only",
            "No semantic addressing or real-world generalization",
            "No sublinear retrieval or C5 cost claim",
        ],
    }
    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    measure()
