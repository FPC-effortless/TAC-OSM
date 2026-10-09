#!/usr/bin/env python3
"""Execute the frozen PERSISTENCE-002 experiment; no post-run model selection."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.contract import load_contract  # noqa: E402
from tac_osm.persistence_002 import PersistencePLM002  # noqa: E402
from tac_osm.persistence_002_benchmark import (  # noqa: E402
    DELAYS,
    assert_valid_pairs,
    benchmark_manifest,
    make_paired_batch,
    pool_fingerprint,
)

EXPERIMENT_ID = "TACOSM-PLM-PERSISTENCE-002"
SEEDS = (0, 1, 2, 3, 4)
STEPS = 600
BATCH_SIZE = 64
EVAL_PAIRS = 300
LR = 0.002
WEIGHT_DECAY = 0.0001
MIN_STATE_CHANGE = 0.0001
TRAIN_NAMESPACE = 1_000_000
EVAL_NAMESPACE = 7_000_000
ARTIFACT = ROOT / "artifacts" / (EXPERIMENT_ID + ".json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, val in model.state_dict().items():
        h.update(key.encode())
        h.update(val.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def preflight() -> tuple[object, dict]:
    raw = json.loads(
        (ROOT / "contracts" / (EXPERIMENT_ID + ".json")).read_text()
    )
    contract = load_contract(EXPERIMENT_ID)
    assert contract.check_consistency() == [], contract.check_consistency()
    assert contract.status == "pre-registered"
    assert contract.seeds == SEEDS
    assert contract.h_levels == DELAYS
    assert contract.steps == STEPS
    assert contract.eval_steps == EVAL_PAIRS
    p = raw["protocol"]
    assert p["batch_size"] == BATCH_SIZE
    assert p["hidden_dim"] == 64 and p["embedding_dim"] == 8
    assert p["learning_rate"] == LR
    assert p["weight_decay"] == WEIGHT_DECAY
    assert p["train_seed_namespace"] == TRAIN_NAMESPACE
    assert p["eval_seed_namespace"] == EVAL_NAMESPACE
    assert raw["thresholds"]["minimum_mean_distractor_state_change"] == MIN_STATE_CHANGE
    assert raw["thresholds"]["delay4_mean"] == 0.90
    assert raw["thresholds"]["delay4_min_seed"] == 0.75
    assert raw["thresholds"]["delay4_gap_mean"] == 0.30
    assert raw["thresholds"]["delay4_pair_disagreement"] == 0.90
    assert raw["thresholds"]["reset_exact_accuracy"] == 0.50
    assert raw["thresholds"]["oracle_exact_accuracy"] == 1.00
    names = [a.name for a in contract.arms]
    assert names == [
        "persistent_state", "reset_before_q2", "swapped_history",
        "oracle_memory", "no_intervening_write"
    ]
    return contract, raw


def train_seed(seed: int) -> tuple[PersistencePLM002, dict]:
    torch.manual_seed(seed)
    model = PersistencePLM002()
    start_hash = parameter_hash(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    g = torch.Generator().manual_seed(TRAIN_NAMESPACE + 100_003 * seed)
    losses = []
    gradient_gate = False
    for step in range(STEPS):
        batch = make_paired_batch(g, pairs=BATCH_SIZE // 2, delay=DELAYS[step % len(DELAYS)])
        trace = model(batch)
        loss = F.cross_entropy(trace.logits, batch.target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            w_in = model.write_cell.weight_ih.grad
            w_recurrent = model.write_cell.weight_hh.grad
            gradient_gate = all(
                v is not None and bool(torch.isfinite(v).all())
                and float(v.abs().sum()) > 0.0
                for v in (w_in, w_recurrent)
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    final_hash = parameter_hash(model)
    return model, {
        "first20_loss": statistics.fmean(losses[:20]),
        "last20_loss": statistics.fmean(losses[-20:]),
        "parameter_initial_sha256": start_hash,
        "parameter_final_sha256": final_hash,
        "parameter_changed": start_hash != final_hash,
        "write_gradient_gate": gradient_gate,
    }


@torch.no_grad()
def evaluate_seed(model: PersistencePLM002, seed: int) -> list[dict]:
    model.eval()
    g = torch.Generator().manual_seed(EVAL_NAMESPACE + 100_003 * seed)
    rows = []
    for delay in DELAYS:
        batch = make_paired_batch(g, pairs=EVAL_PAIRS, delay=delay)
        assert_valid_pairs(batch)
        calls = [0]

        def count_write(_module, _inputs, _output):
            calls[0] += 1

        hook = model.write_cell.register_forward_hook(count_write)
        try:
            trace = model(batch)
        finally:
            hook.remove()
        if calls[0] != delay + 1 or trace.state_write_calls != delay + 1:
            raise AssertionError("intervening writes were not executed")
        update_norm = statistics.fmean(
            float(delta.detach()) for delta in trace.distractor_update_norms
        )
        if not update_norm > MIN_STATE_CHANGE:
            raise AssertionError("state did not change under distractor writes")

        persistent = trace.logits.argmax(dim=-1)
        reset = model.predict(
            torch.zeros_like(trace.state), trace.current_representation
        ).argmax(dim=-1)
        swap_idx = torch.arange(persistent.numel()).reshape(-1, 2).flip(1).flatten()
        swapped = model.predict(
            trace.state[swap_idx], trace.current_representation
        ).argmax(dim=-1)
        no_write = model.predict(
            trace.t0_state, trace.current_representation
        ).argmax(dim=-1)
        oracle = batch.memory_bit ^ batch.current_bit

        def acc(pred):
            return float((pred == batch.target).float().mean())

        # Exact condition is stronger than an approximate chance estimate:
        # every reset prediction must be identical across the paired histories.
        if not torch.equal(reset[::2], reset[1::2]) or acc(reset) != 0.5:
            raise AssertionError("no-memory collision ceiling failed")
        if acc(oracle) != 1.0:
            raise AssertionError("oracle label differs from generator target")

        rows.append({
            "delay": delay,
            "pair_count": EVAL_PAIRS,
            "episode_count": 2 * EVAL_PAIRS,
            "pool_fingerprint": pool_fingerprint(batch),
            "persistent_accuracy": acc(persistent),
            "reset_accuracy": acc(reset),
            "swapped_history_accuracy": acc(swapped),
            "oracle_accuracy": acc(oracle),
            "no_intervening_write_accuracy": acc(no_write),
            "opposite_past_prediction_disagreement": float(
                (persistent[::2] != persistent[1::2]).float().mean()
            ),
            "mean_distractor_write_delta_norm": update_norm,
            "observed_write_invocations": calls[0],
            "write_calls_per_episode": trace.state_write_calls,
            "representation_calls_per_episode": delay + 2,
        })
    return rows


def seed_bootstrap(values: list[float], draws: int = 4000) -> list[float]:
    g = random.Random(20261009)
    samples = sorted(
        statistics.fmean(g.choices(values, k=len(values)))
        for _ in range(draws)
    )
    return [samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]]


def measure() -> dict:
    torch.set_num_threads(2)
    _contract, raw = preflight()
    results = []
    fingerprints = set()
    for seed in SEEDS:
        model, training = train_seed(seed)
        if not training["write_gradient_gate"] or not training["parameter_changed"]:
            raise AssertionError("model training/state write gate failed")
        evaluation = evaluate_seed(model, seed)
        for row in evaluation:
            if row["pool_fingerprint"] in fingerprints:
                raise AssertionError("evaluation pool fingerprint collision")
            fingerprints.add(row["pool_fingerprint"])
        results.append({"seed": seed, "training": training, "evaluation": evaluation})

    rows4 = [r["evaluation"][DELAYS.index(4)] for r in results]
    persistent = [r["persistent_accuracy"] for r in rows4]
    reset = [r["reset_accuracy"] for r in rows4]
    gaps = [a - b for a, b in zip(persistent, reset)]
    disagreement = [r["opposite_past_prediction_disagreement"] for r in rows4]
    interval = seed_bootstrap(gaps)
    limits = raw["thresholds"]
    passed = (
        statistics.fmean(persistent) >= limits["delay4_mean"]
        and min(persistent) >= limits["delay4_min_seed"]
        and statistics.fmean(gaps) >= limits["delay4_gap_mean"]
        and interval[0] > limits["delay4_gap_ci_lower_strict"]
        and statistics.fmean(disagreement) >= limits["delay4_pair_disagreement"]
        and all(x == 0.5 for x in reset)
    )
    output = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "registered_criterion_pass": passed,
        "provenance": {
            "git_head": os.environ.get("GITHUB_SHA", "local-unverified"),
            "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "local-unverified"),
            "contract_sha256": sha256(
                ROOT / "contracts" / (EXPERIMENT_ID + ".json")
            ),
            "benchmark_sha256": sha256(
                ROOT / "src/tac_osm/persistence_002_benchmark.py"
            ),
            "model_sha256": sha256(ROOT / "src/tac_osm/persistence_002.py"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "delay_levels": list(DELAYS),
            "train_steps": STEPS,
            "train_batch_size": BATCH_SIZE,
            "evaluation_pairs_per_delay_per_seed": EVAL_PAIRS,
            "bootstrap_resamples": 4000,
            "train_seed_namespace": TRAIN_NAMESPACE,
            "eval_seed_namespace": EVAL_NAMESPACE,
            "manifest": benchmark_manifest(),
        },
        "seed_results": results,
        "summary": {
            "delay4_persistent_accuracy_mean": statistics.fmean(persistent),
            "delay4_persistent_accuracy_min_seed": min(persistent),
            "delay4_reset_accuracy_mean": statistics.fmean(reset),
            "delay4_mean_paired_gap": statistics.fmean(gaps),
            "delay4_seed_bootstrap_gap_ci95": interval,
            "delay4_opposite_past_disagreement_mean": statistics.fmean(disagreement),
            "all_gradient_gates_pass": True,
            "all_pair_integrity_gates_pass": True,
            "all_real_write_gates_pass": True,
            "all_oracle_gates_pass": True,
            "all_reset_ceiling_gates_pass": True,
            "full_registered_arm_count": len(raw["arms"]),
        },
        "claim_boundary": (
            "Synthetic binary m-XOR-c causal state retention under actual "
            "intervening learned distractor updates only; no semantic memory, "
            "unbounded horizon or selective-computation scaling claim."
        ),
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"] | {
        "registered_criterion_pass": passed
    }, indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    measure()
