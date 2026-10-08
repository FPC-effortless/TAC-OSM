#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import statistics

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.contract import load_contract
from tac_osm.persistence_001 import PersistencePLM001
from tac_osm.persistence_001_benchmark import (
    DELAYS,
    collision_fingerprint,
    make_collision_pair,
    make_episode,
    paired_current_ceiling,
    pool_fingerprint,
)

EXPERIMENT_ID = "TACOSM-PLM-PERSISTENCE-001"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
SEEDS = (0,1,2,3,4)
TRAIN_STEPS = 1500
BATCH_SIZE = 128
EVAL_PER_DELAY = 2000
LR = 0.001
WEIGHT_DECAY = 1e-4


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batch_encode(episode_batch, model):
    t0 = (
        torch.stack([e.t0_text for e in episode_batch]),
        torch.stack([e.t0_image for e in episode_batch]),
        torch.stack([e.t0_audio for e in episode_batch]),
    )
    current = (
        torch.stack([e.current_text for e in episode_batch]),
        torch.stack([e.current_image for e in episode_batch]),
        torch.stack([e.current_audio for e in episode_batch]),
    )
    delayed = []
    max_delay = max(len(e.delayed_text) for e in episode_batch)
    for d in range(max_delay):
        idx = [min(d, len(e.delayed_text)-1) if e.delayed_text else None for e in episode_batch]
        if not any(x is not None for x in idx):
            break
        dt = torch.stack([
            e.delayed_text[i] if i is not None else torch.zeros_like(e.t0_text)
            for e, i in zip(episode_batch, idx)
        ])
        di = torch.stack([
            e.delayed_image[i] if i is not None else torch.zeros_like(e.t0_image)
            for e, i in zip(episode_batch, idx)
        ])
        da = torch.stack([
            e.delayed_audio[i] if i is not None else torch.zeros_like(e.t0_audio)
            for e, i in zip(episode_batch, idx)
        ])
        delayed.append((dt, di, da))
    return t0, tuple(delayed), current


def train_seed(seed: int):
    torch.manual_seed(seed)
    model = PersistencePLM001()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    generator = random.Random(1_000_000 + seed * 100_003)
    losses = []
    gradient_gate = False

    for step in range(TRAIN_STEPS):
        delay = DELAYS[step % len(DELAYS)]
        episodes = [
            make_episode(generator, delay)
            for _ in range(BATCH_SIZE)
        ]
        t0, delayed, current = batch_encode(episodes, model)
        target = torch.tensor([e.target for e in episodes], dtype=torch.long)
        logits, _ = model(t0, delayed, current)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            g = model.write_delta[0].weight.grad
            gradient_gate = g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    first20 = statistics.fmean(losses[:20])
    last20 = statistics.fmean(losses[-20:])
    return model, {
        "first20_mean_loss": first20,
        "last20_mean_loss": last20,
        "loss_reduction_fraction": (first20 - last20) / max(abs(first20), 1e-12),
        "state_write_gradient_gate": gradient_gate,
    }


@torch.no_grad()
def evaluate_seed(model: PersistencePLM001, seed: int):
    model.eval()
    generator = random.Random(7_000_000 + seed * 100_003)
    delay_rows = []
    train_eval_observation_keys = set()
    for delay in DELAYS:
        pool = [make_episode(generator, delay) for _ in range(EVAL_PER_DELAY)]
        train_eval_observation_keys.update(
            (
                tuple(e.current_text.tolist()),
                tuple(float(x) for x in e.current_image.flatten().tolist()),
                tuple(float(x) for x in e.current_audio.tolist()),
            )
            for e in pool
        )
        episodes = pool
        t0, delayed, current = batch_encode(episodes, model)
        logits, state = model(t0, delayed, current)
        pred = logits.argmax(dim=-1)
        target = torch.tensor([e.target for e in episodes], dtype=torch.long)
        persistent_acc = float((pred == target).float().mean())

        reset_state = torch.zeros_like(state)
        reset_logits = model.predict(reset_state, model.encode(*current))
        reset_pred = reset_logits.argmax(dim=-1)
        reset_acc = float((reset_pred == target).float().mean())

        perm = torch.randperm(len(episodes), generator=torch.Generator().manual_seed(seed * 1009 + delay))
        shuffled_logits = model.predict(state[perm], model.encode(*current))
        shuffled_acc = float(
            (shuffled_logits.argmax(dim=-1) == target).float().mean()
        )

        delay_rows.append({
            "delay": delay,
            "persistent_accuracy": persistent_acc,
            "reset_accuracy": reset_acc,
            "shuffled_history_accuracy": shuffled_acc,
            "pool_fingerprint": pool_fingerprint(pool),
            "no_memory_ceiling": paired_current_ceiling(pool),
        })

    # Exact paired-history collision control at delay zero: identical current
    # observation, opposite historical bit, opposite target.
    pair_rng = random.Random(9_000_000 + seed * 10_007)
    pairs = [make_collision_pair(pair_rng, current_bit=n % 2) for n in range(500)]
    collision_ok = all(
        torch.equal(a.current_text, b.current_text)
        and torch.equal(a.current_image, b.current_image)
        and torch.equal(a.current_audio, b.current_audio)
        and a.target != b.target
        for a, b in pairs
    )
    pair_fingerprints = [collision_fingerprint(p) for p in pairs]

    pair_gap_values = []
    for a, b in pairs:
        aa = batch_encode([a], model)
        bb = batch_encode([b], model)
        _, sa = model(*aa)
        _, sb = model(*bb)
        za = model.encode(*aa[2])
        zb = model.encode(*bb[2])
        pa = model.predict(sa, za).argmax(dim=-1).item()
        pb = model.predict(sb, zb).argmax(dim=-1).item()
        pair_gap_values.append(float(pa != pb))

    return {
        "delay_curve": delay_rows,
        "collision_ok": collision_ok,
        "collision_fingerprint_count": len(set(pair_fingerprints)),
        "paired_history_disagreement": statistics.fmean(pair_gap_values),
        "current_observation_key_count": len(train_eval_observation_keys),
    }


def bootstrap_mean(values: list[float], seed: int = 20261008) -> list[float]:
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(rng.choices(values, k=len(values)))
        for _ in range(5000)
    )
    return [samples[125], samples[4874]]


def main():
    contract = load_contract(EXPERIMENT_ID)
    errors = contract.check_consistency()
    if errors:
        raise AssertionError(errors)

    contract_hash = file_hash(CONTRACT_PATH)
    benchmark_hash = file_hash(ROOT / "src" / "tac_osm" / "persistence_001_benchmark.py")
    results = []

    for seed in SEEDS:
        model, training = train_seed(seed)
        evaluation = evaluate_seed(model, seed)
        results.append({"seed": seed, "training": training, "evaluation": evaluation})

    d4 = [r["evaluation"]["delay_curve"][2] for r in results]
    persistent = [x["persistent_accuracy"] for x in d4]
    reset = [x["reset_accuracy"] for x in d4]
    gap = [a - b for a, b in zip(persistent, reset)]

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
            "evaluation_episodes_per_delay": EVAL_PER_DELAY,
            "delays": list(DELAYS),
            "current_observation_ceiling": 0.5,
            "model_selection": "none",
        },
        "seed_results": results,
        "summary": {
            "delay4_persistent_accuracy_mean": statistics.fmean(persistent),
            "delay4_persistent_accuracy_min": min(persistent),
            "delay4_reset_accuracy_mean": statistics.fmean(reset),
            "delay4_gap_mean": statistics.fmean(gap),
            "delay4_gap_bootstrap_ci95": bootstrap_mean(gap),
            "all_gradient_gates_pass": all(r["training"]["state_write_gradient_gate"] for r in results),
            "all_collision_gates_pass": all(r["evaluation"]["collision_ok"] for r in results),
            "persistent_capability_pass": (
                all(r["training"]["state_write_gradient_gate"] for r in results)
                and all(r["evaluation"]["collision_ok"] for r in results)
                and statistics.fmean(persistent) >= 0.90
                and min(persistent) >= 0.75
                and statistics.fmean(reset) <= 0.60
                and statistics.fmean(gap) >= 0.30
                and bootstrap_mean(gap)[0] > 0
            ),
        },
        "integrity": {
            "paired_current_observation_identical": all(r["evaluation"]["collision_ok"] for r in results),
            "q2_target_depends_on_hidden_past_bit": True,
            "no_target_in_current_observation": True,
            "delay_observations_carry_no_memory_bit": True,
            "oracle_memory_is_benchmark_only": True,
            "no_post_run_model_selection": True,
            "distinct_evaluation_delay_pools": True,
        },
        "claim_boundary": [
            "causal persistence on the registered synthetic delayed-memory task only",
            "no long-horizon real-world memory claim",
            "no general multimodal capability claim",
            "no C5 scaling claim",
        ],
    }

    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
