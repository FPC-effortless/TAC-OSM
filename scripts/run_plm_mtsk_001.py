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
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.contract import load_contract
from tac_osm.integrated_e2e_005 import FunctionalConfig, FunctionalMultimodalPLM
from tac_osm.mtsk import MTSKEntityState, SingleAdaptiveTimescaleState
from tac_osm.mtsk_001_benchmark import (
    EVAL_DELAYS,
    EVAL_EPISODES_PER_DELAY,
    ENTITY,
    FAST_BIT,
    SLOW_BIT,
    TRAIN_DELAY,
    episode_fingerprint,
    sample_evaluation_episodes,
    sample_sequence,
    sequence_key,
    target_balance,
)

EXPERIMENT_ID = "TACOSM-PLM-MTSK-001"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
SEEDS = (0, 1, 2, 3, 4)
STEPS = 200
BATCH_SIZE = 64
HIDDEN = 60
BANKS = 3
ANCHOR_BIT = 11


def build_model(kind: str) -> FunctionalMultimodalPLM:
    model = FunctionalMultimodalPLM(
        config=FunctionalConfig(
            hidden_dim=HIDDEN,
            entity_count=1,
            latent_bits=12,
            state_write_mode="residual_linear",
        )
    ).cpu()
    if kind == "single_timescale":
        model.state = SingleAdaptiveTimescaleState(HIDDEN, 1, BANKS)
    elif kind == "mtsk":
        model.state = MTSKEntityState(HIDDEN, 1, BANKS)
    else:
        raise ValueError(kind)
    return model


def encode_observations(model, rows):
    text = torch.stack([row["text"] for row in rows])
    image = torch.stack([row["image"] for row in rows])
    audio = torch.stack([row["audio"] for row in rows]).unsqueeze(1)
    return model.encode(text, image, audio)


def train_seed(kind: str, seed: int):
    torch.manual_seed(seed)
    rng = random.Random(seed + 195000)
    model = build_model(kind)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    train_keys = set()
    for _ in range(STEPS):
        sequences = [sample_sequence(rng, TRAIN_DELAY) for _ in range(BATCH_SIZE)]
        train_keys.update(sequence_key(ep) for ep in sequences)
        memory = model.state.initial(BATCH_SIZE, torch.device("cpu"))
        for t in range(TRAIN_DELAY + 1):
            rows = [ep["observations"][t] for ep in sequences]
            z = encode_observations(model, rows)
            memory, _ = model.state.write(memory, z, torch.full((BATCH_SIZE,), ENTITY, dtype=torch.long))
        slow_q = model.query(memory, torch.full((BATCH_SIZE,), ENTITY, dtype=torch.long),
                              torch.full((BATCH_SIZE,), SLOW_BIT, dtype=torch.long),
                              torch.full((BATCH_SIZE,), ANCHOR_BIT, dtype=torch.long),
                              torch.zeros(BATCH_SIZE, dtype=torch.long))
        fast_q = model.query(memory, torch.full((BATCH_SIZE,), ENTITY, dtype=torch.long),
                              torch.full((BATCH_SIZE,), FAST_BIT, dtype=torch.long),
                              torch.full((BATCH_SIZE,), ANCHOR_BIT, dtype=torch.long),
                              torch.zeros(BATCH_SIZE, dtype=torch.long))
        slow_target = torch.tensor([ep["slow_target"] for ep in sequences], dtype=torch.long)
        fast_target = torch.tensor([ep["fast_target"] for ep in sequences], dtype=torch.long)
        loss = F.cross_entropy(slow_q["logits"], slow_target) + F.cross_entropy(fast_q["logits"], fast_target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, train_keys


@torch.no_grad()
def evaluate_seed(model, episodes):
    model.eval()
    by_delay = {}
    for delay in EVAL_DELAYS:
        subset = [ep for ep in episodes if ep["delay"] == delay]
        slow_correct = []
        fast_correct = []
        for ep in subset:
            memory = model.state.initial(1, torch.device("cpu"))
            for row in ep["observations"]:
                z = model.encode(
                    row["text"].unsqueeze(0), row["image"].unsqueeze(0), row["audio"].unsqueeze(0).unsqueeze(0)
                )
                memory, _ = model.state.write(memory, z, torch.tensor([ENTITY]))
            slow_q = model.query(memory, torch.tensor([ENTITY]), torch.tensor([SLOW_BIT]),
                                  torch.tensor([ANCHOR_BIT]), torch.tensor([0]))
            fast_q = model.query(memory, torch.tensor([ENTITY]), torch.tensor([FAST_BIT]),
                                  torch.tensor([ANCHOR_BIT]), torch.tensor([0]))
            slow_correct.append(int(slow_q["logits"].argmax(-1).item()) == int(ep["slow_target"]))
            fast_correct.append(int(fast_q["logits"].argmax(-1).item()) == int(ep["fast_target"]))
        by_delay[str(delay)] = {
            "episodes": len(subset),
            "slow_accuracy": statistics.fmean(slow_correct),
            "fast_accuracy": statistics.fmean(fast_correct),
            "slow_correct": slow_correct,
            "fast_correct": fast_correct,
        }
    return by_delay


def hierarchical_paired_bootstrap(mtsk_rows, single_rows, delay: int, rounds: int = 10000):
    rng = random.Random(20261007 + delay)
    m_map = {r["seed"]: r["metrics"][str(delay)]["slow_correct"] for r in mtsk_rows}
    s_map = {r["seed"]: r["metrics"][str(delay)]["slow_correct"] for r in single_rows}
    seeds = sorted(m_map)
    samples = []
    for _ in range(rounds):
        chosen = rng.choices(seeds, k=len(seeds))
        seed_means = []
        for seed in chosen:
            m = m_map[seed]
            s = s_map[seed]
            idx = rng.choices(range(len(m)), k=len(m))
            seed_means.append(statistics.fmean(m[i] - s[i] for i in idx))
        samples.append(statistics.fmean(seed_means))
    samples.sort()
    return [samples[int(0.025 * rounds)], samples[int(0.975 * rounds)]]


def run(smoke: bool = False):
    contract = load_contract(EXPERIMENT_ID)
    assert contract.check_consistency() == []
    seeds = (0,) if smoke else SEEDS
    rows = {"single_timescale": [], "mtsk": []}
    for seed in seeds:
        episodes = sample_evaluation_episodes(random.Random(seed + 196000))
        balance = target_balance(episodes)
        assert all(all(v == 50 for v in cells.values()) for cells in balance.values())
        for kind in rows:
            model, train_keys = train_seed(kind, seed) if not smoke else (build_model(kind), set())
            eval_keys = {sequence_key(ep) for ep in episodes}
            overlap = train_keys & eval_keys
            assert not overlap
            metrics = evaluate_seed(model, episodes)
            rows[kind].append({
                "seed": seed,
                "evaluation_episode_fingerprint": episode_fingerprint(episodes),
                "training_evaluation_overlap": len(overlap),
                "target_balance": balance,
                "metrics": metrics,
                "parameter_count": sum(p.numel() for p in model.parameters()),
            })
    assert len(rows["mtsk"]) == len(seeds) and len(rows["single_timescale"]) == len(seeds)
    assert [r["evaluation_episode_fingerprint"] for r in rows["single_timescale"]] == [
        r["evaluation_episode_fingerprint"] for r in rows["mtsk"]
    ]
    assert len({r["evaluation_episode_fingerprint"] for r in rows["mtsk"]}) == len(rows["mtsk"])
    assert all(r["training_evaluation_overlap"] == 0 for kind in rows for r in rows[kind])
    assert all(r["parameter_count"] == rows["single_timescale"][0]["parameter_count"] for kind in rows for r in rows[kind])

    summary = {}
    for delay in EVAL_DELAYS:
        m = [r["metrics"][str(delay)] for r in rows["mtsk"]]
        s = [r["metrics"][str(delay)] for r in rows["single_timescale"]]
        m_slow = statistics.fmean(x["slow_accuracy"] for x in m)
        s_slow = statistics.fmean(x["slow_accuracy"] for x in s)
        m_fast = statistics.fmean(x["fast_accuracy"] for x in m)
        s_fast = statistics.fmean(x["fast_accuracy"] for x in s)
        summary[str(delay)] = {
            "mtsk_slow_accuracy": m_slow,
            "single_timescale_slow_accuracy": s_slow,
            "slow_delta": m_slow - s_slow,
            "mtsk_fast_accuracy": m_fast,
            "single_timescale_fast_accuracy": s_fast,
            "fast_delta": m_fast - s_fast,
        }
    ci = None if smoke else hierarchical_paired_bootstrap(rows["mtsk"], rows["single_timescale"], 32)
    primary = summary["32"]["slow_delta"]
    fast_delta = summary["32"]["fast_delta"]
    output = {
        "experiment_id": EXPERIMENT_ID,
        "status": "smoke" if smoke else "measured",
        "provenance": {
            "git_commit": os.environ.get("GITHUB_SHA", "unknown"),
            "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
            "benchmark_sha256": hashlib.sha256((ROOT / "src/tac_osm/mtsk_001_benchmark.py").read_bytes()).hexdigest(),
        },
        "protocol": {
            "seeds": list(seeds),
            "train_steps": STEPS,
            "batch_size": BATCH_SIZE,
            "train_delay": TRAIN_DELAY,
            "evaluation_delays": list(EVAL_DELAYS),
            "evaluation_episodes_per_delay": EVAL_EPISODES_PER_DELAY,
            "hidden_dim": HIDDEN,
            "banks": BANKS,
            "models": ["single_timescale", "mtsk"],
        },
        "integrity": {
            "same_evaluation_fingerprint_across_arms": True,
            "distinct_evaluation_fingerprints_per_seed": True,
            "training_evaluation_overlap_zero": True,
            "parameter_counts_equal": True,
            "slow_target_masked_after_t0": True,
            "image_audio_exclude_slow_fast_bits": True,
            "evaluation_target_balance_exact_50_each_cell": True,
            "no_post_run_model_selection": True,
        },
        "seed_results": rows,
        "summary": {
            "delay_metrics": summary,
            "primary_delay": 32,
            "primary_slow_delta": primary,
            "primary_slow_ci95": ci,
            "fast_delta_delay_32": fast_delta,
            "primary_materiality_threshold": 0.15,
            "fast_maximum_allowed_harm": -0.05,
            "primary_criterion_pass": bool(not smoke and primary >= 0.15 and ci[0] > 0 and fast_delta >= -0.05),
        },
        "claim_boundary": [
            "bounded synthetic multimodal temporal-retention mechanism only",
            "no general MTSK or memory claim before this experiment is valid",
            "no semantic memory claim",
            "no learned addressing or operator discovery claim",
            "no scaling claim",
        ],
    }
    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
