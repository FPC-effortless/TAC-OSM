#!/usr/bin/env python3
"""Training-only functionality gate for the post-E2E-005 CASM repair.

This script never evaluates E2E-005's retired held-out compositions. It samples
fresh episodes only from the registered training-composition distribution.
Its output is development evidence for functionality, not evidence for
held-out generalization or for the original E2E-005 claim.
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from scripts.run_integrated_e2e_005 import (
    accuracy,
    build_batch,
    write_observations,
)
from tac_osm.functional_repair_signed_casm import FunctionalRepairPLM
from tac_osm.integrated_e2e_005 import OPS
from tac_osm.integrated_e2e_005_benchmark import (
    BATCH_SIZE,
    EVAL_EPISODES,
    TRAIN_COMBOS,
    episode_fingerprint,
    episode_key,
    sample_episode,
)

SEEDS = (0, 1, 2)
STEPS = 300
EVAL_COUNT = EVAL_EPISODES


def environment_outcome(action: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((action >= 0.5).long() == target).float().detach()


def train_batch(model: FunctionalRepairPLM, batch: dict) -> torch.Tensor:
    memory = write_observations(model, batch)

    e1, i1, j1, op1 = batch["q1"]
    out1 = model.query(memory, e1, i1, j1, op1)
    loss1 = F.cross_entropy(out1["logits"], batch["y1"])
    outcome1 = environment_outcome(out1["action"], batch["y1"])
    memory, verifier1 = model.post_action_update(memory, out1, outcome1)

    e2, i2, j2, op2 = batch["q2"]
    out2 = model.query(memory, e2, i2, j2, op2)
    loss2 = F.cross_entropy(out2["logits"], batch["y2"])
    outcome2 = environment_outcome(out2["action"], batch["y2"])
    _, verifier2 = model.post_action_update(memory, out2, outcome2)

    verifier_loss = (
        F.binary_cross_entropy_with_logits(verifier1[:, :1], verifier1[:, 1:])
        + F.binary_cross_entropy_with_logits(verifier2[:, :1], verifier2[:, 1:])
    ) / 2
    return loss1 + loss2 + 0.10 * verifier_loss


def sample_dev_episodes(rng: random.Random, count: int) -> list[tuple]:
    episodes: list[tuple] = []
    for _ in range(count):
        combo1 = rng.choice(TRAIN_COMBOS)
        combo2 = rng.choice(TRAIN_COMBOS)
        ep = sample_episode(rng, combo1=combo1, combo2=combo2)
        episodes.append(ep)
    return episodes


def gradient_surface_probe() -> dict:
    torch.manual_seed(20261006)
    rng = random.Random(42006)
    episodes = [sample_dev_episodes(rng, 1)[0] for _ in range(4)]
    batch = build_batch(episodes)
    model = FunctionalRepairPLM()
    loss = train_batch(model, batch)
    loss.backward()
    required = (
        "text.emb.weight",
        "image.net.0.weight",
        "audio.net.0.weight",
        "rep.fuse.0.weight",
        "state.write_value.weight",
        "casm.decoder.0.weight",
        "verifier.0.weight",
        "write_gate.0.weight",
    )
    failures = []
    params = dict(model.named_parameters())
    for name in required:
        grad = params[name].grad
        if grad is None or not torch.isfinite(grad).all():
            failures.append(name)
    return {"pass": not failures, "missing_or_nonfinite": failures}


def train_seed(seed: int):
    torch.manual_seed(seed)
    rng = random.Random(seed + 50000)
    model = FunctionalRepairPLM().cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.0001,
    )
    training_keys: set[tuple] = set()
    for _ in range(STEPS):
        episodes = sample_dev_episodes(rng, BATCH_SIZE)
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
def evaluate_dev(model: FunctionalRepairPLM, episodes: list[tuple]) -> dict:
    model.eval()
    q1 = 0
    q2 = 0
    verifier = 0
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
        e1, i1, j1, op1, y1 = episode[1]
        out1 = model.query(
            memory,
            torch.tensor([e1]),
            torch.tensor([i1]),
            torch.tensor([j1]),
            torch.tensor([OPS.index(op1)]),
        )
        target1 = torch.tensor([y1])
        outcome1 = environment_outcome(out1["action"], target1)
        updated, verifier1 = model.post_action_update(memory, out1, outcome1)

        e2, i2, j2, op2, y2 = episode[2]
        out2 = model.query(
            updated,
            torch.tensor([e2]),
            torch.tensor([i2]),
            torch.tensor([j2]),
            torch.tensor([OPS.index(op2)]),
        )
        target2 = torch.tensor([y2])
        outcome2 = environment_outcome(out2["action"], target2)
        _, verifier2 = model.post_action_update(updated, out2, outcome2)

        q1 += accuracy(out1["logits"], y1)
        q2 += accuracy(out2["logits"], y2)
        verifier += int(
            ((verifier1[:, :1].sigmoid() >= 0.5).long() == outcome1.long()).item()
        )
        verifier += int(
            ((verifier2[:, :1].sigmoid() >= 0.5).long() == outcome2.long()).item()
        )
    n = len(episodes)
    return {
        "q1_accuracy": q1 / n,
        "q2_accuracy": q2 / n,
        "verifier_accuracy": verifier / (2 * n),
    }


def run() -> dict:
    gradient = gradient_surface_probe()
    if not gradient["pass"]:
        raise AssertionError(f"gradient-surface failure: {gradient}")

    rows = []
    for seed in SEEDS:
        model, training_keys = train_seed(seed)
        rng = random.Random(seed + 250000)
        episodes = sample_dev_episodes(rng, EVAL_COUNT)
        eval_keys = {episode_key(ep) for ep in episodes}
        overlap = training_keys & eval_keys
        if overlap:
            raise AssertionError(f"development train/eval overlap for seed {seed}")
        metrics = evaluate_dev(model, episodes)
        for ep in episodes:
            for q in ep[1:3]:
                if (q[3], q[1], q[2]) not in TRAIN_COMBOS:
                    raise AssertionError("development evaluation used a held-out composition")
        rows.append(
            {
                "seed": seed,
                **metrics,
                "episode_fingerprint": episode_fingerprint(episodes),
                "training_evaluation_semantic_overlap": len(overlap),
            }
        )

    q2 = [row["q2_accuracy"] for row in rows]
    output = {
        "status": "development_only",
        "not_scientific_evidence": True,
        "retired_e2e005_holdout_used": False,
        "evaluation_distribution": "registered E2E-005 TRAIN_COMBOS only",
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "evaluation_episodes_per_seed": EVAL_COUNT,
        "seeds": list(SEEDS),
        "gradient_surface": gradient,
        "seed_results": rows,
        "mean_q2_accuracy": statistics.fmean(q2),
        "min_q2_accuracy": min(q2),
        "functionality_gate": statistics.fmean(q2) >= 0.90 and min(q2) >= 0.80,
    }
    out = (
        Path("artifacts/development")
        / "PLM-FUNCTIONAL-REPAIR-SIGNED-CASM.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    run()
