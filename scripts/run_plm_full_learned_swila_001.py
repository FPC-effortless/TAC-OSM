#!/usr/bin/env python3
"""TACOSM-PLM-FULL-LEARNED-SWILA-001.

The primary full-PLM measurement:
    multimodal perception -> MTSK persistence -> CDL -> learned CASM
    -> action -> environment -> verifier -> repair -> verified write.

No fixed operator table, entity IDs, task-specific state vectors, or oracle
retrieval path is used by the model.

This is a capability run, not an ablation run.  Ablation is intentionally
deferred until the integrated model either passes or yields a clean bounded
negative.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.plm_full_benchmark import (  # noqa: E402
    ACTION_COUNT,
    HISTORY,
    VOCAB_SIZE,
    Episode,
    episode_fingerprint,
    episode_key,
    sample_balanced_episodes,
    sample_history_pairs,
    step_environment,
)
from tac_osm.plm_full_learned import (  # noqa: E402
    FullLearnedPLM,
    FullPLMConfig,
    full_plm_parameter_count,
)

EXPERIMENT_ID = "TACOSM-PLM-FULL-LEARNED-SWILA-001"
SEEDS = (0, 1, 2, 3, 4)
TRAIN_PER_ACTION = 72
EVAL_PER_ACTION = 36
HISTORY_PAIRS = 80
TRAIN_STEPS = 220
BATCH_SIZE = 8
LR = 0.001
WEIGHT_DECAY = 1e-4
PHYSICS_WEIGHT = 0.03
WORLD_WEIGHT = 0.20
VERIFIER_WEIGHT = 0.50
REPAIR_WEIGHT = 0.15
LOAD_WEIGHT = 0.01


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch(episodes: Iterable[Episode]) -> dict[str, torch.Tensor]:
    eps = list(episodes)
    return {
        "text": torch.stack([e.text for e in eps], dim=0),
        "image": torch.stack([e.image for e in eps], dim=0),
        "audio": torch.stack([e.audio for e in eps], dim=0),
    }


def _post_batch(
    model: FullLearnedPLM,
    forward: dict[str, torch.Tensor],
    episodes: list[Episode],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = forward["action_logits"].argmax(dim=-1)
    outcomes = []
    targets_after = []
    valid_repair = []
    post_text = []
    post_image = []
    post_audio = []

    for episode, action in zip(episodes, actions.tolist()):
        _next_state, obs, reward, target_after = step_environment(episode, int(action))
        post_text.append(obs[0])
        post_image.append(obs[1])
        post_audio.append(obs[2])
        outcomes.append(float(reward))
        if target_after is None:
            targets_after.append(0)
            valid_repair.append(0.0)
        else:
            targets_after.append(int(target_after))
            valid_repair.append(1.0)

    outcome = torch.tensor(outcomes, dtype=forward["action_logits"].dtype)
    target_after = torch.tensor(targets_after, dtype=torch.long)
    valid_repair = torch.tensor(
        valid_repair, dtype=forward["action_logits"].dtype
    )
    post_z = model.representation(
        torch.stack(post_text, dim=0),
        torch.stack(post_image, dim=0),
        torch.stack(post_audio, dim=0),
    )
    return outcome, target_after, valid_repair, post_z


def train_seed(seed: int) -> tuple[FullLearnedPLM, dict]:
    random.seed(seed)
    torch.manual_seed(seed)

    cfg = FullPLMConfig(
        vocab_size=VOCAB_SIZE,
        action_count=ACTION_COUNT,
        d_model=96,
        num_heads=4,
        num_experts=6,
        fast_experts=2,
        medium_experts=2,
        slow_experts=2,
        cdl_top_k=2,
        casm_operators=4,
        casm_depth=3,
    )
    model = FullLearnedPLM(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    pool_rng = random.Random(seed * 100003 + 17)
    train_pool = sample_balanced_episodes(pool_rng, TRAIN_PER_ACTION)
    losses: list[float] = []

    for step in range(TRAIN_STEPS):
        batch_eps = [
            train_pool[(step * BATCH_SIZE + j) % len(train_pool)]
            for j in range(BATCH_SIZE)
        ]
        batch = _batch(batch_eps)
        output = model(**batch)

        targets = torch.tensor(
            [e.action for e in batch_eps], dtype=torch.long
        )
        action_loss = F.cross_entropy(output["action_logits"], targets)

        actions = output["action_logits"].argmax(dim=-1)
        outcomes = []
        post_text = []
        post_image = []
        post_audio = []
        post_targets = []
        repair_valid = []
        for episode, action in zip(batch_eps, actions.detach().tolist()):
            _next_state, obs, reward, target_after = step_environment(
                episode, int(action)
            )
            outcomes.append(float(reward))
            post_text.append(obs[0])
            post_image.append(obs[1])
            post_audio.append(obs[2])
            if target_after is None:
                post_targets.append(0)
                repair_valid.append(0.0)
            else:
                post_targets.append(int(target_after))
                repair_valid.append(1.0)

        outcome = torch.tensor(
            outcomes, dtype=output["action_logits"].dtype
        )
        post_z = model.representation(
            torch.stack(post_text),
            torch.stack(post_image),
            torch.stack(post_audio),
        )
        predicted_next = model.osm.predict(
            output["query"],
            actions,
            outcome,
        )
        world_loss = F.mse_loss(predicted_next, post_z.detach())

        step_result = model.verify_and_update(
            output,
            action=actions,
            outcome=outcome,
            post_text=torch.stack(post_text),
            post_image=torch.stack(post_image),
            post_audio=torch.stack(post_audio),
        )
        verifier_loss = F.binary_cross_entropy_with_logits(
            step_result.verifier_logit,
            outcome,
        )
        repair_target = torch.tensor(post_targets, dtype=torch.long)
        repair_mask = torch.tensor(
            repair_valid, dtype=output["action_logits"].dtype
        )
        repair_loss_all = F.cross_entropy(
            step_result.repair_logits,
            repair_target,
            reduction="none",
        )
        repair_loss = (repair_loss_all * repair_mask).sum() / repair_mask.sum().clamp_min(1.0)

        canonical_physics = model.physics_prior_loss(
            output["z_seq"],
            dt=0.08,
        )
        total = (
            action_loss
            + WORLD_WEIGHT * world_loss
            + VERIFIER_WEIGHT * verifier_loss
            + REPAIR_WEIGHT * repair_loss
            + PHYSICS_WEIGHT * canonical_physics
            + LOAD_WEIGHT * output["load_loss"]
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(total.detach()))

    return model, {
        "final_loss": losses[-1],
        "mean_loss_last_20": statistics.fmean(losses[-20:]),
        "train_size": len(train_pool),
    }


def evaluate_seed(
    model: FullLearnedPLM,
    seed: int,
) -> dict:
    random.seed(seed + 50000)
    torch.manual_seed(seed + 50000)
    model.eval()

    eval_rng = random.Random(seed * 200003 + 29)
    eval_pool = sample_balanced_episodes(eval_rng, EVAL_PER_ACTION)
    pair_rng = random.Random(seed * 400009 + 43)
    pairs = sample_history_pairs(pair_rng, HISTORY_PAIRS)

    with torch.no_grad():
        batch = _batch(eval_pool)
        output = model(**batch)
        pred = output["action_logits"].argmax(dim=-1)
        targets = torch.tensor([e.action for e in eval_pool], dtype=torch.long)
        accuracy = float((pred == targets).float().mean())

        outcome = []
        post_text = []
        post_image = []
        post_audio = []
        repair_targets = []
        first_success = []
        repair_success = []

        for episode, action in zip(eval_pool, pred.tolist()):
            _next_state, obs, reward, target_after = step_environment(
                episode, int(action)
            )
            outcome.append(float(reward))
            first_success.append(float(reward))
            post_text.append(obs[0])
            post_image.append(obs[1])
            post_audio.append(obs[2])
            repair_targets.append(-1 if target_after is None else target_after)

        outcome_t = torch.tensor(outcome, dtype=torch.float32)
        post_result = model.verify_and_update(
            output,
            action=pred,
            outcome=outcome_t,
            post_text=torch.stack(post_text),
            post_image=torch.stack(post_image),
            post_audio=torch.stack(post_audio),
        )
        for idx, target in enumerate(repair_targets):
            if target >= 0:
                repaired = int(post_result.repair_logits[idx].argmax().item())
                repair_success.append(float(repaired == target))

        pair_correct = []
        pair_disagreement = []
        for left, right in pairs:
            pair_out_left = model(**left.batch)
            pair_out_right = model(**right.batch)
            pred_left = int(pair_out_left["action_logits"].argmax().item())
            pred_right = int(pair_out_right["action_logits"].argmax().item())
            pair_correct.append(
                float(pred_left == left.action and pred_right == right.action)
            )
            pair_disagreement.append(float(pred_left != pred_right))

    return {
        "seed": seed,
        "eval_size": len(eval_pool),
        "accuracy": accuracy,
        "minimum_class_count": EVAL_PER_ACTION,
        "first_action_success": statistics.fmean(first_success),
        "repair_success_conditional": (
            statistics.fmean(repair_success) if repair_success else 0.0
        ),
        "verified_write_rate": float(
            (post_result.verifier_logit.sigmoid() >= model.config.verifier_threshold)
            .float()
            .mean()
        ),
        "history_pair_exact": statistics.fmean(pair_correct),
        "history_pair_disagreement": statistics.fmean(pair_disagreement),
        "eval_fingerprint": episode_fingerprint(eval_pool),
        "eval_keys": [episode_key(e) for e in eval_pool],
    }


def main() -> None:
    seeds = []
    models = {}
    train_meta = {}
    eval_meta = []

    for seed in SEEDS:
        model, train_info = train_seed(seed)
        models[seed] = model
        train_meta[seed] = train_info
        eval_meta.append(evaluate_seed(model, seed))

    # Cross-seed benchmark integrity: no train/eval overlap within seed and
    # no duplicated evaluation fingerprint across seeds.
    integrity = {"train_eval_overlap": {}, "distinct_eval_fingerprints": True}
    for seed in SEEDS:
        rng = random.Random(seed * 100003 + 17)
        train_pool = sample_balanced_episodes(rng, TRAIN_PER_ACTION)
        train_keys = {episode_key(e) for e in train_pool}
        eval_keys = set(eval_meta[seed]["eval_keys"])
        integrity["train_eval_overlap"][str(seed)] = len(train_keys & eval_keys)

    fps = [row["eval_fingerprint"] for row in eval_meta]
    integrity["distinct_eval_fingerprints"] = len(fps) == len(set(fps))

    manifest = __import__(
        "tac_osm.plm_full_benchmark",
        fromlist=["benchmark_manifest"],
    ).benchmark_manifest()
    contract_path = ROOT / "contracts" / "TACOSM-PLM-FULL-LEARNED-SWILA-001.json"

    accuracies = [r["accuracy"] for r in eval_meta]
    pairs = [r["history_pair_exact"] for r in eval_meta]
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "science_commit": os.environ.get("GITHUB_SHA", "local"),
        "protocol": {
            "seeds": list(SEEDS),
            "train_per_action": TRAIN_PER_ACTION,
            "eval_per_action": EVAL_PER_ACTION,
            "history_pairs": HISTORY_PAIRS,
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "history_length": HISTORY,
            "model_selection": "none",
        },
        "model": {
            "config": vars(FullPLMConfig()),
            "parameter_count": full_plm_parameter_count(models[SEEDS[0]]),
            "privileged_physics_prior": [
                "Hamilton canonical equations",
                "Hamiltonian conservation on passive intervals",
            ],
            "forbidden_scaffolds": [
                "entity identifiers",
                "fixed operator table",
                "gold action input",
                "physical-state supervision",
                "hard-coded retrieval",
            ],
        },
        "benchmark": manifest,
        "benchmark_hash": _sha256_file(
            ROOT / "src" / "tac_osm" / "plm_full_benchmark.py"
        ),
        "contract_hash": _sha256_file(contract_path) if contract_path.exists() else None,
        "integrity": integrity,
        "train": train_meta,
        "evaluation": eval_meta,
        "aggregate": {
            "mean_accuracy": statistics.fmean(accuracies),
            "min_seed_accuracy": min(accuracies),
            "mean_history_pair_exact": statistics.fmean(pairs),
            "min_seed_history_pair_exact": min(pairs),
            "full_plm_gate_pass": (
                statistics.fmean(accuracies) >= 0.70
                and min(accuracies) >= 0.55
                and statistics.fmean(pairs) >= 0.70
                and all(v == 0 for v in integrity["train_eval_overlap"].values())
                and integrity["distinct_eval_fingerprints"]
            ),
        },
    }

    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
