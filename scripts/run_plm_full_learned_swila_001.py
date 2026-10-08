#!/usr/bin/env python3
"""TACOSM-PLM-FULL-LEARNED-SWILA-001.

Primary construction run for the full learned PLM.  The model itself contains
learned representation, persistent MTSK, CDL, structural CASM, OSM, verifier,
repair, and verifier-gated state update.  The benchmark's binary success label
is never passed to the verifier; only continuous post-action outcome evidence
is supplied to the model.

This is a capability run, not an ablation.
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
    step_environment_from_state,
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
SECOND_VERIFIER_WEIGHT = 0.25
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


def _post_observation(
    model: FullLearnedPLM,
    observation: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    return model.representation(
        torch.stack([x[0] for x in observation]),
        torch.stack([x[1] for x in observation]),
        torch.stack([x[2] for x in observation]),
    )


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
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
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
        targets = torch.tensor([e.action for e in batch_eps], dtype=torch.long)
        action_loss = F.cross_entropy(output["action_logits"], targets)

        first_states = []
        first_observations = []
        first_evidence = []
        first_rewards = []
        repair_targets = []
        repair_masks = []

        for episode, action in zip(
            batch_eps, output["action_logits"].argmax(dim=-1).detach().tolist()
        ):
            next_state, obs, evidence, reward, target_after = step_environment(
                episode, int(action)
            )
            first_states.append(next_state)
            first_observations.append(obs)
            first_evidence.append(evidence)
            first_rewards.append(float(reward))
            repair_targets.append(0 if target_after is None else int(target_after))
            repair_masks.append(
                float((not reward) and target_after is not None)
            )

        evidence = torch.tensor(
            first_evidence, dtype=output["action_logits"].dtype
        )
        reward_label = torch.tensor(
            first_rewards, dtype=output["action_logits"].dtype
        )
        post_z = _post_observation(model, first_observations)
        predicted_next = model.osm.predict(
            output["query"],
            output["action_logits"].argmax(dim=-1).detach(),
            evidence,
        )
        world_loss = F.mse_loss(predicted_next, post_z.detach())

        first_step = model.verify_and_update(
            output,
            action=output["action_logits"].argmax(dim=-1).detach(),
            outcome=evidence,
            post_text=torch.stack([x[0] for x in first_observations]),
            post_image=torch.stack([x[1] for x in first_observations]),
            post_audio=torch.stack([x[2] for x in first_observations]),
        )
        verifier_loss = F.binary_cross_entropy_with_logits(
            first_step.verifier_logit, reward_label
        )

        repair_target = torch.tensor(repair_targets, dtype=torch.long)
        repair_mask = torch.tensor(
            repair_masks, dtype=output["action_logits"].dtype
        )
        repair_loss_all = F.cross_entropy(
            first_step.repair_logits, repair_target, reduction="none"
        )
        repair_loss = (
            (repair_loss_all * repair_mask).sum()
            / repair_mask.sum().clamp_min(1.0)
        )

        # Execute the learned repair after a failed first action.  The repair
        # action is selected without access to the hidden target.
        repair_actions = first_step.repair_logits.argmax(dim=-1).detach()
        second_evidence = []
        second_rewards = []
        second_states = []
        second_obs = []
        second_masks = []
        for i, (episode, action, failed) in enumerate(
            zip(batch_eps, repair_actions.tolist(), [not bool(x) for x in first_rewards])
        ):
            if not failed:
                second_evidence.append(0.0)
                second_rewards.append(0.0)
                second_states.append(None)
                second_obs.append(None)
                second_masks.append(0.0)
                continue
            state2, obs2, ev2, reward2, _target2 = step_environment_from_state(
                first_states[i], episode.goal, int(action)
            )
            second_states.append(state2)
            second_obs.append(obs2)
            second_evidence.append(ev2)
            second_rewards.append(float(reward2))
            second_masks.append(1.0)

        second_mask = torch.tensor(
            second_masks, dtype=output["action_logits"].dtype
        )
        if second_mask.sum() > 0:
            valid_obs = [x for x in second_obs if x is not None]
            post2_z = _post_observation(model, valid_obs)
            valid_pre = first_step.memory_read[
                second_mask.bool()
            ]
            valid_actions = repair_actions[second_mask.bool()]
            valid_action_prob = F.one_hot(
                valid_actions, num_classes=ACTION_COUNT
            ).to(output["action_logits"].dtype)
            valid_ev = torch.tensor(
                [x for x, m in zip(second_evidence, second_masks) if m],
                dtype=output["action_logits"].dtype,
            )
            second_reward_label = torch.tensor(
                [x for x, m in zip(second_rewards, second_masks) if m],
                dtype=output["action_logits"].dtype,
            )
            second_verifier_logit = model.verifier(
                valid_pre, post2_z, valid_action_prob, valid_ev
            )
            second_verifier_loss = F.binary_cross_entropy_with_logits(
                second_verifier_logit, second_reward_label
            )
            second_gate = second_verifier_logit.sigmoid()
            model.mtsk.commit_verified(
                first_step.next_memory_state[second_mask.bool()],
                post2_z,
                valid_actions,
                valid_ev,
                second_gate,
            )
        else:
            second_verifier_loss = torch.zeros(
                (), dtype=output["action_logits"].dtype
            )

        canonical_physics = model.physics_prior_loss(output["z_seq"], dt=0.08)
        total = (
            action_loss
            + WORLD_WEIGHT * world_loss
            + VERIFIER_WEIGHT * verifier_loss
            + REPAIR_WEIGHT * repair_loss
            + SECOND_VERIFIER_WEIGHT * second_verifier_loss
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


def evaluate_seed(model: FullLearnedPLM, seed: int) -> dict:
    random.seed(seed + 50000)
    torch.manual_seed(seed + 50000)
    model.eval()

    eval_rng = random.Random(seed * 200003 + 29)
    eval_pool = sample_balanced_episodes(eval_rng, EVAL_PER_ACTION)
    pair_rng = random.Random(seed * 400009 + 43)
    pairs = sample_history_pairs(pair_rng, HISTORY_PAIRS)

    batch = _batch(eval_pool)
    with torch.no_grad():
        output = model(**batch)
        pred = output["action_logits"].argmax(dim=-1)
        targets = torch.tensor([e.action for e in eval_pool], dtype=torch.long)
        accuracy = float((pred == targets).float().mean())

    first_success: list[float] = []
    repair_success: list[float] = []
    closed_loop_success: list[float] = []
    verifier_probs: list[float] = []

    for index, (episode, action) in enumerate(zip(eval_pool, pred.tolist())):
        next_state, obs, evidence, reward, _target_after = step_environment(
            episode, int(action)
        )
        post_z = _post_observation(model, [obs])
        with torch.no_grad():
            action_onehot = F.one_hot(
                torch.tensor([action]), num_classes=ACTION_COUNT
            ).float()
            verifier_logit = model.verifier(
                output["query"][index:index + 1],
                post_z,
                F.one_hot(
                    torch.tensor([action]), num_classes=ACTION_COUNT
                ).float(),
                torch.tensor([evidence], dtype=torch.float32),
            )
            verifier_prob = float(verifier_logit.sigmoid().item())
            repair_logits = model.repair(
                post_z,
                action_onehot,
                torch.tensor([evidence], dtype=torch.float32),
            )
            verifier_probs.append(verifier_prob)

        first_success.append(float(reward))
        if reward:
            closed_loop_success.append(1.0)
            continue

        repair_action = int(repair_logits.argmax(dim=-1).item())
        next2, obs2, evidence2, reward2, _target2 = step_environment_from_state(
            next_state, episode.goal, repair_action
        )
        repair_success.append(float(reward2))
        closed_loop_success.append(float(reward2))

        post2_z = _post_observation(model, [obs2])
        with torch.no_grad():
            repair_prob = F.one_hot(
                torch.tensor([repair_action]), num_classes=ACTION_COUNT
            ).float()
            second_verifier_logit = model.verifier(
                post_z,
                post2_z,
                repair_prob,
                torch.tensor([evidence2], dtype=torch.float32),
            )
            second_prob = second_verifier_logit.sigmoid()
            model.mtsk.commit_verified(
                output["memory_state"][index:index + 1],
                post2_z,
                torch.tensor([repair_action]),
                torch.tensor([evidence2], dtype=torch.float32),
                second_prob,
            )

    pair_correct = []
    pair_disagreement = []
    with torch.no_grad():
        for left, right in pairs:
            left_out = model(**left.batch)
            right_out = model(**right.batch)
            left_pred = int(left_out["action_logits"].argmax().item())
            right_pred = int(right_out["action_logits"].argmax().item())
            pair_correct.append(
                float(left_pred == left.action and right_pred == right.action)
            )
            pair_disagreement.append(float(left_pred != right_pred))

    return {
        "seed": seed,
        "eval_size": len(eval_pool),
        "accuracy": accuracy,
        "first_action_success": statistics.fmean(first_success),
        "repair_attempt_rate": statistics.fmean(
            [1.0 - x for x in first_success]
        ),
        "repair_success_conditional": (
            statistics.fmean(repair_success) if repair_success else 0.0
        ),
        "closed_loop_success": statistics.fmean(closed_loop_success),
        "mean_first_verifier_probability": statistics.fmean(verifier_probs),
        "history_pair_exact": statistics.fmean(pair_correct),
        "history_pair_disagreement": statistics.fmean(pair_disagreement),
        "eval_fingerprint": episode_fingerprint(eval_pool),
        "eval_keys": [episode_key(e) for e in eval_pool],
    }


def main() -> None:
    eval_meta = []
    train_meta = {}
    models = {}

    for seed in SEEDS:
        model, train_info = train_seed(seed)
        models[seed] = model
        train_meta[str(seed)] = train_info
        eval_meta.append(evaluate_seed(model, seed))

    integrity = {"train_eval_overlap": {}, "distinct_eval_fingerprints": True}
    all_train_sets: dict[int, set[tuple]] = {}
    all_eval_sets: dict[int, set[tuple]] = {}
    all_pair_sets: dict[int, set[tuple]] = {}

    for seed in SEEDS:
        train_rng = random.Random(seed * 100003 + 17)
        train_pool = sample_balanced_episodes(train_rng, TRAIN_PER_ACTION)
        train_keys = {episode_key(e) for e in train_pool}
        eval_keys = set(eval_meta[SEEDS.index(seed)]["eval_keys"])
        all_train_sets[seed] = train_keys
        all_eval_sets[seed] = eval_keys
        integrity["train_eval_overlap"][str(seed)] = len(train_keys & eval_keys)

        pair_rng = random.Random(seed * 400009 + 43)
        pairs_for_integrity = sample_history_pairs(pair_rng, HISTORY_PAIRS)
        pair_keys = {
            episode_key(episode)
            for pair in pairs_for_integrity
            for episode in pair
        }
        all_pair_sets[seed] = pair_keys

    integrity["cross_seed_train_overlap"] = {
        f"{a}:{b}": len(all_train_sets[a] & all_train_sets[b])
        for i, a in enumerate(SEEDS)
        for b in SEEDS[i + 1:]
    }
    integrity["cross_seed_eval_overlap"] = {
        f"{a}:{b}": len(all_eval_sets[a] & all_eval_sets[b])
        for i, a in enumerate(SEEDS)
        for b in SEEDS[i + 1:]
    }
    integrity["history_pair_train_overlap"] = {
        str(seed): len(
            all_pair_sets[seed]
            & set().union(*(all_train_sets[s] for s in SEEDS))
        )
        for seed in SEEDS
    }
    integrity["history_pair_eval_overlap"] = {
        str(seed): len(
            all_pair_sets[seed]
            & set().union(*(all_eval_sets[s] for s in SEEDS))
        )
        for seed in SEEDS
    }
    integrity["cross_seed_pair_overlap"] = {
        f"{a}:{b}": len(all_pair_sets[a] & all_pair_sets[b])
        for i, a in enumerate(SEEDS)
        for b in SEEDS[i + 1:]
    }

    fps = [row["eval_fingerprint"] for row in eval_meta]
    integrity["distinct_eval_fingerprints"] = len(fps) == len(set(fps))

    all_integrity_zero = (
        all(v == 0 for v in integrity["train_eval_overlap"].values())
        and all(v == 0 for v in integrity["cross_seed_train_overlap"].values())
        and all(v == 0 for v in integrity["cross_seed_eval_overlap"].values())
        and all(v == 0 for v in integrity["history_pair_train_overlap"].values())
        and all(v == 0 for v in integrity["history_pair_eval_overlap"].values())
        and all(v == 0 for v in integrity["cross_seed_pair_overlap"].values())
        and integrity["distinct_eval_fingerprints"]
    )

    benchmark = __import__(
        "tac_osm.plm_full_benchmark",
        fromlist=["benchmark_manifest"],
    ).benchmark_manifest()
    contract_path = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
    accuracies = [r["accuracy"] for r in eval_meta]
    pairs = [r["history_pair_exact"] for r in eval_meta]
    closed = [r["closed_loop_success"] for r in eval_meta]

    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "science_commit": os.environ.get("GITHUB_SHA", "local"),
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
                "binary success input to verifier",
            ],
        },
        "benchmark": benchmark,
        "benchmark_hash": _sha256_file(
            ROOT / "src" / "tac_osm" / "plm_full_benchmark.py"
        ),
        "contract_hash": _sha256_file(contract_path),
        "integrity": integrity,
        "train": train_meta,
        "evaluation": eval_meta,
        "aggregate": {
            "mean_accuracy": statistics.fmean(accuracies),
            "min_seed_accuracy": min(accuracies),
            "mean_history_pair_exact": statistics.fmean(pairs),
            "min_seed_history_pair_exact": min(pairs),
            "mean_closed_loop_success": statistics.fmean(closed),
            "min_seed_closed_loop_success": min(closed),
            "full_plm_gate_pass": (
                statistics.fmean(accuracies) >= 0.70
                and min(accuracies) >= 0.55
                and statistics.fmean(pairs) >= 0.70
                and statistics.fmean(closed) >= 0.70
                and all_integrity_zero
            ),
        },
        "scientific_note": (
            "Capability measurement only. Ablation is prohibited until this "
            "integrated gate passes with intact provenance and integrity."
        ),
    }

    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
