#!/usr/bin/env python3
"""TACOSM-PLM-FULL-LEARNED-SWILA-OPT-002.

Pre-registered diagnostic companion to the completed
TACOSM-PLM-FULL-LEARNED-SWILA-001 run.

Arms:
  * untrained: exact architecture, zero optimizer updates;
  * trained_full: exact original training protocol;
  * trained_physics_off: exact original protocol with only the physics prior
    weight removed.

This is a diagnosis, not a new capability claim.
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

EXPERIMENT_ID = "TACOSM-PLM-FULL-LEARNED-SWILA-OPT-002"
SEEDS = (0, 1, 2, 3, 4)
TRAIN_PER_ACTION = 72
EVAL_PER_ACTION = 36
HISTORY_PAIRS = 80
TRAIN_STEPS = 220
BATCH_SIZE = 8
LR = 0.001
WEIGHT_DECAY = 1e-4
FULL_PHYSICS_WEIGHT = 0.03
WORLD_WEIGHT = 0.20
VERIFIER_WEIGHT = 0.50
REPAIR_WEIGHT = 0.15
SECOND_VERIFIER_WEIGHT = 0.25
LOAD_WEIGHT = 0.01
LOSS_REDUCTION_THRESHOLD = 0.10
ACCURACY_DELTA_THRESHOLD = 0.05
GENERATOR_SHA256 = "42cd765194492a06bc2180bf2c9462c973ab73d6"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


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


def build_model(seed: int) -> FullLearnedPLM:
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
    return FullLearnedPLM(cfg)


def train_seed(seed: int, physics_weight: float) -> tuple[FullLearnedPLM, dict]:
    random.seed(seed)
    torch.manual_seed(seed)
    model = build_model(seed)
    initial_hash = parameter_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    pool_rng = random.Random(seed * 100003 + 17)
    train_pool = sample_balanced_episodes(pool_rng, TRAIN_PER_ACTION)
    trajectory: list[dict[str, float]] = []

    for step in range(TRAIN_STEPS):
        batch_eps = [
            train_pool[(step * BATCH_SIZE + j) % len(train_pool)]
            for j in range(BATCH_SIZE)
        ]
        batch = _batch(batch_eps)
        targets = torch.tensor([e.action for e in batch_eps], dtype=torch.long)
        output = model(**batch)
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
            first_evidence.append(float(evidence))
            first_rewards.append(float(reward))
            repair_targets.append(0 if target_after is None else int(target_after))
            repair_masks.append(float((not reward) and target_after is not None))

        evidence = torch.tensor(first_evidence, dtype=output["action_logits"].dtype)
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

        repair_actions = first_step.repair_logits.argmax(dim=-1).detach()
        second_evidence = []
        second_rewards = []
        second_obs = []
        second_masks = []
        for i, (episode, action, failed) in enumerate(
            zip(
                batch_eps,
                repair_actions.tolist(),
                [not bool(x) for x in first_rewards],
            )
        ):
            if not failed:
                second_evidence.append(0.0)
                second_rewards.append(0.0)
                second_obs.append(None)
                second_masks.append(0.0)
                continue
            state2, obs2, ev2, reward2, _target2 = step_environment_from_state(
                first_states[i], episode.goal, int(action)
            )
            second_obs.append(obs2)
            second_evidence.append(float(ev2))
            second_rewards.append(float(reward2))
            second_masks.append(1.0)

        second_mask = torch.tensor(
            second_masks, dtype=output["action_logits"].dtype
        )
        if second_mask.sum() > 0:
            valid_obs = [x for x in second_obs if x is not None]
            post2_z = _post_observation(model, valid_obs)
            valid_pre = first_step.memory_read[second_mask.bool()]
            valid_actions = repair_actions[second_mask.bool()]
            valid_action_prob = F.one_hot(
                valid_actions, num_classes=ACTION_COUNT
            ).to(output["action_logits"].dtype)
            valid_ev = torch.tensor(
                [
                    x
                    for x, m in zip(second_evidence, second_masks)
                    if m
                ],
                dtype=output["action_logits"].dtype,
            )
            second_reward_label = torch.tensor(
                [
                    x
                    for x, m in zip(second_rewards, second_masks)
                    if m
                ],
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

        physics_loss = (
            model.physics_prior_loss(output["z_seq"], dt=0.08)
            if physics_weight > 0
            else torch.zeros((), dtype=output["action_logits"].dtype)
        )
        total = (
            action_loss
            + WORLD_WEIGHT * world_loss
            + VERIFIER_WEIGHT * verifier_loss
            + REPAIR_WEIGHT * repair_loss
            + SECOND_VERIFIER_WEIGHT * second_verifier_loss
            + physics_weight * physics_loss
            + LOAD_WEIGHT * output["load_loss"]
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        )
        optimizer.step()

        trajectory.append(
            {
                "step": float(step + 1),
                "total_loss": float(total.detach()),
                "action_loss": float(action_loss.detach()),
                "world_loss": float(world_loss.detach()),
                "verifier_loss": float(verifier_loss.detach()),
                "repair_loss": float(repair_loss.detach()),
                "second_verifier_loss": float(second_verifier_loss.detach()),
                "physics_loss": float(physics_loss.detach()),
                "load_loss": float(output["load_loss"].detach()),
                "gradient_norm_preclip": grad_norm,
            }
        )

    final_hash = parameter_hash(model)
    first20 = statistics.fmean(x["total_loss"] for x in trajectory[:20])
    last20 = statistics.fmean(x["total_loss"] for x in trajectory[-20:])
    return model, {
        "initial_parameter_hash": initial_hash,
        "final_parameter_hash": final_hash,
        "parameter_changed": initial_hash != final_hash,
        "train_size": len(train_pool),
        "first20_mean_total_loss": first20,
        "last20_mean_total_loss": last20,
        "loss_reduction_fraction": (first20 - last20) / max(abs(first20), 1e-12),
        "trajectory": trajectory,
    }


def evaluate_unmodified(model: FullLearnedPLM, seed: int) -> dict:
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
                action_onehot,
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
        _next2, obs2, evidence2, reward2, _target2 = step_environment_from_state(
            next_state, episode.goal, repair_action
        )
        repair_success.append(float(reward2))
        closed_loop_success.append(float(reward2))

        post2_z = _post_observation(model, [obs2])
        with torch.no_grad():
            second_verifier_logit = model.verifier(
                post_z,
                post2_z,
                F.one_hot(
                    torch.tensor([repair_action]), num_classes=ACTION_COUNT
                ).float(),
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


def make_arm(seed: int, arm: str) -> tuple[FullLearnedPLM, dict]:
    if arm == "untrained":
        model = build_model(seed)
        h = parameter_hash(model)
        return model, {
            "initial_parameter_hash": h,
            "final_parameter_hash": h,
            "parameter_changed": False,
            "train_size": 0,
            "first20_mean_total_loss": None,
            "last20_mean_total_loss": None,
            "loss_reduction_fraction": None,
            "trajectory": [],
        }
    if arm == "trained_full":
        return train_seed(seed, FULL_PHYSICS_WEIGHT)
    if arm == "trained_physics_off":
        return train_seed(seed, 0.0)
    raise ValueError(f"unknown arm: {arm}")


def main() -> None:
    integrity = {
        "train_eval_overlap": {},
        "cross_seed_train_overlap": {},
        "cross_seed_eval_overlap": {},
        "history_pair_train_overlap": {},
        "history_pair_eval_overlap": {},
        "cross_seed_pair_overlap": {},
        "distinct_eval_fingerprints": True,
    }
    arm_results: dict[str, list[dict]] = {
        "untrained": [],
        "trained_full": [],
        "trained_physics_off": [],
    }
    train_info: dict[str, dict] = {}

    # The train/eval/pair generators are deterministic functions of the seed,
    # so the same exact pools are reused across arms without sharing weights.
    all_train_sets: dict[int, set[tuple]] = {}
    all_eval_sets: dict[int, set[tuple]] = {}
    all_pair_sets: dict[int, set[tuple]] = {}

    for seed in SEEDS:
        train_rng = random.Random(seed * 100003 + 17)
        train_pool = sample_balanced_episodes(train_rng, TRAIN_PER_ACTION)
        train_keys = {episode_key(e) for e in train_pool}
        all_train_sets[seed] = train_keys

        eval_rng = random.Random(seed * 200003 + 29)
        eval_pool = sample_balanced_episodes(eval_rng, EVAL_PER_ACTION)
        eval_keys = {episode_key(e) for e in eval_pool}
        all_eval_sets[seed] = eval_keys
        integrity["train_eval_overlap"][str(seed)] = len(train_keys & eval_keys)

        pair_rng = random.Random(seed * 400009 + 43)
        pairs = sample_history_pairs(pair_rng, HISTORY_PAIRS)
        pair_keys = {
            episode_key(ep)
            for pair in pairs
            for ep in pair
        }
        all_pair_sets[seed] = pair_keys

        for arm in arm_results:
            model, info = make_arm(seed, arm)
            if arm != "untrained":
                train_info[f"{arm}:{seed}"] = {
                    k: v for k, v in info.items() if k != "trajectory"
                }
            elif not info["parameter_changed"]:
                pass
            result = evaluate_unmodified(model, seed)
            result["arm"] = arm
            result["parameter_hash"] = parameter_hash(model)
            result["parameter_changed"] = info["parameter_changed"]
            result["loss_reduction_fraction"] = info["loss_reduction_fraction"]
            result["first20_mean_total_loss"] = info["first20_mean_total_loss"]
            result["last20_mean_total_loss"] = info["last20_mean_total_loss"]
            arm_results[arm].append(result)

    for i, a in enumerate(SEEDS):
        for b in SEEDS[i + 1:]:
            integrity["cross_seed_train_overlap"][f"{a}:{b}"] = len(
                all_train_sets[a] & all_train_sets[b]
            )
            integrity["cross_seed_eval_overlap"][f"{a}:{b}"] = len(
                all_eval_sets[a] & all_eval_sets[b]
            )
            integrity["cross_seed_pair_overlap"][f"{a}:{b}"] = len(
                all_pair_sets[a] & all_pair_sets[b]
            )
    for seed in SEEDS:
        integrity["history_pair_train_overlap"][str(seed)] = len(
            all_pair_sets[seed] & set().union(*all_train_sets.values())
        )
        integrity["history_pair_eval_overlap"][str(seed)] = len(
            all_pair_sets[seed] & set().union(*all_eval_sets.values())
        )

    seed_fingerprints = [
        arm_results["untrained"][i]["eval_fingerprint"]
        for i in range(len(SEEDS))
    ]
    integrity["distinct_eval_fingerprints_by_seed"] = (
        len(seed_fingerprints) == len(set(seed_fingerprints))
    )

    # A deterministic eval pool must be the same across arms, not unique per
    # arm. The untrained fingerprints are also required to differ across seeds.
    within_seed_same_eval = all(
        arm_results["untrained"][i]["eval_fingerprint"]
        == arm_results["trained_full"][i]["eval_fingerprint"]
        == arm_results["trained_physics_off"][i]["eval_fingerprint"]
        for i in range(len(SEEDS))
    )
    integrity["within_seed_eval_identity_across_arms"] = within_seed_same_eval
    integrity["generator_sha256"] = sha256_file(
        ROOT / "src" / "tac_osm" / "plm_full_benchmark.py"
    )
    integrity["generator_matches_registered"] = (
        integrity["generator_sha256"] == GENERATOR_SHA256
    )

    def mean(arm: str, key: str) -> float:
        return statistics.fmean(row[key] for row in arm_results[arm])

    trained_full_accuracy = mean("trained_full", "accuracy")
    untrained_accuracy = mean("untrained", "accuracy")
    physics_off_accuracy = mean("trained_physics_off", "accuracy")

    full_loss_reduction = statistics.fmean(
        row["loss_reduction_fraction"]
        for row in arm_results["trained_full"]
    )
    physics_off_loss_reduction = statistics.fmean(
        row["loss_reduction_fraction"]
        for row in arm_results["trained_physics_off"]
    )

    diagnostic = {
        "trained_minus_untrained_action_accuracy":
            trained_full_accuracy - untrained_accuracy,
        "trained_parameter_change_all_seeds": all(
            row["parameter_changed"]
            for row in arm_results["trained_full"]
        ),
        "trained_full_mean_loss_reduction": full_loss_reduction,
        "trained_physics_off_mean_loss_reduction":
            physics_off_loss_reduction,
        "physics_off_minus_full_action_accuracy":
            physics_off_accuracy - trained_full_accuracy,
        "training_signal_criterion": (
            full_loss_reduction >= LOSS_REDUCTION_THRESHOLD
            and all(row["parameter_changed"] for row in arm_results["trained_full"])
        ),
        "above_untrained_floor_criterion": (
            trained_full_accuracy - untrained_accuracy
            >= ACCURACY_DELTA_THRESHOLD
        ),
        "physics_prior_material_constraint_criterion": (
            physics_off_accuracy - trained_full_accuracy
            >= ACCURACY_DELTA_THRESHOLD
        ),
    }

    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "science_commit": os.environ.get("GITHUB_SHA", "local"),
        "protocol": {
            "seeds": list(SEEDS),
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "physics_weights": {
                "trained_full": FULL_PHYSICS_WEIGHT,
                "trained_physics_off": 0.0,
            },
            "loss_reduction_threshold": LOSS_REDUCTION_THRESHOLD,
            "accuracy_delta_threshold": ACCURACY_DELTA_THRESHOLD,
        },
        "model": {
            "parameter_count": full_plm_parameter_count(build_model(SEEDS[0])),
            "forbidden_scaffolds": [
                "entity identifiers",
                "fixed operator table",
                "gold action input",
                "physical-state supervision",
                "hard-coded retrieval",
                "binary success input to verifier",
            ],
        },
        "benchmark_generator_sha256": GENERATOR_SHA256,
        "integrity": integrity,
        "train_info": train_info,
        "arms": arm_results,
        "diagnostic": diagnostic,
    }

    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(diagnostic, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
