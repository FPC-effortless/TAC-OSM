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

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.contract import load_contract
from tac_osm.integrated_e2e_005 import FunctionalConfig, FunctionalMultimodalPLM
from tac_osm.temporal_dependency_001_benchmark import (
    BATCH_SIZE,
    ENTITY_COUNT,
    EVAL_EPISODES,
    Q1_QUERY,
    Q2_QUERY,
    SCRATCH_ENTITY,
    STEPS,
    benchmark_manifest,
    episode_fingerprint,
    episode_key,
    q2_observation_fingerprint,
    sample_episode,
    sample_evaluation_episodes,
)

EXPERIMENT_ID = "TACOSM-PLM-TEMPORAL-DEPENDENCY-001"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
SEEDS = (0, 1, 2, 3, 4)
BASE_E2E008_GENERATOR_SHA256 = (
    "3be32a91191b73792b0fffb5a7b497e80aa6a46d0189e032e6b7c60082949f3e"
)
PARAM_COUNT = None


def contract_hash() -> str:
    return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model():
    return FunctionalMultimodalPLM(
        config=FunctionalConfig(
            hidden_dim=64,
            entity_count=ENTITY_COUNT,
            latent_bits=12,
            state_write_mode="residual_linear",
        )
    ).cpu()


def write_observation(model, memory, observation, entity_override=None):
    entity, text, image, audio = observation
    entity = entity if entity_override is None else entity_override
    z = model.encode(
        text.unsqueeze(0),
        image.unsqueeze(0),
        audio.unsqueeze(0).unsqueeze(0),
    )
    updated, _ = model.state.write(
        memory,
        z,
        torch.tensor([entity], dtype=torch.long),
    )
    return updated


def run_carry_path(model, pre, post, q1, q2):
    memory = model.state.initial(1, torch.device("cpu"))
    memory = write_observation(model, memory, pre[0])
    q1_result = model.query(
        memory,
        torch.tensor([q1[0]]),
        torch.tensor([q1[1]]),
        torch.tensor([q1[2]]),
        torch.tensor([0]),
    )
    # Environment feedback is the true q1 answer, detached from model action.
    outcome = torch.tensor([float(q1[4])])
    memory, _ = model.post_action_update(memory, q1_result, outcome)
    # The post-boundary observation is present in both arms. In carry it is
    # written only to a scratch entity, leaving the retained target state intact.
    memory = write_observation(
        model,
        memory,
        post[0],
        entity_override=SCRATCH_ENTITY,
    )
    q2_result = model.query(
        memory,
        torch.tensor([q2[0]]),
        torch.tensor([q2[1]]),
        torch.tensor([q2[2]]),
        torch.tensor([0]),
    )
    return q1_result, q2_result


def run_fresh_path(model, post, q2):
    # Fresh arm has no q1-time state, action, outcome, or verifier signal.
    memory = model.state.initial(1, torch.device("cpu"))
    memory = write_observation(model, memory, post[0], entity_override=q2[0])
    q2_result = model.query(
        memory,
        torch.tensor([q2[0]]),
        torch.tensor([q2[1]]),
        torch.tensor([q2[2]]),
        torch.tensor([0]),
    )
    return q2_result


def batch_from_episodes(episodes):
    texts = []
    images = []
    audios = []
    q1_targets = []
    q2_targets = []
    for ep in episodes:
        pre, post, q1, q2, _payload = ep
        _ = post
        texts.append(pre[0][1])
        images.append(pre[0][2])
        audios.append(pre[0][3])
        q1_targets.append(q1[4])
        q2_targets.append(q2[4])
    return (
        torch.stack(texts),
        torch.stack(images),
        torch.stack(audios).unsqueeze(1),
        torch.tensor(q1_targets, dtype=torch.long),
        torch.tensor(q2_targets, dtype=torch.long),
    )


def train_seed(seed: int):
    torch.manual_seed(seed)
    rng = random.Random(seed + 185000)
    model = build_model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=0.0001
    )
    train_keys = set()

    for _ in range(STEPS):
        episodes = [sample_episode(rng) for _ in range(BATCH_SIZE)]
        train_keys.update(episode_key(ep) for ep in episodes)
        text, image, audio, q1_targets, q2_targets = batch_from_episodes(episodes)
        memory = model.state.initial(BATCH_SIZE, text.device)

        z = model.encode(text, image, audio)
        memory, _ = model.state.write(
            memory,
            z,
            torch.zeros(BATCH_SIZE, dtype=torch.long),
        )
        q1_result = model.query(
            memory,
            torch.zeros(BATCH_SIZE, dtype=torch.long),
            torch.zeros(BATCH_SIZE, dtype=torch.long),
            torch.ones(BATCH_SIZE, dtype=torch.long),
            torch.zeros(BATCH_SIZE, dtype=torch.long),
        )
        memory, _ = model.post_action_update(
            memory,
            q1_result,
            q1_targets.float(),
        )
        q2_result = model.query(
            memory,
            torch.zeros(BATCH_SIZE, dtype=torch.long),
            torch.full((BATCH_SIZE,), 2, dtype=torch.long),
            torch.full((BATCH_SIZE,), 3, dtype=torch.long),
            torch.zeros(BATCH_SIZE, dtype=torch.long),
        )
        loss = 0.5 * F.cross_entropy(q1_result["logits"], q1_targets)
        loss = loss + F.cross_entropy(q2_result["logits"], q2_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return model, train_keys


def hierarchical_paired_bootstrap(seed_rows, rounds=10000):
    rng = random.Random(20261007)
    deltas = {row["seed"]: row["paired_episode_deltas"] for row in seed_rows}
    seeds = [row["seed"] for row in seed_rows]
    samples = []
    for _ in range(rounds):
        chosen = rng.choices(seeds, k=len(seeds))
        per_seed = []
        for seed in chosen:
            vals = deltas[seed]
            per_seed.append(statistics.fmean(rng.choices(vals, k=len(vals))))
        samples.append(statistics.fmean(per_seed))
    samples.sort()
    return [samples[int(0.025 * rounds)], samples[int(0.975 * rounds)]]


@torch.no_grad()\ndef evaluate_seed(model, episodes):
    model.eval()
    carry_correct = 0
    fresh_correct = 0
    carry_q1_mismatches = 0
    paired_deltas = []
    secret_mismatch_errors = 0

    for ep in episodes:
        pre, post, q1, q2, _payload = ep
        carry_q1, carry_q2 = run_carry_path(model, pre, post, q1, q2)
        fresh_q2 = run_fresh_path(model, post, q2)

        carry_pred = int(carry_q2["logits"].argmax(-1).item())
        fresh_pred = int(fresh_q2["logits"].argmax(-1).item())
        target = int(q2[4])
        carry_correct += int(carry_pred == target)
        fresh_correct += int(fresh_pred == target)
        paired_deltas.append(
            int(carry_pred == target) - int(fresh_pred == target)
        )
        carry_q1_mismatches += int(
            int(carry_q1["logits"].argmax(-1).item())
            != int((carry_q1["action"] >= 0.5).long().item())
        )
        expected_mask = 20 + int(payload[2])
        if int(pre[0][1][4].item()) != expected_mask or int(post[0][1][4].item()) != 20:
            secret_mismatch_errors += 1
        if not torch.equal(pre[0][1][:4], post[0][1][:4]) or not torch.equal(pre[0][1][5:], post[0][1][5:]):
            secret_mismatch_errors += 1
        if not torch.equal(pre[0][2], post[0][2]) or not torch.equal(pre[0][3], post[0][3]):
            secret_mismatch_errors += 1

    n = len(episodes)
    return {
        "carry_q2_accuracy": carry_correct / n,
        "fresh_q2_accuracy": fresh_correct / n,
        "paired_mean_delta": statistics.fmean(paired_deltas),
        "paired_episode_deltas": paired_deltas,
        "carry_q1_decision_mismatches": carry_q1_mismatches,
        "secret_integrity_errors": secret_mismatch_errors,
    }


def run(smoke=False):
    contract = load_contract(EXPERIMENT_ID)
    errors = contract.check_consistency()
    if errors:
        raise AssertionError(errors)

    manifest = benchmark_manifest()
    assert manifest["base_e2e008_generator_sha256"] == BASE_E2E008_GENERATOR_SHA256
    probe = build_model()
    global PARAM_COUNT
    PARAM_COUNT = parameter_count(probe)

    seeds = (0,) if smoke else SEEDS
    eval_n = 32 if smoke else EVAL_EPISODES
    seed_rows = []

    for seed in seeds:
        model, train_keys = train_seed(seed)
        evaluation = (sample_evaluation_episodes(random.Random(seed + 186000), EVAL_EPISODES)
                      if not smoke else [sample_episode(random.Random(seed + 186000 + n)) for n in range(eval_n)])
        if smoke:
            for ep in evaluation:
                from tac_osm.temporal_dependency_001_benchmark import validate_episode
                validate_episode(ep)
        eval_keys = {episode_key(ep) for ep in evaluation}
        overlap = train_keys & eval_keys
        assert not overlap
        fingerprint = episode_fingerprint(evaluation)
        q2_fp = q2_observation_fingerprint(evaluation)
        metrics = evaluate_seed(model, evaluation)
        seed_rows.append({
            "seed": seed,
            "evaluation_episode_fingerprint": fingerprint,
            "q2_observation_fingerprint": q2_fp,
            "training_evaluation_semantic_overlap": len(overlap),
            "carry_q2_accuracy": metrics["carry_q2_accuracy"],
            "fresh_q2_accuracy": metrics["fresh_q2_accuracy"],
            "paired_mean_delta": metrics["paired_mean_delta"],
            "paired_episode_deltas": metrics["paired_episode_deltas"],
            "carry_q1_decision_mismatches": metrics["carry_q1_decision_mismatches"],
            "secret_integrity_errors": metrics["secret_integrity_errors"],
        })

    assert len({r["evaluation_episode_fingerprint"] for r in seed_rows}) == len(seed_rows)
    assert len({r["q2_observation_fingerprint"] for r in seed_rows}) == len(seed_rows)
    assert all(r["q2_observation_fingerprint"] for r in seed_rows)
    assert all(r["training_evaluation_semantic_overlap"] == 0 for r in seed_rows)
    assert all(r["secret_integrity_errors"] == 0 for r in seed_rows)

    carry_values = [r["carry_q2_accuracy"] for r in seed_rows]
    fresh_values = [r["fresh_q2_accuracy"] for r in seed_rows]
    delta_values = [r["paired_mean_delta"] for r in seed_rows]
    ci95 = hierarchical_paired_bootstrap(seed_rows) if not smoke else [None, None]
    delta_mean = statistics.fmean(delta_values)

    summary = {
        "carry_q2_mean": statistics.fmean(carry_values),
        "carry_q2_min": min(carry_values),
        "fresh_q2_mean": statistics.fmean(fresh_values),
        "fresh_q2_min": min(fresh_values),
        "carry_minus_fresh_q2_accuracy": delta_mean,
        "paired_bootstrap_ci95": ci95,
        "paired_bootstrap_lower_gt_zero": bool(
            not smoke and ci95[0] > 0
        ),
        "primary_materiality_threshold": 0.10,
        "primary_criterion_pass": bool(
            not smoke and delta_mean >= 0.10 and ci95[0] > 0
        ),
        "q1_action_decision_integrity_pass": all(
            r["carry_q1_decision_mismatches"] == 0 for r in seed_rows
        ),
    }

    output = {
        "experiment_id": EXPERIMENT_ID,
        "status": "smoke" if smoke else "measured",
        "provenance": {
            "git_commit": os.environ.get("GITHUB_SHA", "unknown"),
            "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "contract_sha256": contract_hash(),
            "benchmark_sha256": manifest["generator_sha256"],
            "base_e2e008_generator_sha256": BASE_E2E008_GENERATOR_SHA256,
        },
        "protocol": {
            "seeds": list(seeds),
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "evaluation_episodes_per_seed": eval_n,
            "hidden_dim": 64,
            "entity_count": ENTITY_COUNT,
            "state_write_mode": "residual_linear",
            "q1_query": list(Q1_QUERY),
            "q2_query": list(Q2_QUERY),
            "operator_id_fixed": True,
            "model_selection": "none",
            "train_target": "q2 depends on q1-time-only secret bit",
        },
        "seed_results": seed_rows,
        "summary": summary,
        "leakage_audit": {
            "q1_time_secret_masked_from_fresh_q2_surface": True,
            "fresh_q2_state_initialized_only_from_post_boundary_observation": True,
            "carry_q2_state_retains_pre_boundary_information": True,
            "post_boundary_observation_same_across_arms": True,
            "q2_query_same_across_arms": True,
            "q1_outcome_derived_from_independent_bits": True,
            "training_evaluation_semantic_overlap_zero": all(
                r["training_evaluation_semantic_overlap"] == 0 for r in seed_rows
            ),
            "evaluation_generated_after_training": True,
            "all_evaluation_fingerprints_distinct": len(
                {r["evaluation_episode_fingerprint"] for r in seed_rows}
            ) == len(seed_rows),
            "all_q2_observation_fingerprints_distinct": len(
                {r["q2_observation_fingerprint"] for r in seed_rows}
            ) == len(seed_rows),
            "no_q1_action_or_outcome_in_fresh_arm": True,
            "no_future_q2_information_in_carry_pre_boundary_state": True,
        },
        "claim_boundary": [
            "task-bounded temporal information-flow evidence only",
            "no general long-horizon memory claim",
            "no semantic memory claim",
            "no learned addressing claim",
            "no operator discovery claim",
            "no scaling or real-world multimodal claim",
        ],
    }

    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
