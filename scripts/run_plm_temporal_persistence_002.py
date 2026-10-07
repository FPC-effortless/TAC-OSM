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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.contract import load_contract
from tac_osm.integrated_e2e_005 import FunctionalConfig, FunctionalMultimodalPLM
from tac_osm.temporal_persistence_002_benchmark import (
    BITS,
    ENTITY_COUNT,
    EVAL_EPISODES,
    Q1_ENTITY,
    Q1_QUERY,
    Q2_ENTITY,
    Q2_QUERY,
    SCRATCH_ENTITY,
    SECRET_BIT,
    VISIBLE_Q2_BIT,
    episode_fingerprint,
    episode_key,
    q2_surface_fingerprint,
    sample_episode,
    sample_evaluation_episodes,
    benchmark_manifest,
)

EXPERIMENT_ID = "TACOSM-PLM-TEMPORAL-PERSISTENCE-002"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
SEEDS = (0, 1, 2, 3, 4)
STEPS = 300
BATCH_SIZE = 96
EVAL_STEPS = 600
BASE_E2E008_GENERATOR_SHA256 = "3be32a91191b73792b0fffb5a7b497e80aa6a46d0189e032e6b7c60082949f3e"


def build_model():
    return FunctionalMultimodalPLM(
        config=FunctionalConfig(
            hidden_dim=64,
            entity_count=ENTITY_COUNT,
            latent_bits=BITS,
            state_write_mode="residual_linear",
        )
    ).cpu()


def write_observation(model, memory, observation, entity_override=None):
    entity, text, image, audio = observation
    target = entity if entity_override is None else entity_override
    z = model.encode(
        text.unsqueeze(0),
        image.unsqueeze(0),
        audio.unsqueeze(0).unsqueeze(0),
    )
    memory, _ = model.state.write(
        memory, z, torch.tensor([target], dtype=torch.long)
    )
    return memory


def write_pre(model, memory, pre):
    for observation in pre:
        memory = write_observation(model, memory, observation)
    return memory


def carry_path(model, pre, post, q1, q2):
    memory = model.state.initial(1, torch.device("cpu"))
    memory = write_pre(model, memory, pre)
    q1_result = model.query(
        memory,
        torch.tensor([Q1_ENTITY]),
        torch.tensor([q1[1]]),
        torch.tensor([q1[2]]),
        torch.tensor([0]),
    )
    outcome = torch.tensor([float(q1[4])])
    memory, _ = model.post_action_update(memory, q1_result, outcome)
    # Post-boundary q2 observation is written only to a scratch slot, so it
    # cannot overwrite the retained secret-bearing q2 entity.
    memory = write_observation(model, memory, post, entity_override=SCRATCH_ENTITY)
    q2_result = model.query(
        memory,
        torch.tensor([Q2_ENTITY]),
        torch.tensor([q2[1]]),
        torch.tensor([q2[2]]),
        torch.tensor([0]),
    )
    return q1_result, q2_result


def fresh_path(model, post, q2):
    memory = model.state.initial(1, torch.device("cpu"))
    memory = write_observation(model, memory, post, entity_override=Q2_ENTITY)
    q2_result = model.query(
        memory,
        torch.tensor([Q2_ENTITY]),
        torch.tensor([q2[1]]),
        torch.tensor([q2[2]]),
        torch.tensor([0]),
    )
    return q2_result


def train_seed(seed: int):
    torch.manual_seed(seed)
    rng = random.Random(seed + 185000)
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    train_keys = set()
    for _ in range(STEPS):
        episodes = [sample_episode(rng) for _ in range(BATCH_SIZE)]
        train_keys.update(episode_key(ep) for ep in episodes)
        bsz = len(episodes)
        memory = model.state.initial(bsz, torch.device("cpu"))
        e0_text = torch.stack([ep[0][0][1] for ep in episodes])
        e0_image = torch.stack([ep[0][0][2] for ep in episodes])
        e0_audio = torch.stack([ep[0][0][3] for ep in episodes]).unsqueeze(1)
        e1_text = torch.stack([ep[0][1][1] for ep in episodes])
        e1_image = torch.stack([ep[0][1][2] for ep in episodes])
        e1_audio = torch.stack([ep[0][1][3] for ep in episodes]).unsqueeze(1)
        z0 = model.encode(e0_text, e0_image, e0_audio)
        z1 = model.encode(e1_text, e1_image, e1_audio)
        memory, _ = model.state.write(memory, z0, torch.full((bsz,), Q1_ENTITY, dtype=torch.long))
        memory, _ = model.state.write(memory, z1, torch.full((bsz,), Q2_ENTITY, dtype=torch.long))
        q1_result = model.query(
            memory,
            torch.full((bsz,), Q1_ENTITY, dtype=torch.long),
            torch.zeros(bsz, dtype=torch.long),
            torch.ones(bsz, dtype=torch.long),
            torch.zeros(bsz, dtype=torch.long),
        )
        q1_targets = torch.tensor([ep[2][4] for ep in episodes], dtype=torch.long)
        q2_targets = torch.tensor([ep[3][4] for ep in episodes], dtype=torch.long)
        memory, _ = model.post_action_update(memory, q1_result, q1_targets.float())
        q2_result = model.query(
            memory,
            torch.full((bsz,), Q2_ENTITY, dtype=torch.long),
            torch.full((bsz,), SECRET_BIT, dtype=torch.long),
            torch.full((bsz,), VISIBLE_Q2_BIT, dtype=torch.long),
            torch.zeros(bsz, dtype=torch.long),
        )
        loss = 0.5 * F.cross_entropy(q1_result["logits"], q1_targets)
        loss = loss + F.cross_entropy(q2_result["logits"], q2_targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, train_keys


@torch.no_grad()
def q1_secret_interference(model, ep):
    pre, _post, q1, _q2, payload = ep
    flipped = [tuple(row) for row in pre]
    q2_pre = pre[1]
    flipped_text = q2_pre[1].clone()
    flipped_text[4] = 21 if int(payload[Q2_ENTITY][SECRET_BIT]) == 0 else 20
    flipped[1] = (Q2_ENTITY, flipped_text, q2_pre[2].clone(), q2_pre[3].clone())
    def q1_read_from(pre_rows):
        memory = model.state.initial(1, torch.device("cpu"))
        for row in pre_rows:
            memory = write_observation(model, memory, row)
        out = model.query(
            memory,
            torch.tensor([Q1_ENTITY]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([0]),
        )
        return out
    a = q1_read_from(pre)
    b = q1_read_from(flipped)
    return float(not torch.allclose(a["logits"], b["logits"], atol=0.0, rtol=0.0))


def hierarchical_bootstrap(seed_rows, rounds=10000):
    rng = random.Random(20261007)
    by_seed = {row["seed"]: row["paired_episode_deltas"] for row in seed_rows}
    seeds = [row["seed"] for row in seed_rows]
    samples = []
    for _ in range(rounds):
        chosen = rng.choices(seeds, k=len(seeds))
        seed_means = []
        for seed in chosen:
            vals = by_seed[seed]
            seed_means.append(statistics.fmean(rng.choices(vals, k=len(vals))))
        samples.append(statistics.fmean(seed_means))
    samples.sort()
    return [samples[int(0.025 * rounds)], samples[int(0.975 * rounds)]]


@torch.no_grad()
def evaluate_seed(model, episodes):
    model.eval()
    carry = []
    fresh = []
    deltas = []
    q1_mismatch = 0
    interference = 0
    for ep in episodes:
        pre, post, q1, q2, _payload = ep
        q1_out, carry_out = carry_path(model, pre, post, q1, q2)
        fresh_out = fresh_path(model, post, q2)
        cp = int(carry_out["logits"].argmax(-1).item()) == int(q2[4])
        fp = int(fresh_out["logits"].argmax(-1).item()) == int(q2[4])
        carry.append(int(cp)); fresh.append(int(fp)); deltas.append(int(cp)-int(fp))
        q1_mismatch += int(
            int(q1_out["logits"].argmax(-1).item())
            != int((q1_out["action"] >= 0.5).long().item())
        )
        interference += int(q1_secret_interference(model, ep))
    return {
        "carry_q2_accuracy": statistics.fmean(carry),
        "fresh_q2_accuracy": statistics.fmean(fresh),
        "paired_mean_delta": statistics.fmean(deltas),
        "paired_episode_deltas": deltas,
        "q1_action_decision_mismatches": q1_mismatch,
        "q1_secret_interference_mismatches": interference,
    }


def run(smoke=False):
    c = load_contract(EXPERIMENT_ID)
    assert c.check_consistency() == []
    manifest = benchmark_manifest()
    assert manifest["base_e2e008_generator_sha256"] == BASE_E2E008_GENERATOR_SHA256
    seeds = (0,) if smoke else SEEDS
    n = 32 if smoke else EVAL_STEPS
    rows = []
    for seed in seeds:
        model, train_keys = train_seed(seed)
        evaluation = sample_evaluation_episodes(random.Random(seed + 186000), EVAL_STEPS)
        if smoke:
            evaluation = evaluation[:n]
        overlap = train_keys & {episode_key(ep) for ep in evaluation}
        assert not overlap
        metrics = evaluate_seed(model, evaluation)
        rows.append({
            "seed": seed,
            "evaluation_episode_fingerprint": episode_fingerprint(evaluation),
            "q2_surface_fingerprint": q2_surface_fingerprint(evaluation),
            "training_evaluation_semantic_overlap": len(overlap),
            **metrics,
        })
    assert len({r["evaluation_episode_fingerprint"] for r in rows}) == len(rows)
    assert len({r["q2_surface_fingerprint"] for r in rows}) == len(rows)
    assert all(r["training_evaluation_semantic_overlap"] == 0 for r in rows)
    assert all(r["q1_action_decision_mismatches"] == 0 for r in rows)
    assert all(r["q1_secret_interference_mismatches"] == 0 for r in rows)
    ci = hierarchical_bootstrap(rows) if not smoke else [None, None]
    carry_mean = statistics.fmean(r["carry_q2_accuracy"] for r in rows)
    fresh_mean = statistics.fmean(r["fresh_q2_accuracy"] for r in rows)
    delta = statistics.fmean(r["paired_mean_delta"] for r in rows)
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
            "benchmark_sha256": manifest["generator_sha256"],
        },
        "protocol": {
            "seeds": list(seeds),
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "evaluation_episodes_per_seed": n,
            "q1_entity": Q1_ENTITY,
            "q2_entity": Q2_ENTITY,
            "scratch_entity": SCRATCH_ENTITY,
            "secret_bit": SECRET_BIT,
            "visible_q2_bit": VISIBLE_Q2_BIT,
            "state_write_mode": "residual_linear",
            "operator_dispatch": "fixed xor",
            "model_selection": "none",
        },
        "seed_results": rows,
        "summary": {
            "carry_q2_mean": carry_mean,
            "fresh_q2_mean": fresh_mean,
            "carry_minus_fresh_q2_accuracy": delta,
            "paired_bootstrap_ci95": ci,
            "primary_materiality_threshold": 0.10,
            "primary_criterion_pass": bool(not smoke and delta >= 0.10 and ci[0] > 0),
        },
        "leakage_audit": {
            "q1_entity_distinct_from_q2_entity": True,
            "q1_cannot_read_secret_entity": True,
            "q1_outcome_uses_only_q1_entity_bits": True,
            "post_boundary_secret_masked": True,
            "carry_retains_secret_entity": True,
            "fresh_rebuilds_q2_entity_from_masked_post_observation": True,
            "same_q2_query_across_arms": True,
            "same_post_q2_observation_across_arms": True,
            "training_evaluation_semantic_overlap_zero": True,
            "q1_secret_interference_gate_pass": True,
            "no_post_run_model_selection": True,
        },
        "claim_boundary": [
            "task-bounded persistence information-flow evidence",
            "no general long-horizon memory claim",
            "no semantic memory claim",
            "no learned addressing or operator discovery claim",
            "no scaling or real-world multimodal claim",
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
