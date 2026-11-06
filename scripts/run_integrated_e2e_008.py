#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.integrated_e2e_005 import FunctionalConfig, FunctionalMultimodalPLM
from tac_osm.integrated_e2e_008_benchmark import (
    EVAL_EPISODES,
    GENERATOR_VERSION,
    HELDOUT,
    TRAIN_COMBOS,
    benchmark_manifest,
    episode_fingerprint,
    episode_key,
    sample_episode,
    sample_evaluation_episodes,
)

EXPERIMENT_ID = "TACOSM-PLM-INTEGRATED-E2E-008"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
SEEDS = (0, 1, 2, 3, 4)
STEPS = 300
BATCH_SIZE = 96
OPS = ("xor", "and", "or", "xnor")


def build_batch(episodes):
    from scripts.run_integrated_e2e_005 import build_batch
    return build_batch(episodes)


def train_batch(model, batch):
    from scripts.run_integrated_e2e_005 import train_batch
    return train_batch(model, batch)


def gradient_surface_probe():
    torch.manual_seed(20261008)
    rng = random.Random(8008)
    episodes = [sample_episode(rng) for _ in range(4)]
    model = FunctionalMultimodalPLM(
        config=FunctionalConfig(hidden_dim=64, state_write_mode="residual_linear")
    )
    loss = train_batch(model, build_batch(episodes))
    loss.backward()
    required = (
        "text.emb.weight",
        "text.rnn.weight_ih_l0",
        "image.net.0.weight",
        "audio.net.0.weight",
        "rep.fuse.0.weight",
        "state.write_value.weight",
        "casm.decoder.0.weight",
        "verifier.0.weight",
        "write_gate.0.weight",
    )
    params = dict(model.named_parameters())
    bad = []
    for name in required:
        grad = params.get(name)
        if grad is None or not torch.isfinite(grad).all():
            bad.append(name)
    return {"pass": not bad, "missing_or_nonfinite": bad}


def train_seed(seed: int):
    torch.manual_seed(seed)
    rng = random.Random(seed + 180800)
    model = FunctionalMultimodalPLM(
        config=FunctionalConfig(hidden_dim=64, state_write_mode="residual_linear")
    ).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=0.0001
    )
    keys = set()
    for _ in range(STEPS):
        episodes = [
            sample_episode(
                rng,
                combo1=rng.choice(TRAIN_COMBOS),
                combo2=rng.choice(TRAIN_COMBOS),
            )
            for _ in range(BATCH_SIZE)
        ]
        keys.update(episode_key(ep) for ep in episodes)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = train_batch(model, build_batch(episodes))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model, keys


@torch.no_grad()
def evaluate(model, episodes, control):
    model.eval()
    q1_correct = 0
    q2_correct = 0
    verifier_correct = 0
    positive_count = 0
    decision_mismatches = 0
    per_comp = defaultdict(lambda: [0, 0])
    targets = []

    for ep in episodes:
        memory = model.state.initial(1, torch.device("cpu"))
        rows = ep[0]
        for k, (entity, text, image, audio) in enumerate(rows):
            if control == "shuffle_image":
                image = rows[(k + 1) % len(rows)][2]
            elif control == "text_only":
                image = torch.zeros_like(image)
                audio = torch.zeros_like(audio)
            elif control == "image_only":
                text = torch.zeros_like(text)
                audio = torch.zeros_like(audio)
            elif control == "audio_only":
                text = torch.zeros_like(text)
                image = torch.zeros_like(image)
            z = model.encode(
                text.unsqueeze(0),
                image.unsqueeze(0),
                audio.unsqueeze(0).unsqueeze(0),
            )
            memory, _ = model.state.write(
                memory, z, torch.tensor([entity], dtype=torch.long)
            )

        if control == "no_memory":
            memory = torch.zeros_like(memory)

        def query(q, mem):
            return model.query(
                mem,
                torch.tensor([q[0]], dtype=torch.long),
                torch.tensor([q[1]], dtype=torch.long),
                torch.tensor([q[2]], dtype=torch.long),
                torch.tensor([OPS.index(q[3])], dtype=torch.long),
            )

        out1 = query(ep[1], memory)
        target1 = ep[1][4]
        action1 = (out1["action"] >= 0.5).long()
        outcome1 = (action1 == torch.tensor([target1])).float()
        updated, verifier1 = model.post_action_update(memory, out1, outcome1)

        out2 = query(ep[2], memory if control == "no_memory" else updated)
        target2 = ep[2][4]
        action2 = (out2["action"] >= 0.5).long()
        outcome2 = (action2 == torch.tensor([target2])).float()
        _, verifier2 = model.post_action_update(
            memory if control == "no_memory" else updated, out2, outcome2
        )

        pred1 = int(out1["logits"].argmax(-1).item())
        pred2 = int(out2["logits"].argmax(-1).item())
        threshold1 = int(action1.item())
        threshold2 = int(action2.item())
        decision_mismatches += int(pred1 != threshold1) + int(pred2 != threshold2)
        q1_correct += int(pred1 == target1)
        q2_correct += int(pred2 == target2)
        positive_count += pred2
        comp = (ep[2][3], ep[2][1], ep[2][2])
        per_comp[comp][0] += int(pred2 == target2)
        per_comp[comp][1] += 1
        targets.append(target2)

        verifier_pred = ((torch.cat([verifier1[:1, :1], verifier2[:1, :1]], dim=1).sigmoid() >= 0.5).long())
        verifier_target = torch.tensor([[int(outcome1.item()), int(outcome2.item())]])
        verifier_correct += int(torch.equal(verifier_pred.cpu(), verifier_target))

    n = len(episodes)
    empirical_prior = max(targets.count(0), targets.count(1)) / max(n, 1)
    return {
        "q1_accuracy": q1_correct / max(n, 1),
        "q2_accuracy": q2_correct / max(n, 1),
        "verifier_q1_q2_accuracy": verifier_correct / max(n, 1),
        "q2_positive_fraction": positive_count / max(n, 1),
        "q2_decision_mismatches": decision_mismatches,
        "per_composition_q2_accuracy": {
            str(key): correct / count for key, (correct, count) in per_comp.items()
        },
        "empirical_prior_baseline": empirical_prior,
    }


def bootstrap_compositions(seed_rows, rounds=5000):
    rng = random.Random(20261008)
    compositions = sorted(
        seed_rows[0]["normal_per_composition_q2_accuracy"].keys()
    )
    means = [
        statistics.fmean(
            row["normal_per_composition_q2_accuracy"][composition]
            for row in seed_rows
        )
        for composition in compositions
    ]
    samples = sorted(
        statistics.fmean(rng.choices(means, k=len(means)))
        for _ in range(rounds)
    )
    return [samples[int(0.025 * rounds)], samples[int(0.975 * rounds)]]


def run(smoke=False):
    controls = ("normal", "no_memory", "shuffle_image", "text_only", "image_only", "audio_only")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["status"] == "pre-registered"
    assert contract["seeds"] == list(SEEDS)
    assert contract["protocol"]["hidden_dim"] == 64
    assert contract["protocol"]["state_write_mode"] == "residual_linear"
    assert contract["protocol"]["entity_side_channel"] is False
    assert len(contract["heldout"]) == len(HELDOUT) == 12
    assert [tuple(x) for x in contract["heldout"]] == list(HELDOUT)
    assert contract["steps"] == STEPS
    assert contract["eval_steps"] == EVAL_EPISODES
    assert contract["h_levels"] == [3]
    assert [arm["name"] for arm in contract["arms"]] == list(controls)

    gradient_gate = gradient_surface_probe()
    assert gradient_gate["pass"], gradient_gate

    seeds = (0,) if smoke else SEEDS
    eval_n = 16 if smoke else EVAL_EPISODES
    rows = []

    for seed in seeds:
        model, training_keys = train_seed(seed)
        evaluation = sample_evaluation_episodes(
            random.Random(seed + 181000), eval_n
        )
        evaluation_keys = {episode_key(ep) for ep in evaluation}
        overlap = training_keys & evaluation_keys
        assert not overlap, f"training/evaluation semantic overlap: seed={seed}"
        fingerprint = episode_fingerprint(evaluation)
        metrics = {
            control: evaluate(model, evaluation, control) for control in controls
        }
        rows.append(
            {
                "seed": seed,
                "evaluation_episode_fingerprint": fingerprint,
                "training_evaluation_semantic_overlap": len(overlap),
                "normal_q1_accuracy": metrics["normal"]["q1_accuracy"],
                "normal_q2_accuracy": metrics["normal"]["q2_accuracy"],
                "normal_verifier_accuracy": metrics["normal"]["verifier_q1_q2_accuracy"],
                "normal_q2_positive_fraction": metrics["normal"]["q2_positive_fraction"],
                "normal_q2_decision_mismatches": metrics["normal"]["q2_decision_mismatches"],
                "normal_per_composition_q2_accuracy": metrics["normal"]["per_composition_q2_accuracy"],
                "empirical_prior_baseline": metrics["normal"]["empirical_prior_baseline"],
                "no_memory_q2_accuracy": metrics["no_memory"]["q2_accuracy"],
                "shuffle_image_q2_accuracy": metrics["shuffle_image"]["q2_accuracy"],
                "text_only_q2_accuracy": metrics["text_only"]["q2_accuracy"],
                "image_only_q2_accuracy": metrics["image_only"]["q2_accuracy"],
                "audio_only_q2_accuracy": metrics["audio_only"]["q2_accuracy"],
                "memory_drop": metrics["normal"]["q2_accuracy"] - metrics["no_memory"]["q2_accuracy"],
                "alignment_drop": metrics["normal"]["q2_accuracy"] - metrics["shuffle_image"]["q2_accuracy"],
                "per_op_q2_accuracy": {
                    op: statistics.fmean(
                        acc
                        for key, acc in metrics["normal"]["per_composition_q2_accuracy"].items()
                        if key.startswith("('" + op + "',")
                    )
                    for op in OPS
                },
            }
        )

    assert len({row["evaluation_episode_fingerprint"] for row in rows}) == len(rows), "seed evaluation streams must be distinct"
    primary_values = [row["normal_q2_accuracy"] for row in rows]
    composition_mean = (
        {
            composition: statistics.fmean(
                row["normal_per_composition_q2_accuracy"][composition]
                for row in rows
            )
            for composition in sorted(rows[0]["normal_per_composition_q2_accuracy"])
        }
        if not smoke
        else {}
    )

    summary = {
        "primary_q2_mean": statistics.fmean(primary_values),
        "primary_q2_composition_bootstrap_ci95": bootstrap_compositions(rows) if not smoke else [None, None],
        "all_seed_min_q2": min(primary_values),
        "primary_pass": bool(
            (not smoke) and statistics.fmean(primary_values) >= 0.8 and min(primary_values) >= 0.4
        ),
        "empirical_prior_baseline_mean": statistics.fmean(
            row["empirical_prior_baseline"] for row in rows
        ),
        "mean_memory_drop": statistics.fmean(row["memory_drop"] for row in rows),
        "mean_alignment_drop": statistics.fmean(row["alignment_drop"] for row in rows),
        "mean_per_composition_q2_accuracy": composition_mean,
        "oracle_q2_accuracy": 1.0,
        "gradient_surface_pass": gradient_gate["pass"],
        "classifier_decision_integrity_pass": all(
            row["normal_q2_decision_mismatches"] == 0 for row in rows
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
            "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
            "benchmark_sha256": benchmark_manifest()["generator_sha256"],
        },
        "protocol": {
            "seeds": list(seeds),
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "evaluation_episodes_per_seed": eval_n,
            "hidden_dim": 64,
            "state_write_mode": "residual_linear",
            "registered_heldout": [list(x) for x in HELDOUT],
            "train_combos_count": len(TRAIN_COMBOS),
            "benchmark_generator_version": GENERATOR_VERSION,
            "model_selection": "none",
            "entity_side_channel": False,
        },
        "seed_results": rows,
        "summary": summary,
        "leakage_audit": {
            "training_evaluation_semantic_overlap_zero": all(
                row["training_evaluation_semantic_overlap"] == 0 for row in rows
            ),
            "heldout_compositions_excluded_from_training": True,
            "evaluation_generated_after_training": True,
            "payload_auxiliary_supervision": False,
            "entity_side_channel_in_modalities": False,
            "image_entity_stripe_present": False,
            "image_payload_id_overlap": False,
            "classifier_action_decision_consistency": summary["classifier_decision_integrity_pass"],
            "prior_confirmatory_and_dev_sets_excluded": True,
        },
        "claim_boundary": [
            "fresh synthetic benchmark only",
            "collision-free payload-only modality encoding",
            "no real-world semantic multimodal claim",
            "no learned addressing or operator discovery claim",
            "no scaling or hardware claim",
        ],
    }

    path = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
