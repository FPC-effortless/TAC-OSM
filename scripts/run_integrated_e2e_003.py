#!/usr/bin/env python3
"""Per-arm runner for corrected TACOSM-PLM-INTEGRATED-E2E-003."""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.contract import load_contract
from tac_osm.integrated_e2e_address import AddressDiagnosisConfig, AddressDiagnosisModel
from tac_osm.integrated_e2e_benchmark import (
    BATCH_SIZE,
    EVAL_EPISODES,
    HELDOUT,
    BENCHMARK_GENERATOR_VERSION,
    SEEDS,
    STEPS,
    batchify,
    sample_episode,
    validate_episode,
)

ARMS = {
    "explicit_write": ("explicit", "learned"),
    "explicit_read": ("learned", "explicit"),
    "explicit_both": ("explicit", "explicit"),
}


def validate_contract(arm: str) -> None:
    contract = load_contract("TACOSM-PLM-INTEGRATED-E2E-003")
    contract.require_levels((3,))
    contract.require_seeds(SEEDS)
    contract.require_steps(STEPS)
    contract.require_eval_steps(EVAL_EPISODES)
    contract.require_arms(ARMS.keys())
    if arm not in ARMS or arm not in {a.name for a in contract.arms}:
        raise RuntimeError(f"unregistered arm: {arm}")
    if contract.check_consistency():
        raise RuntimeError("contract consistency failure")


def train_seed(seed: int, write_mode: str, read_mode: str) -> AddressDiagnosisModel:
    torch.manual_seed(seed)
    rng = random.Random(seed + 9001)
    model = AddressDiagnosisModel(
        AddressDiagnosisConfig(write_mode=write_mode, read_mode=read_mode)
    ).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=1e-4
    )
    for _ in range(STEPS):
        episodes = [sample_episode(rng) for _ in range(BATCH_SIZE)]
        batch = batchify(episodes)
        optimizer.zero_grad(set_to_none=True)
        out = model.forward_episode(
            {
                "text": torch.stack(batch["text"], 0),
                "image": torch.stack(batch["image"], 0),
                "audio": torch.stack(batch["audio"], 0),
            },
            torch.stack(batch["entities"], 0),
            torch.stack(batch["payloads"], 0),
            batch["q1"],
            batch["q2"],
        )
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


@torch.no_grad()
def evaluate(model: AddressDiagnosisModel, episodes: list[tuple], control: str = "normal") -> dict[str, float]:
    q1_correct = q2_correct = 0
    attention = 0.0
    write_correct = 0.0
    for ep in episodes:
        validate_episode(ep)
        batch = batchify([ep])
        memory = model.state.initial(1, torch.device("cpu"))
        for k in range(3):
            text = batch["text"][k]
            image = batch["image"][k]
            audio = batch["audio"][k]
            if control == "shuffle_image":
                image = batch["image"][(k + 1) % 3]
            elif control == "text_only":
                image = torch.zeros_like(image)
                audio = torch.zeros_like(audio)
            elif control == "image_only":
                text = torch.full_like(text, 1)
                audio = torch.zeros_like(audio)
            elif control == "audio_only":
                text = torch.full_like(text, 1)
                image = torch.zeros_like(image)
            z, _, _, _ = model.encode(text, image, audio)
            memory, probs = model.observation_write(memory, z, batch["entities"][k])
            write_correct += float(probs.argmax(-1).item() == batch["entities"][k].item())

        if control == "no_memory":
            memory.zero_()

        q1 = batch["q1"]
        q2 = batch["q2"]
        out1 = model.query(memory, *q1[:-1])
        q1_correct += int(out1["logits"].argmax(-1).item() == q1[-1].item())
        attention += float(out1["attention"][0, q1[0].item()].item())

        if control != "no_memory":
            memory, _ = model.post_action_update(memory, out1, q1[-1])

        out2 = model.query(memory, *q2[:-1])
        q2_correct += int(out2["logits"].argmax(-1).item() == q2[-1].item())

    n = len(episodes)
    return {
        "q1_accuracy": q1_correct / n,
        "q2_accuracy": q2_correct / n,
        "target_memory_attention": attention / n,
        "write_target_address_accuracy": write_correct / (n * 3),
    }


def episode_fingerprint(episodes: list[tuple]) -> str:
    import hashlib
    h = hashlib.sha256()
    for obs, q1, q2, payload in episodes:
        h.update(repr(tuple((int(e), tuple(int(x) for x in payload[e])) for e in sorted(payload))).encode())
        h.update(repr(q1).encode())
        h.update(repr(q2).encode())
    return h.hexdigest()


def seed_bootstrap(values: list[float], samples: int = 5000, seed: int = 20261002) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    )
    return [draws[int(0.025 * samples)], draws[int(0.975 * samples) - 1]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--output", default="artifacts/TACOSM-PLM-INTEGRATED-E2E-003.json")
    args = parser.parse_args()
    validate_contract(args.arm)

    models = [(seed, train_seed(seed, *ARMS[args.arm])) for seed in SEEDS]
    results = []
    rng_by_seed = {seed: random.Random(100000 + seed) for seed in SEEDS}
    for seed in SEEDS:
        rng = rng_by_seed[seed]
        episodes = []
        for _ in range(EVAL_EPISODES):
            combo1, combo2 = rng.sample(HELDOUT, 2)
            ep = sample_episode(rng, combo1=combo1, combo2=combo2)
            validate_episode(ep)
            episodes.append(ep)

        evaluation_fingerprint = episode_fingerprint(episodes)
        normal = evaluate(dict(models)[seed], episodes)
        row = {"seed": seed, args.arm: normal["q2_accuracy"], "evaluation_episode_fingerprint": evaluation_fingerprint}
        row[f"{args.arm}_q1_accuracy"] = normal["q1_accuracy"]
        row[f"{args.arm}_attention"] = normal["target_memory_attention"]
        row[f"{args.arm}_write_target_address_accuracy"] = normal["write_target_address_accuracy"]

        if args.arm == "explicit_both":
            for control in ("no_memory", "shuffle_image", "text_only", "image_only", "audio_only"):
                row[f"{control}_explicit_both_q2"] = evaluate(
                    dict(models)[seed], episodes, control
                )["q2_accuracy"]
        results.append(row)

    values = [row[args.arm] for row in results]
    output = {
        "experiment_id":"TACOSM-PLM-INTEGRATED-E2E-003",
        "status":"measured",
        "arm":args.arm,
        "provenance":{
            "git_commit":os.environ.get("GITHUB_SHA","unknown"),
            "workflow_run_id":os.environ.get("GITHUB_RUN_ID","unknown"),
            "python_version":platform.python_version(),
            "torch_version":torch.__version__,
            "platform":platform.platform()
        },
        "protocol":{
            "seeds":SEEDS,
            "steps":STEPS,
            "batch_size":BATCH_SIZE,
            "heldout_compositions":HELDOUT,
            "evaluation_episodes_per_seed":EVAL_EPISODES,
            "benchmark_generator_version":BENCHMARK_GENERATOR_VERSION,
            "registered_arms":["explicit_write","explicit_read","explicit_both"]
        },
        "leakage_audit":{
            "q1_entity_rule":"q1 target entity is entities[0]",
            "q2_entity_rule":"q2 target entity is entities[1]",
            "q1_q2_entities_distinct":True,
            "heldout_compositions_excluded_from_training":True,
            "train_eval_rng_streams_disjoint":True,
            "evaluation_generated_after_training":True,
            "controls_reuse_exact_same_episode_objects_for_explicit_both":True,
            "pre_action_payload_bits_available":False,
            "auxiliary_payload_supervision_is_training_only":True,
            "post_action_feedback_is_executed_action_correctness":True,
            "pre_action_query_fields":["entity","i","j","op"],
            "forbidden_pre_action_fields":["answer","environment_outcome","verifier_target"]
        },
        "seed_results":results,
        "summary":{
            "q2_accuracy_mean":statistics.fmean(values),
            "q2_accuracy_seed_bootstrap_ci95":seed_bootstrap(values),
            "min_seed_q2_accuracy":min(values)
        }
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print(json.dumps(output,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
