#!/usr/bin/env python3
"""Authoritative runner for TACOSM-PLM-INTEGRATED-E2E-002."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import platform
import random
import statistics
import sys

import torch
from torch import nn

ROOT = __file__
sys.path.insert(0, str(__import__("pathlib").Path(ROOT).resolve().parents[1]))

from tac_osm.integrated_e2e import OPS
from tac_osm.contract import load_contract
from tac_osm.integrated_e2e_address import AddressDiagnosisConfig, AddressDiagnosisModel
from scripts.run_integrated_e2e_001 import (
    BITS, HELDOUT, BATCH_SIZE, STEPS, batchify, sample_episode,
)


SEEDS = (0, 1, 2, 3, 4)
ARMS = {
    "explicit_write": ("explicit", "learned"),
    "explicit_read": ("learned", "explicit"),
    "explicit_both": ("explicit", "explicit"),
}


def train_seed(seed: int, write_mode: str, read_mode: str) -> AddressDiagnosisModel:
    torch.manual_seed(seed)
    rng = random.Random(seed + 9001)
    model = AddressDiagnosisModel(
        AddressDiagnosisConfig(
            write_mode=write_mode,
            read_mode=read_mode,
        )
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
            batch["q1"], batch["q2"],
        )
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


@torch.no_grad()
def evaluate(model: AddressDiagnosisModel, episodes: list[tuple], control: str = "normal") -> dict[str, float]:
    totals = 0
    attention_mass = 0.0
    write_correct = 0.0
    for ep in episodes:
        batch = batchify([ep])
        memory = model.state.initial(1, torch.device("cpu"))
        for k in range(3):
            image = batch["image"][k]
            if control == "shuffle_image":
                image = batch["image"][(k + 1) % 3]
            elif control == "text_only":
                image = torch.zeros_like(image)
                batch_text = batch["text"][k]
            else:
                batch_text = batch["text"][k]
            text = batch_text
            audio = batch["audio"][k]
            if control == "text_only":
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
            write_correct += float(
                (probs.argmax(-1).item() == batch["entities"][k].item())
            )
        if control == "no_memory":
            memory.zero_()
        q1 = batch["q1"]; q2 = batch["q2"]
        out1 = model.query(memory, *q1[:-1])
        if control == "no_memory":
            # no outcome update can restore the deleted state
            pass
        else:
            memory, _ = model.post_action_update(memory, out1, q1[-1])
        out2 = model.query(memory, *q2[:-1])
        totals += int(out2["logits"].argmax(-1).item() == q2[-1].item())
        attention_mass += float(out2["attention"][0, q2[0].item()].item())
    n = len(episodes)
    return {
        "q2_accuracy": totals / n,
        "target_memory_attention": attention_mass / n,
        "write_target_accuracy": write_correct / (n * 3),
    }


def bootstrap(values: list[float], samples: int = 5000, seed: int = 20261002) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    )
    return [
        means[int(0.025 * samples)],
        means[int(0.975 * samples) - 1],
    ]


def validate_contract() -> None:
    contract = load_contract("TACOSM-PLM-INTEGRATED-E2E-002")
    contract.require_levels((3,))
    contract.require_seeds(SEEDS)
    contract.require_steps(STEPS)
    contract.require_eval_steps(400)
    contract.require_arms(tuple(ARMS.keys()))
    problems = contract.check_consistency()
    if problems:
        raise RuntimeError(
            "E2E-002 contract consistency failure: " + " | ".join(problems)
        )


def main() -> None:
    validate_contract()
    seed_models = {
        arm: [(seed, train_seed(seed, *modes)) for seed in SEEDS]
        for arm, modes in ARMS.items()
    }
    rng_by_seed = {
        seed: random.Random(100000 + seed)
        for seed in SEEDS
    }
    results = []
    for seed in SEEDS:
        eval_episodes = []
        rng = rng_by_seed[seed]
        for _ in range(400):
            combo1, combo2 = rng.sample(HELDOUT, 2)
            eval_episodes.append(sample_episode(rng, combo1=combo1, combo2=combo2))
        row = {"seed": seed}
        for arm, models in seed_models.items():
            model = dict(models)[seed]
            normal = evaluate(model, eval_episodes, "normal")
            row[arm] = normal["q2_accuracy"]
            row[f"{arm}_attention"] = normal["target_memory_attention"]
            row[f"{arm}_write_accuracy"] = normal["write_target_accuracy"]
        # Controls only for explicit_both to keep the diagnostic focused.
        both = dict(seed_models["explicit_both"])[seed]
        for control in ("no_memory", "shuffle_image", "text_only", "image_only", "audio_only"):
            row[f"{control}_explicit_both_q2"] = evaluate(
                both, eval_episodes, control
            )["q2_accuracy"]
        results.append(row)

    both_values = [r["explicit_both"] for r in results]
    output = {
        "experiment_id":"TACOSM-PLM-INTEGRATED-E2E-002",
        "status":"measured",
        "provenance":{
            "git_commit":os.environ.get("GITHUB_SHA","unknown"),
            "pr_head_sha":os.environ.get("GITHUB_HEAD_SHA","unknown"),
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
            "evaluation_episodes_per_seed":400,
            "frozen_e2e001_mean":0.5200
        },
        "seed_results":results,
        "summary":{
            "explicit_both_mean_q2":statistics.fmean(both_values),
            "explicit_both_seed_bootstrap_ci95":bootstrap(both_values),
            "explicit_both_primary_pass":statistics.fmean(both_values)>=0.80,
            "explicit_both_min_seed":min(both_values),
            "frozen_e2e001_mean":0.5200
        }
    }
    out = __import__("pathlib").Path(ROOT).resolve().parents[1] / "artifacts" / "TACOSM-PLM-INTEGRATED-E2E-002.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print(json.dumps(output,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
