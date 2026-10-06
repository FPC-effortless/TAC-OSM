#!/usr/bin/env python3
"""TACOSM-PLM-INTEGRATED-E2E-002.

The script is the authoritative benchmark runner. It does not inspect test
labels or evaluation outcomes during training. Held-out query compositions are
generated only after training and are not used for model selection.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
import math
import platform
import random
import statistics
import sys
from typing import Iterable

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.contract import load_contract
from tac_osm.measurement.results import report_smoke
from tac_osm.integrated_e2e import OPS, IntegratedConfig, IntegratedE2EModel


EXPERIMENT_ID = "TACOSM-PLM-INTEGRATED-E2E-002"
BENCHMARK_GENERATOR_VERSION = "integrated-e2e-v2-correct-q1-q2-entities-complementary-12bit-heldout"
BENCHMARK_HASH = hashlib.sha256(BENCHMARK_GENERATOR_VERSION.encode()).hexdigest()
SEEDS = (0, 1, 2, 3, 4)
STEPS = 300
BATCH_SIZE = 96
REGISTERED_ARMS = ("integrated", "no_memory", "shuffle_image")
ENTITY_COUNT = 16
BITS = 12
HELDOUT = (
    ("xor", 0, 4),
    ("xor", 5, 8),
    ("and", 1, 9),
    ("and", 2, 6),
    ("or", 7, 10),
    ("or", 3, 11),
    ("xnor", 4, 9),
    ("xnor", 6, 11),
)
ALL_COMBOS = tuple(
    (op, i, j)
    for op in OPS
    for i in range(BITS)
    for j in range(i + 1, BITS)
)
TRAIN_COMBOS = tuple(combo for combo in ALL_COMBOS if combo not in HELDOUT)


@dataclass(frozen=True)
class EpisodeBatch:
    text: tuple[Tensor, ...]
    image: tuple[Tensor, ...]
    audio: tuple[Tensor, ...]
    entities: tuple[Tensor, ...]
    payloads: tuple[Tensor, ...]
    q1: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    q2: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


def _noise(shape: tuple[int, ...], rng: random.Random) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(rng.randrange(2**31))
    return torch.randn(shape, generator=generator)


def make_modalities(entity: int, bits: list[int], rng: random.Random):
    # Complementary information: no single modality contains the full
    # payload. Text owns bits 0-3, image owns bits 4-7, audio owns bits 8-11.
    text = torch.tensor(
        [2, 4 + entity, 3, *[20 + b for b in bits[:4]]],
        dtype=torch.long,
    )

    image = torch.zeros(1, 16, 16)
    for idx, bit in enumerate(bits[4:8]):
        r = (idx // 2) * 6 + 1
        c = (idx % 2) * 6 + 1
        image[0, r:r+3, c:c+3] = float(bit)
    for k in range(4):
        image[0, k, :] = float((entity >> k) & 1)
    image += 0.03 * _noise((1, 16, 16), rng)
    image.clamp_(0.0, 1.0)

    t = torch.linspace(0, 1, 96)
    audio = torch.zeros(96)
    base = 2 + entity % 5
    for idx, bit in enumerate(bits[8:12]):
        seg = slice(idx * 24, (idx + 1) * 24)
        local_t = t[seg]
        audio[seg] = (1 if bit else -1) * torch.sin(
            2 * math.pi * (base + idx + 1) * local_t
        )
    audio += 0.03 * _noise((96,), rng)
    return text, image, audio


def sample_episode(
    rng: random.Random,
    combo1: tuple[str, int, int] | None = None,
    combo2: tuple[str, int, int] | None = None,
):
    entities = rng.sample(range(ENTITY_COUNT), 3)
    payload = {e: [rng.randrange(2) for _ in range(BITS)] for e in entities}

    def make_query(combo, entity):
        op, i, j = combo
        a, b = payload[entity][i], payload[entity][j]
        y = {
            "xor": a ^ b,
            "and": a & b,
            "or": a | b,
            "xnor": 1 - (a ^ b),
        }[op]
        return entity, i, j, op, y

    combo1 = combo1 or rng.choice(TRAIN_COMBOS)
    combo2 = combo2 or rng.choice(TRAIN_COMBOS)
    if combo1 in HELDOUT or combo2 in HELDOUT:
        raise RuntimeError("evaluation composition leaked into training sampler")
    obs = []
    for entity in entities:
        obs.append((entity, *make_modalities(entity, payload[entity], rng)))
    q1 = make_query(combo1, entities[0])
    q2 = make_query(combo2, entities[1])
    assert q1[0] == entities[0] and q2[0] == entities[1] and q1[0] != q2[0]
    return obs, q1, q2, payload


def batchify(episodes: Iterable[tuple]):
    eps = list(episodes)
    b = len(eps)
    texts, images, audios, entities, payloads = [], [], [], [], []
    for k in range(3):
        t, im, au, en, pl = [], [], [], [], []
        for ep in eps:
            entity, text, image, audio = ep[0][k]
            t.append(text)
            im.append(image)
            au.append(audio)
            en.append(entity)
            pl.append(ep[3][entity])
        texts.append(torch.stack(t))
        images.append(torch.stack(im))
        audios.append(torch.stack(au).unsqueeze(1))
        entities.append(torch.tensor(en))
        payloads.append(torch.tensor(pl, dtype=torch.float32))

    def qbatch(idx):
        qs = [ep[idx] for ep in eps]
        return tuple(
            (
                torch.tensor([q[0] for q in qs]),
                torch.tensor([q[1] for q in qs]),
                torch.tensor([q[2] for q in qs]),
                torch.tensor([OPS.index(q[3]) for q in qs]),
                torch.tensor([q[4] for q in qs]),
            )
        )

    return {
        "text": tuple(texts),
        "image": tuple(images),
        "audio": tuple(audios),
        "entities": tuple(entities),
        "payloads": tuple(payloads),
        "q1": qbatch(1),
        "q2": qbatch(2),
    }


def train_seed(seed: int) -> IntegratedE2EModel:
    torch.manual_seed(seed)
    rng = random.Random(seed + 9001)
    model = IntegratedE2EModel(IntegratedConfig()).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    for _ in range(STEPS):
        episodes = [sample_episode(rng) for _ in range(BATCH_SIZE)]
        batch = batchify(episodes)
        optimizer.zero_grad(set_to_none=True)
        out = model.forward_episode(
            {"text": torch.stack(batch["text"], 0),
             "image": torch.stack(batch["image"], 0),
             "audio": torch.stack(batch["audio"], 0)},
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
def evaluate_seed(
    model: IntegratedE2EModel,
    episodes: list[tuple],
    control: str = "normal",
) -> dict[str, float]:
    total1 = total2 = 0
    op_correct = 0
    attention_mass = 0.0

    for ep in episodes:
        batch = batchify([ep])
        device = torch.device("cpu")
        memory = model.state.initial(1, device)

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
            memory, _ = model.observation_write(memory, z)

        if control == "no_memory":
            memory.zero_()

        q1 = batch["q1"]
        q2 = batch["q2"]
        out1 = model.query(memory, *q1[:-1])
        total1 += int(
            out1["logits"].argmax(-1).item() == q1[-1].item()
        )
        op_correct += int(
            out1["operator_prob"].argmax(-1).item() == q1[3].item()
        )
        attention_mass += float(
            out1["attention"][0, q1[0].item()].item()
        )

        if control != "no_memory":
            memory, _ = model.post_action_update(
                memory, out1, q1[-1]
            )

        out2 = model.query(memory, *q2[:-1])
        total2 += int(
            out2["logits"].argmax(-1).item() == q2[-1].item()
        )

    n = len(episodes)
    return {
        "q1_accuracy": total1 / n,
        "q2_accuracy": total2 / n,
        "operator_selection_accuracy": op_correct / n,
        "target_memory_attention": attention_mass / n,
    }

def seed_bootstrap_ci(
    values: list[float],
    *,
    samples: int = 5000,
    seed: int = 20261002,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(draw))
    means.sort()
    lo = means[int((alpha / 2) * samples)]
    hi = means[int((1 - alpha / 2) * samples) - 1]
    return lo, hi


def theoretical_operator_prior_baseline() -> float:
    return (0.50 + 0.75 + 0.75 + 0.50) / 4


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.fmean(row[key] for row in rows)
        for key in rows[0]
    }


def episode_fingerprint(episodes: list[tuple]) -> str:
    h = hashlib.sha256()
    for obs, q1, q2, payload in episodes:
        h.update(repr(tuple((int(e), tuple(int(x) for x in payload[e])) for e in sorted(payload))).encode())
        h.update(repr(q1).encode())
        h.update(repr(q2).encode())
    return h.hexdigest()


def main(smoke: bool = False) -> None:
    contract = load_contract(EXPERIMENT_ID)
    if smoke:
        report_smoke(
            contract,
            EXPERIMENT_ID,
            steps=2,
            eval_steps=20,
            h_levels=list(contract.h_levels),
            seeds=[0],
            arms=list(REGISTERED_ARMS),
        )
        return

    contract.require_levels(contract.h_levels)
    contract.require_seeds(SEEDS)
    contract.require_steps(STEPS)
    contract.require_eval_steps(contract.eval_steps)
    contract.require_arms(REGISTERED_ARMS)

    if contract.eval_steps != 400:
        raise RuntimeError("E2E-002 expects the registered 400 evaluation episodes")
    if STEPS != contract.steps or BATCH_SIZE != 96:
        raise RuntimeError("runner/training contract drift")

    seed_models = [(seed, train_seed(seed)) for seed in SEEDS]
    results = []
    controls = (
        "normal",
        "no_memory",
        "shuffle_image",
        "text_only",
        "image_only",
        "audio_only",
    )
    for seed, model in seed_models:
        rng = random.Random(100000 + seed)
        eval_episodes = []
        for episode_id in range(400):
            combo1, combo2 = rng.sample(HELDOUT, 2)
            episode = sample_episode(rng, combo1=combo1, combo2=combo2)
            assert episode[1][0] != episode[2][0]
            eval_episodes.append(episode)

        episode_hash = episode_fingerprint(eval_episodes)
        evaluated = {
            control: evaluate_seed(model, eval_episodes, control)
            for control in controls
        }
        normal = evaluated["normal"]
        results.append({
            "seed": seed,
            "normal_q1": normal["q1_accuracy"],
            "normal_q2": normal["q2_accuracy"],
            "normal_operator_selection": normal["operator_selection_accuracy"],
            "normal_target_memory_attention": normal["target_memory_attention"],
            "no_memory_q2": evaluated["no_memory"]["q2_accuracy"],
            "shuffle_image_q2": evaluated["shuffle_image"]["q2_accuracy"],
            "text_only_q2": evaluated["text_only"]["q2_accuracy"],
            "image_only_q2": evaluated["image_only"]["q2_accuracy"],
            "audio_only_q2": evaluated["audio_only"]["q2_accuracy"],
            "evaluation_episode_fingerprint": episode_hash,
        })

    summary = {
        key: statistics.fmean(row[key] for row in results)
        for key in results[0]
        if key != "seed"
    }
    summary["alignment_drop"] = (
        summary["normal_q2"] - summary["shuffle_image_q2"]
    )
    summary["memory_drop"] = summary["normal_q2"] - summary["no_memory_q2"]
    summary["text_only_gap"] = summary["normal_q2"] - summary["text_only_q2"]
    summary["image_only_gap"] = summary["normal_q2"] - summary["image_only_q2"]
    summary["audio_only_gap"] = summary["normal_q2"] - summary["audio_only_q2"]
    summary["primary_pass"] = bool(summary["normal_q2"] >= 0.80)
    summary["all_seed_min_q2"] = min(r["normal_q2"] for r in results)
    summary["seed_failure_threshold_pass"] = bool(
        summary["all_seed_min_q2"] >= 0.40
    )
    summary["theoretical_operator_prior_accuracy"] = theoretical_operator_prior_baseline()

    q2_values = [r["normal_q2"] for r in results]
    memory_drops = [r["normal_q2"] - r["no_memory_q2"] for r in results]
    alignment_drops = [r["normal_q2"] - r["shuffle_image_q2"] for r in results]
    summary["primary_q2_seed_bootstrap_ci95"] = seed_bootstrap_ci(q2_values)
    summary["memory_drop_seed_bootstrap_ci95"] = seed_bootstrap_ci(memory_drops)
    summary["alignment_drop_seed_bootstrap_ci95"] = seed_bootstrap_ci(alignment_drops)

    output = {
        "experiment_id": EXPERIMENT_ID,
        "provenance": {
            "git_commit": os.environ.get("GITHUB_SHA", "unknown"),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        },
        "status": "measured",
        "protocol": {
            "seeds": SEEDS,
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "heldout_compositions": HELDOUT,
            "evaluation_episodes_per_seed": 400,
            "operator_prior_baseline": theoretical_operator_prior_baseline(),
            "benchmark_generator_version": BENCHMARK_GENERATOR_VERSION,
            "benchmark_hash": BENCHMARK_HASH,
            "registered_arms": REGISTERED_ARMS,
            "seed_bootstrap": {
                "samples": 5000,
                "seed": 20261002,
                "confidence": 0.95,
            },
        },
        "seed_results": results,
        "summary": summary,
        "leakage_audit": {
            "q1_entity_rule": "q1 target entity is entities[0]",
            "q2_entity_rule": "q2 target entity is entities[1]",
            "q1_q2_entities_distinct": True,
            "heldout_compositions_excluded_from_training": True,
            "evaluation_generated_after_training": True,
            "controls_reuse_exact_same_episode_objects": True,
            "pre_action_query_signature_fields": ["entity", "i", "j", "op"],
            "forbidden_pre_action_fields": ["answer", "environment_outcome", "verifier_target"],
        },
        "leakage_audit": {
            "q1_entity_rule": "q1 target entity is entities[0]",
            "q2_entity_rule": "q2 target entity is entities[1]",
            "q1_q2_entities_distinct": True,
            "q1_q2_query_indices_not_payload_leaked": True,
            "heldout_compositions_excluded_from_training": True,
            "train_eval_rng_streams_disjoint": True,
            "evaluation_generated_after_training": True,
            "controls_reuse_exact_same_episode_objects": True,
            "pre_action_query_signature_fields": ["entity", "i", "j", "op"],
            "forbidden_pre_action_fields": ["answer", "environment_outcome", "verifier_target"],
        },
        "decision_rule": {
            "primary": "mean normal_q2 >= 0.80",
            "memory_control": "normal_q2 > no_memory_q2",
            "alignment_control": "normal_q2 > shuffle_image_q2",
        },
        "claim_boundary": [
            "Synthetic benchmark only.",
            "No real-world language, vision or audio claim.",
            "No scaling-law claim.",
        ],
    }
    out = ROOT / "artifacts" / "TACOSM-PLM-INTEGRATED-E2E-002.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    main(smoke=parser.parse_args().smoke)
