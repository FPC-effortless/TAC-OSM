#!/usr/bin/env python3
"""TACOSM-PLM-INTEGRATED-E2E-001.

The script is the authoritative benchmark runner. It does not inspect test
labels or evaluation outcomes during training. Held-out query compositions are
generated only after training and are not used for model selection.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import math
import random
import statistics
import sys
from typing import Iterable

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.integrated_e2e import OPS, IntegratedConfig, IntegratedE2EModel


SEEDS = (0, 1, 2, 3, 4)
STEPS = 300
BATCH_SIZE = 96
ENTITY_COUNT = 16
BITS = 8
TRAIN_COMBOS = tuple(
    (op, i, j)
    for op in OPS
    for i in range(BITS)
    for j in range(BITS)
    if i != j
)
HELDOUT = (
    ("xor", 0, 1),
    ("xor", 2, 3),
    ("and", 4, 5),
    ("and", 6, 7),
    ("or", 0, 7),
    ("or", 1, 6),
    ("xnor", 2, 5),
    ("xnor", 3, 4),
)


@dataclass(frozen=True)
class EpisodeBatch:
    text: tuple[Tensor, ...]
    image: tuple[Tensor, ...]
    audio: tuple[Tensor, ...]
    entities: tuple[Tensor, ...]
    payloads: tuple[Tensor, ...]
    q1: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    q2: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


def make_modalities(entity: int, bits: list[int], rng: random.Random):
    text = torch.tensor(
        [2, 4 + entity, *[20 + b for b in bits], *([1] * 0)],
        dtype=torch.long,
    )
    image = torch.zeros(1, 16, 16)
    for idx, bit in enumerate(bits):
        r = (idx // 4) * 3 + 1
        c = (idx % 4) * 3 + 1
        image[0, r:r+2, c:c+2] = float(bit)
    for k in range(4):
        image[0, k, :] = float((entity >> k) & 1)
    image += 0.03 * torch.randn_like(image)
    image.clamp_(0.0, 1.0)

    t = torch.linspace(0, 1, 96)
    audio = torch.zeros(96)
    base = 2 + entity % 5
    for idx, bit in enumerate(bits):
        seg = slice(idx * 12, (idx + 1) * 12)
        audio[seg] = (1 if bit else -1) * torch.sin(
            2 * math.pi * (base + idx + 1) * t[seg]
        )
    audio += 0.03 * torch.randn_like(audio)
    return text, image, audio


def sample_episode(
    rng: random.Random,
    combo1: tuple[str, int, int] | None = None,
    combo2: tuple[str, int, int] | None = None,
):
    entities = rng.sample(range(ENTITY_COUNT), 3)
    payload = {e: [rng.randrange(2) for _ in range(BITS)] for e in entities}

    def make_query(combo):
        op, i, j = combo
        a, b = payload[entities[0]][i], payload[entities[0]][j]
        y = {
            "xor": a ^ b,
            "and": a & b,
            "or": a | b,
            "xnor": 1 - (a ^ b),
        }[op]
        return entities[0], i, j, op, y

    combo1 = combo1 or rng.choice(TRAIN_COMBOS)
    combo2 = combo2 or rng.choice(TRAIN_COMBOS)
    obs = []
    for entity in entities:
        obs.append((entity, *make_modalities(entity, payload[entity], rng)))
    return obs, make_query(combo1), make_query(combo2), payload


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
def evaluate_seed(model: IntegratedE2EModel, seed: int, control: str = "normal") -> dict[str, float]:
    rng = random.Random(100000 + seed)
    total1 = total2 = 0
    for _ in range(400):
        combo1, combo2 = rng.sample(HELDOUT, 2)
        ep = sample_episode(rng, combo1=combo1, combo2=combo2)
        batch = batchify([ep])
        memory = model.state.initial(1, torch.device("cpu"))
        for k in range(3):
            image = batch["image"][k]
            if control == "shuffle_image":
                image = batch["image"][(k + 1) % 3]
            z, _, _, _ = model.encode(
                batch["text"][k], image, batch["audio"][k]
            )
            memory, _ = model.observation_write(memory, z)
        if control == "no_memory":
            memory.zero_()
        out1 = model.query(
            memory, *batch["q1"][:-1]
        )
        total1 += int(out1["logits"].argmax(-1).item() == batch["q1"][-1].item())
        if control != "no_memory":
            memory, _ = model.post_action_update(
                memory, out1, batch["q1"][-1]
            )
        out2 = model.query(memory, *batch["q2"][:-1])
        total2 += int(out2["logits"].argmax(-1).item() == batch["q2"][-1].item())
    return {"q1_accuracy": total1 / 400, "q2_accuracy": total2 / 400}


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.fmean(row[key] for row in rows)
        for key in rows[0]
    }


def main() -> None:
    seed_models = [(seed, train_seed(seed)) for seed in SEEDS]
    results = []
    for seed, model in seed_models:
        normal = evaluate_seed(model, seed, "normal")
        no_memory = evaluate_seed(model, seed, "no_memory")
        shuffled = evaluate_seed(model, seed, "shuffle_image")
        results.append({
            "seed": seed,
            "normal_q1": normal["q1_accuracy"],
            "normal_q2": normal["q2_accuracy"],
            "no_memory_q2": no_memory["q2_accuracy"],
            "shuffle_image_q2": shuffled["q2_accuracy"],
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
    output = {
        "experiment_id": "TACOSM-PLM-INTEGRATED-E2E-001",
        "status": "measured",
        "protocol": {
            "seeds": SEEDS,
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "heldout_compositions": HELDOUT,
        },
        "seed_results": results,
        "summary": summary,
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
    out = ROOT / "artifacts" / "TACOSM-PLM-INTEGRATED-E2E-001.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
