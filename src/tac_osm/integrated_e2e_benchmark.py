"""Canonical synthetic benchmark generator for corrected integrated E2E studies."""
from __future__ import annotations

import math
import random
from typing import Iterable

import torch
from torch import Tensor

from .integrated_e2e import OPS

SEEDS = (0, 1, 2, 3, 4)
STEPS = 300
BATCH_SIZE = 96
ENTITY_COUNT = 16
BITS = 12
EVAL_EPISODES = 400
BENCHMARK_GENERATOR_VERSION = "integrated-e2e-benchmark-v3-entity-corrected-heldout"

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


def _noise(shape: tuple[int, ...], rng: random.Random) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(rng.randrange(2**31))
    return torch.randn(shape, generator=generator)


def make_modalities(
    entity: int, bits: list[int], rng: random.Random
) -> tuple[Tensor, Tensor, Tensor]:
    text = torch.tensor(
        [2, 4 + entity, 3, *[20 + b for b in bits[:4]]],
        dtype=torch.long,
    )

    image = torch.zeros(1, 16, 16)
    for idx, bit in enumerate(bits[4:8]):
        row = (idx // 2) * 6 + 1
        col = (idx % 2) * 6 + 1
        image[0, row:row + 3, col:col + 3] = float(bit)
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

    def make_query(combo: tuple[str, int, int], entity: int):
        op, i, j = combo
        a, b = payload[entity][i], payload[entity][j]
        y = {
            "xor": a ^ b,
            "and": a & b,
            "or": a | b,
            "xnor": 1 - (a ^ b),
        }[op]
        # Critical temporal benchmark invariant: the supplied entity is the
        # target entity. q1 and q2 intentionally use different entities.
        return entity, i, j, op, y

    combo1 = combo1 or rng.choice(TRAIN_COMBOS)
    combo2 = combo2 or rng.choice(TRAIN_COMBOS)
    q1 = make_query(combo1, entities[0])
    q2 = make_query(combo2, entities[1])
    if q1[0] == q2[0]:
        raise AssertionError("corrected benchmark requires distinct q1/q2 entities")

    obs = []
    for entity in entities:
        obs.append((entity, *make_modalities(entity, payload[entity], rng)))
    return obs, q1, q2, payload


def batchify(episodes: Iterable[tuple]):
    eps = list(episodes)
    if not eps:
        raise ValueError("cannot batchify an empty episode collection")

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

    def qbatch(idx: int):
        qs = [ep[idx] for ep in eps]
        return (
            torch.tensor([q[0] for q in qs]),
            torch.tensor([q[1] for q in qs]),
            torch.tensor([q[2] for q in qs]),
            torch.tensor([OPS.index(q[3]) for q in qs]),
            torch.tensor([q[4] for q in qs]),
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


def validate_episode(episode: tuple) -> None:
    _, q1, q2, _ = episode
    if q1[0] == q2[0]:
        raise AssertionError("q1 and q2 must target distinct entities")
    q1_composition = (q1[3], q1[1], q1[2])
    q2_composition = (q2[3], q2[1], q2[2])
    if q1_composition not in HELDOUT:
        raise AssertionError(
            "evaluation q1 composition is not registered held-out"
        )
    if q2_composition not in HELDOUT:
        raise AssertionError(
            "evaluation q2 composition is not registered held-out"
        )
    if q1_composition in TRAIN_COMBOS or q2_composition in TRAIN_COMBOS:
        raise AssertionError("held-out composition leaked into training set")
