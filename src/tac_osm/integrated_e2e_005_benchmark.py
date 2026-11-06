"""Canonical benchmark for TACOSM-PLM-INTEGRATED-E2E-005.

The runner fingerprints this exact source file and the validator recomputes the
fingerprint before accepting a result.
"""
from __future__ import annotations

import hashlib
import io
import math
import random
from typing import Iterable

import torch
from torch import Tensor

OPS = ("xor", "and", "or", "xnor")
SEEDS = (0, 1, 2, 3, 4)
STEPS = 300
BATCH_SIZE = 96
ENTITY_COUNT = 16
BITS = 12
EVAL_EPISODES = 400
NOISE = 0.03

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
GENERATOR_VERSION = "integrated-e2e-v3-functional-no-aux-supervision"


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
    image += NOISE * _noise((1, 16, 16), rng)
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
    audio += NOISE * _noise((96,), rng)
    return text, image, audio


def make_query(
    payload: dict[int, list[int]],
    combo: tuple[str, int, int],
    entity: int,
) -> tuple[int, int, int, str, int]:
    op, i, j = combo
    a = payload[entity][i]
    b = payload[entity][j]
    answer = {
        "xor": a ^ b,
        "and": a & b,
        "or": a | b,
        "xnor": 1 - (a ^ b),
    }[op]
    return entity, i, j, op, answer


def sample_episode(
    rng: random.Random,
    *,
    combo1: tuple[str, int, int] | None = None,
    combo2: tuple[str, int, int] | None = None,
) -> tuple[list[tuple[int, Tensor, Tensor, Tensor]], tuple, tuple, dict[int, list[int]]]:
    entities = rng.sample(range(ENTITY_COUNT), 3)
    payload = {e: [rng.randrange(2) for _ in range(BITS)] for e in entities}
    combo1 = combo1 or rng.choice(TRAIN_COMBOS)
    combo2 = combo2 or rng.choice(TRAIN_COMBOS)
    q1 = make_query(payload, combo1, entities[0])
    q2 = make_query(payload, combo2, entities[1])
    if q1[0] == q2[0]:
        raise AssertionError("q1 and q2 must target distinct entities")
    observations = [
        (entity, *make_modalities(entity, payload[entity], rng))
        for entity in entities
    ]
    return observations, q1, q2, payload


def sample_evaluation_episodes(rng: random.Random, count: int) -> list[tuple]:
    episodes = []
    for n in range(count):
        q1_combo = HELDOUT[n % len(HELDOUT)]
        q2_combo = HELDOUT[(n + 3) % len(HELDOUT)]
        if q1_combo == q2_combo:
            q2_combo = HELDOUT[(n + 4) % len(HELDOUT)]
        episode = sample_episode(rng, combo1=q1_combo, combo2=q2_combo)
        validate_episode(episode)
        episodes.append(episode)
    return episodes


def validate_episode(episode: tuple) -> None:
    observations, q1, q2, payload = episode
    entities = [row[0] for row in observations]
    if len(set(entities)) != 3:
        raise AssertionError("episode must contain three unique entities")
    if q1[0] != entities[0] or q2[0] != entities[1] or q1[0] == q2[0]:
        raise AssertionError("q1/q2 target entity invariant violated")
    for q in (q1, q2):
        composition = (q[3], q[1], q[2])
        if composition not in HELDOUT:
            raise AssertionError("evaluation query is not registered held-out")
        if composition in TRAIN_COMBOS:
            raise AssertionError("held-out composition leaked into training")
        a, b = payload[q[0]][q[1]], payload[q[0]][q[2]]
        expected = {
            "xor": a ^ b,
            "and": a & b,
            "or": a | b,
            "xnor": 1 - (a ^ b),
        }[q[3]]
        if q[4] != expected:
            raise AssertionError("stored answer does not match target entity payload")


def episode_key(episode: tuple) -> tuple:
    observations, q1, q2, payload = episode
    entity_key = tuple(row[0] for row in observations)
    payload_key = tuple(
        (int(entity), tuple(int(x) for x in payload[entity]))
        for entity in entity_key
    )
    return (
        entity_key,
        payload_key,
        (q1[3], q1[1], q1[2]),
        (q2[3], q2[1], q2[2]),
    )


def episode_fingerprint(episodes: Iterable[tuple]) -> str:
    buffer = io.BytesIO()
    serial = []
    for episode in episodes:
        observations, q1, q2, payload = episode
        serial.append(
            {
                "entities": [int(row[0]) for row in observations],
                "payload": {
                    str(k): list(map(int, v))
                    for k, v in sorted(payload.items())
                },
                "q1": list(q1),
                "q2": list(q2),
                "text": [row[1].tolist() for row in observations],
                "image": [row[2].tolist() for row in observations],
                "audio": [row[3].tolist() for row in observations],
            }
        )
    torch.save(serial, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def generator_hash() -> str:
    with open(__file__, "r", encoding="utf-8") as handle:
        return hashlib.sha256(handle.read().encode("utf-8")).hexdigest()


def benchmark_manifest() -> dict:
    return {
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": generator_hash(),
        "ops": list(OPS),
        "heldout": [list(x) for x in HELDOUT],
        "train_combos_count": len(TRAIN_COMBOS),
        "eval_episodes_per_seed": EVAL_EPISODES,
        "entity_count": ENTITY_COUNT,
        "latent_bits": BITS,
    }
