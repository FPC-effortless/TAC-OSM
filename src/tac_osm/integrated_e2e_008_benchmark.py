"""Clean E2E-008 benchmark: payload-only text/image/audio encodings.

The prior E2E-007 image encoding placed an entity-ID stripe in rows 0-3 while
payload patches for image bits 4-5 occupied rows 1-3. This generator removes
the side channel and collision entirely. Entity identity is represented only
by the explicit state-write/query scaffold in the harness.
"""
from __future__ import annotations

import hashlib
import io
import math
import random
from typing import Iterable

import torch

OPS = ("xor", "and", "or", "xnor")
SEEDS = (0, 1, 2, 3, 4)
STEPS = 300
BATCH_SIZE = 96
ENTITY_COUNT = 16
BITS = 12
EVAL_EPISODES = 600
NOISE = 0.03

SEALED_E2E005 = (
    ("xor", 0, 4), ("xor", 5, 8), ("and", 1, 9), ("and", 2, 6),
    ("or", 7, 10), ("or", 3, 11), ("xnor", 4, 9), ("xnor", 6, 11),
)
DEV_COMBOS = (
    ("xor", 0, 5), ("xor", 0, 6), ("xor", 0, 7), ("xor", 0, 8),
    ("xor", 0, 9), ("xor", 0, 10), ("xor", 0, 11), ("xor", 1, 4),
    ("and", 0, 4), ("and", 0, 5), ("and", 0, 6), ("and", 0, 7),
    ("and", 0, 8), ("and", 0, 9), ("and", 0, 10), ("and", 0, 11),
    ("or", 0, 4), ("or", 0, 5), ("or", 0, 6), ("or", 0, 7),
    ("or", 0, 8), ("or", 0, 9), ("or", 0, 10), ("or", 0, 11),
    ("xnor", 0, 4), ("xnor", 0, 5), ("xnor", 0, 6), ("xnor", 0, 7),
    ("xnor", 0, 8), ("xnor", 0, 9), ("xnor", 0, 10), ("xnor", 0, 11),
)
E2E006_HELDOUT = (
    ("xor", 1, 5), ("xor", 2, 9), ("and", 4, 10), ("and", 3, 8),
    ("or", 1, 7), ("or", 5, 10), ("xnor", 2, 7), ("xnor", 6, 9),
)
E2E007_HELDOUT = (
    ("xor", 1, 6), ("xor", 2, 10), ("and", 4, 11), ("and", 5, 9),
    ("or", 2, 8), ("or", 6, 10), ("xnor", 1, 10), ("xnor", 3, 9),
)
HELDOUT = (
    ("xor", 2, 4), ("xor", 4, 8), ("xor", 1, 9),
    ("and", 3, 5), ("and", 6, 10), ("and", 2, 11),
    ("or", 2, 4), ("or", 4, 8), ("or", 1, 9),
    ("xnor", 3, 5), ("xnor", 6, 10), ("xnor", 2, 11),
)
EXCLUDED = set(SEALED_E2E005) | set(DEV_COMBOS) | set(E2E006_HELDOUT) | set(E2E007_HELDOUT) | set(HELDOUT)
ALL_COMBOS = tuple((op, i, j) for op in OPS for i in range(BITS) for j in range(i + 1, BITS))
TRAIN_COMBOS = tuple(c for c in ALL_COMBOS if c not in EXCLUDED)
GENERATOR_VERSION = "integrated-e2e-v7-collision-free-payload-only"

assert len(HELDOUT) == 12
assert set(HELDOUT).isdisjoint(set(SEALED_E2E005))
assert set(HELDOUT).isdisjoint(set(DEV_COMBOS))
assert set(HELDOUT).isdisjoint(set(E2E006_HELDOUT))
assert set(HELDOUT).isdisjoint(set(E2E007_HELDOUT))
assert set(HELDOUT).isdisjoint(set(TRAIN_COMBOS))

def _noise(shape: tuple[int, ...], rng: random.Random) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(rng.randrange(2**31))
    return torch.randn(shape, generator=g)

def make_modalities(bits: list[int], rng: random.Random):
    text = torch.tensor([2, 3, *[20 + b for b in bits[:4]]], dtype=torch.long)
    image = torch.zeros(1, 16, 16)
    for idx, bit in enumerate(bits[4:8]):
        row = (idx // 2) * 6 + 4
        col = (idx % 2) * 6 + 1
        image[0, row:row + 3, col:col + 3] = float(bit)
    image += NOISE * _noise((1, 16, 16), rng)
    image.clamp_(0.0, 1.0)

    t = torch.linspace(0, 1, 96)
    audio = torch.zeros(96)
    for idx, bit in enumerate(bits[8:12]):
        seg = slice(idx * 24, (idx + 1) * 24)
        local_t = t[seg]
        audio[seg] = (1 if bit else -1) * torch.sin(
            2 * math.pi * (2 + idx + 1) * local_t
        )
    audio += NOISE * _noise((96,), rng)
    return text, image, audio

def make_query(payload, combo, entity):
    op, i, j = combo
    a, b = payload[entity][i], payload[entity][j]
    answer = {"xor": a ^ b, "and": a & b, "or": a | b, "xnor": 1 - (a ^ b)}[op]
    return entity, i, j, op, answer

def sample_episode(rng: random.Random, *, combo1=None, combo2=None):
    entities = rng.sample(range(ENTITY_COUNT), 3)
    payload = {e: [rng.randrange(2) for _ in range(BITS)] for e in entities}
    combo1 = combo1 or rng.choice(TRAIN_COMBOS)
    combo2 = combo2 or rng.choice(TRAIN_COMBOS)
    q1 = make_query(payload, combo1, entities[0])
    q2 = make_query(payload, combo2, entities[1])
    observations = [(entity, *make_modalities(payload[entity], rng)) for entity in entities]
    return observations, q1, q2, payload

def sample_evaluation_episodes(rng: random.Random, count: int):
    episodes = []
    for n in range(count):
        q1_combo = HELDOUT[n % len(HELDOUT)]
        q2_combo = HELDOUT[(n + 5) % len(HELDOUT)]
        if q2_combo == q1_combo:
            q2_combo = HELDOUT[(n + 6) % len(HELDOUT)]
        ep = sample_episode(rng, combo1=q1_combo, combo2=q2_combo)
        validate_episode(ep)
        episodes.append(ep)
    return episodes

def validate_episode(ep):
    obs, q1, q2, payload = ep
    entities = [row[0] for row in obs]
    assert len(set(entities)) == 3
    assert q1[0] == entities[0] and q2[0] == entities[1] and q1[0] != q2[0]
    for q in (q1, q2):
        comp = (q[3], q[1], q[2])
        assert comp in HELDOUT
        assert comp not in TRAIN_COMBOS
        a, b = payload[q[0]][q[1]], payload[q[0]][q[2]]
        expected = {"xor": a ^ b, "and": a & b, "or": a | b, "xnor": 1 - (a ^ b)}[q[3]]
        assert q[4] == expected

def episode_key(ep):
    obs, q1, q2, payload = ep
    entity_key = tuple(row[0] for row in obs)
    payload_key = tuple((int(e), tuple(int(x) for x in payload[e])) for e in entity_key)
    return entity_key, payload_key, (q1[3], q1[1], q1[2]), (q2[3], q2[1], q2[2])

def episode_fingerprint(episodes: Iterable[tuple]) -> str:
    serial = []
    for ep in episodes:
        obs, q1, q2, payload = ep
        serial.append({
            "entities": [int(row[0]) for row in obs],
            "payload": {str(k): list(map(int, v)) for k, v in sorted(payload.items())},
            "q1": list(q1), "q2": list(q2),
            "text": [row[1].tolist() for row in obs],
            "image": [row[2].tolist() for row in obs],
            "audio": [row[3].tolist() for row in obs],
        })
    b = io.BytesIO()
    torch.save(serial, b)
    return hashlib.sha256(b.getvalue()).hexdigest()

def generator_hash() -> str:
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def benchmark_manifest():
    return {
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": generator_hash(),
        "ops": list(OPS),
        "heldout": [list(x) for x in HELDOUT],
        "train_combos_count": len(TRAIN_COMBOS),
        "eval_episodes_per_seed": EVAL_EPISODES,
        "entity_side_channel": False,
        "image_entity_stripe": False,
        "image_payload_id_overlap": False,
        "image_payload_rows": [4, 10]
    }
