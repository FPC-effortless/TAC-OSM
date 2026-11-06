"""Latent-operator induction benchmark built on the clean E2E-008 payload task.

The target operator is never passed to the model. Each episode supplies a
permuted four-row truth-table context (a, b, y) from which the model must infer
one of XOR/AND/OR/XNOR, then apply it to a held-out target entity/bit pair.

Entity addressing remains explicit by design. This lane removes only the
operator-ID dispatch scaffold.
"""
from __future__ import annotations

import hashlib
import io
import math
import random
from typing import Iterable

import torch

from .integrated_e2e_008_benchmark import (
    BITS,
    ENTITY_COUNT,
    EVAL_EPISODES,
    EXCLUDED,
    HELDOUT,
    TRAIN_COMBOS,
    make_modalities,
)

OPS = ("xor", "and", "or", "xnor")
SEEDS = (0, 1, 2, 3, 4)
STEPS = 300
BATCH_SIZE = 96
GENERATOR_VERSION = "integrated-e2e-v8-latent-operator-truth-table"
BASE_E2E008_GENERATOR_SHA256 = "3be32a91191b73792b0fffb5a7b497e80aa6a46d0189e032e6b7c60082949f3e"
SUPPORT_PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))


def apply_op(op: str, a: int, b: int) -> int:
    return {
        "xor": a ^ b,
        "and": a & b,
        "or": a | b,
        "xnor": 1 - (a ^ b),
    }[op]


def make_support_context(op: str, rng: random.Random) -> tuple[tuple[int, int, int], ...]:
    rows = [(a, b, apply_op(op, a, b)) for a, b in SUPPORT_PAIRS]
    rng.shuffle(rows)
    return tuple(rows)


def sample_episode(
    rng: random.Random,
    *,
    combo2: tuple[str, int, int] | None = None,
):
    entities = rng.sample(range(ENTITY_COUNT), 3)
    payload = {e: [rng.randrange(2) for _ in range(BITS)] for e in entities}
    combo2 = combo2 or rng.choice(TRAIN_COMBOS)
    op, i, j = combo2
    target_entity = entities[1]
    answer = apply_op(op, payload[target_entity][i], payload[target_entity][j])
    q2 = (target_entity, i, j, answer)
    observations = [
        (entity, *make_modalities(payload[entity], rng))
        for entity in entities
    ]
    support = make_support_context(op, rng)
    return observations, q2, support, op, payload


def sample_evaluation_episodes(rng: random.Random, count: int):
    episodes = []
    for n in range(count):
        combo = HELDOUT[n % len(HELDOUT)]
        ep = sample_episode(rng, combo2=combo)
        validate_episode(ep)
        episodes.append(ep)
    return episodes


def validate_episode(ep) -> None:
    observations, q2, support, hidden_op, payload = ep
    entities = [row[0] for row in observations]
    assert len(set(entities)) == 3
    assert q2[0] == entities[1]
    assert hidden_op in OPS
    assert (hidden_op, q2[1], q2[2]) in HELDOUT
    assert tuple(sorted(support)) == tuple(
        sorted((a, b, apply_op(hidden_op, a, b)) for a, b in SUPPORT_PAIRS)
    )
    a = payload[q2[0]][q2[1]]
    b = payload[q2[0]][q2[2]]
    assert q2[3] == apply_op(hidden_op, a, b)


def episode_key(ep):
    observations, q2, support, hidden_op, payload = ep
    entity_key = tuple(row[0] for row in observations)
    payload_key = tuple(
        (int(e), tuple(int(x) for x in payload[e]))
        for e in entity_key
    )
    return (
        entity_key,
        payload_key,
        tuple(map(int, q2[:3])),
        int(q2[3]),
        tuple(support),
        hidden_op,
    )


def episode_fingerprint(episodes: Iterable[tuple]) -> str:
    serial = []
    for ep in episodes:
        observations, q2, support, hidden_op, payload = ep
        serial.append(
            {
                "entities": [int(row[0]) for row in observations],
                "payload": {
                    str(k): list(map(int, v))
                    for k, v in sorted(payload.items())
                },
                "q2": list(q2),
                "support": [list(row) for row in support],
                "hidden_op": hidden_op,
                "text": [row[1].tolist() for row in observations],
                "image": [row[2].tolist() for row in observations],
                "audio": [row[3].tolist() for row in observations],
            }
        )
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
        "base_e2e008_generator_sha256": BASE_E2E008_GENERATOR_SHA256,
        "ops": list(OPS),
        "heldout": [list(x) for x in HELDOUT],
        "train_combos_count": len(TRAIN_COMBOS),
        "eval_episodes_per_seed": EVAL_EPISODES,
        "support_pairs": [list(x) for x in SUPPORT_PAIRS],
        "support_order_randomized": True,
        "operator_id_in_model_input": False,
        "entity_side_channel_in_modalities": False,
    }
