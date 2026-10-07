"""Learned content-addressed retrieval benchmark.

Explicit entity IDs are removed from model inputs. Each entity receives an
episode-local random key independent of payload content. The query carries the
same key for its target entity. Observation order is randomized and q2 target
slots are balanced so fixed-position shortcuts cannot solve the task.

Operator IDs remain query-supplied/oracle-controlled to isolate addressing.
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
    HELDOUT,
    TRAIN_COMBOS,
    make_modalities,
)

OPS = ("xor", "and", "or", "xnor")
SEEDS = (0, 1, 2, 3, 4)
STEPS = 300
BATCH_SIZE = 96
EVAL_EPISODES = 600
ADDRESS_DIM = 16
GENERATOR_VERSION = "integrated-e2e-v9-learned-content-address"
BASE_E2E008_GENERATOR_SHA256 = "3be32a91191b73792b0fffb5a7b497e80aa6a46d0189e032e6b7c60082949f3e"


def apply_op(op: str, a: int, b: int) -> int:
    return {"xor": a ^ b, "and": a & b, "or": a | b, "xnor": 1 - (a ^ b)}[op]


def make_address_key(rng: random.Random) -> tuple[float, ...]:
    values = [rng.gauss(0.0, 1.0) for _ in range(ADDRESS_DIM)]
    norm = math.sqrt(sum(x * x for x in values))
    return tuple(x / norm for x in values)


def make_payload_and_observations(rng: random.Random, entities: list[int]):
    payload = {e: [rng.randrange(2) for _ in range(BITS)] for e in entities}
    keyed = {e: make_address_key(rng) for e in entities}
    rows = []
    for entity in entities:
        text, image, audio = make_modalities(payload[entity], rng)
        rows.append((keyed[entity], text, image, audio, entity))
    rng.shuffle(rows)
    return payload, keyed, rows


def make_query(payload, keyed, entity: int, combo):
    op, i, j = combo
    answer = apply_op(op, payload[entity][i], payload[entity][j])
    return keyed[entity], i, j, OPS.index(op), answer, entity


def sample_episode(
    rng: random.Random,
    *,
    combo1=None,
    combo2=None,
    eval_index: int | None = None,
):
    entities = rng.sample(range(ENTITY_COUNT), 3)
    payload, keyed, rows = make_payload_and_observations(rng, entities)

    if eval_index is None:
        target_slots = list(range(3))
        rng.shuffle(target_slots)
        q1_slot, q2_slot = target_slots[:2]
    else:
        q2_slot = eval_index % 3
        q1_slot = (q2_slot + 1) % 3

    row_entity_order = [row[4] for row in rows]
    q1_entity = entities[q1_slot]
    q2_entity = entities[q2_slot]
    assert q1_entity != q2_entity

    combo1 = combo1 or rng.choice(TRAIN_COMBOS)
    combo2 = combo2 or rng.choice(TRAIN_COMBOS)
    q1 = make_query(payload, keyed, q1_entity, combo1)
    q2 = make_query(payload, keyed, q2_entity, combo2)

    observations = [
        (key, text, image, audio)
        for key, text, image, audio, _entity in rows
    ]
    target_slots_after_shuffle = {
        "q1_slot": row_entity_order.index(q1_entity),
        "q2_slot": row_entity_order.index(q2_entity),
    }
    return observations, q1, q2, payload, target_slots_after_shuffle


def sample_evaluation_episodes(rng: random.Random, count: int):
    episodes = []
    for n in range(count):
        combo1 = TRAIN_COMBOS[n % len(TRAIN_COMBOS)]
        combo2 = HELDOUT[n % len(HELDOUT)]
        ep = sample_episode(
            rng,
            combo1=combo1,
            combo2=combo2,
            eval_index=n,
        )
        validate_episode(ep)
        episodes.append(ep)
    return episodes


def validate_episode(ep):
    observations, q1, q2, payload, slots = ep
    assert len(observations) == 3
    keys = [row[0] for row in observations]
    assert len(set(keys)) == 3

    q1_key, q1_i, q1_j, q1_op, q1_answer, q1_entity = q1
    q2_key, q2_i, q2_j, q2_op, q2_answer, q2_entity = q2

    assert q1_entity != q2_entity
    assert q1_key in keys and q2_key in keys and q1_key != q2_key
    assert q1_op in range(4) and q2_op in range(4)
    assert q1_answer == apply_op(
        OPS[q1_op], payload[q1_entity][q1_i], payload[q1_entity][q1_j]
    )
    assert q2_answer == apply_op(
        OPS[q2_op], payload[q2_entity][q2_i], payload[q2_entity][q2_j]
    )
    assert (OPS[q2_op], q2_i, q2_j) in HELDOUT
    assert slots["q1_slot"] in range(3)
    assert slots["q2_slot"] in range(3)
    assert slots["q1_slot"] != slots["q2_slot"]


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(x) for x in value)
    if isinstance(value, tuple):
        return tuple(_freeze(x) for x in value)
    return value


def episode_key(ep):
    observations, q1, q2, payload, slots = ep
    return (
        tuple(tuple(float(x) for x in row[0]) for row in observations),
        tuple(
            (
                _freeze(row[1].tolist()),
                _freeze(row[2].tolist()),
                _freeze(row[3].tolist()),
            )
            for row in observations
        ),
        tuple(q1[:-1]),
        tuple(q2[:-1]),
        tuple(sorted((int(k), tuple(int(x) for x in v)) for k, v in payload.items())),
        tuple(sorted(slots.items())),
    )


def episode_fingerprint(episodes: Iterable[tuple]) -> str:
    serial = []
    for ep in episodes:
        observations, q1, q2, payload, slots = ep
        serial.append(
            {
                "addresses": [list(map(float, row[0])) for row in observations],
                "text": [row[1].tolist() for row in observations],
                "image": [row[2].tolist() for row in observations],
                "audio": [row[3].tolist() for row in observations],
                "q1": list(q1),
                "q2": list(q2),
                "payload": {
                    str(k): list(map(int, v))
                    for k, v in sorted(payload.items())
                },
                "slots": slots,
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
        "address_dim": ADDRESS_DIM,
        "address_source": "episode-local random Gaussian vector independent of payload",
        "entity_id_in_model_input": False,
        "observation_order_randomized": True,
        "q2_target_slot_balanced_across_evaluation": True,
        "q1_q2_target_slots_distinct": True,
    }
