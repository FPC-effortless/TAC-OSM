"""Fresh confirmatory benchmark for the selected PLM configuration.

The generator reuses the frozen multimodal episode encoding from E2E-005 but
uses a brand-new held-out composition set. The sealed E2E-005 held-out set and
the registered development set are permanently excluded from E2E-006 training.
"""
from __future__ import annotations

import hashlib
import io
import math
import random
from typing import Iterable

import torch
from torch import Tensor

from .integrated_e2e_005_benchmark import (
    ALL_COMBOS,
    BITS,
    ENTITY_COUNT,
    NOISE,
    OPS,
    _noise,
    make_modalities,
)

SEEDS = (0, 1, 2, 3, 4)

SEALED_E2E005 = (
    ("xor", 0, 4), ("xor", 5, 8),
    ("and", 1, 9), ("and", 2, 6),
    ("or", 7, 10), ("or", 3, 11),
    ("xnor", 4, 9), ("xnor", 6, 11),
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

HELDOUT = (
    ("xor", 1, 5),
    ("xor", 2, 9),
    ("and", 4, 10),
    ("and", 3, 8),
    ("or", 1, 7),
    ("or", 5, 10),
    ("xnor", 2, 7),
    ("xnor", 6, 9),
)

ALL_EXCLUDED = set(SEALED_E2E005) | set(DEV_COMBOS) | set(HELDOUT)
TRAIN_COMBOS = tuple(c for c in ALL_COMBOS if c not in ALL_EXCLUDED)
GENERATOR_VERSION = "integrated-e2e-v5-fresh-heldout-selected-capacity64"

assert set(SEALED_E2E005).isdisjoint(DEV_COMBOS)
assert set(HELDOUT).isdisjoint(set(SEALED_E2E005))
assert set(HELDOUT).isdisjoint(set(DEV_COMBOS))
assert len(SEALED_E2E005) == 8
assert len(DEV_COMBOS) == 32
assert len(HELDOUT) == 8
assert len(TRAIN_COMBOS) == len(ALL_COMBOS) - 48

def make_query(payload: dict[int, list[int]], combo: tuple[str,int,int], entity: int):
    op, i, j = combo
    a, b = payload[entity][i], payload[entity][j]
    answer = {"xor":a ^ b, "and":a & b, "or":a | b, "xnor":1-(a ^ b)}[op]
    return entity, i, j, op, answer

def sample_episode(rng: random.Random, *, combo1=None, combo2=None):
    entities = rng.sample(range(ENTITY_COUNT), 3)
    payload = {e:[rng.randrange(2) for _ in range(BITS)] for e in entities}
    combo1 = combo1 or rng.choice(TRAIN_COMBOS)
    combo2 = combo2 or rng.choice(TRAIN_COMBOS)
    q1 = make_query(payload, combo1, entities[0])
    q2 = make_query(payload, combo2, entities[1])
    observations = [
        (e, *make_modalities(e, payload[e], rng)) for e in entities
    ]
    return observations, q1, q2, payload

def validate_episode(ep):
    obs,q1,q2,payload=ep
    entities=[r[0] for r in obs]
    assert len(set(entities)) == 3
    assert q1[0] == entities[0] and q2[0] == entities[1] and q1[0] != q2[0]
    for q in (q1,q2):
        combo=(q[3],q[1],q[2])
        assert combo in HELDOUT
        assert combo not in SEALED_E2E005 and combo not in DEV_COMBOS and combo not in TRAIN_COMBOS
        a,b=payload[q[0]][q[1]],payload[q[0]][q[2]]
        expected={"xor":a ^ b,"and":a & b,"or":a | b,"xnor":1-(a ^ b)}[q[3]]
        assert q[4] == expected

def sample_evaluation_episodes(rng: random.Random, count: int):
    episodes=[]
    for n in range(count):
        q1_combo=HELDOUT[n % len(HELDOUT)]
        q2_combo=HELDOUT[(n*3 + 1) % len(HELDOUT)]
        if q2_combo == q1_combo:
            q2_combo=HELDOUT[(n*3 + 2) % len(HELDOUT)]
        ep=sample_episode(rng,combo1=q1_combo,combo2=q2_combo)
        validate_episode(ep)
        episodes.append(ep)
    return episodes

def episode_key(ep):
    obs,q1,q2,payload=ep
    ents=tuple(r[0] for r in obs)
    payload_key=tuple((int(e),tuple(map(int,payload[e]))) for e in ents)
    return ents,payload_key,(q1[3],q1[1],q1[2]),(q2[3],q2[1],q2[2])

def episode_fingerprint(episodes: Iterable[tuple]) -> str:
    buf=io.BytesIO()
    serial=[]
    for ep in episodes:
        obs,q1,q2,payload=ep
        serial.append({
            "entities":[int(r[0]) for r in obs],
            "payload":{str(k):list(map(int,v)) for k,v in sorted(payload.items())},
            "q1":list(q1),"q2":list(q2),
            "text":[r[1].tolist() for r in obs],
            "image":[r[2].tolist() for r in obs],
            "audio":[r[3].tolist() for r in obs],
        })
    torch.save(serial,buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()

def generator_hash() -> str:
    return hashlib.sha256(__file_content__().encode("utf-8")).hexdigest()

def __file_content__() -> str:
    with open(__file__,"r",encoding="utf-8") as h:
        return h.read()

def benchmark_manifest():
    base_path = __import__("pathlib").Path(__file__).resolve().parent / "integrated_e2e_005_benchmark.py"
    # The dependency file is separately recorded by the runner; this manifest
    # hashes the fresh selector/validation wrapper itself.
    return {
        "generator_version":GENERATOR_VERSION,
        "generator_sha256":generator_hash(),
        "base_generator_sha256":hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "sealed_e2e005_count":len(SEALED_E2E005),
        "development_count":len(DEV_COMBOS),
        "heldout_count":len(HELDOUT),
        "train_combos_count":len(TRAIN_COMBOS),
        "heldout":[list(x) for x in HELDOUT],
        "dev_combos":[list(x) for x in DEV_COMBOS],
        "sealed_e2e005":[list(x) for x in SEALED_E2E005],
    }
