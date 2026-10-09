"""Pure persistence benchmark: q1 cannot access the q2-time secret entity.
The secret is carried in entity 1, while q1 acts only on entity 0. After the
q1 boundary the q2 entity's observable payload is masked. Carry retains entity
1's pre-boundary state; fresh reconstructs entity 1 only from the masked post
observation.
"""
from __future__ import annotations

import hashlib
import io
import math
import random
from typing import Iterable

import torch
from torch import Tensor

BITS = 12
ENTITY_COUNT = 3
Q1_ENTITY = 0
Q2_ENTITY = 1
SCRATCH_ENTITY = 2
SECRET_BIT = 2
VISIBLE_Q2_BIT = 3
Q1_QUERY = (Q1_ENTITY, 0, 1, "xor")
Q2_QUERY = (Q2_ENTITY, SECRET_BIT, VISIBLE_Q2_BIT, "xor")
EVAL_EPISODES = 600
NOISE = 0.03
GENERATOR_VERSION = "temporal-persistence-v2-separated-secret-entity"
BASE_E2E008_GENERATOR_SHA256 = "3be32a91191b73792b0fffb5a7b497e80aa6a46d0189e032e6b7c60082949f3e"


def _noise(shape: tuple[int, ...], rng: random.Random) -> Tensor:
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
        audio[seg] = (1 if bit else -1) * torch.sin(
            2 * math.pi * (idx + 2) * t[seg]
        )
    audio += NOISE * _noise((96,), rng)
    return text, image, audio


def apply_op(a: int, b: int) -> int:
    return a ^ b


def mask_secret(observation):
    entity, text, image, audio = observation
    masked = text.clone()
    masked[4] = 20
    return entity, masked, image.clone(), audio.clone()


def sample_episode(
    rng: random.Random,
    *,
    secret: int | None = None,
    visible_q2_bit: int | None = None,
):
    payload = {
        Q1_ENTITY: [rng.randrange(2) for _ in range(BITS)],
        Q2_ENTITY: [rng.randrange(2) for _ in range(BITS)],
    }
    if secret is not None:
        payload[Q2_ENTITY][SECRET_BIT] = int(secret)
    if visible_q2_bit is not None:
        payload[Q2_ENTITY][VISIBLE_Q2_BIT] = int(visible_q2_bit)
    pre = [
        (Q1_ENTITY, *make_modalities(payload[Q1_ENTITY], rng)),
        (Q2_ENTITY, *make_modalities(payload[Q2_ENTITY], rng)),
    ]
    post = mask_secret(pre[1])
    q1 = (*Q1_QUERY, apply_op(payload[Q1_ENTITY][0], payload[Q1_ENTITY][1]))
    q2 = (*Q2_QUERY, apply_op(payload[Q2_ENTITY][SECRET_BIT], payload[Q2_ENTITY][VISIBLE_Q2_BIT]))
    return pre, post, q1, q2, payload


def sample_evaluation_episodes(rng: random.Random, count: int):
    assert count == EVAL_EPISODES
    episodes = []
    for n in range(count):
        cell = n // 150
        secret = cell // 2
        visible = cell % 2
        ep = sample_episode(rng, secret=secret, visible_q2_bit=visible)
        validate_episode(ep)
        episodes.append(ep)
    return episodes


def validate_episode(ep) -> None:
    pre, post, q1, q2, payload = ep
    assert {row[0] for row in pre} == {Q1_ENTITY, Q2_ENTITY}
    assert q1[:4] == Q1_QUERY
    assert q2[:4] == Q2_QUERY
    assert q1[4] == apply_op(payload[Q1_ENTITY][0], payload[Q1_ENTITY][1])
    assert q2[4] == apply_op(payload[Q2_ENTITY][SECRET_BIT], payload[Q2_ENTITY][VISIBLE_Q2_BIT])
    q2_pre = next(row for row in pre if row[0] == Q2_ENTITY)
    assert int(q2_pre[1][4]) == 20 + payload[Q2_ENTITY][SECRET_BIT]
    assert int(post[1][4]) == 20
    assert torch.equal(q2_pre[1][:4], post[1][:4])
    assert torch.equal(q2_pre[1][5:], post[1][5:])
    assert torch.equal(q2_pre[2], post[2])
    assert torch.equal(q2_pre[3], post[3])


def episode_key(ep) -> tuple:
    pre, post, q1, q2, payload = ep
    return (
        tuple((int(e), tuple(int(x) for x in payload[e])) for e in sorted(payload)),
        tuple(q1),
        tuple(q2),
        tuple(post[1].tolist()),
        tuple(post[2].flatten().tolist()),
        tuple(post[3].tolist()),
    )


def episode_fingerprint(episodes: Iterable[tuple]) -> str:
    serial = []
    for ep in episodes:
        pre, post, q1, q2, payload = ep
        serial.append({
            "payload": {str(k): list(map(int, v)) for k, v in sorted(payload.items())},
            "q1": list(q1),
            "q2": list(q2),
            "pre": [
                {"entity": int(e), "text": t.tolist(), "image": im.tolist(), "audio": au.tolist()}
                for e, t, im, au in pre
            ],
            "post": {"entity": int(post[0]), "text": post[1].tolist(), "image": post[2].tolist(), "audio": post[3].tolist()},
        })
    b = io.BytesIO()
    torch.save(serial, b)
    return hashlib.sha256(b.getvalue()).hexdigest()


def q2_surface_fingerprint(episodes: Iterable[tuple]) -> str:
    serial = []
    for ep in episodes:
        post = ep[1]
        q2 = ep[3]
        serial.append({
            "entity": int(post[0]),
            "text": post[1].tolist(),
            "image": post[2].tolist(),
            "audio": post[3].tolist(),
            "q2": list(q2[:4]),
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
        "base_e2e008_generator_sha256": BASE_E2E008_GENERATOR_SHA256,
        "entity_count": ENTITY_COUNT,
        "q1_entity": Q1_ENTITY,
        "q2_entity": Q2_ENTITY,
        "scratch_entity": SCRATCH_ENTITY,
        "secret_bit": SECRET_BIT,
        "visible_q2_bit": VISIBLE_Q2_BIT,
        "evaluation_secret_visible_pair_balance": 150,
        "q1_cannot_read_q2_entity_by_construction": True,
        "post_boundary_secret_masked": True,
    }
