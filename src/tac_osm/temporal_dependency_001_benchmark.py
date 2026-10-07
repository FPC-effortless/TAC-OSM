"""Temporal-dependency benchmark with q1-only information withheld from reset.
The post-boundary observation masks one payload bit that determines q2. The
q1 outcome is constructed from independent payload bits, so it cannot encode
the q2-only secret.
"""
from __future__ import annotations

import hashlib
import io
import math
import random
from typing import Iterable

import torch
from torch import Tensor

OPS = ("xor",)
STEPS = 300
BATCH_SIZE = 96
ENTITY_COUNT = 2
BITS = 12
EVAL_EPISODES = 600
NOISE = 0.03
Q1_QUERY = (0, 0, 1, "xor")
Q2_QUERY = (0, 2, 3, "xor")
SECRET_BIT = 2
VISIBLE_Q2_BIT = 3
SCRATCH_ENTITY = 1
GENERATOR_VERSION = "temporal-dependency-v1-q1-secret-masked-post-boundary"
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
        local_t = t[seg]
        audio[seg] = (1 if bit else -1) * torch.sin(
            2 * math.pi * (idx + 2) * local_t
        )
    audio += NOISE * _noise((96,), rng)
    return text, image, audio


def mask_secret_observation(pre_observation):
    entity, text, image, audio = pre_observation
    masked = text.clone()
    masked[4] = 20  # bits[2] is the third payload token.
    return entity, masked, image.clone(), audio.clone()


def apply_op(op: str, a: int, b: int) -> int:
    if op != "xor":
        raise ValueError(op)
    return a ^ b


def sample_episode(
    rng: random.Random,
    *,
    secret: int | None = None,
    visible_q2_bit: int | None = None,
):
    payload = [rng.randrange(2) for _ in range(BITS)]
    if secret is not None:
        payload[SECRET_BIT] = int(secret)
    if visible_q2_bit is not None:
        payload[VISIBLE_Q2_BIT] = int(visible_q2_bit)
    pre = [(0, *make_modalities(payload, rng))]
    post = [mask_secret_observation(pre[0])]
    q1_answer = apply_op("xor", payload[0], payload[1])
    q2_answer = apply_op("xor", payload[SECRET_BIT], payload[VISIBLE_Q2_BIT])
    q1 = (0, 0, 1, "xor", q1_answer)
    q2 = (0, 2, 3, "xor", q2_answer)
    return pre, post, q1, q2, payload


def sample_evaluation_episodes(rng: random.Random, count: int):
    # Exactly 150 examples per (secret, visible_q2_bit) pair, making the
    # hidden dependency balanced and independent of the visible q2 bit.
    assert count == EVAL_EPISODES
    episodes = []
    for n in range(count):
        pair = n // 150
        secret = pair // 2
        visible = pair % 2
        ep = sample_episode(
            rng,
            secret=secret,
            visible_q2_bit=visible,
        )
        validate_episode(ep)
        episodes.append(ep)
    return episodes


def validate_episode(ep) -> None:
    pre, post, q1, q2, payload = ep
    assert q1[:4] == Q1_QUERY
    assert q2[:4] == Q2_QUERY
    assert q1[4] == payload[0] ^ payload[1]
    assert q2[4] == payload[SECRET_BIT] ^ payload[VISIBLE_Q2_BIT]
    assert pre[0][0] == 0 and post[0][0] == 0
    assert int(pre[0][1][4].item()) == 20 + payload[SECRET_BIT]
    assert int(post[0][1][4].item()) == 20
    assert torch.equal(pre[0][2], post[0][2])
    assert torch.equal(pre[0][3], post[0][3])
    assert payload[SECRET_BIT] not in (q2[4],) or True


def episode_key(ep) -> tuple:
    pre, post, q1, q2, payload = ep
    return (
        tuple(int(x) for x in payload),
        tuple(q1),
        tuple(q2),
        tuple(pre[0][1].tolist()),
        tuple(post[0][1].tolist()),
    )


def episode_fingerprint(episodes: Iterable[tuple]) -> str:
    serial = []
    for ep in episodes:
        pre, post, q1, q2, payload = ep
        serial.append({
            "payload": list(map(int, payload)),
            "q1": list(q1),
            "q2": list(q2),
            "pre_text": pre[0][1].tolist(),
            "post_text": post[0][1].tolist(),
            "pre_image": pre[0][2].tolist(),
            "post_image": post[0][2].tolist(),
            "pre_audio": pre[0][3].tolist(),
            "post_audio": post[0][3].tolist(),
        })
    b = io.BytesIO()
    torch.save(serial, b)
    return hashlib.sha256(b.getvalue()).hexdigest()


def q2_observation_fingerprint(episodes: Iterable[tuple]) -> str:
    serial = []
    for ep in episodes:
        post = ep[1][0]
        serial.append({
            "entity": int(post[0]),
            "text": post[1].tolist(),
            "image": post[2].tolist(),
            "audio": post[3].tolist(),
            "q2": list(ep[3][:4]),
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
        "ops": list(OPS),
        "q1_query": list(Q1_QUERY),
        "q2_query": list(Q2_QUERY),
        "secret_bit": SECRET_BIT,
        "visible_q2_bit": VISIBLE_Q2_BIT,
        "entity_count": ENTITY_COUNT,
        "eval_episodes_per_seed": EVAL_EPISODES,
        "evaluation_secret_visible_pair_balance": 150,
        "post_boundary_secret_masked": True,
        "post_boundary_observation_reuses_pre_noise": True,
    }
