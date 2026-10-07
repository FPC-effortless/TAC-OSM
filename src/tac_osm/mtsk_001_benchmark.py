"""MTSK-001 controlled multimodal temporal-retention benchmark."""
from __future__ import annotations

import hashlib
import io
import math
import random
from collections import Counter
from typing import Iterable

import torch
from torch import Tensor

BITS = 12
ENTITY_COUNT = 1
ENTITY = 0
SLOW_BIT = 0
FAST_BIT = 1
ANCHOR_BIT = 11
MASK_TOKEN = 39
TRAIN_DELAY = 8
EVAL_DELAYS = (1, 2, 4, 8, 16, 32)
EVAL_EPISODES_PER_DELAY = 200
CELLS_PER_DELAY = 50
NOISE = 0.03
GENERATOR_VERSION = "mtsk-001-dual-timescale-overwrite-v2"


def _noise(shape: tuple[int, ...], rng: random.Random) -> Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(rng.randrange(2**31))
    return torch.randn(shape, generator=g)


def make_modalities(bits: list[int], rng: random.Random, *, slow_masked: bool) -> tuple[Tensor, Tensor, Tensor]:
    text = torch.tensor([2, 3, 20 + bits[0], 20 + bits[1], 20 + bits[2], 20 + bits[3]], dtype=torch.long)
    if slow_masked:
        text[2] = MASK_TOKEN

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
        audio[seg] = (1 if bit else -1) * torch.sin(2 * math.pi * (idx + 2) * t[seg])
    audio += NOISE * _noise((96,), rng)
    return text, image, audio


def sample_sequence(
    rng: random.Random, delay: int, *, slow: int | None = None, final_fast: int | None = None
) -> dict:
    slow_value = rng.randrange(2) if slow is None else int(slow)
    observations = []
    for step in range(delay + 1):
        bits = [rng.randrange(2) for _ in range(BITS)]
        bits[SLOW_BIT] = slow_value
        if step == delay and final_fast is not None:
            bits[FAST_BIT] = int(final_fast)
        masked = step > 0
        text, image, audio = make_modalities(bits, rng, slow_masked=masked)
        observations.append({
            "step": step,
            "text": text,
            "image": image,
            "audio": audio,
            "slow": slow_value,
            "fast": int(bits[FAST_BIT]),
            "slow_masked": masked,
        })
    return {
        "delay": delay,
        "observations": observations,
        "slow_target": slow_value,
        "fast_target": observations[-1]["fast"],
    }


def validate_sequence(sequence: dict) -> None:
    obs = sequence["observations"]
    assert len(obs) == sequence["delay"] + 1
    assert int(obs[0]["text"][2]) == 20 + sequence["slow_target"]
    assert int(obs[0]["text"][3]) == 20 + obs[0]["fast"]
    for row in obs[1:]:
        assert int(row["text"][2]) == MASK_TOKEN
        assert int(row["text"][3]) == 20 + row["fast"]


def sequence_key(sequence: dict) -> str:
    serial = {
        "delay": int(sequence["delay"]),
        "slow_target": int(sequence["slow_target"]),
        "fast_target": int(sequence["fast_target"]),
        "observations": [
            {"text": row["text"].tolist(), "image": row["image"].tolist(), "audio": row["audio"].tolist()}
            for row in sequence["observations"]
        ],
    }
    buf = io.BytesIO()
    torch.save(serial, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def sample_evaluation_episodes(rng: random.Random) -> list[dict]:
    episodes = []
    for delay in EVAL_DELAYS:
        for slow in (0, 1):
            for fast in (0, 1):
                for _ in range(CELLS_PER_DELAY):
                    ep = sample_sequence(rng, delay, slow=slow, final_fast=fast)
                    validate_sequence(ep)
                    episodes.append(ep)
    rng.shuffle(episodes)
    assert len(episodes) == len(EVAL_DELAYS) * EVAL_EPISODES_PER_DELAY
    return episodes


def target_balance(episodes: Iterable[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for delay in EVAL_DELAYS:
        subset = [ep for ep in episodes if ep["delay"] == delay]
        counts = Counter((int(ep["slow_target"]), int(ep["fast_target"])) for ep in subset)
        out[str(delay)] = {f"{a}{b}": int(counts[(a, b)]) for a in (0, 1) for b in (0, 1)}
    return out


def episode_fingerprint(episodes: Iterable[dict]) -> str:
    serial = []
    for ep in episodes:
        serial.append({
            "delay": int(ep["delay"]),
            "slow_target": int(ep["slow_target"]),
            "fast_target": int(ep["fast_target"]),
            "observations": [
                {"text": row["text"].tolist(), "image": row["image"].tolist(), "audio": row["audio"].tolist()}
                for row in ep["observations"]
            ],
        })
    buf = io.BytesIO()
    torch.save(serial, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def generator_hash() -> str:
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def benchmark_manifest() -> dict:
    return {
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": generator_hash(),
        "entity_count": ENTITY_COUNT,
        "entity": ENTITY,
        "slow_bit": SLOW_BIT,
        "fast_bit": FAST_BIT,
        "anchor_bit": ANCHOR_BIT,
        "mask_token": MASK_TOKEN,
        "train_delay": TRAIN_DELAY,
        "eval_delays": list(EVAL_DELAYS),
        "eval_episodes_per_delay": EVAL_EPISODES_PER_DELAY,
        "cells_per_delay": CELLS_PER_DELAY,
        "slow_source": "text_only",
        "slow_masked_post_t0": True,
        "image_audio_exclude_slow_fast_bits": True,
    }
