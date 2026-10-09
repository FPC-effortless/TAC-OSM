from __future__ import annotations

import hashlib
import io
import math
import random
from dataclasses import dataclass

import torch

BITS = 1
DELAYS = (0, 1, 4, 8)
TEXT_LEN = 4
IMAGE_SIZE = 8
AUDIO_LEN = 32
NOISE = 0.02


@dataclass(frozen=True)
class PersistenceEpisode:
    t0_text: torch.Tensor
    t0_image: torch.Tensor
    t0_audio: torch.Tensor
    delayed_text: tuple[torch.Tensor, ...]
    delayed_image: tuple[torch.Tensor, ...]
    delayed_audio: tuple[torch.Tensor, ...]
    current_text: torch.Tensor
    current_image: torch.Tensor
    current_audio: torch.Tensor
    memory_bit: int
    current_bit: int
    target: int


def _noise(shape: tuple[int, ...], rng: random.Random) -> torch.Tensor:
    g = torch.Generator().manual_seed(rng.randrange(2**31))
    return torch.randn(shape, generator=g)


def encode_modalities(bit: int, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text = torch.tensor([2, 11 + int(bit), 3, 7], dtype=torch.long)
    image = torch.full((1, IMAGE_SIZE, IMAGE_SIZE), float(bit))
    image = image + NOISE * _noise(image.shape, rng)
    image.clamp_(0.0, 1.0)
    t = torch.linspace(0, 1, AUDIO_LEN)
    audio = (2 * bit - 1) * torch.sin(2 * math.pi * 3 * t)
    audio = audio + NOISE * _noise(audio.shape, rng)
    return text, image, audio


def fixed_current_modalities(bit: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(900_000 + bit)
    return encode_modalities(bit, rng)


def make_episode(rng: random.Random, delay: int) -> PersistenceEpisode:
    memory_bit = rng.randrange(2)
    current_bit = rng.randrange(2)
    t0 = encode_modalities(memory_bit, rng)
    delayed = tuple(
        (
            torch.zeros(TEXT_LEN, dtype=torch.long),
            torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE),
            torch.zeros(AUDIO_LEN),
        )
        for _ in range(delay)
    )
    current = encode_modalities(current_bit, rng)
    return PersistenceEpisode(
        t0_text=t0[0],
        t0_image=t0[1],
        t0_audio=t0[2],
        delayed_text=tuple(x[0] for x in delayed),
        delayed_image=tuple(x[1] for x in delayed),
        delayed_audio=tuple(x[2] for x in delayed),
        current_text=current[0],
        current_image=current[1],
        current_audio=current[2],
        memory_bit=memory_bit,
        current_bit=current_bit,
        target=memory_bit ^ current_bit,
    )


def make_collision_pair(rng: random.Random, current_bit: int = 0):
    current_rng = random.Random(rng.randrange(2**31))
    current = encode_modalities(current_bit, current_rng)
    episodes = []
    for memory_bit in (0, 1):
        local = random.Random(rng.randrange(2**31))
        t0 = encode_modalities(memory_bit, local)
        target = memory_bit ^ current_bit
        episodes.append(
            PersistenceEpisode(
                t0_text=t0[0],
                t0_image=t0[1],
                t0_audio=t0[2],
                delayed_text=(),
                delayed_image=(),
                delayed_audio=(),
                current_text=current[0].clone(),
                current_image=current[1].clone(),
                current_audio=current[2].clone(),
                memory_bit=memory_bit,
                current_bit=current_bit,
                target=target,
            )
        )
    return tuple(episodes)


def sample_evaluation_pool(rng: random.Random, count_per_delay: int):
    out = {}
    for delay in DELAYS:
        out[delay] = [make_episode(rng, delay) for _ in range(count_per_delay)]
    return out


def paired_current_ceiling(pool: list[PersistenceEpisode]) -> float:
    # For each fixed current bit, m is balanced. Therefore any predictor that
    # observes only current modalities has conditional target accuracy 0.5.
    return 0.5


def episode_bytes(ep: PersistenceEpisode) -> bytes:
    b = io.BytesIO()
    torch.save(ep, b)
    return b.getvalue()


def pool_fingerprint(pool: list[PersistenceEpisode]) -> str:
    h = hashlib.sha256()
    for ep in pool:
        h.update(episode_bytes(ep))
    return h.hexdigest()


def collision_fingerprint(pair: tuple[PersistenceEpisode, PersistenceEpisode]) -> str:
    h = hashlib.sha256()
    for ep in pair:
        h.update(ep.current_text.numpy().tobytes())
        h.update(ep.current_image.numpy().tobytes())
        h.update(ep.current_audio.numpy().tobytes())
    return h.hexdigest()


def benchmark_manifest() -> dict:
    return {
        "delays": list(DELAYS),
        "text_len": TEXT_LEN,
        "image_size": IMAGE_SIZE,
        "audio_len": AUDIO_LEN,
        "current_observation_has_memory_bit": False,
        "paired_current_observations_identical": True,
        "no_memory_conditional_ceiling": 0.5,
        "target_rule": "memory_bit XOR current_bit",
        "memory_and_current_balanced": True,
    }
