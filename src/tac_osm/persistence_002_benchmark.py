"""PERSISTENCE-002: balanced, paired pasts with real intervening observations.

This generator constructs model-visible observations from latent m/c/d bits.
Within each adjacent pair, only t0's bit changes; all later tensors are
byte-identical, while targets are complementary. The generated oracle labels
never enter any model method.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor

DELAYS = (1, 2, 4, 8, 16)
IMAGE_SIDE = 8
AUDIO_LEN = 32
TEXT_LEN = 4
NOISE_SD = 0.05


@dataclass(frozen=True)
class PairedBatch:
    t0: tuple[Tensor, Tensor, Tensor]
    distractors: tuple[tuple[Tensor, Tensor, Tensor], ...]
    current: tuple[Tensor, Tensor, Tensor]
    memory_bit: Tensor
    current_bit: Tensor
    target: Tensor
    delay: int

    @property
    def pair_count(self) -> int:
        return self.memory_bit.numel() // 2


def _modalities(
    bits: Tensor, phase: int, generator: torch.Generator, *, pair_noise: bool
) -> tuple[Tensor, Tensor, Tensor]:
    if bits.ndim != 1:
        raise ValueError("bits must be rank one")
    n = bits.numel()
    if pair_noise and n % 2:
        raise ValueError("paired observation requires even number of episodes")
    draw = n // 2 if pair_noise else n
    tokens = torch.stack((
        torch.full_like(bits, phase),
        11 + bits,
        torch.full_like(bits, 3),
        torch.full_like(bits, 7),
    ), dim=1)
    image_noise = torch.randn(draw, 1, IMAGE_SIDE, IMAGE_SIDE, generator=generator)
    audio_noise = torch.randn(draw, AUDIO_LEN, generator=generator)
    if pair_noise:
        image_noise = image_noise.repeat_interleave(2, dim=0)
        audio_noise = audio_noise.repeat_interleave(2, dim=0)
    image = (bits.float().view(n, 1, 1, 1) + NOISE_SD * image_noise).clamp(0, 1)
    t = torch.arange(AUDIO_LEN).float() / AUDIO_LEN
    wave = torch.sin(2 * torch.pi * 3 * t)
    audio = (2 * bits.float().unsqueeze(1) - 1) * wave + NOISE_SD * audio_noise
    return tokens, image, audio


def make_paired_batch(
    generator: torch.Generator, *, pairs: int, delay: int
) -> PairedBatch:
    if pairs < 1 or delay not in DELAYS:
        raise ValueError("pairs must be positive and delay must be registered")
    m = torch.tensor([0, 1], dtype=torch.long).repeat(pairs)
    c = torch.randint(0, 2, (pairs,), generator=generator).repeat_interleave(2)
    t0 = _modalities(m, 1, generator, pair_noise=False)
    distractors = []
    for _ in range(delay):
        d = torch.randint(0, 2, (pairs,), generator=generator).repeat_interleave(2)
        distractors.append(_modalities(d, 2, generator, pair_noise=True))
    current = _modalities(c, 3, generator, pair_noise=True)
    batch = PairedBatch(
        t0=t0,
        distractors=tuple(distractors),
        current=current,
        memory_bit=m,
        current_bit=c,
        target=m ^ c,
        delay=delay,
    )
    assert_valid_pairs(batch)
    return batch


def assert_valid_pairs(batch: PairedBatch) -> None:
    """Fail closed on identity or label leaks in every *evaluated* episode."""
    if batch.memory_bit.numel() != 2 * batch.pair_count:
        raise AssertionError("incomplete pairs")
    if batch.delay != len(batch.distractors) or batch.delay not in DELAYS:
        raise AssertionError("incorrect intervening step count")
    if not bool((batch.memory_bit[::2] == 0).all()):
        raise AssertionError("left history must have m=0")
    if not bool((batch.memory_bit[1::2] == 1).all()):
        raise AssertionError("right history must have m=1")
    if not torch.equal(batch.current_bit[::2], batch.current_bit[1::2]):
        raise AssertionError("current bit differs across pair")
    if not torch.equal(batch.target, batch.memory_bit ^ batch.current_bit):
        raise AssertionError("target is not m XOR c")
    if not bool((batch.target[::2] != batch.target[1::2]).all()):
        raise AssertionError("targets must be opposite")
    if torch.equal(batch.t0[0][::2], batch.t0[0][1::2]):
        raise AssertionError("the two pasts are not different")
    for observation in (*batch.distractors, batch.current):
        for modal in observation:
            if not torch.equal(modal[::2], modal[1::2]):
                raise AssertionError("paired histories have different later inputs")
    for obs in batch.distractors:
        if not bool((obs[0][:, 0] == 2).all()):
            raise AssertionError("non-distractor observation in intervening steps")
    if not bool((batch.current[0][:, 0] == 3).all()):
        raise AssertionError("current-role marker invalid")


def pool_fingerprint(batch: PairedBatch) -> str:
    hash_ = hashlib.sha256()
    for obs in (batch.t0, *batch.distractors, batch.current):
        for tensor in obs:
            hash_.update(tensor.contiguous().numpy().tobytes())
    for tensor in (batch.memory_bit, batch.current_bit, batch.target):
        hash_.update(tensor.contiguous().numpy().tobytes())
    return hash_.hexdigest()


def benchmark_manifest() -> dict:
    return {
        "delays": list(DELAYS),
        "paired_current_and_distractors_byte_identical": True,
        "paired_pasts_opposite_memory_bits": True,
        "paired_targets_opposite": True,
        "intervening_observations": "independent random distractor bits",
        "oracle_inputs": "benchmark only",
        "label_rule": "m XOR c",
        "no_memory_pair_ceiling": 0.5,
    }
