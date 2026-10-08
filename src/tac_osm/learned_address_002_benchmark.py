"""Dedicated generator for TACOSM-PLM-LEARNED-ADDRESS-002.

The only information used by the learned scorer is a query key and a set of
stored unit keys. Payloads are generated independently and never enter the
scorer. Structured queries use a fixed anisotropic channel that is part of the
registered corruption model, not a task-specific semantic label.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterator

import torch
import torch.nn.functional as F

ADDRESS_DIM = 16
MEMORY_SIZES = (3, 8, 16, 32, 64, 128, 256)
ISOTROPIC_SIGMAS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)
STRUCTURED_SIGMAS = (0.0, 0.05, 0.1, 0.2, 0.4)
STRUCTURED_HIGH_NOISE_M = 32
STRUCTURED_HIGH_NOISE_SIGMAS = (0.2, 0.4)
TRIALS = 10_000
STRUCTURED_DIAGONAL = torch.tensor([2.0] * 8 + [0.5] * 8)


def unit_keys(generator: torch.Generator, batch: int, memory: int) -> torch.Tensor:
    keys = torch.randn(batch, memory, ADDRESS_DIM, generator=generator)
    return F.normalize(keys, dim=-1)


def balanced_targets(generator: torch.Generator, batch: int, memory: int) -> torch.Tensor:
    base = torch.arange(batch, dtype=torch.long) % memory
    return base[torch.randperm(batch, generator=generator)]


def make_trial_batch(
    generator: torch.Generator,
    *,
    batch: int,
    memory: int,
    sigma: float,
    structured: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = unit_keys(generator, batch, memory)
    target = balanced_targets(generator, batch, memory)
    target_key = keys[torch.arange(batch), target]
    if structured:
        signal = target_key * STRUCTURED_DIAGONAL.to(keys.device, keys.dtype)
    else:
        signal = target_key
    query = F.normalize(
        signal + sigma * torch.randn(batch, ADDRESS_DIM, generator=generator),
        dim=-1,
    )
    # Independent payload verifies that address and stored content can be
    # generated separately. It is intentionally not consumed by the scorer.
    payload = torch.randn(batch, memory, 8, generator=generator)
    return keys, query, target, payload


def fingerprint_batch(
    keys: torch.Tensor,
    query: torch.Tensor,
    target: torch.Tensor,
    payload: torch.Tensor,
) -> str:
    h = hashlib.sha256()
    for tensor in (keys, query, target, payload):
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def generator_sha256() -> str:
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()


def benchmark_manifest() -> dict:
    return {
        "generator_version": "learned-address-002-v1",
        "generator_sha256": generator_sha256(),
        "address_dim": ADDRESS_DIM,
        "memory_sizes": list(MEMORY_SIZES),
        "isotropic_sigmas": list(ISOTROPIC_SIGMAS),
        "structured_sigmas": list(STRUCTURED_SIGMAS),
        "structured_diagonal": [2.0] * 8 + [0.5] * 8,
        "trials_per_condition": TRIALS,
        "payload_is_independent": True,
        "payload_enters_scorer": False,
        "model_observes_only_query_and_keys": True,
    }
