from __future__ import annotations

import json
from pathlib import Path

import torch

from tac_osm.learned_address_002_benchmark import (
    ADDRESS_DIM,
    STRUCTURED_DIAGONAL,
    generator_sha256,
    make_trial_batch,
)


def test_benchmark_hash_matches_contract_pin():
    contract = json.loads(
        Path("contracts/TACOSM-PLM-LEARNED-ADDRESS-002.json").read_text()
    )
    assert contract["benchmark_generator_sha256"] == generator_sha256()


def test_unit_keys_and_queries_are_normalized():
    gen = torch.Generator().manual_seed(123)
    keys, query, target, payload = make_trial_batch(
        gen, batch=32, memory=8, sigma=0.2, structured=True
    )
    assert keys.shape == (32, 8, ADDRESS_DIM)
    assert query.shape == (32, ADDRESS_DIM)
    assert target.shape == (32,)
    assert payload.shape == (32, 8, 8)
    assert torch.allclose(keys.norm(dim=-1), torch.ones(32, 8), atol=1e-5)
    assert torch.allclose(query.norm(dim=-1), torch.ones(32), atol=1e-5)


def test_structured_channel_is_exactly_registered():
    assert STRUCTURED_DIAGONAL.tolist() == [2.0] * 8 + [0.5] * 8


def test_trial_generation_is_deterministic_for_a_frozen_seed():
    def sample(seed: int):
        gen = torch.Generator().manual_seed(seed)
        return make_trial_batch(
            gen, batch=16, memory=32, sigma=0.4, structured=True
        )

    a = sample(99)
    b = sample(99)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_payload_is_not_a_scorer_input():
    gen = torch.Generator().manual_seed(7)
    keys, query, target, payload = make_trial_batch(
        gen, batch=8, memory=3, sigma=0.2, structured=True
    )
    assert payload.requires_grad is False
    raw = torch.einsum("bd,bmd->bm", query, keys)
    assert raw.shape == (8, 3)
    assert target.max().item() < 3
