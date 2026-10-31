"""Regression fixtures for configuration ownership and long-clock RoPE math."""
from dataclasses import replace

import pytest
import torch

from research.langcache104.model import Config104, DeepCacheTAC104, State104


def candidate(**changes):
    config = Config104(d_model=8, n_layers=1, n_heads=1, key_dim=4,
                       value_dim=4, write_rank=1, ffn_dim=12,
                       window=3, chunk_size=2, **changes)
    return DeepCacheTAC104(config).eval()


def test_same_shape_different_cache_semantics_reject_carried_state():
    unrotated = candidate(local_rope=False)
    rotated = candidate(local_rope=True)
    observations = torch.tensor([[1, 2]], dtype=torch.long)
    _, state = unrotated(observations, owners=("owner",))
    assert state.config_signature != rotated.config_signature
    with pytest.raises(ValueError, match="configuration"):
        rotated(observations, state=state, owners=("owner",))


def test_config_signature_v2_roundtrip_detach_and_reject_malformed_payload():
    model = candidate()
    _, state = model(torch.tensor([[1, 2]]), owners=("owner",))
    payload = state.payload()
    assert payload["schema"] == "tac-osm.langcache104-state.v2"
    restored = State104.from_payload(payload)
    assert restored.config_signature == state.config_signature == state.detach().config_signature
    assert len(restored.config_signature) == 64
    missing = dict(payload); missing.pop("config_signature")
    with pytest.raises(ValueError):
        State104.from_payload(missing)
    for signature in [None, "BAD", "F"*64, "g"*64]:
        with pytest.raises(ValueError):
            State104.from_payload({**payload, "config_signature": signature})
    with pytest.raises(ValueError, match="schema"):
        State104.from_payload({**payload, "schema": "tac-osm.langcache104-state.v1"})
    with pytest.raises(ValueError, match="configuration"):
        model(torch.tensor([[3]]), state=replace(restored, config_signature="0"*64), owners=("owner",))


def test_long_clock_fp32_local_rope_keeps_adjacent_positions_distinct():
    model = candidate()
    block, config = model.blocks[0], model.config
    torch.manual_seed(104)
    source = torch.randn(1, 1, config.d_model).expand(1, 2, -1)
    initial = model.start(("owner",)).layers[0]
    clock = 2**24
    _, keys, _ = block.local_attention(source, initial.recent_key, initial.recent_value, clock)
    unrotated = block.local_key(source).reshape(1, 2, config.n_heads, config.key_dim).transpose(1, 2)
    positions = torch.tensor([clock, clock+1], dtype=torch.float64)
    angle = positions[:, None] * block.local_frequencies.double()[None, :]
    cosine, sine = angle.cos().float()[None, None], angle.sin().float()[None, None]
    x, y = unrotated[..., ::2], unrotated[..., 1::2]
    expected = torch.stack((cosine*x-sine*y, sine*x+cosine*y), -1).flatten(-2)
    torch.testing.assert_close(keys, expected, rtol=0, atol=0)
    assert not torch.equal(keys[:, :, 0], keys[:, :, 1])


def test_long_clock_chunk_step_parity_and_out_of_range_fail_closed():
    torch.manual_seed(104)
    model = candidate()
    owners = ("owner",)
    _, state = model(torch.tensor([[1, 2]]), owners=owners)
    state = replace(state, clock=2**24+1)
    source = torch.tensor([[3, 4, 5]])
    full, full_state = model(source, state=state, owners=owners)
    pieces = []
    for token in source.split(1, 1):
        out, state = model.step(token, state=state, owners=owners)
        pieces.append(out)
    torch.testing.assert_close(full, torch.cat(pieces, 1), rtol=3e-5, atol=3e-5)
    for key, value in full_state.layers[0].tensors().items():
        torch.testing.assert_close(value, state.layers[0].tensors()[key], rtol=3e-5, atol=3e-5)
    with pytest.raises(ValueError, match="exact float64 integer range"):
        model(torch.tensor([[6]]), state=replace(state, clock=2**53), owners=owners)
