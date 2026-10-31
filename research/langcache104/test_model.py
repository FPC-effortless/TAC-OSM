"""Actual-source differential and leakage fixtures, without optimization."""
from __future__ import annotations

from dataclasses import replace
import io
import math

import pytest
import torch

from research.langcache104.model import Config104, DeepCacheTAC104, State104
from research.langcache104.probe_adapter import adapter_for, patch_state, state_views
from tac_osm.research_probe import probe_dependencies, probe_entity_isolation, probe_memory


def small_model(dtype=torch.float64, **kwargs):
    torch.manual_seed(104)
    return DeepCacheTAC104(Config104(d_model=12, n_layers=2, n_heads=2, key_dim=4,
        value_dim=4, write_rank=2, ffn_dim=19, window=5, chunk_size=3, **kwargs)).to(dtype=dtype).eval()


def observations(length=11, batch=2):
    return torch.arange(batch*length, dtype=torch.long).reshape(batch, length) * 7 % 256


def assert_state(a, b, atol=1e-10, rtol=1e-9):
    assert a.owners == b.owners and a.clock == b.clock
    av, bv = state_views(a), state_views(b)
    assert av.keys() == bv.keys()
    for name in av:
        torch.testing.assert_close(av[name], bv[name], atol=atol, rtol=rtol, msg=name)


@pytest.mark.parametrize("length", [1, 3, 11])
def test_model_chunk_recurrent_forward_and_all_parameter_gradients(length):
    model = small_model()
    x = observations(length)
    owners = ("stream-A", "stream-B")
    actual, actual_state = model(x, owners=owners, mode="chunk")
    expected, expected_state = model(x, owners=owners, mode="recurrent")
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)
    assert_state(actual_state, expected_state)
    weight = torch.linspace(.3, 1.7, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    parameters = tuple(model.parameters())
    ga = torch.autograd.grad((actual*weight).mean(), parameters)
    gb = torch.autograd.grad((expected*weight).mean(), parameters)
    for (name, _), a, b in zip(model.named_parameters(), ga, gb):
        assert bool(torch.isfinite(a).all()), name
        torch.testing.assert_close(a, b, atol=3e-10, rtol=3e-8, msg=name)


def test_model_input_state_gradient_parity():
    model = small_model()
    owners = ("stream-A", "stream-B")
    _, state = model(observations(4), owners=owners)
    leaves = {name: tensor.detach().clone().requires_grad_()
              for name, tensor in state_views(state).items()}
    for name, tensor in leaves.items():
        state = patch_state(state, name, tensor)
    x = observations(7)
    a, _ = model(x, state=state, owners=owners, mode="chunk")
    b, _ = model(x, state=state, owners=owners, mode="recurrent")
    weight = torch.linspace(.7, 1.3, a.numel(), dtype=a.dtype).reshape_as(a)
    ga = torch.autograd.grad((a*weight).mean(), tuple(leaves.values()))
    gb = torch.autograd.grad((b*weight).mean(), tuple(leaves.values()))
    for name, x, y in zip(leaves, ga, gb):
        assert bool(torch.isfinite(x).all()), name
        assert bool((x != 0).any()), name
        torch.testing.assert_close(x, y, atol=3e-10, rtol=4e-8, msg=name)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_model_streaming_serialization_and_future_causality(dtype):
    model = small_model(dtype)
    x, owners = observations(), ("stream-A", "stream-B")
    all_logits, all_state = model(x, owners=owners)
    state, pieces = None, []
    for begin, end in [(0, 2), (2, 7), (7, 8), (8, 11)]:
        logits, state = model(x[:, begin:end], owners=owners, state=state)
        pieces.append(logits)
    tolerance = 3e-5 if dtype == torch.float32 else 3e-10
    torch.testing.assert_close(torch.cat(pieces, 1), all_logits, atol=tolerance, rtol=tolerance)
    assert_state(state, all_state, atol=tolerance, rtol=tolerance)
    file = io.BytesIO()
    torch.save(state.payload(), file)
    file.seek(0)
    restored = State104.from_payload(torch.load(file, weights_only=True))
    assert_state(restored, state, atol=0, rtol=0)
    tail = observations(1)
    a, _ = model.step(tail, state=state, owners=owners)
    b, _ = model.step(tail, state=restored, owners=owners)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    poisoned = x.clone(); poisoned[:, 7:] = (poisoned[:, 7:]+127) % 256
    altered, _ = model(poisoned, owners=owners)
    torch.testing.assert_close(all_logits[:, :7], altered[:, :7], atol=tolerance, rtol=tolerance)


def test_model_write_mask_freezes_compressed_lanes_but_not_visible_local_kv():
    model = small_model()
    owners = ("stream-A", "stream-B")
    _, initial = model(observations(3), owners=owners)
    before = {k: v.detach().clone() for k, v in state_views(initial).items()}
    _, final = model(observations(5), state=initial, owners=owners,
                     write_mask=torch.zeros(2, 5, dtype=torch.bool))
    for name, old in before.items():
        if ".recent_" not in name:
            torch.testing.assert_close(state_views(final)[name], old, rtol=0, atol=0)
        torch.testing.assert_close(state_views(initial)[name], old, rtol=0, atol=0)
    assert final.clock == initial.clock+5


def test_model_owners_precision_and_invalid_state_fail_closed():
    model = small_model()
    x, owners = observations(2), ("stream-A", "stream-B")
    _, state = model(x, owners=owners)
    for wrong in [("stream-B", "stream-A"), ("stream-A", "stream-A")]:
        with pytest.raises(ValueError):
            model(x, state=state, owners=wrong)
    with pytest.raises(ValueError): model(x, state=state)
    with pytest.raises(ValueError): model(x.float(), owners=owners)
    with pytest.raises(ValueError): model(torch.full((2, 1), 256), owners=owners)
    with pytest.raises(ValueError): model(x, state=replace(state, clock=-1), owners=owners)
    with pytest.raises(ValueError): model(x, write_mask=torch.ones_like(x))
    with pytest.raises(ValueError): model.to(torch.bfloat16)(x, owners=owners)


def test_model_probe_adapter_actual_state_isolation_and_gradients():
    model = small_model()
    sample = {"history": observations(4), "query": observations(1),
              "owners": ("stream-A", "stream-B")}
    adapter = adapter_for(model)
    assert probe_entity_isolation(model, sample, adapter, tolerance=1e-9)["status"] == "PASS"
    result = probe_dependencies(model, sample, adapter)
    for sources in result["heads"].values():
        assert all(source["connected"] and source["finite"] for source in sources.values())
    assert probe_memory(model, sample, adapter)["query_does_not_write_state"]


def test_optional_rotary_off_has_no_disconnected_parameters():
    model = small_model(rotary=False)
    logits, _ = model(observations(5))
    gradients = torch.autograd.grad(logits.square().mean(), tuple(model.parameters()))
    assert all(bool(torch.isfinite(g).all()) for g in gradients)


def test_inference_state_owns_only_bounded_tensor_storage():
    model = small_model()
    with torch.no_grad():
        _, state = model(observations(129))
    views = state_views(state)
    for name, tensor in views.items():
        assert tensor.untyped_storage().nbytes() == tensor.numel()*tensor.element_size(), name
    assert state.state_bytes() == sum(t.untyped_storage().nbytes() for t in views.values())


def test_local_attention_independent_equations_and_parameter_gradients():
    """Explicit per-query softmax oracle includes carried-cache offset and RoPE."""
    torch.manual_seed(104)
    model = small_model()
    block, config = model.blocks[0], model.config
    inputs = torch.randn(2, 7, config.d_model, dtype=torch.float64, requires_grad=True)
    old_key = torch.randn(2, config.n_heads, 4, config.key_dim, dtype=torch.float64,
                          requires_grad=True)
    old_value = torch.randn(2, config.n_heads, 4, config.value_dim, dtype=torch.float64,
                            requires_grad=True)
    actual, _, _ = block.local_attention(inputs, old_key, old_value, clock=19)
    q = (inputs @ block.local_query.weight.T).reshape(2, 7, config.n_heads, config.key_dim)
    k = (inputs @ block.local_key.weight.T).reshape(2, 7, config.n_heads, config.key_dim)
    v = (inputs @ block.local_value.weight.T).reshape(2, 7, config.n_heads, config.value_dim)
    def rotate(z):
        pieces = []
        for time in range(7):
            coordinates = []
            for pair in range(config.key_dim//2):
                # Frequencies are fixed source buffers; their finite-precision
                # values are part of source, not recomputed exact-real values.
                angle = (19+time)*block.local_frequencies[pair]
                x, y = z[:, time, :, 2*pair], z[:, time, :, 2*pair+1]
                coordinates.extend((angle.cos()*x-angle.sin()*y,
                                    angle.sin()*x+angle.cos()*y))
            pieces.append(torch.stack(coordinates, -1))
        return torch.stack(pieces, 1)
    q, k = rotate(q), rotate(k)
    keys = torch.cat((old_key.transpose(1, 2), k), 1)
    values = torch.cat((old_value.transpose(1, 2), v), 1)
    outputs = []
    for time in range(7):
        end = 4+time+1
        start = max(0, end-config.window)
        scores = (keys[:, start:end] * q[:, time, None]).sum(-1)/math.sqrt(config.key_dim)
        weights = scores.softmax(1)
        outputs.append((weights[..., None]*values[:, start:end]).sum(1))
    expected = torch.stack(outputs, 1)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)
    arguments = (inputs, old_key, old_value, block.local_query.weight,
                 block.local_key.weight, block.local_value.weight)
    weight = torch.linspace(.3, 1.7, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    ga = torch.autograd.grad((actual*weight).sum(), arguments)
    gb = torch.autograd.grad((expected*weight).sum(), arguments)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-9)
