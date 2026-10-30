"""ARCHFUSE089 — constructed no-fit memory-operator falsifiers, not a benchmark."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "research/mathematical-engineering/archfuse089/operators089.py"
spec = importlib.util.spec_from_file_location("operators089", FILE)
op = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = op
spec.loader.exec_module(op)


def test_kda_reduces_to_native_oriented_scalar_gate():
    torch.manual_seed(89)
    m = torch.randn(8, 8, dtype=torch.float64)
    k = torch.nn.functional.normalize(torch.randn(8, dtype=m.dtype), dim=0)
    v = torch.randn(8, dtype=m.dtype)
    rho = torch.tensor(.997, dtype=m.dtype)
    eta = torch.tensor(.13, dtype=m.dtype)
    base = rho * m
    native = base + eta * (v - base @ k).unsqueeze(-1) * k.unsqueeze(0)
    assert torch.allclose(op.channel_delta(m, k, v, torch.full_like(k, rho), eta), native, rtol=0, atol=1e-14)
    assert torch.allclose(op.scalar_delta(m, k, v, rho, eta), native, rtol=0, atol=1e-14)


def test_kda_channel_specific_retention_and_correction():
    m = torch.eye(8, dtype=torch.float64)
    k = torch.nn.functional.normalize(torch.ones(8, dtype=m.dtype), dim=0)
    alpha = torch.tensor([1., 0., 1., 0., 1., 0., 1., 0.], dtype=m.dtype)
    result = op.channel_delta(m, k, torch.zeros(8, dtype=m.dtype), alpha, torch.tensor(0., dtype=m.dtype))
    assert torch.equal(result, torch.diag(alpha))
    after = op.channel_delta(m, k, torch.zeros(8, dtype=m.dtype), alpha, torch.tensor(.25, dtype=m.dtype))
    before = torch.linalg.vector_norm((m * alpha) @ k)
    assert torch.linalg.vector_norm(after @ k) < before


def test_kda_gradient_and_invalid_gate_are_detected():
    m = torch.randn(8, 8, requires_grad=True)
    k = torch.randn(8)
    v = torch.randn(8)
    alpha = torch.nn.Parameter(torch.full((8,), .95))
    eta = torch.nn.Parameter(torch.tensor(.3))
    op.channel_delta(m, k, v, alpha, eta).square().mean().backward()
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in (m, alpha, eta))
    with pytest.raises(ValueError):
        op.channel_delta(m, k, v, torch.ones(8) * 1.01, eta)


def test_rotating_slow_state_preserves_norm_without_decay():
    x = torch.tensor([[1., 0.], [0., 1.]])
    y = op.rotation_ema(x, torch.zeros_like(x), torch.ones(2), torch.tensor([.4, -.6]))
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-7)
    assert y.numel() == x.numel()
    assert torch.allclose(op.rotation_ema(x, torch.zeros_like(x), torch.zeros(2),
                                          torch.tensor([.4, -.6])), torch.zeros_like(x))


def test_titans_local_update_matches_autograd_negative_gradient():
    torch.manual_seed(8)
    m = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    k = torch.randn(3, dtype=m.dtype)
    v = torch.randn(5, dtype=m.dtype)
    loss = .5 * ((m @ k - v) ** 2).sum()
    grad = torch.autograd.grad(loss, m)[0]
    zero = torch.zeros_like(m)
    new, momentum = op.titans_surprise_step(m.detach(), zero, k, v, .07, 0., 0.)
    assert torch.allclose(new, m.detach() - .07 * grad, atol=1e-13, rtol=0)
    assert torch.allclose(momentum, new - m.detach())


def test_mlstm_running_state_matches_weighted_history():
    torch.manual_seed(11)
    keys = torch.randn(5, 3)
    values = torch.randn(5, 4)
    state = torch.zeros(4, 3)
    norm = torch.zeros(3)
    for i in range(len(keys)):
        state, norm = op.mlstm_step(state, norm, keys[i], values[i], torch.tensor(.8), torch.tensor(.2))
    explicit = sum((.2 * .8 ** (4-i)) * values[i].unsqueeze(-1) * keys[i].unsqueeze(0) for i in range(5))
    assert torch.allclose(state, explicit, atol=2e-7)
    assert torch.isfinite(op.mlstm_read(state, norm, torch.ones(3))).all()


def test_lightning_outer_product_chunk_associativity():
    torch.manual_seed(12)
    keys = torch.randn(18, 3, dtype=torch.float64)
    values = torch.randn(18, 4, dtype=torch.float64)
    state = torch.zeros(4, 3, dtype=torch.float64)
    for k, v in zip(keys, values):
        state = op.linear_attention_step(state, k, v)
    chunk1 = values[:9].T @ keys[:9]
    chunk2 = values[9:].T @ keys[9:]
    assert torch.allclose(state, chunk1 + chunk2, atol=2e-14)
    q = torch.randn(3, dtype=torch.float64)
    assert torch.allclose(state @ q, (chunk1 + chunk2) @ q, atol=3e-14)


def test_ring_is_bounded_exact_owner_isolated_and_prequential():
    r = op.ExactRing("owner-A", 7)
    for byte in (1, 2, 3, 4, 1, 2, 3):
        r = r.append("owner-A", byte)
    assert r.continuations((1, 2)) == (3, 3)
    assert r.continuations((2, 3)) == (4,)
    assert len(r.bytes_seen) == 7
    with pytest.raises(ValueError):
        r.append("owner-B", 99)
    for byte in (1, 2, 9):
        r = r.append("owner-A", byte)
    assert r.continuations((1, 2)) == (3, 9)
    assert len(r.bytes_seen) == 7


def test_bounded_attention_cannot_see_poisoned_future_tokens():
    torch.manual_seed(18)
    k = torch.randn(12, 4)
    v = torch.randn(12, 6)
    q = torch.randn(4)
    a = op.bounded_attention(k[:7], v[:7], q, 4)
    kp, vp = k.clone(), v.clone()
    kp[7:] += 1000
    vp[7:] -= 999
    b = op.bounded_attention(kp[:7], vp[:7], q, 4)
    assert torch.equal(a, b)
    assert op.bounded_attention(k, v, q, 4).shape == (6,)


def test_selective_verification_target_blind_budgeted_ties():
    a = torch.tensor([[.9, .1], [.9, .1], [.5, .5], [.5, .5]])
    b = torch.tensor([[.1, .9], [.9, .1], [.5, .5], [.5, .5]])
    top = op.selective_indices(a, b, 2)
    assert top[0] == 0 and len(top) == 2 and len(set(top)) == 2
    assert op.selective_indices(a, b, 0) == ()
    assert op.selective_indices(a, b, 4) == (0, 2, 3, 1)
    # Two confidently wrong estimators can agree; disagreement is not truth.
    wrong = torch.tensor([[.99, .01]])
    assert op.selective_indices(wrong, wrong, 1) == (0,)
    with pytest.raises(ValueError):
        op.selective_indices(a, b, 5)


def test_ternary_weight_codes_and_ideal_packing_only():
    w = torch.tensor([-.91, -.2, 0., .1, .8])
    codes, scale = op.ternary_absmean(w)
    assert set(codes.tolist()).issubset({-1, 0, 1}) and float(scale) > 0
    assert op.ternary_ideal_min_bytes(100000) == 19813
    assert op.ternary_ideal_min_bytes(0) == 0
    with pytest.raises(ValueError):
        op.ternary_absmean(torch.tensor([float("nan")]))


def test_original_langmem086_untouched_graph_and_state_bytes():
    sys.path.insert(0, str(ROOT / "research/mathematical-engineering/langmem086"))
    pytest.importorskip("compact_tac", reason="Only available in full TAC-OSM checkout")
    from compact_tac import CompactDualClippedTAC086
    m = CompactDualClippedTAC086().eval()
    assert m.counts()["trainable"] == 100521
    assert m.start(("owner-A",)).tensor_bytes() == 1920
    with torch.no_grad():
        output, state = m(torch.tensor([[4, 8, 16]]), owners=("owner-A",))
    assert output.shape == (1, 3, 256) and state.clock == 3
