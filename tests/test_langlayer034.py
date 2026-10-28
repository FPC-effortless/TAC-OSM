"""No-fit mechanism tests for LANGLAYER034. No authentic-data score claims."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/mathematical-engineering/langlayer034"))
from layered_transition import CausalResidualTACBlock, LayeredContextTAC034
from context_transition import ContextTransition026


def test_size_budget_and_depth():
    assert ContextTransition026().parameter_count() == 500122
    counts = {d: LayeredContextTAC034(depth=d).parameter_count() for d in (2, 3)}
    assert counts == {2: 499725, 3: 499934}
    assert max(counts.values()) / min(counts.values()) - 1 < .01
    assert abs(counts[2] - 500122) / 500122 < .01
    for d in (2, 3):
        m = LayeredContextTAC034(depth=d)
        assert m.architecture_record()["deployed_learned_parameters"] == counts[d]
        assert len(m.layernorm) == d
        assert all(p.requires_grad for p in m.parameters())
    with pytest.raises(ValueError): LayeredContextTAC034(depth=4)


@pytest.mark.parametrize("depth", [2, 3])
def test_forward_finite_and_source_prefix_causality(depth):
    torch.manual_seed(3400 + depth)
    m = LayeredContextTAC034(depth=depth).double().eval()
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [9, 8, 7, 6, 5, 4, 3, 2]])
    with torch.no_grad():
        logits, trace = m(x, return_trace=True)
        y = x.clone()
        y[:, 5:] = torch.tensor([[17, 19, 23], [37, 41, 43]])
        changed = m(y)
        unchanged_replay = m(x)
    assert logits.shape == (2, 8, 256) and torch.isfinite(logits).all()
    assert torch.allclose(logits[:, :5], changed[:, :5], atol=1e-11, rtol=1e-11)
    assert torch.equal(logits, unchanged_replay)
    assert torch.equal(trace["native_write_key"][:, 1:], trace["native_query_key"][:, :-1])
    assert not trace["native_admission"][:, 0].any()
    assert torch.equal(trace["native_memory"][:, 0], torch.zeros_like(trace["native_memory"][:, 0]))


def test_original_source_admission_is_preserved():
    torch.manual_seed(3404)
    m = LayeredContextTAC034(depth=2).double()
    x = torch.tensor([[11, 12, 13, 14, 15, 16]])
    mask = torch.tensor([[True, False, True, False, True, False]])
    _, trace = m(x, write_mask=mask, return_trace=True)
    assert not trace["native_admission"][0, 0]
    assert not trace["native_admission"][0, 1]
    assert torch.equal(trace["native_memory"][:, 1], trace["native_memory"][:, 0])
    assert torch.equal(trace["native_memory"][:, 3], trace["native_memory"][:, 2])
    assert torch.equal(trace["native_memory"][:, 5], trace["native_memory"][:, 4])
    with pytest.raises(ValueError):
        m(x, start=1)


@pytest.mark.parametrize("depth", [2, 3])
def test_real_gradient_paths_each_residual_layer(depth):
    torch.manual_seed(3470 + depth)
    m = LayeredContextTAC034(depth=depth).double()
    x = torch.randint(0, 256, (2, 9))
    y = m(x)
    y.square().mean().backward()
    required = (
        "input_proj.weight", "forget.weight", "offer.weight",
        "read_gate.weight", "concept_gate.weight", "concept_offer.weight",
        "concept_q.weight", "concept_k.weight", "concept_to_state.weight",
        "output.weight", "residual_gain",
    )
    for i in range(1, depth):
        block = m.layernorm[i]
        for name in required:
            p = dict(block.named_parameters())[name]
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"layer {i} {name} missing gradient"
            assert p.grad.abs().max() > 0, f"layer {i} {name} disconnected"
    assert m.context_lag_weights.grad.abs().max() > 0
    assert m.ffn_gate.weight.grad.abs().max() > 0


def test_scalar_affine_inclusive_scan_equals_chronological_and_concepts_solve():
    torch.manual_seed(3420)
    block = CausalResidualTACBlock(280, 24).double().eval()
    z = torch.randn(2, 7, 280, dtype=torch.float64)
    with torch.no_grad():
        _, trace = block(z, return_trace=True)
        u = torch.tanh(block.input_proj(block.input_norm(z)))
        a = .995 * torch.sigmoid(block.forget(u))
        v = torch.tanh(block.offer(u))
        state = torch.zeros_like(a[:, 0])
        expected = []
        for j in range(z.shape[1]):
            state = a[:, j] * state + (1-a[:, j]) * v[:, j]
            expected.append(state)
        reference = torch.stack(expected, dim=1)
    assert torch.allclose(reference, trace["state"], atol=1e-13, rtol=1e-13)
    assert torch.equal(trace["prior_state"][:, 0], torch.zeros_like(trace["prior_state"][:, 0]))
    assert torch.allclose(trace["prior_state"][:, 1:], trace["state"][:, :-1])
    L = trace["concept_laplacian"]
    assert torch.allclose(L, L.transpose(-1, -2), atol=1e-13)
    assert torch.linalg.eigvalsh(1.25*torch.eye(6, dtype=L.dtype)+.25*L).min() >= 1.249999


def test_block_rejects_wrong_shape():
    layer = CausalResidualTACBlock()
    with pytest.raises(ValueError):
        layer(torch.randn(2, 0, 280))
    with pytest.raises(ValueError):
        layer(torch.randn(2, 4, 48))
