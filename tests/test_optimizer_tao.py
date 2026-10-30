"""Constructed numerical/engineering checks only; not trained TAC evidence."""
import copy
import math

import pytest
import torch

from tac_osm.optimizer_tao import TAO, ns5, build_tao


def test_update_shape_rms_all_modes():
    torch.manual_seed(42)
    for mode in ("muon", "adamuon", "normuon", "pre_ns"):
        for shape in ((16, 32), (32, 16), (16, 16), (8, 4, 3, 3)):
            p = torch.nn.Parameter(torch.zeros(shape))
            o = TAO([{"params": [p], "kind": "orthogonal"}], mode=mode, lr=1e-2, weight_decay=0)
            p.grad = torch.randn_like(p)
            o.step()
            assert p.shape == shape and torch.isfinite(p).all()
            assert math.isclose(float(p.detach().square().mean().sqrt()), 0.002, abs_tol=2e-6)
            assert o.state[p]["step"] == 1
            expected = (shape[0], 1) if mode == "normuon" else (shape[0], math.prod(shape[1:]))
            if mode != "muon":
                assert o.state[p]["second_moment"].shape == expected
            else:
                assert "second_moment" not in o.state[p]


def test_ns5_rectangular_and_zero():
    for shape in ((12, 64), (64, 12), (32, 32)):
        u = ns5(torch.randn(shape))
        assert u.shape == shape and torch.isfinite(u).all()
    assert torch.count_nonzero(ns5(torch.zeros(12, 24))) == 0


def test_adamw_fallback_matches_pytorch():
    torch.manual_seed(11)
    p = torch.nn.Parameter(torch.randn(7))
    q = torch.nn.Parameter(p.detach().clone())
    a = TAO([p], lr=.001, weight_decay=.02)
    b = torch.optim.AdamW([q], lr=.001, weight_decay=.02, foreach=False)
    for _ in range(7):
        grad = torch.randn_like(p)
        p.grad = grad.clone()
        q.grad = grad.clone()
        a.step()
        b.step()
        assert torch.allclose(p, q, atol=5e-7, rtol=1e-6)


def test_routing_excludes_memory_and_output_even_when_matrices():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(12, 16)
            self.ff = torch.nn.Linear(16, 32)
            self.recurrent_transition = torch.nn.Linear(16, 16)
            self.lm_head = torch.nn.Linear(32, 12)
    m = Model()
    o = build_tao(m, orthogonal_names=["ff.weight"], mode="normuon")
    orth = next(x["params"] for x in o.param_groups if x["kind"] == "orthogonal")
    adam = next(x["params"] for x in o.param_groups if x["kind"] == "adamw")
    assert len(orth) == 1 and orth[0] is m.ff.weight
    assert any(p is m.recurrent_transition.weight for p in adam)
    assert any(p is m.lm_head.weight for p in adam)
    assert any(p is m.embed.weight for p in adam)
    with pytest.raises(ValueError):
        build_tao(m, orthogonal_names=["missing.weight"])
    with pytest.raises(ValueError):
        build_tao(m, orthogonal_names=["ff.bias"])


def test_checkpoint_resume_same_updates():
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.randn(24, 12))
    a = TAO([{"params": [p], "kind": "orthogonal"}], mode="normuon")
    grads = [torch.randn_like(p) for _ in range(5)]
    for g in grads[:3]:
        p.grad = g.clone()
        a.step()
    saved = copy.deepcopy(a.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    b = TAO([{"params": [q], "kind": "orthogonal"}], mode="normuon")
    b.load_state_dict(saved)
    for g in grads[3:]:
        p.grad = g.clone()
        q.grad = g.clone()
        a.step()
        b.step()
    assert torch.equal(p, q)


def test_fails_closed_on_nonfinite_gradient_without_partial_update():
    p = torch.nn.Parameter(torch.ones(4, 4))
    q = torch.nn.Parameter(torch.ones(4))
    a = TAO([{"params": [p], "kind": "orthogonal"}, {"params": [q], "kind": "adamw"}])
    p.grad = torch.ones_like(p)
    q.grad = torch.tensor([0., 0., float("nan"), 0.])
    with pytest.raises(ValueError):
        a.step()
    assert torch.equal(p, torch.ones_like(p)) and torch.equal(q, torch.ones_like(q))


def test_zero_gradient_does_not_nan_or_drift_without_wd():
    p = torch.nn.Parameter(torch.ones(12, 8))
    o = TAO([{"params": [p], "kind": "orthogonal"}], mode="adamuon", weight_decay=0)
    p.grad = torch.zeros_like(p)
    o.step()
    assert torch.equal(p, torch.ones_like(p))
