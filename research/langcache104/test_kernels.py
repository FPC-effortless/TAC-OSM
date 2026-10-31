"""Constructed numerical fixtures only; no fits or capability measurements."""
from __future__ import annotations

import pytest
import torch

from research.langcache104.kernels import associative_read, rotary_scan


def bank_oracle(q, k, v, a, w, memory, mass, epsilon=1e-4):
    """Independent unvectorized-time recurrence, with explicit rank summation."""
    outputs, evidence = [], []
    for t in range(q.shape[1]):
        update = sum(v[:, t, :, r, :, None] * k[:, t, :, r, None, :]
                     for r in range(k.shape[3])) / k.shape[3]
        addresses = sum(k[:, t, :, r] for r in range(k.shape[3])) / k.shape[3]
        memory = a[:, t, :, None, None] * memory + w[:, t, :, None, None] * update
        mass = a[:, t, :, None] * mass + w[:, t, :, None] * addresses
        denom = (mass * q[:, t]).sum(-1)
        outputs.append((memory * q[:, t, :, None, :]).sum(-1) /
                       (denom[..., None] + epsilon))
        evidence.append(denom)
    return torch.stack(outputs, 1), memory, mass, torch.stack(evidence, 1)


def rotary_oracle(a, theta, offer, initial):
    out = []
    for t in range(offer.shape[1]):
        paired = initial.reshape(*initial.shape[:-1], -1, 2)
        c, s = theta[:, t].cos(), theta[:, t].sin()
        rotated = torch.stack((c*paired[..., 0]-s*paired[..., 1],
                               s*paired[..., 0]+c*paired[..., 1]), -1).flatten(-2)
        initial = a[:, t] * rotated + (1-a[:, t]) * offer[:, t]
        out.append(initial)
    return torch.stack(out, 1)


def bank_fixture(length=9, rank=2, dtype=torch.float64):
    g = torch.Generator().manual_seed(104)
    make = lambda *shape: torch.randn(*shape, generator=g, dtype=dtype)
    q = make(2, length, 2, 3).softmax(-1)
    k = make(2, length, 2, rank, 3).softmax(-1)
    v = make(2, length, 2, rank, 4).tanh()
    a = make(2, length, 2).sigmoid()
    w = (1-a) * make(2, length, 2).sigmoid()
    mass = make(2, 2, 3).sigmoid()
    memory = make(2, 2, 4, 3).tanh() * mass[..., None, :]
    return q, k, v, a, w, memory, mass


@pytest.mark.parametrize("length,rank,chunk", [(1, 1, 1), (3, 2, 2), (9, 3, 4), (17, 2, 6)])
def test_bank_independent_forward_and_cone(length, rank, chunk):
    args = bank_fixture(length, rank)
    expected = bank_oracle(*args)
    actual = associative_read(*args, chunk_size=chunk, mode="chunk")
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-10, atol=2e-11)
    reads, memory, mass, _ = actual
    assert bool((memory.abs() <= mass[..., None, :] + 1e-12).all())
    assert bool((mass >= 0).all()) and bool((mass <= 1+1e-12).all())
    assert bool((reads.abs() <= 1+1e-12).all())


def test_bank_all_input_gradients():
    args = tuple(x.detach().requires_grad_() for x in bank_fixture())
    actual = associative_read(*args, chunk_size=4, mode="chunk")
    expected = bank_oracle(*args)
    def objective(result):
        return sum((x * torch.linspace(.3, 1.7, x.numel(), dtype=x.dtype).reshape_as(x)).sum()
                   for x in result)
    ga = torch.autograd.grad(objective(actual), args)
    gb = torch.autograd.grad(objective(expected), args)
    for a, b in zip(ga, gb):
        assert bool(torch.isfinite(a).all())
        torch.testing.assert_close(a, b, rtol=3e-9, atol=3e-10)


def test_bank_zero_retention_identity_and_long_decay():
    args = list(bank_fixture(length=257))
    args[3] = torch.full_like(args[3], .999)
    args[3][:, 5] = 0
    args[3][:, 6] = 1
    args[4] = (1-args[3]) * .5
    for a, b in zip(associative_read(*args, chunk_size=16, mode="chunk"), bank_oracle(*args)):
        torch.testing.assert_close(a, b, rtol=3e-10, atol=3e-11)
    args[3] = torch.ones_like(args[3]); args[4] = torch.zeros_like(args[4])
    _, memory, mass, _ = associative_read(*args, chunk_size=15, mode="chunk")
    torch.testing.assert_close(memory, args[5], atol=0, rtol=0)
    torch.testing.assert_close(mass, args[6], atol=0, rtol=0)


@pytest.mark.parametrize("length", [1, 2, 3, 11, 64])
def test_rotary_independent_forward_and_gradients(length):
    torch.manual_seed(104)
    a = torch.randn(2, length, 2, 1, dtype=torch.float64).sigmoid().requires_grad_()
    theta = torch.randn(2, length, 2, 3, dtype=torch.float64).requires_grad_()
    offer = torch.randn(2, length, 2, 6, dtype=torch.float64).tanh().requires_grad_()
    initial = torch.randn(2, 2, 6, dtype=torch.float64).requires_grad_()
    args = (a, theta, offer, initial)
    actual = rotary_scan(*args, mode="parallel")
    expected = rotary_oracle(*args)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    weight = torch.linspace(.5, 1.5, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    for x, y in zip(torch.autograd.grad((actual*weight).sum(), args),
                    torch.autograd.grad((expected*weight).sum(), args)):
        torch.testing.assert_close(x, y, rtol=2e-9, atol=2e-10)


def test_rotary_no_write_identity_and_parity_rotation():
    initial = torch.tensor([[[1., 0.]]], dtype=torch.float64)
    a = torch.ones(1, 4, 1, 1, dtype=torch.float64)
    theta = torch.full((1, 4, 1, 1), torch.pi, dtype=torch.float64)
    offer = torch.zeros(1, 4, 1, 2, dtype=torch.float64)
    states = rotary_scan(a, theta, offer, initial, mode="parallel")
    torch.testing.assert_close(states[0, :, 0, 0], torch.tensor([-1.,1.,-1.,1.], dtype=torch.float64))
    freeze = rotary_scan(a, torch.zeros_like(theta), offer, initial, mode="parallel")
    torch.testing.assert_close(freeze, initial[:, None].expand_as(freeze), atol=0, rtol=0)
