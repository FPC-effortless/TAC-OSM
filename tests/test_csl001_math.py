"""Independent real reference vs actual PyTorch operators; not Lean proof."""
import importlib.util
from pathlib import Path

import numpy as np
import torch
from tac_osm.csl001 import CSL001, _bounded_scan

MATH = Path(__file__).resolve().parents[1] / 'research/mathematical-engineering/csl001/math_reference.py'
spec = importlib.util.spec_from_file_location('csl001_math_reference', MATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reference_scan_exact_forward_and_gradient_boundary():
    rng = np.random.default_rng(314)
    z = rng.normal(size=(2, 80, 4, 16)).astype('float32') * 0.05
    initial = rng.normal(size=(2, 4, 16)).astype('float32') * 0.1
    decay = rng.uniform(0.85, 0.99, size=(4, 16)).astype('float32')
    expected = module.serial_retention(z.astype('float64'), initial.astype('float64'), decay.astype('float64'))
    ztorch = torch.tensor(z, requires_grad=True)
    out, end = _bounded_scan(ztorch, torch.tensor(initial), torch.tensor(decay), 64)
    np.testing.assert_allclose(out.detach().numpy(), expected, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(end.detach().numpy(), expected[:, -1], rtol=3e-5, atol=3e-6)
    out.mean().backward()
    assert ztorch.grad is not None and torch.isfinite(ztorch.grad).all()
    # Finite-difference of an actual source operator at one legal coordinate.
    idx = (0, 3, 1, 2)
    eps = 1e-3
    plus, minus = z.copy(), z.copy()
    plus[idx] += eps
    minus[idx] -= eps
    f_plus = module.serial_retention(plus.astype('float64'), initial.astype('float64'), decay.astype('float64')).mean()
    f_minus = module.serial_retention(minus.astype('float64'), initial.astype('float64'), decay.astype('float64')).mean()
    np.testing.assert_allclose(ztorch.grad[idx].item(), (f_plus-f_minus)/(2*eps), rtol=3e-3, atol=1e-6)


def test_reference_causal_depthwise_agrees_with_live_model_conv():
    model = CSL001()
    block = model.blocks[0]
    x = torch.randn(2, 15, model.config.width)
    prev = torch.randn(2, 2, model.config.width)
    with torch.no_grad():
        ref, tail = module.causal_depthwise(x.numpy(), prev.numpy(), block.conv.weight[:, 0].numpy())
        actual = block.conv(torch.cat((prev, x), dim=1).transpose(1,2)).transpose(1,2)
    np.testing.assert_allclose(actual.numpy(), ref, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(tail, torch.cat((prev,x),dim=1)[:, -2:].numpy())


def test_reference_rejects_nonconvex_decay():
    z = np.zeros((1, 2, 2, 2))
    initial = np.zeros((1, 2, 2))
    for decay in (np.zeros((2, 2)), np.ones((2, 2))):
        try:
            module.serial_retention(z, initial, decay)
        except ValueError:
            pass
        else:
            raise AssertionError('must reject excluded endpoints')
