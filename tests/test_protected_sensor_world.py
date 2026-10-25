import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from tac_osm.variants.median_split_sensor_world import MedianSplitForecast
from tac_osm.variants.protected_sensor_world import ProtectedSensorWorld
from tac_osm.variants.reconstruction_sensor_world import ReconstructionSensorWorld
from scripts.me105_graph_reference import serial as reference_serial


torch.set_num_threads(1)


def sample(dtype=torch.float32):
    torch.manual_seed(104)
    values = torch.randn(2, 17, 13, dtype=dtype)
    observed = torch.rand(2, 17, 13) > 0.22
    observed[:, -1, 0] = False
    ages = torch.rand_like(values)
    clock = torch.randn(2, 17, 6, dtype=dtype).tanh()
    owners = torch.tensor([1, 8])
    hours = torch.arange(17)[None].expand(2, -1)
    return values, observed, ages, clock, owners, hours


def test_parameter_budget_and_zero_head_exact_parent_warm_start():
    torch.manual_seed(14)
    parent = ReconstructionSensorWorld()
    model = ProtectedSensorWorld()
    model.load_state_dict(parent.state_dict(), strict=False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 100349
    assert sum(p.numel() for p in model.parameters()) == 100349
    model.eval(); parent.eval()
    x = sample()
    old, old_state, old_trace = parent(*x, return_trace=True)
    new, new_state, trace = model(*x, return_trace=True)
    torch.testing.assert_close(new.log_location, old.log_location, rtol=0, atol=0)
    torch.testing.assert_close(new.left_scale, old.log_scale, rtol=0, atol=0)
    torch.testing.assert_close(new.right_scale, old.log_scale, rtol=0, atol=0)
    torch.testing.assert_close(trace['width_factors'], torch.ones_like(trace['width_factors']))
    torch.testing.assert_close(new_state.memory, old_state.memory, rtol=0, atol=0)
    ps = torch.tensor([0.05, 0.10, 0.50, 0.90, 0.95])
    expected = torch.expm1(old.log_location[..., None] + old.log_scale[..., None]
                           * torch.special.ndtri(ps))
    torch.testing.assert_close(model.raw_quantiles(new, ps), expected.clamp_min(0), rtol=1e-6, atol=1e-6)
    default_q = model.raw_quantiles(new)
    assert default_q.shape == (2, 6, 3)
    old_z = torch.tensor([-1.6448536269514722, 0.0, 1.6448536269514722])
    legacy_q = torch.expm1(old.log_location[..., None] + old.log_scale[..., None] * old_z).clamp_min(0)
    torch.testing.assert_close(default_q, legacy_q, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_independent_reference_actual_source_forward_states_and_gradients(dtype):
    torch.manual_seed(31)
    model = ProtectedSensorWorld().to(dtype)
    with torch.no_grad():
        model.width_head.weight.normal_(0, 0.025)
        model.width_head.bias.normal_(0, 0.025)
        model.median_head.weight.normal_(0, 0.025)
        model.median_head.bias.normal_(0, 0.025)
    x = sample(dtype)
    xa = [v.detach().clone().requires_grad_(v.is_floating_point()) if v.is_floating_point()
          else v.clone() for v in x]
    xb = [v.detach().clone().requires_grad_(v.is_floating_point()) if v.is_floating_point()
          else v.clone() for v in x]
    actual, state, trace = model(*xa, return_trace=True)
    (mu, left, right, source_sum, source_mass, evidence_sum, memory,
     ref_trace) = reference_serial(model, *xb[:5])
    rtol, atol = ((2e-5, 2e-6) if dtype == torch.float32 else (1e-9, 1e-10))
    for a, b in ((actual.log_location, mu), (actual.left_scale, left),
                 (actual.right_scale, right), (state.source_sum, source_sum),
                 (state.source_mass, source_mass), (state.evidence_sum, evidence_sum),
                 (state.memory, memory)):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    for name in ('base_log_scale', 'width_features', 'width_factors', 'left_scale', 'right_scale'):
        torch.testing.assert_close(trace[name], ref_trace[name], rtol=rtol, atol=atol)
    loss_a = actual.log_location.square().mean() + actual.left_scale.log().mean() + actual.right_scale.log().mean()
    loss_b = mu.square().mean() + left.log().mean() + right.log().mean()
    params = tuple(model.parameters())
    grads_a = torch.autograd.grad(loss_a, (*params, *[v for v in xa if v.is_floating_point()]), retain_graph=True)
    grads_b = torch.autograd.grad(loss_b, (*params, *[v for v in xb if v.is_floating_point()]))
    for a, b in zip(grads_a, grads_b):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        assert torch.isfinite(a).all()


def test_quantiles_order_cdf_consistency_and_zero_atom():
    model = ProtectedSensorWorld().double()
    with torch.no_grad():
        model.target_mean.copy_(torch.tensor([0.3, -0.2, 0.5, 0.0, 0.7, -0.1], dtype=torch.float64))
        model.target_scale.copy_(torch.tensor([0.5, 0.8, 0.3, 1.1, 0.6, 0.9], dtype=torch.float64))
        model.width_head.bias.copy_(torch.linspace(-0.5, 0.5, 12, dtype=torch.float64))
    f, _, _ = model(*sample(torch.float64), return_trace=True)
    p = torch.tensor([0.001, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.999], dtype=torch.float64)
    q = model.raw_quantiles(f, p)
    assert torch.all(q[..., 1:] >= q[..., :-1]) and torch.all(q >= 0)
    got = model.cdf(f, q)
    # The zero censoring atom makes inverse-CDF values below its mass equal 0.
    target = p.expand_as(got)
    atom = model.cdf(f, torch.zeros_like(f.log_location))[..., None]
    torch.testing.assert_close(torch.where(target <= atom, torch.zeros_like(got), got),
                               torch.where(target <= atom, torch.zeros_like(target), target),
                               rtol=1e-8, atol=1e-8)
    assert torch.equal(model.cdf(f, -torch.ones_like(f.log_location)), torch.zeros_like(f.log_location))
    torch.testing.assert_close(model.cdf(f, torch.zeros_like(f.log_location)), atom[..., 0])


def test_chunked_parity_and_width_head_does_not_change_sensor_memory():
    model = ProtectedSensorWorld().double()
    x = sample(torch.float64)
    whole, whole_state, _ = model(*x, return_trace=True)
    _, first_state = model(*(v[:, :8] if v.ndim >= 2 else v for v in x))
    chunk, chunk_state, _ = model(*(v[:, 8:] if v.ndim >= 2 else v for v in x),
                                  state=first_state, return_trace=True)
    for a, b in ((whole.log_location, chunk.log_location), (whole.left_scale, chunk.left_scale),
                 (whole.right_scale, chunk.right_scale), (whole_state.memory, chunk_state.memory)):
        torch.testing.assert_close(a, b, rtol=2e-8, atol=2e-9)
    before = whole_state.memory.detach().clone()
    with torch.no_grad():
        model.width_head.weight.add_(0.5)
    _, changed_state = model(*x)
    torch.testing.assert_close(changed_state.memory, before, rtol=0, atol=0)


def test_nonzero_width_changes_tails_but_never_the_parent_median_location():
    model = ProtectedSensorWorld()
    x = sample()
    base, _, _ = model(*x, return_trace=True)
    with torch.no_grad():
        model.width_head.bias[:6].fill_(-0.7)
        model.width_head.bias[6:].fill_(0.7)
    split, _, _ = model(*x, return_trace=True)
    torch.testing.assert_close(split.log_location, base.log_location, rtol=0, atol=0)
    assert torch.all(split.left_scale < split.right_scale)
    torch.testing.assert_close(model.raw_quantiles(split, [0.5])[..., 0],
                               model.raw_quantiles(base, [0.5])[..., 0], rtol=0, atol=0)


def test_split_normal_halves_each_integrate_to_one_half_and_nll_is_finite():
    model = ProtectedSensorWorld().double()
    f = model(*sample(torch.float64))[0]
    grid = torch.linspace(-12, 12, 240001, dtype=torch.float64)
    for scale in (f.left_scale[0, 0], f.right_scale[0, 0]):
        left = torch.exp(-0.5 * (grid[grid <= 0] / scale).square()) / (scale * math.sqrt(2 * math.pi))
        right = torch.exp(-0.5 * (grid[grid >= 0] / scale).square()) / (scale * math.sqrt(2 * math.pi))
        assert abs(torch.trapezoid(left, grid[grid <= 0]).item() - 0.5) < 2e-5
        assert abs(torch.trapezoid(right, grid[grid >= 0]).item() - 0.5) < 3e-5
    targets = torch.tensor([[0., 1.5, 0., 2.0, 3., 0.], [0.2, 0., 0.4, 0., 1., 0.]], dtype=torch.float64)
    mask = torch.ones_like(targets, dtype=torch.bool)
    loss = model.nll(f, targets, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.width_head.weight.grad is not None
    assert torch.isfinite(model.width_head.weight.grad).all()


def test_probability_and_target_validation():
    model = ProtectedSensorWorld()
    f = model(*sample())[0]
    with pytest.raises(ValueError, match='probabilities'):
        model.raw_quantiles(f, [0., 0.5, 1.])
    with pytest.raises(ValueError, match='raw_values'):
        model.cdf(f, torch.zeros(1, 2))
    with pytest.raises(ValueError, match='raw_targets'):
        model.nll(f, torch.full((2, 6), float('nan')), torch.ones((2, 6), dtype=torch.bool))
    with pytest.raises(ValueError, match='nonnegative'):
        model.nll(f, -torch.ones((2, 6)), torch.ones((2, 6), dtype=torch.bool))


def test_zero_atom_uses_right_side_when_threshold_is_above_median():
    model = ProtectedSensorWorld().double()
    f = MedianSplitForecast(
        torch.full((1, 6), -1.0, dtype=torch.float64),
        torch.full((1, 6), 0.25, dtype=torch.float64),
        torch.full((1, 6), 1.5, dtype=torch.float64),
    )
    zero = torch.zeros((1, 6), dtype=torch.float64)
    atom = torch.special.ndtr(torch.full((1, 6), 1.0 / 1.5, dtype=torch.float64))
    torch.testing.assert_close(model.cdf(f, zero), atom)
    torch.testing.assert_close(model.nll(f, zero, torch.ones_like(zero, dtype=torch.bool)),
                               -torch.special.log_ndtr(torch.full_like(zero, 1.0 / 1.5)).mean())
    assert torch.all(model.raw_quantiles(f)[..., 1] == 0)


def test_source_floor_and_missing_relative_scale_survive_extreme_shrink():
    model = ProtectedSensorWorld(floor_rate=torch.full((6,), 0.25)).double()
    with torch.no_grad():
        model.width_head.bias.fill_(-100.)
    x = sample(torch.float64)
    pred, _, trace = model(*x, return_trace=True)
    for scale in (pred.left_scale, pred.right_scale):
        assert torch.all(scale.square() + 1e-14 >= trace['floor_variance'])
        assert torch.all(scale >= trace['protected_floor_scale'])
        assert torch.all(scale[~x[1][:,-1,:6]] >= 0.8 * trace['base_log_scale'][~x[1][:,-1,:6]])
    assert (trace['width_factors'] < 0.251).all()


def test_median_correction_bounded_and_point_gradient_reaches_appropriate_paths():
    model = ProtectedSensorWorld().double()
    x = sample(torch.float64)
    model.target_mean.fill_(3.)
    pred, _, trace = model(*x, return_trace=True)
    median = model.raw_quantiles(pred)[...,1]
    grads = torch.autograd.grad(median.sum(), (model.median_head.weight, model.width_head.weight, model.encoder.weight), allow_unused=True)
    assert grads[0].abs().sum() > 0
    assert grads[1] is None or grads[1].abs().sum() == 0
    assert grads[2].abs().sum() > 0
    with torch.no_grad():
        model.median_head.bias.fill_(100.)
    changed, _, tr = model(*x, return_trace=True)
    assert torch.all(tr['median_correction'].abs() <= 0.5)
    torch.testing.assert_close(changed.log_location-tr['inherited_log_location'], torch.full_like(changed.log_location,0.5))


def test_zero_floor_has_finite_input_gradients_including_zero_age():
    model = ProtectedSensorWorld().double()
    x = list(sample(torch.float64))
    x[2] = torch.zeros_like(x[2], requires_grad=True)
    f, _ = model(*x)
    grad = torch.autograd.grad(f.left_scale.sum()+f.right_scale.sum()+f.log_location.sum(), x[2])[0]
    assert torch.isfinite(grad).all()
