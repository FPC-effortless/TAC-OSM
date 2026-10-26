"""No-fit implementation contracts for the ME109 native candidate."""
import sys
from pathlib import Path
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tac_osm.variants.point_priority_sensor_world import PointPrioritySensorWorld
from tac_osm.variants.residual_role_sensor_world import ResidualRoleSensorWorld
from tests.test_protected_sensor_world import sample

torch.set_num_threads(1)


def pair(dtype):
    torch.manual_seed(109)
    parent = PointPrioritySensorWorld(floor_rate=torch.full((6,), .01)).to(dtype)
    with torch.no_grad():
        for head in (parent.current_head, parent.width_head, parent.median_head):
            head.weight.normal_(0, .02)
            head.bias.normal_(0, .02)
    model = ResidualRoleSensorWorld().to(dtype)
    result = model.load_state_dict(parent.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert len(result.missing_keys) == 12
    return parent, model


def state_tensors(state):
    return [v for v in vars(state).values() if isinstance(v, torch.Tensor)]


def exact(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_nonzero_inherited_head_forward_trace_state_and_gradient_parity(dtype):
    parent, model = pair(dtype)
    xa = list(sample(dtype)); xb = list(sample(dtype))
    for i in (0, 2, 3):
        xa[i] = xa[i].clone().requires_grad_()
        xb[i] = xb[i].clone().requires_grad_()
    a, sa, ta = parent(*xa, return_trace=True)
    b, sb, tb = model(*xb, return_trace=True)
    for u, v in zip(a, b): exact(u, v)
    for u, v in zip(state_tensors(sa), state_tensors(sb)): exact(u, v)
    assert ta.keys() == tb.keys()
    for key in ta: exact(ta[key], tb[key])
    def loss(f, t):
        return (f.log_location.square().sum() + f.left_scale.log().sum()
                + f.right_scale.log().sum() + t['current_mean'].sum()
                + t['current_variance'].sum())
    loss(a, ta).backward(); loss(b, tb).backward()
    tol = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=1e-9, atol=1e-10)
    for name, p in parent.named_parameters():
        q = dict(model.named_parameters())[name]
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None: torch.testing.assert_close(p.grad, q.grad, **tol)
    for i in (0, 2, 3): torch.testing.assert_close(xa[i].grad, xb[i].grad, **tol)
    for role in ('current', 'width', 'median'):
        residual = getattr(model, role + '_residual')
        for p in residual.input.parameters():
            assert torch.count_nonzero(p.grad) == 0
        for p in residual.output.parameters():
            assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_budget_state_schema_and_exact_zero_outputs():
    parent, model = pair(torch.float32)
    assert sum(p.numel() for p in model.parameters()) == 104663
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 104663
    assert dict(parent.named_buffers()).keys() == dict(model.named_buffers()).keys()
    for name in ('current_head', 'width_head', 'median_head'):
        assert type(getattr(model, name)) is torch.nn.Linear
    for name in ('current_residual', 'width_residual', 'median_residual'):
        residual = getattr(model, name)
        assert residual.input.in_features == 108 and residual.input.out_features == 12
        assert torch.count_nonzero(residual.output.weight) == 0
        assert torch.count_nonzero(residual.output.bias) == 0


@pytest.mark.parametrize('mode', ['all', 'none', 'mixed'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_mask_legality_chunk_state_parity_and_pure_interleaved_calls(mode, dtype):
    _, model = pair(dtype)
    # Exercise the active graph, beyond the zero-delta initialization.
    with torch.no_grad():
        for name in ('current_residual', 'width_residual', 'median_residual'):
            getattr(model, name).output.weight.normal_(0, .03)
    x = list(sample(dtype))
    if mode != 'mixed': x[1].fill_(mode == 'all')
    dirty = list(x)
    dirty[0] = torch.where(x[1], x[0], torch.full_like(x[0], 7.5))
    f, s, t = model(*dirty, return_trace=True)
    for value in f: assert torch.isfinite(value).all()
    assert torch.all(f.left_scale > 0) and torch.all(f.right_scale > 0)
    assert torch.all(f.left_scale >= t['protected_floor_scale'])
    assert torch.all(f.right_scale >= t['protected_floor_scale'])
    assert torch.all(t['median_correction'].abs() <= .5)
    clean = model(*x)[0]
    for a, b in zip(f, clean): exact(a, b)
    def chunk(start, end):
        return [v[:, start:end] if v.ndim >= 2 else v for v in dirty]
    _, first = model(*chunk(0, 8))
    # An unrelated call cannot alter the mask used by resumed decoding.
    model(*sample(dtype))
    g, gs = model(*chunk(8, None), state=first)
    tol = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=2e-8, atol=2e-9)
    for a, b in zip(f, g): torch.testing.assert_close(a, b, **tol)
    for a, b in zip(state_tensors(s), state_tensors(gs)): torch.testing.assert_close(a, b, **tol)
    view = model.uncertainty_forecast(f, t)
    exact(view.log_location, f.log_location)
    grad = torch.autograd.grad(view.log_location.sum(), model.median_residual.output.weight)[0]
    assert torch.count_nonzero(grad) == 0


def test_role_context_uses_actual_mask_and_existing_age_features():
    _, model = pair(torch.float64)
    x = sample(torch.float64)
    contexts = []
    handles = [getattr(model, name + '_residual').register_forward_pre_hook(
        lambda module, args: contexts.append(args[0]))
        for name in ('current', 'width', 'median')]
    _, _, trace = model(*x, return_trace=True)
    for handle in handles: handle.remove()
    assert len(contexts) == 3
    for context in contexts:
        exact(context[:, :102], trace['width_features'])
        exact(context[:, 102:], x[1][:, -1, :6].to(torch.float64))
        exact(context[:, 88:94], x[2][:, -1, :6])
    assert not any('mask' in key for key in vars(model))
