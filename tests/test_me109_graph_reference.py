"""No-fit independent full-graph numerical checks for the ME109 equations."""
from contextlib import ExitStack
from unittest.mock import patch
import pytest
import torch
from tac_osm.variants.residual_role_sensor_world import ResidualRoleSensorWorld
from scripts.me109_graph_reference import serial
from tests.test_protected_sensor_world import sample


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('mask_case', ['mixed', 'last_missing', 'all_missing'])
def test_full_nonzero_residual_graph_parameter_and_input_gradients(dtype, mask_case):
    torch.manual_seed(109)
    model = ResidualRoleSensorWorld(torch.full((6,), .1)).to(dtype)
    model.target_mean.fill_(3.)
    # Exercise actual learned residual paths instead of testing only zero-start.
    with torch.no_grad():
        for head in (model.current_residual, model.width_residual, model.median_residual):
            head.output.weight.normal_(0, .025)
            head.output.bias.normal_(0, .025)
        model.width_head.weight.normal_(0, .01)
        model.median_head.weight.normal_(0, .01)
    values, observed, ages, clock, owners, hours = sample(dtype)
    if mask_case == 'last_missing': observed[:, -1] = False
    if mask_case == 'all_missing': observed[:] = False
    values.requires_grad_(); ages.requires_grad_(); clock.requires_grad_()
    x = values, observed, ages, clock, owners, hours
    forecast, state, trace = model(*x, return_trace=True)
    with ExitStack() as stack:
        for head in (model.current_residual, model.width_residual, model.median_residual,
                     model.current_head, model.width_head, model.median_head):
            stack.enter_context(patch.object(head, 'forward', side_effect=AssertionError('reference called new/parent head')))
        stack.enter_context(patch.object(model, 'forward', side_effect=AssertionError('reference called model')))
        mu, left, right, ss, sm, es, memory, rt = serial(model, *x[:5])
    rtol, atol = (2e-5, 2e-6) if dtype == torch.float32 else (1e-9, 1e-10)
    for actual, expected in zip(
        (forecast.log_location, forecast.left_scale, forecast.right_scale,
         state.source_sum, state.source_mass, state.evidence_sum, state.memory),
        (mu, left, right, ss, sm, es, memory)):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    actual_loss = sum(z.square().sum() for z in forecast)
    expected_loss = sum(z.square().sum() for z in (mu, left, right))
    parameters = tuple(model.parameters())
    assert len(parameters) == 45
    targets = parameters + (values, ages, clock)
    actual_grads = torch.autograd.grad(actual_loss, targets, retain_graph=True, allow_unused=True)
    expected_grads = torch.autograd.grad(expected_loss, targets, allow_unused=True)
    for actual, expected, target in zip(actual_grads, expected_grads, targets):
        actual = torch.zeros_like(target) if actual is None else actual
        expected = torch.zeros_like(target) if expected is None else expected
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert (forecast.left_scale >= trace['protected_floor_scale']).all()
    assert (forecast.right_scale >= trace['protected_floor_scale']).all()
    assert (trace['median_correction'].abs() <= .5).all()
    assert (rt['residual_context'][:, -6:] == observed[:, -1, :6]).all()


def test_masked_finite_values_do_not_enter_reference_context():
    model = ResidualRoleSensorWorld().double()
    values, observed, ages, clock, owners, hours = sample(torch.float64)
    baseline, _ = model(values, observed, ages, clock, owners, hours)
    values[~observed] = 7.99
    actual, _ = model(values, observed, ages, clock, owners, hours)
    expected = serial(model, values, observed, ages, clock, owners)
    for a, b, old in zip(actual, expected[:3], baseline):
        torch.testing.assert_close(a, old, rtol=0, atol=0)
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)
