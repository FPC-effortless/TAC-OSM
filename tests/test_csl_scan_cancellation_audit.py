"""Audit enforcement and independent serial derivative fixtures, not training."""
import pytest
import torch
from scripts.audit_csl_scan_cancellation import (
    CASES, decimal_serial, difference, evaluate, fixture, objective, serial_torch,
)


@pytest.mark.parametrize("case", CASES)
def test_fixture_is_finite_bounded_numerator_mass(case):
    z, carry, rate, gate = fixture(case)
    assert z.shape == (1, 65, 2, 2, 2)
    assert torch.isfinite(z).all() and torch.isfinite(carry).all()
    assert (z[..., 0].abs() <= z[..., 1]).all() and (z[..., 1] >= 0).all()
    assert (carry[..., 0].abs() <= carry[..., 1]).all() and (carry[..., 1] >= 0).all()
    assert ((rate > 0) & (rate <= .5)).all() and ((gate >= 0) & (gate <= 1)).all()


def test_gate_rejects_nonfinite_shape_broadcast_and_real_mismatch():
    assert difference(torch.tensor([float('nan')]), torch.ones(1), 1e-5, 1e-5)["status"] == "FAIL_NONFINITE"
    assert difference(torch.ones(2), torch.ones(1), 1e-5, 1e-5)["status"] == "FAIL_SHAPE_OR_EMPTY"
    assert difference(torch.zeros(1), torch.ones(1), 1e-5, 1e-5)["status"] == "FAIL_TOLERANCE"
    assert difference(torch.zeros(1), torch.zeros(1), 1e-5, 1e-5)["status"] == "PASS"


def test_decimal_reference_independently_matches_serial_double_forward():
    args = fixture('constructed_cancellation')
    expected = decimal_serial(args)
    actual = serial_torch(*(a.double() for a in args), 7)[0]
    assert difference(actual, expected, 1e-12, 1e-12)["status"] == "PASS"


@pytest.mark.parametrize("arg", range(4))
def test_serial_double_all_argument_gradients_against_directional_fd(arg):
    args = [a.double() for a in fixture('alternating')]
    args[2] *= .8
    args[3] *= .5
    result = evaluate(serial_torch, args, 7, 'linear_state', torch.float64)
    step = 1e-5
    plus, minus = [a.clone() for a in args], [a.clone() for a in args]
    direction = torch.ones_like(args[arg]) / args[arg].numel() ** .5
    plus[arg] += step * direction
    minus[arg] -= step * direction
    fd = (objective(serial_torch(*plus, 7)[0], 'linear_state') -
          objective(serial_torch(*minus, 7)[0], 'linear_state')) / (2 * step)
    ad = (result[2][arg] * direction).sum()
    assert torch.allclose(fd, ad, atol=1e-8, rtol=1e-6)


def test_closed_gate_hold_is_numeric_not_assumed_bitwise():
    args = fixture('signed_zero_hold')
    out, end = serial_torch(*(a.double() for a in args), 1)
    assert torch.equal(out, args[1].double().unsqueeze(1).expand_as(out))
    assert torch.equal(end, args[1].double())


def test_source_default_objective_uses_the_actual_config_epsilon():
    from tac_osm.csl125_r2 import CSL125R2Config
    assert CSL125R2Config().denominator_epsilon == 1e-8
    value = torch.tensor([[[[[1e-10, 1e-10]]]]], dtype=torch.float64)
    assert objective(value, 'normalized_read_source_default') != objective(value, 'normalized_read')
    with pytest.raises(ValueError, match='unregistered'):
        objective(value, 'unregistered')
