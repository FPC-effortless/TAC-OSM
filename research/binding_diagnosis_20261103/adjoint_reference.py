"""Independent stdlib scalar-coordinate equations; not a full model reference."""
import math


def memory_candidate(previous, residual, direction, gain):
    return previous + gain*residual*direction


def memory_gain_adjoint(residual, direction, upstream=1.):
    return upstream*residual*direction


def precision_coordinate(previous, z_row, z_column, leverage, gain):
    return previous-gain*z_row*z_column/(1+gain*leverage)


def precision_gain_adjoint(z_row, z_column, leverage, gain, upstream=1.):
    return -upstream*z_row*z_column/(1+gain*leverage)**2


def softmax_two_head_credit(a, b, residual, direction):
    p = 1/(1+math.exp(b-a))
    magnitude = residual*direction*p*(1-p)
    return (-magnitude, magnitude)


def verify():
    eps = 1e-6
    expected = memory_gain_adjoint(2., 1.)
    numerical = (memory_candidate(0., 2., 1., eps)-memory_candidate(0., 2., 1., 0.))/eps
    if abs(expected-numerical) > 1e-10:
        raise ValueError("independent memory difference failed")
    expected_p = precision_gain_adjoint(1., 1., 1., 0.)
    numerical_p = (precision_coordinate(1., 1., 1., 1., eps)-1)/eps
    if abs(expected_p-numerical_p) > 2e-6:
        raise ValueError("independent precision difference failed")
    return {"memory_gain_at_zero": expected, "precision_gain_at_zero": expected_p, "two_head_unselected_write_credit": softmax_two_head_credit(.4, -.2, 3., 1.), "numerical_right_differences": [numerical, numerical_p], "scope": "scalar-coordinate candidate equations only; full model, argmax derivative and IEEE proof OPEN"}


if __name__ == "__main__":
    import json
    print(json.dumps(verify(), indent=2))
