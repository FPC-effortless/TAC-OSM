import pytest

from tac_osm.information_bounds import (
    balanced_collision_floor,
    collision_cauchy_lower_bound,
    partition_success_upper_bound,
    signature_alphabet_ceiling,
    terminal_success_upper_bound,
)


@pytest.mark.parametrize(
    ("m", "q", "expected"),
    [(32, 16, 2.0), (64, 16, 4.0), (512, 16, 32.0), (512, 256, 2.0), (512, 4096, 1.0)],
)
def test_balanced_collision_floor(m, q, expected):
    assert balanced_collision_floor(m, q) == pytest.approx(expected)


def test_balanced_floor_respects_cauchy_bound():
    for m, q in [(32, 3), (128, 16), (512, 64), (512, 4096)]:
        assert balanced_collision_floor(m, q) >= collision_cauchy_lower_bound(m, q)


def test_terminal_success_ceiling():
    assert terminal_success_upper_bound(512, 16, 8) == pytest.approx(0.25)
    assert terminal_success_upper_bound(512, 64, 8) == pytest.approx(1.0)


def test_fixed_partition_success_matches_bucket_cap():
    assert partition_success_upper_bound([32] * 16, 8) == pytest.approx(0.25)
    assert partition_success_upper_bound([8] * 64, 8) == pytest.approx(1.0)


def test_binary_signature_ceiling():
    assert signature_alphabet_ceiling(4) == 16
    assert signature_alphabet_ceiling(6) == 64
    assert signature_alphabet_ceiling(12) == 4096


@pytest.mark.parametrize(
    "fn,args",
    [
        (collision_cauchy_lower_bound, (0, 16)),
        (collision_cauchy_lower_bound, (16, 0)),
        (balanced_collision_floor, (0, 16)),
        (terminal_success_upper_bound, (512, 16, 0)),
        (signature_alphabet_ceiling, (-1,)),
    ],
)
def test_invalid_inputs_fail_closed(fn, args):
    with pytest.raises(ValueError):
        fn(*args)

def test_budget_utility_has_complementarity_counterexample():
    # Four hypotheses; three binary probes. Feature 0 is constant, while
    # feature 1 splits {0,1}|{2,3} and feature 2 splits {0,2}|{1,3}.
    # With B=1, U({2})-U({}) = 1/4, but
    # U({1,2})-U({1}) = 1/2. Marginal value increases after observing
    # another probe, violating submodularity.
    hypotheses = range(4)
    feature = {
        0: (0, 0, 0, 0),
        1: (1, 1, 0, 0),
        2: (1, 0, 1, 0),
    }

    def u(selected):
        buckets = {}
        for h in hypotheses:
            signature = tuple(feature[i][h] for i in selected)
            buckets[signature] = buckets.get(signature, 0) + 1
        return sum(min(1, n) for n in buckets.values()) / 4

    assert u(()) == 0.25
    assert u((2,)) - u(()) == 0.25
    assert u((1, 2)) - u((1,)) == 0.50
    assert u((2,)) - u(()) < u((1, 2)) - u((1,))
