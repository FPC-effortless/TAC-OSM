import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from me102_gap_data import augmentation, floor_rates


def data(phase='fit'):
    # Small deterministic source-shaped fixture with known 24-hour pairs.
    raw = np.empty((2, 80, 7), dtype=np.float32)
    for site in range(2):
        for t in range(80):
            raw[site, t] = (np.arange(7) + 1) * (t + site + 1)
    observed = np.ones_like(raw, dtype=bool)
    observed[:, 7:10, 1] = False
    raw[~observed] = np.nan
    logged = np.log1p(raw)
    mean = np.nanmean(logged, axis=(0, 1))
    std = np.nanstd(logged, axis=(0, 1)).clip(.0001)
    values = np.nan_to_num((logged - mean) / std, nan=0).clip(-8, 8).astype(np.float32)
    ages = np.zeros_like(values)
    last = np.full((2, 7), -1, dtype=np.int64)
    for t in range(80):
        last = np.where(observed[:, t], t, last)
        ages[:, t] = np.log1p(np.minimum(t - last, 256)) / np.log(257)
    angles = np.arange(80, dtype=np.float32) * (2 * np.pi / 24)
    clock = np.broadcast_to(np.stack((np.sin(angles), np.cos(angles)), axis=-1), (2, 80, 2)).copy()
    hours = np.broadcast_to(np.arange(80), (2, 80)).copy()
    return dict(phase=phase, raw=raw, values=values, observed=observed, ages=ages,
                mean=mean.astype(np.float32), std=std.astype(np.float32),
                clock=clock, hours=hours)


def test_floor_rates_counts_only_valid_lag24_fit_pairs():
    d = data()
    result = floor_rates(d)
    assert set(result) == {'rates', 'counts'}
    assert result['rates'].shape == result['counts'].shape == (6,)
    assert result['counts'][0] == 2 * (80 - 24)
    assert result['counts'][1] == 2 * ((80 - 24) - 3)
    assert np.isfinite(result['rates']).all()
    logs = np.log1p(d['raw'][:, :, :6].astype(np.float64))
    for j in range(6):
        valid = d['observed'][:, 24:, j] & d['observed'][:, :-24, j]
        diff = (logs[:, 24:, j] - logs[:, :-24, j]) / d['std'][j]
        expected = max(1e-4, float(np.mean(diff[valid] ** 2)))
        assert result['rates'][j] == pytest.approx(expected)
    with pytest.raises(ValueError, match='phase'):
        floor_rates(data('report'))


def test_augmentation_hides_authentic_current_value_and_preserves_labels():
    d = data()
    sid, at, length = 0, 40, 16
    class FixedRng:
        def __init__(self):
            self.coins = iter((.9, .1))

        def random(self):
            return next(self.coins)  # augment and choose the all-six branch

        def choice(self, choices, size=None):
            if size is None:
                return np.asarray(choices)[0]
            return np.full(size, 64)

    batch = augmentation(d, [(sid, at)], length, FixedRng())
    values, obs, ages, clock, ids, hours, y, ym, truth, rm = batch
    assert len(batch) == 10
    assert values.shape == obs.shape == ages.shape == (1, length, 7)
    assert clock.shape == (1, length, 2)
    assert ids.tolist() == [sid]
    assert hours[0, -1] == at
    assert rm[0].all()
    for channel in range(6):
        assert truth[0, channel] == d['values'][sid, at, channel]
        hidden = ~obs[0, :, channel]
        left = int(np.flatnonzero(hidden)[0])
        assert hidden[left:].all()
        assert np.all(values[0, left:, channel] == 0)
        global_left = at - (length - left) + 1
        prior = np.flatnonzero(d['observed'][sid, :global_left, channel])
        last_real = int(prior[-1]) if len(prior) else -10**9
        times = np.arange(at - length + 1 + left, at + 1)
        expected_age = np.log1p(np.minimum(times - last_real, 256)) / np.log(257)
        np.testing.assert_allclose(ages[0, left:, channel], expected_age, rtol=0, atol=1e-7)
    assert np.array_equal(ym[0], np.isfinite(d['raw'][sid, at + 6, :6]))
    np.testing.assert_array_equal(y[0], np.nan_to_num(d['raw'][sid, at + 6, :6], nan=0))


def test_augmentation_inputs_ages_and_labels_ignore_future_perturbations():
    a, b = data(), data()
    sid, at = 0, 40
    # Change all source-derived future fields after the origin. A batch ending
    # at the origin must remain byte-for-byte identical, including labels at +6.
    b['raw'][sid, at + 7:] = 1e6
    b['values'][sid, at + 7:] = -7
    b['observed'][sid, at + 7:] = False
    b['ages'][sid, at + 7:] = 1
    x = augmentation(a, [(sid, at)], 16, np.random.default_rng(9))
    z = augmentation(b, [(sid, at)], 16, np.random.default_rng(9))
    for left, right in zip(x, z):
        np.testing.assert_array_equal(left, right)


def test_augmentation_rejects_report_and_missing_current_is_unmasked():
    d = data('report')
    with pytest.raises(ValueError, match='phase'):
        augmentation(d, [(0, 40)], 16, np.random.default_rng(1))

    d = data()
    d['observed'][0, 40, :6] = False
    d['values'][0, 40, :6] = 0
    class SingleRng:
        def __init__(self):
            self.coins = iter((.9, .99))

        def random(self):
            return next(self.coins)

        def choice(self, choices, size=None):
            if size is None:
                return np.asarray(choices)[0]
            return np.full(size, 4)

    x = augmentation(d, [(0, 40)], 8, SingleRng())
    assert not x[-1].any()
    assert not x[-2].any()
    np.testing.assert_array_equal(x[0][0], d['values'][0, 33:41])
    np.testing.assert_array_equal(x[1][0], d['observed'][0, 33:41])

    d = data()
    x = augmentation(d, [(0, 40)], 8, SingleRng())
    assert x[-1][0].tolist() == [True, False, False, False, False, False]
    assert not x[1][0, -1, 0]
    assert x[1][0, -1, 1:6].all()

    class CleanRng:
        def random(self):
            return .1

        def choice(self, choices, size=None):
            return np.full(size, 4)

    clean = augmentation(d, [(0, 40)], 8, CleanRng())
    np.testing.assert_array_equal(clean[0][0], d['values'][0, 33:41])
    np.testing.assert_array_equal(clean[1][0], d['observed'][0, 33:41])
    assert not clean[-1].any()

    with pytest.raises(ValueError, match='full FIT history window'):
        augmentation(data(), [(0, 6)], 8, np.random.default_rng(1))
