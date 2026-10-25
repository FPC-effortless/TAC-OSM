"""FIT-only natural-gap and artificial-gap data helpers for ME102."""
from __future__ import annotations

import numpy as np

EPSILON = 1e-4
GAP_LENGTHS = np.asarray((4, 8, 16, 32, 64), dtype=np.int64)
POLLUTANTS = 6


def _fit_only(d):
    if d.get('phase') != 'fit':
        raise ValueError("ME102 gap helpers only accept phase='fit' data")


def floor_rates(d):
    """Return the 24-hour standardized log-difference floor and pair counts.

    For pollutant j, this is the mean of (z[t,j] - z[t-24,j])**2 over
    observed endpoint pairs in the FIT cohort. The returned counts make sparse
    estimates explicit. Values use ME100's FIT standard deviation for
    ``log1p(raw)``; the 1e-4 epsilon is only a lower bound on the final rate.
    """
    _fit_only(d)
    raw = np.asarray(d['raw'])
    observed = np.asarray(d['observed'], dtype=bool)
    if raw.ndim != 3 or raw.shape[-1] < POLLUTANTS or observed.shape != raw.shape:
        raise ValueError('expected [site,time,feature] raw and observed arrays')
    std = np.asarray(d.get('std'), dtype=np.float64)
    if std.shape[0] < POLLUTANTS or not np.isfinite(std[:POLLUTANTS]).all() or np.any(std[:POLLUTANTS] <= 0):
        raise ValueError('expected positive ME100 FIT standard deviations for six pollutants')
    if raw.shape[1] <= 24:
        return {'rates': np.full(POLLUTANTS, EPSILON, dtype=np.float64),
                'counts': np.zeros(POLLUTANTS, dtype=np.int64)}

    x = raw[..., :POLLUTANTS].astype(np.float64)
    obs = observed[..., :POLLUTANTS] & np.isfinite(x)
    logs = np.zeros_like(x)
    logs[obs] = np.log1p(x[obs])
    pair_mask = obs[:, 24:] & obs[:, :-24]
    counts = pair_mask.sum(axis=(0, 1)).astype(np.int64)
    sq = ((logs[:, 24:] - logs[:, :-24]) / std[:POLLUTANTS]) ** 2
    totals = np.where(pair_mask, sq, 0.0).sum(axis=(0, 1))
    rates = np.maximum(EPSILON,
                       np.divide(totals, counts, out=np.zeros(POLLUTANTS), where=counts > 0))
    return {'rates': rates.astype(np.float64), 'counts': counts}


def augmentation(d, origins, length, rng):
    """Build an FIT batch with causal artificial gaps on observed pollutants.

    Returns ``values, observed, ages, clock, ids, hours, yfuture,
    maskfuture, current_truth, reconstruction_mask`` as NumPy arrays ready for
    ``torch.as_tensor``. Origins are ``(site_id, hour_index)`` pairs. A hidden
    channel is selected only from the six pollutants recorded at the current
    origin; its original normalized value remains available in
    ``current_truth`` for the auxiliary reconstruction objective.
    """
    _fit_only(d)
    if not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError('length must be a positive integer')
    origins = list(origins)
    if not origins:
        raise ValueError('origins must be nonempty')
    ids = np.asarray([int(p[0]) for p in origins], dtype=np.int64)
    at = np.asarray([int(p[1]) for p in origins], dtype=np.int64)
    values_all = np.asarray(d['values'])
    observed_all = np.asarray(d['observed'], dtype=bool)
    raw = np.asarray(d['raw'])
    nsite, ntime, nfeature = values_all.shape
    if (np.any(ids < 0) or np.any(ids >= nsite) or np.any(at < length - 1)
            or np.any(at >= ntime - 6)):
        raise ValueError('origin lacks a full FIT history window or six-hour target')
    start = at[:, None] - np.arange(length - 1, -1, -1, dtype=np.int64)[None, :]
    values = values_all[ids[:, None], start].copy()
    observed = observed_all[ids[:, None], start].copy()
    ages = np.asarray(d['ages'])[ids[:, None], start].copy()
    clock = np.asarray(d['clock'])[ids[:, None], start].copy()
    hours = np.asarray(d['hours'])[ids[:, None], start].copy()

    truth = np.zeros((len(origins), POLLUTANTS), dtype=np.float32)
    reconstruction_mask = np.zeros((len(origins), POLLUTANTS), dtype=bool)
    source_current = observed_all[ids, at, :POLLUTANTS]
    current_channels = np.asarray(d['values'])[ids, at, :POLLUTANTS]
    truth[:] = np.where(source_current, current_channels, 0)
    hidden_channels = np.zeros((len(origins), POLLUTANTS), dtype=bool)
    for b in range(len(origins)):
        # Half of origins keep their natural observation pattern, providing
        # ordinary fully observed histories at the inference distribution.
        if rng.random() < 0.5:
            continue
        candidates = np.flatnonzero(source_current[b])
        # Keep entirely unobserved origins natural: there is no auxiliary
        # target, and fabricating a block would only corrupt their history.
        if not len(candidates):
            continue
        if rng.random() < 0.65:
            hidden_channels[b, :] = True
        else:
            hidden_channels[b, int(rng.choice(candidates))] = True
        reconstruction_mask[b] = source_current[b] & hidden_channels[b]

    lengths = np.asarray(rng.choice(GAP_LENGTHS, size=len(origins)), dtype=np.int64)
    lengths = np.minimum(lengths, int(length))
    for b in range(len(origins)):
        channels = np.flatnonzero(hidden_channels[b])
        if not len(channels):
            continue
        gap = int(lengths[b])
        left = int(length) - gap
        # Mask the full artificial block, including naturally missing slots.
        # This keeps the block boundary independent of source values.
        observed[b, left:, channels] = False
        values[b, left:, channels] = 0

        for channel in channels:
            # The anchor is the latest genuine reading strictly before the
            # hidden block, found across the full FIT past. Nothing inside the
            # hidden block can refresh that channel's age.
            global_left = int(at[b]) - gap + 1
            prior = np.flatnonzero(observed_all[int(ids[b]), :max(global_left, 0), channel])
            last_real = int(prior[-1]) if len(prior) else -10**9
            for j in range(left, int(length)):
                t = int(start[b, j])
                stale = min(max(t - last_real, 0), 256)
                ages[b, j, channel] = np.log1p(stale) / np.log(257.0)

    y = raw[ids, at + 6, :POLLUTANTS].astype(np.float32)
    maskfuture = np.isfinite(y)
    yfuture = np.nan_to_num(y, nan=0.0).astype(np.float32)
    return (values.astype(np.float32), observed, ages.astype(np.float32),
            clock.astype(np.float32), ids, hours, yfuture, maskfuture,
            truth, reconstruction_mask)
