"""Pure original-FIT sampling; monitors are ancestor-exposed internal diagnostics."""
import numpy as np
STEPS, BATCH, GAP_PER_BATCH = 4096, 16, 2
MONITOR_SEED, FIT_STOP = 110900, 16104
MONITOR_SCOPE = 'original FIT internal monitor; ancestor-exposed, not fresh or independent'


def _origins(value, label):
    arr = np.asarray(value)
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f'{label} must contain exact integer [station,hour] pairs')
    if (np.any(arr[:, 0] < 0) or np.any(arr[:, 0] >= 12) or
            np.any(arr[:, 1] < 255) or np.any(arr[:, 1] >= FIT_STOP - 6)):
        raise ValueError(f'{label} lacks original FIT station/history/target eligibility')
    rows = [tuple(map(int, row)) for row in arr]
    if len(set(rows)) != len(rows): raise ValueError(f'{label} contains duplicate origins')
    return rows


def _pools(d):
    if d.get('phase') != 'fit': raise ValueError('ME110 sampling accepts original FIT pools only')
    gap = _origins(d['gap'], 'gap'); general = _origins(d['general'], 'general')
    gap_set = set(gap)
    if not gap_set.issubset(general): raise ValueError('original FIT gap must belong to general pool')
    return gap, [row for row in general if row not in gap_set]


def make_monitor(d, *, gap_size=64, general_size=256):
    gap, general = _pools(d)
    if any(not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1 for n in (gap_size, general_size)):
        raise ValueError('monitor sizes must be positive integers')
    if len(gap) <= gap_size or len(general) <= general_size:
        raise ValueError('monitor must leave both update pools nonempty')
    rng = np.random.default_rng(MONITOR_SEED)
    return np.asarray([gap[i] for i in rng.permutation(len(gap))[:gap_size]] +
                      [general[i] for i in rng.permutation(len(general))[:general_size]], dtype=np.int64)


def _update_pools(d, monitor):
    gap, general = _pools(d); held = set(_origins(monitor, 'monitor'))
    if not held.issubset(set(gap) | set(general)): raise ValueError('monitor outside FIT pools')
    pools = [[row for row in pool if row not in held] for pool in (gap, general)]
    if any(not pool for pool in pools): raise ValueError('monitor exhausted update pool')
    return pools


def _steps(nsteps):
    if not isinstance(nsteps, (int, np.integer)) or isinstance(nsteps, bool) or nsteps < 1:
        raise ValueError('nsteps must be a positive integer')


def _cycle(pool, count, rng):
    chunks = []; left = count
    while left:
        size = min(left, len(pool)); chunks.append(np.asarray(pool, dtype=np.int64)[rng.permutation(len(pool))[:size]])
        left -= size
    return np.concatenate(chunks)


def make_plan(d, seed, monitor, *, nsteps=STEPS):
    _steps(nsteps); gap, general = _update_pools(d, monitor)
    if nsteps == STEPS and len(general) < STEPS * 14:
        raise ValueError('full protocol requires 57344 distinct general update origins')
    rng = np.random.default_rng(seed)
    a = _cycle(gap, nsteps * 2, rng).reshape(nsteps, 2, 2)
    b = _cycle(general, nsteps * 14, rng).reshape(nsteps, 14, 2)
    return np.concatenate((a, b), axis=1)


def validate_plan(plan, d, monitor, *, nsteps=STEPS):
    _steps(nsteps); arr = np.asarray(plan)
    if arr.shape != (nsteps, BATCH, 2) or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError('plan must contain nsteps x 16 x 2 integer origins')
    gap, general = _update_pools(d, monitor); counts = []
    for values, pool in ((arr[:, :2], gap), (arr[:, 2:], general)):
        rows = [tuple(map(int, row)) for row in values.reshape(-1, 2)]
        if not set(rows).issubset(pool): raise ValueError('monitor, wrong-stratum, or non-FIT origin')
        for start in range(0, len(rows), len(pool)):
            chunk = rows[start:start + len(pool)]
            if len(set(chunk)) != len(chunk): raise ValueError('origin repeated before pool exhaustion')
        counts.append(len(set(rows)))
    if nsteps == STEPS and counts[1] != STEPS * 14:
        raise ValueError('full protocol requires 57344 distinct general update origins')
    return dict(scope=MONITOR_SCOPE, steps=nsteps, batch=BATCH, natural_gap_per_batch=2,
                general_per_batch=14, unique_gap_origins=counts[0], unique_general_origins=counts[1],
                monitor_origins=len(monitor), eligible_history=256)
