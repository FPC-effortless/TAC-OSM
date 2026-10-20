"""Exact discrete reconstruction; at most one float64 step for log1p only."""
import numpy as np


def compare_inputs(actual, expected, schema, telemetry):
    a = actual.detach().cpu().numpy(); b = expected.detach().cpu().numpy()
    assert a.dtype == b.dtype == np.float64 and a.shape == b.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    first = sum(len(schema['vocabulary'][name])+2 for name in schema['categorical'])
    log_columns = [first+2*i for i in range(len(schema['numeric']))]
    exact = np.ones(a.shape[1], dtype=bool); exact[log_columns] = False
    assert np.array_equal(a[:, exact], b[:, exact]), 'Categorical/missing indicators must match exactly'
    aa = a[:, log_columns].copy(); bb = b[:, log_columns].copy()
    assert (aa >= 0).all() and (bb >= 0).all()
    # Both signs of zero represent the same numerical zero, as in assert_close.
    aa[aa == 0] = 0.; bb[bb == 0] = 0.
    av = aa.view(np.uint64); bv = bb.view(np.uint64)
    steps = np.maximum(av, bv)-np.minimum(av, bv)
    maximum = int(steps.max()) if steps.size else 0
    assert maximum <= 1, f'log1p reconstruction differs by {maximum} float64 steps'
    telemetry['values'] += int(a.size)
    telemetry['max_abs'] = max(telemetry['max_abs'], float(np.max(np.abs(a-b))) if a.size else 0.)
    telemetry['max_ULP'] = max(telemetry['max_ULP'], maximum)
