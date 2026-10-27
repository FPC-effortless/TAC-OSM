#!/usr/bin/env python3
"""GP-UNC-002 constructed-only CPU filter timing, not production latency.

Runs BOTH exact-precision recomputation reference and rank-one update
on identical generated arrays, comparing predictions before reporting timing.
This script must not fetch original data, train neural models or open TEST.
"""
from __future__ import annotations
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.gp_rankone_nofit import RankOneProjectedResidualGP
from tac_osm.gp_sparse_nofit import ProjectedResidualGP


def generated(m, *, updates=100):
    rng = random.Random(20261027 + m)
    z = [[rng.uniform(-3, 3), rng.uniform(-3, 3)] for _ in range(m)]
    rows = [([rng.uniform(-3, 3), rng.uniform(-3, 3)], rng.uniform(-2, 2))
            for _ in range(updates)]
    query = [0.25, -0.25]
    return z, rows, query


def timed(cls, z, rows, query):
    model = cls(z, noise_variance=.15, jitter=1e-8)
    start = time.perf_counter()
    for i, (x, y) in enumerate(rows):
        kw = dict(feature_at=i, label_at=i+.75, cutoff=i+1.)
        if cls is RankOneProjectedResidualGP:
            kw["issued_at"] = i + .25
        model.observe(x, y, **kw)
    elapsed = time.perf_counter() - start
    pred = model.predict(query)
    return elapsed, pred


def main():
    result = {
        "schema": "gp-unc-002.constructed-cpu-timing.v1",
        "evidence": "CONSTRUCTED_ALGORITHM_ONLY_NOT_REAL_WORLD_QUALIFICATION",
        "rows": [],
        "hardware_uncontrolled": True,
        "fit_real_models": False,
    }
    for m in (4, 8, 16, 32):
        z, rows, query = generated(m)
        fast, slow = [], []
        for repeat in range(3):
            # Alternate execution order to reduce trivial cache-order bias.
            classes = (RankOneProjectedResidualGP, ProjectedResidualGP) if repeat%2==0 else (
                ProjectedResidualGP, RankOneProjectedResidualGP)
            values = {}
            for cls in classes:
                elapsed, pred = timed(cls, z, rows, query)
                values[cls] = (elapsed, pred)
            tfast, pfast = values[RankOneProjectedResidualGP]
            tslow, pslow = values[ProjectedResidualGP]
            if (abs(pfast.mean - pslow.mean) > 1e-8 or
                    abs(pfast.latent_variance - pslow.latent_variance) > 1e-8):
                raise AssertionError("numerical parity failure; refuse timing report")
            fast.append(tfast)
            slow.append(tslow)
        row = {
            "inducing_points": m,
            "updates": len(rows),
            "repeats": len(fast),
            "rankone_median_seconds": statistics.median(fast),
            "precision_ref_median_seconds": statistics.median(slow),
            "reference_over_rankone_runtime_ratio": statistics.median(slow)/statistics.median(fast),
            "prediction_parity_tolerance": 1e-8,
            "observed_not_hardware_generalized": True,
        }
        result["rows"].append(row)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
