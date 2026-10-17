"""Paired crossed-seed/world development inference; never multiply world count."""
from __future__ import annotations
import numpy as np


def crossed_draws(seeds, worlds, strata=None, replicates=2000, seed=5530001):
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(seeds, np.ones(seeds)/seeds, size=replicates)/seeds
    groups = [np.arange(worlds)] if strata is None else [np.flatnonzero(np.asarray(strata) == k) for k in sorted(set(strata))]
    ids = np.concatenate([rng.choice(g, size=(replicates, len(g)), replace=True) for g in groups], axis=1)
    return weights, ids


def paired_summary(costs, baseline, oracle=None, strata=None, replicates=2000):
    costs = np.asarray(costs, dtype=float)
    if costs.ndim == 1:
        costs = costs[None]
    baseline = np.asarray(baseline, dtype=float)
    weights, ids = crossed_draws(*costs.shape, strata, replicates)
    draws = np.take_along_axis(weights@costs, ids, axis=1).mean(1)
    base = baseline[ids].mean(1)
    delta = draws-base
    result = {"world_clusters": costs.shape[1], "model_seeds": costs.shape[0],
              "mean_cost": float(costs.mean()), "mean_baseline_cost": float(baseline.mean()),
              "paired_delta": float((costs-baseline).mean()),
              "paired_delta_95pct": np.quantile(delta, [.025, .975]).tolist(),
              "resampling": "Crossed: paired world clusters, global seed resample within each replicate; same world draws for controls/oracle"}
    if oracle is not None:
        oracle = np.asarray(oracle, dtype=float)
        denominator = float((baseline-oracle).mean())
        denominator_draw = (baseline[ids]-oracle[ids]).mean(1)
        good = denominator_draw > 1e-9
        fraction = (base-draws)[good]/denominator_draw[good]
        result.update({"oracle_gap": denominator,
                       "fraction_oracle_gap_closed": float((baseline-costs.mean(0)).mean()/denominator) if denominator > 1e-9 else None,
                       "fraction_95pct": np.quantile(fraction, [.025, .975]).tolist() if good.any() else None,
                       "fraction_invalid_denominator_rate": float((~good).mean())})
    return result
