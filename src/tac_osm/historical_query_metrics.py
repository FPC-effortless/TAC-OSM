"""Scoring-only query specificity. No fitted transforms or model inputs here."""
from __future__ import annotations

import math
import random


def window_scores(predictions, targets, observed):
    if (len(predictions) != 8 or len(targets) != 8 or len(observed) != 8
            or any(type(v) is not bool for v in observed)
            or not observed[0] or any(not math.isfinite(p) for p in predictions)):
        raise ValueError("eight finite legal-query outputs and observed original anchor required")
    indices = [i for i, valid in enumerate(observed) if valid]
    if any(targets[i] is None or not math.isfinite(targets[i]) for i in indices):
        raise ValueError("actual observed source targets required")
    y = targets[0]
    mean_all = sum(predictions) / 8
    result = {
        "observed_targets": len(indices),
        "H64_abs_error": abs(predictions[0] - y),
        "H64_wrong_clock_mean_abs_error": sum(abs(p - y) for p in predictions[1:]) / 7,
        "H64_collapsed_query_abs_error": abs(mean_all - y),
        "all_query_MAE": sum(abs(predictions[i] - targets[i]) for i in indices) / len(indices),
        "circular_wrong_query_MAE": sum(abs(predictions[(i + 1) % 8] - targets[i]) for i in indices) / len(indices),
        "collapsed_query_MAE": sum(abs(mean_all - targets[i]) for i in indices) / len(indices),
        "prediction_range": max(predictions) - min(predictions),
        "observed_target_range": max(targets[i] for i in indices) - min(targets[i] for i in indices),
        "centered_MSE": None, "centered_null_MSE": None,
        "centered_status": "INCONCLUSIVE_TOO_FEW_OBSERVED_TARGETS",
    }
    if len(indices) >= 2:
        pm = sum(predictions[i] for i in indices) / len(indices)
        ym = sum(targets[i] for i in indices) / len(indices)
        result.update(
            centered_MSE=sum(((predictions[i] - pm) - (targets[i] - ym)) ** 2 for i in indices) / len(indices),
            centered_null_MSE=sum((targets[i] - ym) ** 2 for i in indices) / len(indices),
            centered_status="MEASURED_SCORING_ONLY",
        )
    return result


def station_gain_ci(stations, baseline, actual, *, draws=4000, seed=15015):
    if not (len(stations) == len(baseline) == len(actual)) or not stations:
        raise ValueError("aligned source station/error rows required")
    if any(not math.isfinite(v) for v in baseline + actual):
        raise ValueError("finite original errors required")
    groups = [[i for i, s in enumerate(stations) if s == owner] for owner in sorted(set(stations))]
    if len(groups) < 3 or draws < 2:
        raise ValueError("at least three source-station clusters required")
    rng = random.Random(seed)
    difference = [a - b for a, b in zip(baseline, actual)]
    resamples = []
    for _ in range(draws):
        idx = [i for _ in groups for i in rng.choice(groups)]
        resamples.append(sum(difference[i] for i in idx) / len(idx))
    resamples.sort()
    return [resamples[int(.025 * (draws - 1))], resamples[int(.975 * (draws - 1))]]


def compare_errors(stations, baseline, actual):
    ci = station_gain_ci(stations, baseline, actual)
    base = sum(baseline) / len(baseline)
    measured = sum(actual) / len(actual)
    gain = (base - measured) / max(base, 1e-12)
    return {"relative_error_gain": gain, "station_bootstrap_95_error_gain": ci,
            "gate_pass": gain >= .05 and ci[0] > 0}
