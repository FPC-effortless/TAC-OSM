#!/usr/bin/env python3
"""Real-source, deterministic classical 6-feature ridge forecast (not TAC-OSM).

Developer/exposed CAL and DEV only. Never accesses UCI374 protected future
row labels; outputs original Git-checkpoint candidate JSON and score receipt.
No architecture superiority or independent test claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import uci_appliances_real_dev as source

PINNED_UCI374_SHA = "2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a"
LAMBDA = 1.0
FEATURES = 6


def feature(values, i):
    if i < 144:
        raise source.SourceError("past-lag feature unavailable")
    angle = 2 * math.pi * (i % 144) / 144
    return [values[i], values[i - 1], values[i - 6], values[i - 144],
            math.sin(angle), math.cos(angle)]


def solve(matrix, rhs):
    n = len(rhs)
    data = [matrix[i][:] + [rhs[i]] for i in range(n)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda j: abs(data[j][k]))
        data[k], data[pivot] = data[pivot], data[k]
        if abs(data[k][k]) < 1e-12:
            raise source.SourceError("ill-conditioned design")
        for j in range(k + 1, n):
            factor = data[j][k] / data[k][k]
            for c in range(k, n + 1):
                data[j][c] -= factor * data[k][c]
    w = [0.] * n
    for i in range(n - 1, -1, -1):
        w[i] = (data[i][n] - sum(data[i][j] * w[j] for j in range(i + 1, n))) / data[i][i]
        if not math.isfinite(w[i]):
            raise source.SourceError("nonfinite fit coefficient")
    return w


def fit(values):
    if len(values) < source.DEV_END:
        raise source.SourceError("original UCI374 source CAL and DEV partition required")
    fit_indices = range(144, source.FIT_END - source.H)
    n = len(fit_indices)
    xs = [feature(values, i) for i in fit_indices]
    means = [sum(x[j] for x in xs) / n for j in range(FEATURES)]
    scales = [max(math.sqrt(sum((x[j] - means[j]) ** 2 for x in xs) / n), 1e-6)
              for j in range(FEATURES)]
    gram = [[0.] * (FEATURES + 1) for _ in range(FEATURES + 1)]
    rhs = [0.] * (FEATURES + 1)
    for i, x in zip(fit_indices, xs):
        z = [1.] + [(x[j] - means[j]) / scales[j] for j in range(FEATURES)]
        y = values[i + source.H]
        for j in range(FEATURES + 1):
            rhs[j] += z[j] * y
            for k in range(j, FEATURES + 1):
                gram[j][k] += z[j] * z[k]
    for j in range(FEATURES + 1):
        for k in range(j):
            gram[j][k] = gram[k][j]
        if j:
            gram[j][j] += LAMBDA
    weights = solve(gram, rhs)
    return {"model_schema": "model01-uci374-lag-ridge-original-v1",
            "model_type": "classical_fit_only_ridge",
            "source_zip_sha256": PINNED_UCI374_SHA,
            "dataset_doi": source.DOI,
            "fit_index_range": [144, source.FIT_END - source.H],
            "fit_examples": n, "horizon_steps": source.H,
            "features": ["latest", "lag_1", "lag_6", "lag_144", "sin_day", "cos_day"],
            "regularizer": LAMBDA, "normalizer_fit_means": means,
            "normalizer_fit_scales": scales,
            "coefficients": weights,
            "status": "REAL_SOURCE_FIT_UNQUALIFIED_NO_INDEPENDENT_TEST"}


def forecast(model, values, i):
    if model.get("model_schema") != "model01-uci374-lag-ridge-original-v1":
        raise source.SourceError("unknown trained checkpoint")
    if model.get("source_zip_sha256") != PINNED_UCI374_SHA:
        raise source.SourceError("incorrect original source reference")
    x = feature(values, i)
    means, scales, w = (model[k] for k in
                        ("normalizer_fit_means", "normalizer_fit_scales", "coefficients"))
    if len(means) != FEATURES or len(scales) != FEATURES or len(w) != FEATURES + 1:
        raise source.SourceError("invalid coefficient schema")
    p = w[0] + sum(w[j + 1] * (x[j] - means[j]) / scales[j]
                   for j in range(FEATURES))
    if not math.isfinite(p):
        raise source.SourceError("nonfinite prediction")
    return max(0., p)


def evaluate(model, values):
    cal = list(source.eligible_indices(source.FIT_END, source.CAL_END))
    dev = list(source.eligible_indices(source.CAL_END, source.DEV_END))
    residuals = [abs(forecast(model, values, i) - values[i + source.H]) for i in cal]
    q, rank = source.quantile(residuals)
    metrics = {}
    for label, indices in (("cal", cal), ("dev", dev)):
        total = 0.
        sq = 0.
        inside = 0
        interval_score = 0.
        for i in indices:
            y = values[i + source.H]
            p = forecast(model, values, i)
            err = abs(p - y)
            total += err
            sq += err * err
            inside += err <= q
            interval_score += 2 * q + 20 * max(err - q, 0)
        metrics[label] = {"n": len(indices), "mae_wh": total / len(indices),
                          "rmse_wh": math.sqrt(sq / len(indices)),
                          "interval_width_wh": 2 * q,
                          "nominal_90_coverage": inside / len(indices),
                          "interval_score_90": interval_score / len(indices)}
    return {"study": "MODEL01-UCI374-REAL-CLASSICAL-RIDGE-FIT-DEV-001",
            "evidence_tier": "AUTHENTIC_SOURCE_FIT_EXPOSED_DEV_ONLY",
            "model_type": "classical_ridge_not_TAC_OSM",
            "source_zip_sha256": PINNED_UCI374_SHA,
            "fit_examples": model["fit_examples"], "calibration_q_90_wh": q,
            "calibration_rank": rank, "cal": metrics["cal"], "dev": metrics["dev"],
            "independent_test_scored": False, "protected_test_targets_parsed": False,
            "scientific_promotion": "NOT_QUALIFIED",
            "warnings": [
                "source is previously published and development score is exposed",
                "this ridge recipe was registered after persistence DEV was already inspected",
                "single original house and overlapping horizons prevent independent-case confidence",
                "original publisher on-device/real-time availability not proven",
                "source-original Git checkpoint plus exact numerical replay required before acceptance",
                "no architecture parity or learned TAC-OSM superiority inferred",
            ]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-checkpoint", type=Path, required=True)
    p.add_argument("--out-report", type=Path, required=True)
    p.add_argument("--replay-only", action="store_true")
    args = p.parse_args(argv)
    try:
        archive = source.fetch_official_zip()
        if source.digest(archive) != PINNED_UCI374_SHA:
            raise source.SourceError("original source bytes differ from frozen publisher ZIP")
        values = source.original_rows(archive)
        if args.replay_only:
            model = json.loads(args.out_checkpoint.read_text())
        else:
            if args.out_checkpoint.exists():
                raise source.SourceError("refuse to replace original model checkpoint")
            model = fit(values)
            args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with args.out_checkpoint.open("x", encoding="utf-8") as f:
                json.dump(model, f, sort_keys=True, indent=2, allow_nan=False)
        report = evaluate(model, values)
        if args.out_report.exists():
            raise source.SourceError("refuse to replace original scoring receipt")
        with args.out_report.open("x", encoding="utf-8") as f:
            json.dump(report, f, sort_keys=True, indent=2, allow_nan=False)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        print("ORIGINAL_CHECKPOINT_JSON=" + json.dumps(model, sort_keys=True, allow_nan=False))
        return 0
    except (source.SourceError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "reason": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
