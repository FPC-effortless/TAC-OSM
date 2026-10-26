"""Read-only ME109 saved training-curve audit; no data loading or fitting.

Prints JSON to stdout. Context labels follow the original runner's modulo-three
schedule. Curve windows are equal record counts, not identical examples.
Only trusted repository checkpoints should be supplied (torch pickle format).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def curve_summary(indices, losses, window=256):
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 2 * window:
        raise ValueError("insufficient records for disjoint matched windows")
    half = indices[len(indices) // 2:]
    middle_start = (len(indices) - window) // 2
    windows = {"first": indices[:window],
               "middle": indices[middle_start:middle_start + window],
               "last": indices[-window:]}
    result = {}
    for key in losses[0]:
        values = np.asarray([r[key] for r in losses], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite saved {key}")
        means = {name: float(values[ix].mean()) for name, ix in windows.items()}
        # OLS x is the actual optimizer-step coordinate, preserving schedule spacing.
        slope = float(np.polyfit(half + 1, values[half], 1)[0])
        result[key] = dict(means, last_minus_first=means["last"] - means["first"],
                           last_half_ols_slope_per_1000_steps=1000 * slope)
    return dict(records=len(indices), window_records=window,
                window_step_bounds={name: [int(ix[0] + 1), int(ix[-1] + 1)]
                                    for name, ix in windows.items()},
                last_half_step_bounds=[int(half[0] + 1), int(half[-1] + 1)],
                metrics=result)


def summarize(checkpoint_path, window=256):
    p = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    losses = p["losses"]
    n = len(losses)
    if p["steps"] != n:
        raise ValueError("checkpoint step/loss mismatch")
    # Exact scalar objective from me109_residual_training.role_objective.
    losses = [dict(row, objective=.5 * row["point"] + row["nll"]
                   + .1 * row["location_mse"] + .3 * row["reconstruction_nll"]
                   + .25 * row["interval"]) for row in losses]
    plan = np.asarray(p["plan"])
    if plan.shape != (n, 16, 2):
        raise ValueError("unexpected ME109 origin plan shape")
    unique, counts = np.unique(plan.reshape(-1, 2), axis=0, return_counts=True)
    receipt = json.loads(checkpoint_path.with_name("FIT_RECEIPT.json").read_text())
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_sha256 != receipt["checkpoint_sha256"]:
        raise ValueError("checkpoint hash differs from original fit receipt")
    guards = receipt["guard_aggregate"]
    fractions = {key: dict(count=int(value), fraction_per_step=float(value / n))
                 for key, value in guards.items()
                 if key.endswith((".projected", ".rounding_fallback", ".applied_rounding_fallback"))}
    return dict(seed=p["seed"], steps=n, checkpoint_sha256=checkpoint_sha256, checkpoint="/".join(checkpoint_path.parts[-3:]),
                exposure=dict(example_draws=int(plan.shape[0] * plan.shape[1]),
                              unique_station_hour_origins=len(unique),
                              repeat_draw_fraction=float(1 - len(unique) / counts.sum()),
                              maximum_draws_per_origin=int(counts.max())),
                all_contexts=curve_summary(np.arange(n), losses, window),
                by_context={str(context): curve_summary(np.arange(offset, n, 3), losses, window)
                            for offset, context in enumerate((8, 64, 256))},
                guard_counts_and_fractions=fractions,
                clipping_frequency="UNKNOWN: runner clips norm at 1 each step but saves no preclip norm",
                optimizer_steps=sorted({int(s["step"]) for s in p["optimizer"]["state"].values()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1]
                        / "research/replay-bundles/ME109/residual_guarded")
    parser.add_argument("--window", type=int, default=256)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("window must be positive")
    torch.set_num_threads(1)
    paths = sorted(args.bundle.glob("seed-*/final.pt"))
    if not paths:
        parser.error("no checkpoints found")
    print(json.dumps(dict(scope="saved stochastic FIT batches only; no validation convergence claim",
                          rows=[summarize(path, args.window) for path in paths]), indent=2,
                     allow_nan=False))


if __name__ == "__main__":
    main()
