#!/usr/bin/env python3
"""Independent recomputation for temporal-scale transfer results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXPERIMENT_ID = "TACOSM-PLM-TDBU-TIMESCALE-TRANSFER-001"
ARMS = ("no_state", "one_timescale", "two_timescale", "three_timescale", "distributed_eight")
SEEDS = (0, 1, 2, 3, 4)


def bootstrap_difference(a: np.ndarray, b: np.ndarray, seed: int = 1991, rounds: int = 20000) -> tuple[float, float, float]:
    diffs = a - b
    rng = np.random.default_rng(seed)
    draws = rng.choice(diffs, size=(rounds, len(diffs)), replace=True).mean(axis=1)
    return float(np.mean(diffs)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    args = parser.parse_args()
    payload = json.loads(Path(args.result).read_text(encoding="utf-8"))
    primary = payload["endpoints"]["primary"]
    rows = payload["per_seed"]["rows"]

    def vals(arm: str) -> np.ndarray:
        return np.asarray(
            [r["success"] for r in rows if r["H"] == 256 and r["arm"] == arm],
            dtype=np.float64,
        )

    eight = vals("distributed_eight")
    three = vals("three_timescale")
    no_state = vals("no_state")
    assert len(eight) == len(three) == len(no_state) == 5
    delta, lo, hi = bootstrap_difference(eight, three)

    assert np.isclose(delta, primary["distributed_eight_minus_three_mean"])
    assert np.allclose([lo, hi], primary["seed_bootstrap_95ci"])
    assert np.isclose(float(eight.mean() - no_state.mean()), primary["distributed_eight_minus_no_state_mean"])

    means = {
        arm: {
            str(H): float(np.mean([r["success"] for r in rows if r["arm"] == arm and r["H"] == H]))
            for H in (64, 256, 512)
        }
        for arm in ARMS
    }
    print(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "primary_recomputed": {
            "distributed_eight_mean": float(eight.mean()),
            "three_timescale_mean": float(three.mean()),
            "no_state_mean": float(no_state.mean()),
            "difference": delta,
            "bootstrap_95ci": [lo, hi],
        },
        "means_by_H": means,
        "parameter_counts": payload["endpoints"]["parameter_counts"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
