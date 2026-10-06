#!/usr/bin/env python3
"""Independent metric recomputation for TDBU-MTSK-001 raw records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="results/TACOSM-PLM-TDBU-MTSK-001.json")
    args = parser.parse_args()
    raw = json.loads(Path(args.path).read_text(encoding="utf-8"))
    rows = raw["rows"]

    required_arms = {"no_state", "single_timescale", "two_timescale", "mtsk"}
    assert {r["arm"] for r in rows} == required_arms
    assert sorted(set(raw["parameter_counts"])) == [86]

    def cell(arm: str, H: int) -> np.ndarray:
        values = [r["success"] for r in rows if r["arm"] == arm and r["H"] == H]
        assert len(values) == 5
        return np.asarray(values, dtype=np.float64)

    mtsk = cell("mtsk", 256)
    single = cell("single_timescale", 256)
    no_state = cell("no_state", 256)
    diff = mtsk - single

    rng = np.random.default_rng(991)
    draws = rng.choice(diff, size=(20_000, len(diff)), replace=True).mean(axis=1)
    ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]

    assert all(r["checkpoint_hash"] != r["initial_checkpoint_hash"] for r in [
        {
            "checkpoint_hash": row["checkpoint_hash"],
            "initial_checkpoint_hash": raw["initial_checkpoint_hashes"][f'256:{row["seed"]}:{row["arm"]}']
        } for row in rows if row["H"] == 256
    ])

    reset = np.asarray([r["reset_success"] for r in rows if r["H"] == 256 and r["arm"] == "mtsk"], dtype=np.float64)
    shuffle = np.asarray([r["shuffle_success"] for r in rows if r["H"] == 256 and r["arm"] == "mtsk"], dtype=np.float64)

    recomputed = {
        "H": 256,
        "mtsk_mean": float(mtsk.mean()),
        "single_mean": float(single.mean()),
        "no_state_mean": float(no_state.mean()),
        "mtsk_minus_single_mean": float(diff.mean()),
        "mtsk_minus_single_seed_values": diff.tolist(),
        "seed_bootstrap_95ci": ci,
        "mtsk_minus_no_state_mean": float(mtsk.mean() - no_state.mean()),
        "reset_mean": float(reset.mean()),
        "shuffle_mean": float(shuffle.mean()),
    }
    print(json.dumps(recomputed, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
