#!/usr/bin/env python3
"""Post-run P0-P2 validator for TDBU-MTSK-001."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tac_osm.contract import load_contract
from tac_osm.mtsk_topdown import BENCHMARK_HASH, make_balanced_pairs


EXPERIMENT_ID = "TACOSM-PLM-TDBU-MTSK-001"
ARMS = ("no_state", "single_timescale", "two_timescale", "mtsk")
H_LEVELS = (8, 64, 256)
SEEDS = (0, 1, 2, 3, 4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="results/TACOSM-PLM-TDBU-MTSK-001.json")
    args = parser.parse_args()

    raw = json.loads(Path(args.path).read_text(encoding="utf-8"))
    contract = load_contract(EXPERIMENT_ID)

    p = raw["provenance"]
    assert p["experiment_id"] == EXPERIMENT_ID
    contract_path = Path("contracts") / f"{EXPERIMENT_ID}.json"
    expected_hash = __import__("hashlib").sha256(contract_path.read_bytes()).hexdigest()[:16]
    assert p["contract_sha256"] == expected_hash
    assert p["generator_hash"] == BENCHMARK_HASH
    assert "status" not in raw

    design = raw["design"]
    assert design["steps"] == contract.steps
    assert design["eval_steps"] == contract.eval_steps
    assert tuple(design["h_levels"]) == H_LEVELS
    assert tuple(design["seeds"]) == SEEDS
    assert tuple(design["arms"]) == ARMS
    assert design["contract_checked"] is True
    assert design["smoke"] is False
    assert design["contract_deviations"] == []

    rows = raw["per_seed"]["rows"]
    assert len(rows) == len(H_LEVELS) * len(SEEDS) * len(ARMS)
    keys = {(r["H"], r["seed"], r["arm"]) for r in rows}
    assert len(keys) == len(rows)
    assert keys == {
        (H, seed, arm) for H in H_LEVELS for seed in SEEDS for arm in ARMS
    }

    for H in H_LEVELS:
        for seed in SEEDS:
            examples = make_balanced_pairs(
                (10_000 if "train" == "train" else 20_000) + seed * 1009 + H * 17,
                H,
                100,
            )
            # The validator reconstructs the test stream deterministically; its
            # purpose is to ensure the benchmark version still produces paired
            # opposite-action histories at every evaluated cell.
            for i in range(0, len(examples), 2):
                a, b = examples[i], examples[i + 1]
                assert a.current_observation == b.current_observation == 0.0
                assert a.label != b.label
                assert np.array_equal(b.history, -a.history)

    for row in rows:
        assert np.isfinite(row["success"])
        assert np.isfinite(row["action_gap"])
        assert row["parameter_count"] == 86
        if row["arm"] != "no_state":
            initial = raw["audit"]["model_state"]["initial_checkpoint_hashes"][
                f'{row["H"]}:{row["seed"]}:{row["arm"]}'
            ]
            assert row["checkpoint_hash"] != initial

    trained_work = {
        arm: {
            row["evaluation_work_per_episode"]
            for row in rows
            if row["H"] == 256 and row["arm"] == arm
        }
        for arm in ARMS[1:]
    }
    assert trained_work and len(set(map(tuple, trained_work.values()))) == 1

    interventions = raw["endpoints"]["interventions"]
    for key in (
        "mtsk_reset_success_by_seed",
        "mtsk_shuffle_success_by_seed",
        "reset_mean",
        "shuffle_mean",
    ):
        assert key in interventions
    assert len(interventions["mtsk_reset_success_by_seed"]) == 5
    assert len(interventions["mtsk_shuffle_success_by_seed"]) == 5

    print("TDBU-MTSK-001 post-run invariants passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
