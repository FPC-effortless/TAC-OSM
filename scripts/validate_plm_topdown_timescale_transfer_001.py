#!/usr/bin/env python3
"""Post-run invariant validator for temporal-scale transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tac_osm.measurement.results import contract_fingerprint
from tac_osm.timescale_transfer import (
    ARMS,
    BENCHMARK_HASH,
    TEST_FILTER_ALPHAS,
    TRAIN_FILTER_ALPHAS,
)

EXPERIMENT_ID = "TACOSM-PLM-TDBU-TIMESCALE-TRANSFER-001"
ROOT = Path(__file__).resolve().parent.parent
EXPECTED_CONTRACT = contract_fingerprint(ROOT / "contracts" / f"{EXPERIMENT_ID}.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    args = parser.parse_args()
    raw = json.loads(Path(args.result).read_text(encoding="utf-8"))

    assert raw["provenance"]["experiment_id"] == EXPERIMENT_ID
    assert raw["provenance"]["contract_sha256"] == EXPECTED_CONTRACT
    assert raw["provenance"]["git_commit"] != "UNKNOWN"
    assert raw["endpoints"]["generator_hash"] == BENCHMARK_HASH
    assert raw["audit"]["benchmark"]["filter_sets_disjoint"] is True
    assert set(np.round(raw["audit"]["benchmark"]["train_filter_alphas"], 12)).isdisjoint(
        set(np.round(raw["audit"]["benchmark"]["test_filter_alphas"], 12))
    )

    rows = raw["per_seed"]["rows"]
    assert len(rows) == 3 * 5 * 5
    assert len({(r["H"], r["seed"], r["arm"]) for r in rows}) == len(rows)
    assert {r["H"] for r in rows} == {64, 256, 512}
    assert {r["seed"] for r in rows} == {0, 1, 2, 3, 4}
    assert {r["arm"] for r in rows} == set(ARMS)
    assert all(np.isfinite(r["success"]) and np.isfinite(r["action_gap"]) for r in rows)
    assert all(r["parameter_count"] == 86 for r in rows)
    assert raw["endpoints"]["parameter_counts"] == {arm: 86 for arm in ARMS}
    assert len(TRAIN_FILTER_ALPHAS) == 7 and len(TEST_FILTER_ALPHAS) == 7

    primary = raw["endpoints"]["primary"]
    assert len(primary["distributed_eight_success_by_seed"]) == 5
    assert len(primary["three_timescale_success_by_seed"]) == 5
    assert len(primary["no_state_success_by_seed"]) == 5
    assert primary["materiality_threshold"] == 0.10

    interventions = raw["endpoints"]["interventions"]
    assert len(interventions["distributed_eight_reset_success_by_seed"]) == 5
    assert len(interventions["distributed_eight_shuffle_success_by_seed"]) == 5
    print("PASS: temporal-scale transfer post-run invariants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
