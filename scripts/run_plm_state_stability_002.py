#!/usr/bin/env python3
"""TACOSM-PLM-STATE-STABILITY-002."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.plm_research_phases import (
    StateStabilityRow,
    aggregate_rows,
    run_no_verifier_control,
    run_state_stability,
)
import statistics

EXPERIMENT_ID = "TACOSM-PLM-STATE-STABILITY-002"
SEEDS = tuple(range(5))
HORIZON = 256
RATES = (0.0, 0.01, 0.05, 0.10, 0.20, 1.0)


def main() -> None:
    rows = [
        run_state_stability(
            seed=seed, horizon=HORIZON, false_accept_rate=rate
        )
        for seed in SEEDS
        for rate in RATES
    ]
    rows.extend(
        run_no_verifier_control(seed=seed, horizon=HORIZON)
        for seed in SEEDS
    )
    by_rate = []
    for rate in RATES:
        group = [r for r in rows if r.arm == "false_accept" and r.false_accept_rate == rate]
        by_rate.append({
            "false_accept_rate": rate,
            "wrong_write_rate_mean": statistics.fmean(r.wrong_write_rate for r in group),
            "wrong_verified_writes_mean": statistics.fmean(r.wrong_verified_writes for r in group),
            "verified_writes_mean": statistics.fmean(r.verified_writes for r in group),
            "final_state_size_mean": statistics.fmean(r.final_state_size for r in group),
        })

    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "lineage": {
            "stacked_from": "research/plm-unified-phase-001",
            "purpose": "isolate persistent-state stability",
            "not_reopened": [
                "C5 representation-arm selection",
                "C5 LSH operating-point selection",
            ],
        },
        "protocol": {
            "seeds": SEEDS,
            "horizon": HORIZON,
            "false_accept_rates": RATES,
            "control": "no_verifier",
        },
        "metrics": [
            "wrong_verified_writes",
            "wrong_write_rate",
            "verified_writes",
            "final_state_size",
            "retractions",
            "supersessions",
        ],
        "rows": [r.__dict__ for r in rows],
        "summary_by_arm": aggregate_rows(rows, "arm"),
        "dose_response": by_rate,
        "decision_rules": [
            "Strict rate=0 is the verifier-gated control.",
            "No-verifier accepts every proposal and is an upper contamination control.",
            "Injected false acceptance is a stress process, not a real-world verifier-rate estimate.",
            "Finite horizon does not imply indefinite stability or divergence.",
        ],
    }
    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
