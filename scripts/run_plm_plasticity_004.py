#!/usr/bin/env python3
"""TACOSM-PLM-PLASTICITY-004."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.plm_research_phases import (
    aggregate_rows,
    run_plasticity_sequence,
)

EXPERIMENT_ID = "TACOSM-PLM-PLASTICITY-004"
SEEDS = tuple(range(5))


def main() -> None:
    rows = [
        row
        for seed in SEEDS
        for row in run_plasticity_sequence(seed=seed)
    ]
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "lineage": {
            "stacked_from": "research/plm-unified-phase-001",
            "mechanism_source": "slow shared + fast specialist plasticity hypothesis",
        },
        "protocol": {
            "seeds": SEEDS,
            "train_domains": (0, 1, 2),
            "heldout_domain": 3,
            "examples_per_domain": 200,
            "dimension": 8,
        },
        "arms": [
            "global_fast",
            "specialist_fast",
            "slow_shared_fast_specialist",
        ],
        "metrics": [
            "retention by trained domain",
            "mean final retention",
            "early adaptation error",
            "heldout accuracy",
            "shared update L1",
            "specialist update L1",
        ],
        "rows": [row.__dict__ for row in rows],
        "summary_by_arm": aggregate_rows(rows, "arm"),
        "decision_rules": [
            "This tests a plasticity mechanism, not reproduction of an external Mini-AGI implementation.",
            "Retention and rapid adaptation are reported separately.",
            "Held-out domain accuracy is a generalization control.",
            "Lower parameter movement is not itself a capability result.",
            "No specialization capability is promoted without held-out support.",
        ],
    }
    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
