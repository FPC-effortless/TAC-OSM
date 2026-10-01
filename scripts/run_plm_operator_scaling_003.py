#!/usr/bin/env python3
"""TACOSM-PLM-OPERATOR-SCALING-003."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.plm_research_phases import (
    aggregate_rows,
    operator_scaling_row,
)

EXPERIMENT_ID = "TACOSM-PLM-OPERATOR-SCALING-003"
SEEDS = tuple(range(5))
E_LEVELS = (4, 16, 64, 256, 1024, 4096, 16384)


def main() -> None:
    rows = [
        operator_scaling_row(seed=seed, e=e)
        for seed in SEEDS
        for e in E_LEVELS
    ]
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "lineage": {
            "stacked_from": "research/plm-unified-phase-001",
            "purpose": "isolate computation/operator addressing over E",
        },
        "protocol": {
            "seeds": SEEDS,
            "E": E_LEVELS,
            "max_admitted": 8,
            "key_dim": 16,
            "query_noise": 0.05,
            "index_levels": (4, 8, 12),
        },
        "cost_accounting": {
            "indexed": "probes + raw_candidates + admitted_candidates",
            "full_scan": "E operator records",
            "break_even": "indexed < full_scan",
        },
        "rows": [r.__dict__ for r in rows],
        "summary": aggregate_rows(rows, "E"),
        "decision_rules": [
            "Full scan is the capability-preserving cost floor.",
            "Admission count is not a cost metric by itself.",
            "Finite E measurements are descriptive, not asymptotic proofs.",
            "Target admission is independent of cost.",
        ],
    }
    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
