#!/usr/bin/env python3
"""TACOSM-PLM-C5-STATE-ADMISSION-006."""
from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.c5_admission_scaling_audit import build_population_extended
from tac_osm.c5_state_addressing import C5PersistentStateAdmission, make_persistent_query

EXPERIMENT_ID = "TACOSM-PLM-C5-STATE-ADMISSION-006"
SEEDS = tuple(range(5))
M_LEVELS = (1024, 2048, 4096, 8192)
TRIALS = 64


def main() -> None:
    rows = []
    for seed in SEEDS:
        adapter = C5PersistentStateAdmission(seed=seed)
        for m in M_LEVELS:
            candidates = build_population_extended(seed, m)
            target_descriptor = candidates[seed % len(candidates)].descriptor
            target_ids = tuple(
                c.key for c in candidates if c.descriptor == target_descriptor
            )
            recalls = []
            raw = []
            admitted = []
            ops = []
            for trial in range(TRIALS):
                query = make_persistent_query(
                    target_descriptor,
                    seed=seed + 10000 * m,
                    step=trial,
                )
                result = adapter.lookup(
                    query=query,
                    m=m,
                    target_ids=target_ids,
                )
                recalls.append(float(result.target_admitted))
                raw.append(result.raw_candidates)
                admitted.append(result.admitted_candidates)
                ops.append(result.routing_ops)
            rows.append({
                "seed": seed,
                "M": m,
                "K90": adapter.k90,
                "rho_from_c5": adapter.rho,
                "requested_tables": result.requested_tables,
                "actual_tables": result.actual_tables,
                "target_admission_rate": statistics.fmean(recalls),
                "mean_raw_candidates": statistics.fmean(raw),
                "mean_admitted_candidates": statistics.fmean(admitted),
                "mean_routing_ops": statistics.fmean(ops),
                "full_scan_cost_proxy": float(m),
                "routing_fraction": statistics.fmean(ops) / m,
            })

    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "lineage": {
            "stacked_from": "research/plm-unified-phase-001",
            "source_mechanism": "existing C5 OR-LSH + dense CDL machinery",
            "purpose": "test settled C5 admission mechanism as persistent PLM state addressing",
            "not_reopened": [
                "C5 representation-arm selection",
                "C5 LSH operating-point search",
            ],
        },
        "protocol": {
            "seeds": SEEDS,
            "M": M_LEVELS,
            "trials_per_cell": TRIALS,
            "calibration_m": 1024,
            "K90_alpha": 0.10,
            "lsh_cap": 128,
        },
        "decision_rules": [
            "C5-derived rho is an inherited mechanism parameter, not a new search variable.",
            "The target query is evaluator-owned and never passed to the router as an answer.",
            "Admission recall and routing work are reported separately.",
            "Finite M levels do not imply asymptotic scaling.",
            "A successful state-admission transfer is not a semantic-language result.",
        ],
        "rows": rows,
    }
    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
