#!/usr/bin/env python3
"""Run TACOSM-C5-COMPOSITION-BUDGET-AUDIT-001."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tac_osm.c5_composition_budget_audit import main_measure

def main() -> None:
    result = main_measure()
    out = Path("artifacts/TACOSM-C5-COMPOSITION-BUDGET-AUDIT-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("TACOSM-C5-COMPOSITION-BUDGET-AUDIT-001")
    print("calibration/held-out disjoint:", result["scope"]["calibration_heldout_disjoint_verified"])
    print("control L90:", result["control_scaling"]["pooled_L90"])
    print("control mean expected executions:", result["control_scaling"]["pooled_mean_expected_dense_executions"])
    for arm in result["architectures"]:
        row = result["arms"][arm]["0"]["dense"][4] if arm == "control" else result["arms"][arm]["0"]["1024"]
        print(f"{arm}: M1024 Top1={row['top1']:.4f} P90={row['P90_rank']:.4f} mean_rank={row['mean_rank']:.4f}")

if __name__ == "__main__":
    main()
