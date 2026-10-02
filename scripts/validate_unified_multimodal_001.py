#!/usr/bin/env python3
"""Static preflight for TACOSM-UNIFIED-MULTIMODAL-001.

This is not a capability measurement. It checks contract shape and the
pre-action information boundary before numerical training is allowed.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"contracts"/"TACOSM-UNIFIED-MULTIMODAL-001.json"
RUNNER=ROOT/"scripts"/"run_unified_multimodal_001.py"

REQUIRED=(
    "experiment_id","title","question","hypothesis","arms","endpoints",
    "decision_rule","interpretation_order","h_levels","seeds","steps","eval_steps"
)

def main():
    obj=json.loads(CONTRACT.read_text())
    missing=[k for k in REQUIRED if k not in obj]
    if missing: raise SystemExit(f"missing contract keys: {missing}")
    if sum(bool(e.get("primary")) for e in obj["endpoints"]) != 1:
        raise SystemExit("contract must contain exactly one primary endpoint")
    if obj["status"] != "pre_registered":
        raise SystemExit("contract is not marked pre_registered")
    if not obj.get("held_constant"):
        raise SystemExit("held_constant must be non-empty")
    if set(obj["modalities"]) != {"language","image","audio"}:
        raise SystemExit("modalities are not exactly language/image/audio")
    if not obj.get("leakage_controls"):
        raise SystemExit("leakage controls missing")

    tree=ast.parse(RUNNER.read_text())
    forbidden={"gold","gold_index","target_action","acceptable_actions","true_edge_set","semantic_id"}
    public_calls={"encode","predict","discover","retrieve","confidence"}
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            if node.name in public_calls:
                names={a.arg for a in node.args.args}
                bad=sorted(n for n in names if n in forbidden)
                if bad:
                    raise SystemExit(f"forbidden truth argument(s) in {node.name}: {bad}")

    text=RUNNER.read_text()
    required_literals=("proposal miss", "verified", "TEST_WORLD_SEED", "TRAIN_WORLD_SEED", "VAL_WORLD_SEED")
    missing_literals=[x for x in required_literals if x not in text]
    if missing_literals:
        raise SystemExit(f"runner missing required integrity markers: {missing_literals}")

    print("UNIFIED-MULTIMODAL-001 PREFLIGHT: PASS")
    print("contract schema: PASS")
    print("single primary endpoint: PASS")
    print("modality set: PASS")
    print("runner public truth-argument audit: PASS")
    print("registered split/provenance markers: PASS")

if __name__=="__main__":
    main()
