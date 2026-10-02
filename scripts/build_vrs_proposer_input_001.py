#!/usr/bin/env python3
"""Build the proposer-visible VRS-001 input artifact.

This script intentionally emits only public representation inputs:
the frozen proposer-facing domain description, calibration states, and the
registered state/task rows. It never serializes the evaluator's task score,
safety labels, target identities, truth tables, verifier results, or outcomes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_verified_representation_synthesis_001.py"
DOMAIN = ROOT / "research" / "vrs-001" / "PROPOSER-DOMAIN-001.md"
OUT = ROOT / "research" / "vrs-001" / "proposer_input_001.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("vrs_runner_input", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load VRS runner")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load_runner()
    domain_hash = hashlib.sha256(DOMAIN.read_bytes()).hexdigest()
    rows = []
    queries = []
    for seed in mod.SEEDS:
        for m in mod.M_LEVELS:
            rows.extend(asdict(s) for s in mod.states(seed, m))
            queries.extend(asdict(t) for t in mod.tasks(seed, m))

    payload = {
        "experiment_id": mod.CONTRACT,
        "input_version": "0.1",
        "domain_artifact": "research/vrs-001/PROPOSER-DOMAIN-001.md",
        "domain_sha256": domain_hash,
        "calibration_states": [asdict(s) for s in mod.CALIBRATION_STATES],
        "state_rows": rows,
        "task_rows": queries,
        "forbidden_input_classes": [
            "action outcomes",
            "post-action states",
            "target identities",
            "truth tables",
            "success labels",
            "benchmark scores",
            "verifier outputs",
        ],
        "provenance": {
            "generator": "scripts/build_vrs_proposer_input_001.py",
            "domain_visibility": "public-proposer-only",
            "outcome_visibility": "none",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"domain_sha256={domain_hash}")
    print(f"state_rows={len(rows)} task_rows={len(queries)}")


if __name__ == "__main__":
    main()
