#!/usr/bin/env python3
"""Offline source-side reciprocal PNDS mathematics links check.

Validates *this snapshot*, not remote PNDS parity nor proof of PyTorch/Lean
equivalence. The original source commit, PR, gate and negative evidence remain
in TAC-OSM. Pair with PNDS/scripts/validate_variant_math.py during review.
"""
import json
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINKS = ROOT / "research/pnds-variant-math-links.json"
HISTORICAL_SENTINELS = {
    "CORE-DIAGKEY004", "SYSTEM1-ME076", "SYSTEM1-ME077A",
    "SYSTEM1-ME079", "PHYSICAL-ME080", "PHYSICAL-ME094",
    "PHYSICAL-ME095", "ENTERPRISE-ME069", "ENTERPRISE-ME073",
    "TEMPORAL-V1", "EDGE-V1", "COUNTERFACTUAL-V1",
    "ESWAM-ESWAM010", "ESWAM-ESWAM012", "ESWAM-ESWAM013",
    "ESWAM-ESWAM014", "QUAL-001",
}


def validate(path=LINKS):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "pnds.tac_osm.variant_math_links.v1":
        raise ValueError("unexpected PNDS contract version")
    if data.get("canonical_authority") != {
        "mathematics": "FPC-effortless/pnds-lean",
        "source_and_evidence": "FPC-effortless/TAC-OSM"}:
        raise ValueError("incorrect source/formal authority")
    rows = data["variants"]
    ids = [v["id"] for v in rows]
    if len(ids) != len(set(ids)) or not HISTORICAL_SENTINELS <= set(ids):
        raise ValueError("duplicate or missing original research variant")
    for row in rows:
        name = row["id"]
        t, p = row.get("tac", {}), row.get("pnds", {})
        if (t.get("repository") != "FPC-effortless/TAC-OSM" or
            not re.fullmatch("[0-9a-f]{40}", t.get("commit", "")) or
            not isinstance(t.get("pull_request"), int) or t["pull_request"] < 1):
            raise ValueError(f"{name}: missing TAC original source PR/commit")
        if (p.get("repository") != "FPC-effortless/pnds-lean" or
            p.get("lean_path") != "PNDS/TACOSMVariantContracts.lean" or
            p.get("python_path") != "foundation/v1/variant_math.py"):
            raise ValueError(f"{name}: missing paired PNDS formulations")
        if (not row.get("kernel") or not row.get("formula") or
            row.get("formalization_status") != "PARTIAL_SHARED_KERNEL_NOT_WHOLE_MODEL" or
            row.get("lean_full_graph_proved") is not False or
            row.get("python_source_equivalence_verified") is not False or
            row.get("real_world_capability_inferred") is not False):
            raise ValueError(f"{name}: incomplete or falsely promoted semantics")
    prospective = data.get("prospective_graph_index")
    prospective_count = 0
    if prospective is not None:
        map_path = ROOT / prospective["tac_source_map_path"]
        raw = map_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != prospective["source_map_sha256"]:
            raise ValueError("Prospective source/equation registry digest mismatch")
        mapping = json.loads(raw)
        records = mapping.get("variants", [])
        if (len(records) != 6 or len(set(prospective["ids"])) != 6 or
                set(prospective["ids"]) != {r["id"] for r in records}):
            raise ValueError("All six prospective configurations must remain registered")
        for record in records:
            if (not re.fullmatch("[0-9a-f]{40}", record.get("source_commit", "")) or
                    record.get("source_repository") != "FPC-effortless/TAC-OSM" or
                    any(record.get(k) is not False for k in ("lean_full_graph_proved",
                        "python_global_source_equivalence_proved", "trained_capability_established"))):
                raise ValueError("Missing source pin or unsupported prospective graph promotion")
        prospective_count = len(records)
    return {"source_pinned_versions": len(rows), "reciprocal_manifest_locally_valid": True,
            "prospective_graph_configurations": prospective_count,
            "remote_equivalence_tested": False, "whole_model_proof_established": False}


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
