#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

from tac_osm.integrated_e2e_008_benchmark import HELDOUT, generator_hash

p = Path(sys.argv[1])
x = json.loads(p.read_text(encoding="utf-8"))
root = p.parents[1]
contract_path = root / "contracts" / "TACOSM-PLM-INTEGRATED-E2E-008.json"
contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
assert x["provenance"]["contract_sha256"] == contract_sha
assert x["provenance"]["benchmark_sha256"] == generator_hash()
assert x["protocol"]["registered_heldout"] == [list(c) for c in HELDOUT]
assert x["protocol"]["entity_side_channel"] is False
assert len({row["evaluation_episode_fingerprint"] for row in x["seed_results"]}) == len(x["seed_results"]) == 5
assert x["experiment_id"] == "TACOSM-PLM-INTEGRATED-E2E-008"
assert x["status"] == "measured"
s = x["summary"]
assert s["gradient_surface_pass"]
assert s["classifier_decision_integrity_pass"]
assert s["oracle_q2_accuracy"] == 1.0
assert len(s["primary_q2_composition_bootstrap_ci95"]) == 2
assert x["leakage_audit"]["training_evaluation_semantic_overlap_zero"]
assert x["leakage_audit"]["heldout_compositions_excluded_from_training"]
assert x["leakage_audit"]["evaluation_generated_after_training"]
assert x["leakage_audit"]["entity_side_channel_in_modalities"] is False
assert x["leakage_audit"]["image_entity_stripe_present"] is False
assert x["leakage_audit"]["image_payload_id_overlap"] is False
assert len(x["seed_results"]) == 5
assert len(s["mean_per_composition_q2_accuracy"]) == 12
assert all(r["training_evaluation_semantic_overlap"] == 0 for r in x["seed_results"])
assert all(r["normal_q2_decision_mismatches"] == 0 for r in x["seed_results"])
print(json.dumps({
    "primary_q2_mean": s["primary_q2_mean"],
    "min_seed_q2": s["all_seed_min_q2"],
    "composition_bootstrap_ci95": s["primary_q2_composition_bootstrap_ci95"]
}, indent=2))
