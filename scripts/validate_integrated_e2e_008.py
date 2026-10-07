#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from pathlib import Path

x = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert x["experiment_id"] == "TACOSM-PLM-INTEGRATED-E2E-008"
assert x["status"] == "measured"
s = x["summary"]
assert s["gradient_surface_pass"]
assert s["classifier_decision_integrity_pass"]
assert s["oracle_q2_accuracy"] == 1.0
assert s["primary_q2_seed_bootstrap_ci95"] if "primary_q2_seed_bootstrap_ci95" in s else True
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
