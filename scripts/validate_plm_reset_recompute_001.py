#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from tac_osm.integrated_e2e_008_benchmark import HELDOUT, GENERATOR_VERSION, generator_hash

path = Path(sys.argv[1])
root = path.parents[1]
artifact = json.loads(path.read_text(encoding="utf-8"))
contract_path = root / "contracts" / "TACOSM-PLM-PERSISTENCE-RESET-RECOMPUTE-001.json"
contract = json.loads(contract_path.read_text(encoding="utf-8"))

assert artifact["experiment_id"] == "TACOSM-PLM-PERSISTENCE-RESET-RECOMPUTE-001"
assert artifact["status"] == "measured"
assert artifact["provenance"]["contract_sha256"] == hashlib.sha256(contract_path.read_bytes()).hexdigest()
assert artifact["provenance"]["benchmark_sha256"] == generator_hash()
assert artifact["reference"]["benchmark_generator_version"] == GENERATOR_VERSION
assert artifact["reference"]["heldout"] == [list(x) for x in HELDOUT]
assert artifact["protocol"]["reference_experiment_id"] == "TACOSM-PLM-INTEGRATED-E2E-008"
for row in artifact["seed_results"]:\n    expected = contract["legacy_reference"]["evaluation_fingerprints"][str(row["seed"])]\n    assert row["evaluation_episode_fingerprint"] == expected\nassert len(artifact["seed_results"]) == 5
assert len({r["evaluation_episode_fingerprint"] for r in artifact["seed_results"]}) == 5
assert all(r["training_evaluation_semantic_overlap"] == 0 for r in artifact["seed_results"])
assert all(r["q1_intervention_mismatches"] == 0 for r in artifact["seed_results"])
assert all(r["recompute_identity_failures"] == 0 for r in artifact["seed_results"])

s = artifact["summary"]
assert len(s["paired_composition_bootstrap"]["ci95"]) == 2
assert artifact["leakage_audit"]["reference_benchmark_identity_pass"]
assert artifact["leakage_audit"]["reference_evaluation_fingerprints_match"]
assert artifact["leakage_audit"]["training_evaluation_semantic_overlap_zero"]
assert artifact["leakage_audit"]["same_evaluation_objects_within_seed"]
assert artifact["leakage_audit"]["reset_recompute_receives_observations_only"]
assert artifact["leakage_audit"]["q1_outcome_entered_recomputed_state"] is False
assert artifact["leakage_audit"]["q1_target_entered_recomputed_state"] is False
assert artifact["leakage_audit"]["q1_action_entered_recomputed_state"] is False
assert artifact["leakage_audit"]["verifier_signal_entered_recomputed_state"] is False
assert artifact["leakage_audit"]["recomputed_state_identity_pass"]
assert artifact["leakage_audit"]["q1_prediction_action_outcome_pair_integrity_pass"]
print(json.dumps({
    "normal_q2_mean": s["normal_q2_mean"],
    "reset_recompute_q2_mean": s["reset_recompute_q2_mean"],
    "gap": s["normal_minus_reset_recompute_gap"],
    "paired_composition_bootstrap_ci95": s["paired_composition_bootstrap"]["ci95"],
    "temporal_carry_supported": s["temporal_carry_supported"],
}, indent=2))
