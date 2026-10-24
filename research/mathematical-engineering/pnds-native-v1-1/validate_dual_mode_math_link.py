#!/usr/bin/env python3
"""Fail-closed dual-mode artifact/provenance and evidence-tier checks.

Does not prove that a passing neural graph has useful real-world skill.
"""
import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[3]
m=json.loads((Path(__file__).with_name("dual-mode-math-link-v1-5.json")).read_text())
assert m["schema"]=="pnds.tac_osm.dual_mode_evidence_v1.5"
assert m["streaming_state"]["fp32_values_per_sample"]==152
assert m["streaming_state"]["fp32_bytes_excluding_count_and_weights"]==608
assert m["historical_17_variant_registry_unchanged"] is True
assert m["original_aloha"]["matched_training_flops"] is False
assert m["original_aloha"]["fair_architecture_superiority"] is False
assert m["original_aloha"]["exposed_test_fresh_skill_qualification"] is False
assert m["entity_admission"]["real_outcome_forecast_gain"] is False
assert m["entity_admission"]["dynamic_eviction_compatible_with_parallel_scan"] is False
assert m["formal"]["full_17_model_source_theorems"]=="OPEN"
assert m["formal"]["full_tensor_fp_nonlinear_equivalence"]=="OPEN"
count=0
for key,blob in m["artifact_blobs"].items():
    if not key.startswith("TAC/"):continue
    f=root/key.removeprefix("TAC/")
    if not f.exists():raise AssertionError(f"declared source missing: {key}")
    actual=subprocess.check_output(["git","hash-object",str(f)],text=True).strip()
    if actual!=blob:raise AssertionError(f"source changed from pinned proven source: {key}")
    count+=1
assert count>=5
print(json.dumps({"status":"PASS_FALSE_PROMOTION_GUARD",
  "source_files_exact_blob_checked":count,
  "real_skill_claim":"OPEN","source_historical_17_variants_preserved":True},sort_keys=True))
