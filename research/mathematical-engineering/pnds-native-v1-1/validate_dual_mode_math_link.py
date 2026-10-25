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

# Reciprocal v1.7 negative real evidence and exact source blob guard.
v17=json.loads((Path(__file__).with_name("temporal-retention-link-v1-7.json")).read_text())
assert v17["schema"]=="pnds.tac_osm.temporal_memory_retention_v1.7"
assert v17["architecture_status"]=="NATIVE_INSPIRED_EXPERIMENTAL_UNPROMOTED"
assert v17["stable_timescale"]["parameter_count"]==93324
assert v17["controlled_retention"]["parameter_count"]==93324
assert v17["controlled_retention"]["useful_factual_recall_guaranteed"] is False
assert v17["authentic_data"]["flop_parity_certified"] is False
assert v17["authentic_data"]["independent_real_skill_qualified"] is False
assert v17["caveats"]["real_world_superiority"] is False
assert v17["caveats"]["all_historical_variant_registry_untouched"] is True
for baseline,trial in [
    (v17["authentic_data"]["baseline_all_mean_dev_mse"],
     v17["authentic_data"]["timescale_dev_mse"]),
    (v17["authentic_data"]["baseline_all_mean_dev_mse"],
     v17["authentic_data"]["controlled_gate_dev_mse"])]:
    assert all(b<t for b,t in zip(baseline,trial))
for path,blob in {
    "src/tac_osm/variants/stable_timescale_world.py":v17["stable_timescale"]["source_git_blob"],
    "src/tac_osm/variants/retention_controlled_world.py":v17["controlled_retention"]["source_git_blob"],
    "research/mathematical-engineering/pnds-native-v1-1/TIMESCALE_RETENTION_RESULTS_V1_7.md":
      v17["evidence"]["tac_report_blob"]
}.items():
    actual=subprocess.check_output(["git","hash-object",str(root/path)],text=True).strip()
    assert actual==blob, "v1.7 source drift: "+path

