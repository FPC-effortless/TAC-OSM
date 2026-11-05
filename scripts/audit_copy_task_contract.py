#!/usr/bin/env python3
"""Source/fixture audit only; no Torch, checkpoint, heldout or model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.copy_task_contract import audit_cases, prospective_controls, validate_case


def run(output):
    if output.exists():
        raise FileExistsError("preserve existing audit evidence")
    protocol = json.loads((ROOT / "research/copy_contract_20261105/PROTOCOL.json").read_text())
    path = ROOT / protocol["source_fixture"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol["source_fixture_sha256"]:
        raise ValueError("original source fixture changed")
    original = json.loads(raw)
    training_values = {validate_case(c)[2] for c in original}
    controls = prospective_controls(training_values)
    source_report = audit_cases(original, training_values)
    control_reports = {family: audit_cases([c for c in controls if c["family"] == family], training_values)
                       for family in sorted({c["family"] for c in controls})}
    if source_report["same_prefix_query_selection_witnesses"] or source_report["literal_last_record_correct"] != 8:
        raise ValueError("original task no longer matches the frozen singleton fixture")
    for family in ("query_selection", "latest_write"):
        report = control_reports[family]
        if (not report["same_prefix_query_selection_witnesses"]
                or report["literal_last_record_correct"] == report["cases"]):
            raise ValueError("future controls fail to distinguish query-blind copying")
    if any(r["present_value_overlap_with_training"] for r in control_reports.values()):
        raise ValueError("prospective value novelty check failed")
    if "torch" in sys.modules or path.read_bytes() != raw:
        raise RuntimeError("source-only or immutable-fixture boundary failed")
    output.mkdir(parents=True)
    (output / "PROSPECTIVE_ENGINEERING_FIXTURES.json").write_text(json.dumps(controls, indent=2) + "\n")
    result = {"study": "COPYCONTRACT20261105", "scope": "CONSTRUCTED_ENGINEERING_NOT_CAPABILITY",
              "status": "PASS_SOURCE_TASK_CONTRACT_AUDIT", "source_fixture_sha256": hashlib.sha256(raw).hexdigest(),
              "original_fixture": source_report, "prospective_controls": control_reports,
              "prospective_case_count": len(controls), "prospective_controls_already_exposed": True,
              "old_gates_changed": False, "707_allocation_changed_or_consumed": False,
              "models_constructed": 0, "checkpoint_loads": 0, "optimizer_updates": 0,
              "GPU_minutes": 0, "protected_TEST_opened": False, "independently_qualified": False,
              "neural_IEEE_AD_source_correspondence": "OPEN_UNCHANGED",
              "device_measurements": "NOT_RUN_NO_PHYSICAL_DEVICE",
              "next_use": "Requires a new separately reviewed protocol; never silently add controls to707"}
    (output / "RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "original_cases": len(original),
                      "original_query_witnesses": 0, "prospective_engineering_cases": len(controls),
                      "models_constructed": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
