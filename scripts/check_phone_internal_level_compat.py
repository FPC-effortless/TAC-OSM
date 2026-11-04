#!/usr/bin/env python3
"""Read-only planner/source integration check; never activate a study."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.phone_research_ops import decide


def check():
    protocol = json.loads((ROOT / "research/phone_level_20261104/PROTOCOL.json").read_text())
    for path, digest in protocol["source_sha256"].items():
        if path != "src/tac_osm/phone_research_ops.py":
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
                raise ValueError("preserved source changed: " + path)
    helper = ROOT / "src/tac_osm/research_execution.py"
    vendored = ROOT / "research/collect012/policy/src/tac_osm/research_execution.py"
    if helper.read_bytes() != vendored.read_bytes():
        raise ValueError("restored policy differs from frozen original")
    original = json.loads((ROOT / "research/phone_queue/local_binding_gru512_20261103.json").read_text())
    legacy = decide(original)
    internal = decide({**original, "execution_level": "internal_research"})
    if legacy["execution_level"] != "independent_qualification":
        raise ValueError("legacy policy default changed")
    if internal["blockers"] != ["source-authorized real data and preregistered scientific plan required"]:
        raise ValueError("missing ordinary real-source prerequisite was bypassed")
    for decision in (legacy, internal):
        if decision["state"] != "BLOCKED_EVIDENCE_OR_DESIGN" or decision["backend"] != "github_actions":
            raise ValueError("unreviewed execution route")
        for key in ("approved_to_train", "approved_to_open_test", "automatic_submissions"):
            if decision[key] is not False:
                raise ValueError("planner granted execution authority")
    status = json.loads((ROOT / "research/local_binding_gru512_20261103/DRAFT_STATUS.json").read_text())
    if status["fitting_released"] is not False or status["root_RUN_REQUEST_present"] is not False:
        raise ValueError("draft is no longer the unlaunched candidate")
    if "torch" in sys.modules:
        raise RuntimeError("source-only check imported Torch")
    return {"status": "PASS_SOURCE_ONLY_PLANNER_COMPATIBILITY",
            "legacy_decision": legacy, "explicit_internal_copy_decision": internal,
            "preserved_original_rejection": "research/local_binding_gru512_20261103/PREFLIGHT_PLAN.json",
            "policy_helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
            "planner_sha256": hashlib.sha256((ROOT / "src/tac_osm/phone_research_ops.py").read_bytes()).hexdigest(),
            "synthetic_exception_admitted_by_this_check": False,
            "execution_review_completed": False, "models_constructed": 0,
            "checkpoint_loads": 0, "optimizer_updates": 0, "GPU_minutes": 0,
            "protected_TEST_opened": False, "independently_qualified": False,
            "neural_Lean_Python_IEEE_AD": "UNCHANGED_OPEN",
            "physical_device": {"target_device": "NOT_AVAILABLE",
                                "operator_support": "NOT_RUN", "numerical_agreement": "NOT_RUN",
                                "latency_p50_p95": "NOT_RUN", "RAM": "NOT_RUN",
                                "energy": "NOT_RUN", "airplane_mode": "NOT_RUN"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing evidence")
    result = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "models_constructed": 0,
                      "synthetic_candidate_still_blocked": True}))


if __name__ == "__main__":
    main()
