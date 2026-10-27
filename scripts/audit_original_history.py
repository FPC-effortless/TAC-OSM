#!/usr/bin/env python3
"""Preflight or explicitly execute the HISTORY-AUDIT014 frozen FIT audit.

No source/checkpoint is read, model imported or evaluation run until external
source AND evaluator permission verifies against the exact plan/code binding.
Default is preflight. This program never trains or launches a GPU job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.history_audit_authorization import (  # noqa: E402
    AuditBlocked, audit_binding, verify_authorizations,
)
from tac_osm.phone_research_ops import decide, load_proposal  # noqa: E402

PROPOSAL = Path("research/phone_queue/original-history-audit014.json")
EXECUTION_FILES = (
    "scripts/audit_original_history.py", "scripts/original_history_trace.py",
    "src/tac_osm/history_audit_authorization.py",
    "src/tac_osm/historical_read_controls.py", "src/tac_osm/phone_research_ops.py",
)
EVIDENCE_BLOCKERS = {
    "independent signed prefit reward/source/ACL evidence required at execution",
    "independent evaluator and protected evaluation authorization required",
}


def preflight(source_receipt=None, evaluator_receipt=None):
    proposal = load_proposal(PROPOSAL, ROOT)
    plan = json.loads((ROOT / proposal["study_manifest"]).read_text())
    decision = decide(proposal)
    if (proposal["kind"] != "evaluation" or decision["backend"] != "github_actions"
            or decision["gpu_budget_minutes"] != 0
            or set(decision["blockers"]) - EVIDENCE_BLOCKERS
            or plan.get("no_new_model_fits") is not True
            or plan.get("no_architecture_superiority_claim") is not True):
        raise AuditBlocked("unsupported or unsafe original-source audit plan")
    execution = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                 for p in EXECUTION_FILES}
    binding = audit_binding(plan, proposal, execution)
    report = {"schema": "tac-osm.original-history-preflight.v1",
              "compute_decision": decision, "authorization_binding": binding,
              "execution_files": execution, "evaluation_executed": False,
              "scientific_qualification": False}
    try:
        permission = verify_authorizations(binding,
            {"source": source_receipt, "evaluator": evaluator_receipt},
            {"source": os.environ.get("TAC_OSM_SOURCE_VERIFIER_PUBLIC_KEY_HEX"),
             "evaluator": os.environ.get("TAC_OSM_EVALUATOR_VERIFIER_PUBLIC_KEY_HEX")})
        report["authorization"] = permission
        report["status"] = "READY_AUTHORIZED_FROZEN_DIAGNOSTIC"
    except AuditBlocked as exc:
        report.update(status="BLOCKED_INDEPENDENT_AUTHORIZATION", reason=str(exc))
    return plan, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-authorization", type=Path)
    parser.add_argument("--evaluator-authorization", type=Path)
    parser.add_argument("--implementation-root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("immutable audit output already exists")
    report = {"status": "BLOCKED", "evaluation_executed": False,
              "scientific_qualification": False}
    try:
        read = lambda p: json.loads(p.read_text()) if p else None
        plan, report = preflight(read(args.source_authorization), read(args.evaluator_authorization))
        if args.execute and report["status"] == "READY_AUTHORIZED_FROZEN_DIAGNOSTIC":
            if not args.source or not args.implementation_root:
                raise AuditBlocked("pinned implementation checkout and original source required")
            # Deliberately lazy: no PyTorch/model/source import before permission.
            from original_history_trace import run_original_audit
            report.update(evaluation_executed=True, evaluation_completed=False)
            report["audit"] = run_original_audit(plan, args.implementation_root, args.source)
            report.update(status="MEASURED_FROZEN_DIAGNOSTIC_NOT_QUALIFIED", evaluation_executed=True)
            report["evaluation_completed"] = True
    except (AuditBlocked, ValueError, OSError, RuntimeError, ImportError,
            AssertionError, KeyError, subprocess.SubprocessError) as exc:
        report.update(status="BLOCKED", reason=str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as fp:
        json.dump(report, fp, indent=2, sort_keys=True, allow_nan=False)
        fp.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "evaluation_executed", "scientific_qualification")}))
    return 2 if report["status"].startswith("BLOCKED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
