#!/usr/bin/env python3
"""Read-only, fail-closed report for the bounded capability idea queue."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "research/ideas/CAPABILITY_QUEUE.json"


def registration_state(idea: dict) -> tuple[str, str]:
    """Inspect a registration using the repository validator's API only."""
    relative = idea.get("preregistration_path")
    if not isinstance(relative, str) or not relative.strip():
        return "BLOCKED/DESIGN_ONLY", "preregistration_path is missing"
    manifest_path = (ROOT / relative).resolve()
    if not manifest_path.is_relative_to(ROOT):
        return "BLOCKED/DESIGN_ONLY", "preregistration_path escapes repository root"
    if not manifest_path.is_file():
        return "BLOCKED/DESIGN_ONLY", f"registration not present: {relative}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "BLOCKED/DESIGN_ONLY", f"registration unreadable or invalid JSON: {exc}"

    # Importing and calling check_manifest avoids invoking the validator CLI or
    # any subprocess. Keep the repository root first for its scripts namespace.
    sys.path.insert(0, str(ROOT))
    try:
        from scripts.validate_research_parity import check_manifest

        result = check_manifest(manifest, phase="preflight")
    except Exception as exc:  # validator reports malformed/ineligible manifests
        return "BLOCKED/DESIGN_ONLY", f"preflight rejected: {exc}"
    finally:
        if sys.path and sys.path[0] == str(ROOT):
            sys.path.pop(0)
    return "PREFLIGHT_VALID", str(result.get("status", "PASS"))


def build_report(queue: dict) -> dict:
    ideas = queue.get("ideas")
    if not isinstance(ideas, list):
        raise ValueError("queue.ideas must be an array")
    if queue.get("active_model_families") != 6:
        raise ValueError("queue must preserve exactly six active model families")

    rows = []
    for idea in sorted(ideas, key=lambda row: row.get("priority", 10**9)):
        if not isinstance(idea, dict):
            raise ValueError("each idea must be an object")
        state, detail = registration_state(idea)
        declared = idea.get("status", "DESIGN_ONLY")
        rows.append({
            "id": idea.get("id"),
            "title": idea.get("title"),
            "priority": idea.get("priority"),
            "declared_status": declared,
            "registration_status": state,
            "registration_detail": detail,
        })

    candidates = [row for row in rows if row["registration_status"] == "PREFLIGHT_VALID"
                  and not row["declared_status"].startswith(("COMPLETED", "ARCHIVED", "INTERRUPTED", "REJECTED"))]
    return {
        "schema_version": "tac-osm.capability-queue-report.v1",
        "active_model_families": 6,
        "recommended_max_concurrent_fits": queue.get("queue_policy", {}).get("recommended_max_concurrent_fits", 2),
        "ideas": rows,
        "preflight_valid_candidates": candidates,
        "fit_authorized_by_queue": False,
        "independent_live_gates": ["source qualification", "mathematical engineering", "actual implementation/resource checks", "applicable data-access controls"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the full report as JSON")
    args = parser.parse_args()
    try:
        queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
        report = build_report(queue)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"queue error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Capability idea queue (read-only)")
        for row in report["ideas"]:
            print(f"{row['priority']}. {row['id']} — {row['title']}: {row['registration_status']} ({row['registration_detail']})")
        candidates = report["preflight_valid_candidates"]
        if candidates:
            print("Preflight-valid candidates:")
            for candidate in candidates:
                print(f"- {candidate['id']} — {candidate['title']}")
        else:
            print("Preflight-valid candidates: none")
        print(f"Recommended concurrent-fit limit: {report['recommended_max_concurrent_fits']}")
        print("Fit authorized by queue: no")
        print("Source, math, implementation/resource, and data-access gates must be checked independently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
