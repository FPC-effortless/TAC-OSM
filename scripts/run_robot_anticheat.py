#!/usr/bin/env python3
"""Run the TAC-OSM real-robot causal-window anti-cheat gate.

Examples:
  python scripts/run_robot_anticheat.py --fixture
  python scripts/run_robot_anticheat.py --windows-jsonl windows.jsonl \
      --expected-sha256 <pre-pinned-original-file-digest> --run-id real-001 \
      --output anticheat.json

JSONL must contain full RealLoggedWindow fields from the SOURCE-OWNED
real_windows_from_original_episode adapter. JSON file hashing verifies bytes
against a pre-pinned receipt, NOT that a publisher supplied genuine records.
Do not treat --fixture as a model result or scientific score.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.real_robot_transition_contract import RealLoggedWindow  # noqa: E402
from tac_osm.robot_anticheat import audit_robot_windows  # noqa: E402
from tac_osm.research_anticheat import AuditGateError  # noqa: E402


def fixture_windows() -> list[RealLoggedWindow]:
    """Constructed contract smoke case, never real-world evidence."""
    zero = (0.0,) * 14
    return [RealLoggedWindow(
        observed_history_state=(zero, zero),
        previously_executed_actions=(zero,),
        current_measured_state=zero,
        current_logged_action=zero,
        currently_human_controlled=False,
        next_measured_state=(1.0,) * 14,
        new_intervention_onset=False,
        current_frame_index=1, episode_index=0)]


def load_windows(path: Path, expected_sha256: str) -> list[RealLoggedWindow]:
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != expected_sha256:
        raise ValueError("source SHA256 does not match preregistered receipt")
    allowed = {f.name for f in fields(RealLoggedWindow)}
    results = []
    for n, line in enumerate(source.splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != allowed:
            raise ValueError(f"JSONL row {n}: window fields missing/extra")
        for name in ("observed_history_state", "previously_executed_actions"):
            row[name] = tuple(tuple(v) for v in row[name])
        for name in ("current_measured_state", "current_logged_action",
                     "next_measured_state"):
            row[name] = tuple(row[name])
        results.append(RealLoggedWindow(**row))
    if not results:
        raise ValueError("empty source windows")
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--fixture", action="store_true")
    group.add_argument("--windows-jsonl", type=Path)
    p.add_argument("--expected-sha256")
    p.add_argument("--run-id", default="contract-fixture-no-fit")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    if args.windows_jsonl and (
            not args.expected_sha256 or len(args.expected_sha256) != 64):
        p.error("--expected-sha256 (pre-pinned 64-character hex digest) required")
    try:
        windows = (fixture_windows() if args.fixture else
                   load_windows(args.windows_jsonl, args.expected_sha256))
        ledger = audit_robot_windows(windows, run_id=args.run_id)
        result = ledger.report()
        result["source_provenance"] = ("constructed_fixture_not_capability"
                                       if args.fixture else
                                       "provided_hash_matches_file_only_publisher_unverified")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        # Never turn a broken loader into a passing integrity assertion.
        print(json.dumps({"schema": "tac-osm-anticheat/v1",
                          "status": "FAIL", "run_id": args.run_id,
                          "reason": "source_loader_rejected: " + str(exc)}))
        return 2
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    try:
        ledger.assert_clean()
    except AuditGateError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
