#!/usr/bin/env python3
"""Reward hacking no-fit evidence reviewer; no new fits, no sealed TEST access.

Modes:
  python scripts/run_reward_integrity.py --fixture
  python scripts/run_reward_integrity.py --baseline left.json \
    --candidate right.json --policy policy.json \
    --expected-baseline-sha256 SHA --expected-candidate-sha256 SHA \
    --expected-policy-sha256 SHA

A paired receipt report will remain BLOCKED unless a trusted FIT/DEV runner
also performs independent goal/reward recomputation, side-channel poisoning
and out-of-band protected-asset verification. The CLI never self-certifies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.research_reward_integrity import (  # noqa: E402
    audit_proxy_goal_alignment, reward_audit_ledger, sha256,
)


def pinned_json(path: Path, expected: str):
    if len(expected) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected):
        raise ValueError("independently expected SHA256 required")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest().lower() != expected.lower():
        raise ValueError("independently pinned JSON input changed")
    return json.loads(payload)


def fixture():
    def id_(x): return sha256(x.encode())
    def receipt(name, reward, goal):
        return {
            "schema": "tac-osm-reward-audit/v1", "run_id": name,
            "source_sha256": id_("data"), "cohort_sha256": id_("episodes"),
            "reward_scorer_sha256": id_("reward"),
            "goal_evaluator_sha256": id_("independent-goals"),
            "goal_contract_sha256": id_("frozen contract"),
            "outcome_authority": "external_source",
            "rows": [
                {"id_sha256": id_("one"), "group": "difficult",
                 "horizon": "H512", "reward": reward, "goal": goal,
                 "decision_cost": .1, "safety_violation": False},
            ],
        }
    policy = {
        "min_proxy_gain": .05, "min_goal_gain": 0.,
        "max_goal_regression": 0., "max_added_safety_violations": 0,
        "max_added_decision_cost": .1, "goal_direction": "higher",
        "min_group_samples": 1, "required_groups": ["difficult"],
    }
    hacked = audit_proxy_goal_alignment(
        receipt("baseline", .5, .5),
        receipt("proxy-optimizer", .99, .4), policy)
    safe = audit_proxy_goal_alignment(
        receipt("baseline", .5, .5),
        receipt("independent-improver", .9, .7), policy)
    if hacked.status != "FAIL" or safe.status != "PASS":
        raise ValueError("constructed falsifier failed to detect obvious proxy gaming")
    report = reward_audit_ledger("constructed-no-fit-reward-hacking-negative-control",
                                 [hacked]).report()
    return {
        "schema": "tac-osm-reward-fixture/v1",
        "qualification": "constructed_fixture_not_real_world_reward_hacking_evidence",
        "fixture_falsification": "PASS",
        "hacked_case_status": hacked.status,
        "honest_improvement_status": safe.status,
        "reward_promotion_status": report["status"],
        "missing_live_checks": report["missing"],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", action="store_true")
    mode.add_argument("--baseline", type=Path)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--policy", type=Path)
    p.add_argument("--expected-baseline-sha256")
    p.add_argument("--expected-candidate-sha256")
    p.add_argument("--expected-policy-sha256")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    try:
        if args.fixture:
            if any((args.baseline, args.candidate, args.policy,
                    args.expected_policy_sha256, args.expected_baseline_sha256,
                    args.expected_candidate_sha256)):
                raise ValueError("fixture cannot accept external experiment receipts")
            report, exitcode = fixture(), 0
        else:
            if not all((args.candidate, args.policy, args.expected_policy_sha256,
                        args.expected_baseline_sha256, args.expected_candidate_sha256)):
                raise ValueError("baseline/candidate/policy and three out-of-band hashes required")
            baseline = pinned_json(args.baseline, args.expected_baseline_sha256)
            candidate = pinned_json(args.candidate, args.expected_candidate_sha256)
            policy = pinned_json(args.policy, args.expected_policy_sha256)
            alignment = audit_proxy_goal_alignment(baseline, candidate, policy)
            ledger = reward_audit_ledger(
                str(candidate.get("run_id", "invalid-run")), [alignment])
            report = ledger.report()
            report["qualification"] = (
                "partial_receipt_review_only_not_independent_goal_and_reward_verification")
            exitcode = 1 if report["status"] == "FAIL" else 2
        output = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        print(output, end="")
        return exitcode
    except (ValueError, TypeError, KeyError, RuntimeError, OSError,
            json.JSONDecodeError) as exc:
        print(json.dumps({"schema": "tac-osm-reward-audit/error-v1",
                          "status": "BLOCKED", "reason": type(exc).__name__,
                          "qualification": "no_reward_promotion"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
