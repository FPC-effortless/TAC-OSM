#!/usr/bin/env python3
"""Real original robot FIT evidence -> explicit reward-eligibility BLOCKER.

This consumes the source-audit receipt from the *actual* original FIT
parquets. The source has real observed qpos/action/takeover transitions, NOT
independently verified task rewards, autonomous policy value or final success.

A successfully audited dataset must remain BLOCKED for reward-optimization
claims. This negative gate cannot issue an evaluator attestation or unlock
prospective training. Nonconforming/missing evidence fails the source audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "tac-osm.original-robot-reward-readiness.v1"
SOURCE_SCHEMA = "tac-osm.real-robot-full-FIT.v1"
REVISION = "9290f5a96dc91ea58256e81d9ce2dd9621890b51"


def _hash(x: object) -> bool:
    return (isinstance(x, str) and len(x) == 64
            and all(c in "0123456789abcdef" for c in x.lower()))


def inspect_original_fit(receipt: dict) -> dict:
    """Only qualified original-source facts, no invented independent rewards."""
    if not isinstance(receipt, dict):
        raise ValueError("source receipt must be JSON object")
    if receipt.get("schema_version") != SOURCE_SCHEMA or (
            receipt.get("frozen_publisher_revision") != REVISION):
        raise ValueError("original publisher revision or receipt schema changed")
    if receipt.get("full_FIT_source_qualified") is not True:
        raise ValueError("original physical FIT source not qualified")
    if any(receipt.get(k) is not False for k in (
            "real_DEV_TEST_parquets_ever_opened", "full_DEV_TEST_physical_source_qualified",
            "model_capability_qualified", "architecture_parity_preflight_qualified",
            "source_file_order_is_known_capture_time")):
        raise ValueError("frozen source qualification status is not as advertised")
    if any(receipt.get(k) is not True for k in (
            "no_real_scalar_rewards_or_verified_task_success",
            "onset_is_human_judgment_not_objectively_verified_failure",
            "no_causal_action_benefit_claim",
            "partitions_fixed_before_any_train_or_outcome_comparison")):
        raise ValueError("source now claims unsupported reward/safety/causal evidence")
    if (receipt.get("training_models_fitted") != 0 or
            receipt.get("optimizer_steps") != 0):
        raise ValueError("source-only receipt was mixed with trained-model claims")
    files = receipt.get("original_FIT_source_file_checksums")
    if not isinstance(files, list) or len(files) != 325:
        raise ValueError("325 original FIT source entries required")
    ids = set()
    for file in files:
        if not isinstance(file, dict) or set(file) != {
                "original_episode_index", "source_parquet_SHA256",
                "authentic_frame_count"}:
            raise ValueError("malformed original per-episode SHA256 receipt")
        idx = file["original_episode_index"]
        if type(idx) is not int or not 0 <= idx <= 504 or idx in ids:
            raise ValueError("duplicate or invalid original episode")
        ids.add(idx)
        if not _hash(file["source_parquet_SHA256"]) or (
                type(file["authentic_frame_count"]) is not int
                or file["authentic_frame_count"] <= 0):
            raise ValueError("invalid FIT source digest or frame count")
    if not _hash(receipt.get("all_original_fit_parquet_receipts_SHA256")):
        raise ValueError("entire FIT evidence manifest not hashed")
    # Same digest canonicalization as scripts/audit_full_real_robot_fit.py.
    serialized = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(serialized).hexdigest() != receipt[
            "all_original_fit_parquet_receipts_SHA256"].lower():
        raise ValueError("source evidence manifest hash mismatch")
    support = receipt.get("real_FIT_target_support")
    if not isinstance(support, dict) or support.get(
            "source_original_FIT_episode_count") != 325:
        raise ValueError("original source row count/episode provenance inconsistent")
    if not isinstance(support.get("horizon_support"), dict):
        raise ValueError("missing real physical transition horizons")
    return {
        "schema": SCHEMA,
        "source": "BodenAI RW-RL HIL original FIT physical trajectories",
        "source_revision": REVISION,
        "verified_fit_episodes": 325,
        "original_fit_receipt_sha256": receipt[
            "all_original_fit_parquet_receipts_SHA256"],
        "real_world_prediction_targets": [
            "observed_next_joint_state",
            "observed_next_human_control_onset_with_authority_mask",
        ],
        "independent_reward_evaluator": "UNAVAILABLE",
        "verified_external_task_goal": "UNAVAILABLE",
        "operator_controlled_reward_signing": "UNAVAILABLE",
        "status": "BLOCKED_NO_INDEPENDENT_REAL_REWARD_OR_TASK_GOAL",
        "reward_checkpoint_promotion_eligible": False,
        "model_training_authorized_by_reward_integrity": False,
        "explanation": (
            "Genuine physical state transitions and human takeover annotations "
            "are not independently verified scalar task reward or policy success. "
            "An external source-qualified task goal and access-separated evaluator "
            "are needed before reward-based model qualification."),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.fit_receipt.read_text())
        result = inspect_original_fit(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
        # Exit 0 means the *negative evidence gate* correctly blocked.
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema": SCHEMA,
                          "status": "INCONCLUSIVE_SOURCE_EVIDENCE_INVALID",
                          "reward_checkpoint_promotion_eligible": False,
                          "error_type": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
