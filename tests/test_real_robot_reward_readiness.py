"""Original real FIT source availability is NOT task reward evidence.

All constructed rows below are software-only fixtures; authentic source
trajectories are audited separately by the original FIT CI workflow.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.audit_full_real_robot_fit import build_full_fit_receipt
from scripts.qualify_real_robot_cohort import (
    EXPECTED_EPISODES, HISTORY_LENGTHS, metadata_only_rows, partition_metadata)
from scripts.check_real_robot_reward_readiness import inspect_original_fit


def fixture_source_receipt():
    raw = ("\n".join(
        json.dumps({"episode_index": i, "length": 300})
        for i in range(EXPECTED_EPISODES)) + "\n").encode()
    parts = partition_metadata(metadata_only_rows(raw))
    one = {"frames": 300, "policy_frames": 300, "human_frames": 0,
           "takeover_onsets_all": 0,
           "histories": {str(h): {
               "real_observed_state_action_next_state_pairs": 300-h,
               "operator_policy_eligible_onset_examples": 300-h,
               "actual_human_takeover_onset_positives": 0,
               "observed_no_takeover_negatives": 300-h,
               "masked_already_human_frames": 0,
           } for h in HISTORY_LENGTHS}}
    rows = {i: deepcopy(one) for i, _ in parts["FIT"]}
    hashes = {i: hashlib.sha256(f"constructed:{i}".encode()).hexdigest()
              for i in rows}
    return build_full_fit_receipt({"fixture_metadata": "a" * 64},
                                  parts, hashes, rows)


def test_qualified_physical_source_does_not_grant_reward_or_task_success():
    d = inspect_original_fit(fixture_source_receipt())
    assert d["verified_fit_episodes"] == 325
    assert d["reward_checkpoint_promotion_eligible"] is False
    assert d["model_training_authorized_by_reward_integrity"] is False
    assert d["status"] == "BLOCKED_NO_INDEPENDENT_REAL_REWARD_OR_TASK_GOAL"
    assert d["real_world_prediction_targets"] == [
        "observed_next_joint_state",
        "observed_next_human_control_onset_with_authority_mask",
    ]


@pytest.mark.parametrize("field,wrong", [
    ("no_real_scalar_rewards_or_verified_task_success", False),
    ("onset_is_human_judgment_not_objectively_verified_failure", False),
    ("real_DEV_TEST_parquets_ever_opened", True),
    ("model_capability_qualified", True),
    ("full_FIT_source_qualified", False),
    ("optimizer_steps", 5),
    ("architecture_parity_preflight_qualified", True),
    ("frozen_publisher_revision", "unverified revision"),
])
def test_source_tampering_or_reward_promotion_claim_invalidates_readiness(field, wrong):
    receipt = fixture_source_receipt()
    receipt[field] = wrong
    with pytest.raises(ValueError):
        inspect_original_fit(receipt)


def test_original_parquet_manifest_hash_and_duplicate_falsifiers():
    corrupted = fixture_source_receipt()
    corrupted["original_FIT_source_file_checksums"][0]["source_parquet_SHA256"] = "a" * 64
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        inspect_original_fit(corrupted)
    duplicate = fixture_source_receipt()
    duplicate["original_FIT_source_file_checksums"][1]["original_episode_index"] = (
        duplicate["original_FIT_source_file_checksums"][0]["original_episode_index"])
    with pytest.raises(ValueError, match="duplicate"):
        inspect_original_fit(duplicate)


def test_actual_cli_reports_blocked_reward_qualification_not_synthetic_success(tmp_path):
    source_path = tmp_path / "source.json"
    out_path = tmp_path / "reward-gate.json"
    source_path.write_text(json.dumps(fixture_source_receipt()))
    r = subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[1] /
                            "scripts/check_real_robot_reward_readiness.py"),
        "--fit-receipt", str(source_path),
        "--output", str(out_path)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(out_path.read_text())["reward_checkpoint_promotion_eligible"] is False
    source_path.write_text("{}")
    r = subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[1] /
                            "scripts/check_real_robot_reward_readiness.py"),
        "--fit-receipt", str(source_path),
        "--output", str(out_path)],
        capture_output=True, text=True)
    assert r.returncode == 2
    assert json.loads(r.stdout)["status"] == "INCONCLUSIVE_SOURCE_EVIDENCE_INVALID"


def test_actual_source_workflow_archives_explicit_reward_blocker():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/full-real-robot-fit-source-qualification.yml").read_text()
    assert "scripts/check_real_robot_reward_readiness.py" in workflow
    assert "results/real-robot-reward-readiness-blocker.json" in workflow
    assert 'assert evidence["reward_checkpoint_promotion_eligible"] is False' in workflow
    assert "tests/test_real_robot_reward_readiness.py" in workflow
