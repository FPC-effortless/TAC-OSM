"""Constructed no-fit tests for saved Kaggle dual-T4 receipts (no API/GPU)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_kaggle_dual_t4_smoke import (
    EXPECTED_SQUARE_SUM, ROOT, SmokeReceiptError, audit, main,
)


def receipts(tmp_path):
    source = ROOT / "research/kaggle_workloads/dual_t4_smoke.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    files = {
        "phone-research-runtime.json": {
            "schema": "tac-osm.kaggle-t4x2-runtime.v1",
            "run_id": "dual-t4-readiness",
            "requested_accelerator": "NvidiaTeslaT4",
            "observed": {"count": 2, "names": ["Tesla T4", "Tesla T4"],
                         "accelerator": "NvidiaTeslaT4"},
            "workload_sha256": digest,
            "strategy": "seed_shard",
            "seeds": [101, 102],
            "status": "COMPLETED_UNQUALIFIED_REQUIRES_INDEPENDENT_EVALUATION",
            "results": [{"worker_index": 0, "exit_code": 0},
                        {"worker_index": 1, "exit_code": 0}],
            "real_world_claim_authorized": False,
        },
    }
    for seed, slot in [(101, 0), (102, 1)]:
        files[f"seed-{seed}.json"] = {
            "schema": "tac-osm.no-fit-t4-smoke.v1",
            "seed": seed, "physical_slot": slot, "device": "Tesla T4",
            "numeric_diagnostic": EXPECTED_SQUARE_SUM,
            "source_provenance": "constructed_tensor_only",
            "model_training": False,
            "capability_claim_authorized": False,
        }
    for name, data in files.items():
        (tmp_path / name).write_text(json.dumps(data))
    return files


def test_exact_two_workers_and_workload_sha_pass_only_self_reported(tmp_path):
    receipts(tmp_path)
    result = audit(tmp_path)
    assert result["status"] == "SELF_REPORTED_TWO_WORKER_RECEIPTS_CONSISTENT"
    assert result["self_reported_two_worker_execution"]
    assert result["independent_remote_attestation"] is False
    assert result["gpu_device_provenance_independently_verified"] is False
    assert result["scientific_qualification"] == "NONE"
    assert len(result["files_sha256"]) == 3


@pytest.mark.parametrize("name,field,value", [
    ("phone-research-runtime.json", "status", "RUNNING"),
    ("phone-research-runtime.json", "workload_sha256", "wrong"),
    ("phone-research-runtime.json", "observed",
     {"count": 1, "names": ["Tesla T4"], "accelerator": "NvidiaTeslaT4"}),
    ("seed-101.json", "physical_slot", 1),
    ("seed-102.json", "numeric_diagnostic", 0),
    ("seed-101.json", "model_training", True),
])
def test_corrupted_or_unqualified_receipts_fail_closed(tmp_path, name, field, value):
    data = receipts(tmp_path)
    corrupted = copy.deepcopy(data[name])
    corrupted[field] = value
    (tmp_path / name).write_text(json.dumps(corrupted))
    with pytest.raises(SmokeReceiptError):
        audit(tmp_path)


def test_missing_receipt_fails_and_cli_writes_unverified_report(tmp_path):
    receipts(tmp_path)
    (tmp_path / "seed-102.json").unlink()
    report = tmp_path / "report" / "verify.json"
    assert main(["--output-dir", str(tmp_path), "--report", str(report)]) == 2
    result = json.loads(report.read_text())
    assert result["status"] == "INCOMPLETE_NOT_VERIFIED"
    assert result["scientific_qualification"] == "NONE"


def test_protected_workflow_is_read_only_and_does_not_request_gpu():
    yml = (ROOT / ".github/workflows/phone-kaggle-smoke-audit.yml").read_text()
    assert "workflow_dispatch:" in yml
    assert "environment: kaggle-research" in yml
    assert "kaggle kernels output" in yml
    assert "--file-pattern" in yml
    assert "kaggle kernels push" not in yml
    assert "GPU_EXECUTION_ENABLED" not in yml
    assert "--accelerator" not in yml
    assert "if: always()" in yml
    assert "verification.json" in yml
