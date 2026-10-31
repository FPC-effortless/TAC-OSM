"""No-fit phone research orchestration tests: do not contact Kaggle or train."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.phone_research_ops import (
    FAMILIES, GPU_ACCELERATOR, ResearchPlanError, decide, load_proposal, phone_markdown,
)
from scripts.stage_kaggle_research import package

def proposal(**changes):
    base = {
        "schema": "tac-osm.phone-research.v1",
        "run_id": "reviewed-study-01", "variant": "System 1",
        "kind": "no_fit", "issue_number": 472,
        "estimated_cpu_minutes": 5, "requires_cuda": False,
        "gpu_speedup_expected": False, "estimated_gpu_minutes": 0,
        "gpu_strategy": None, "seeds": [], "real_source_sha256": None,
        "study_manifest": None, "workload": None, "diagnostic": "lab",
        "dataset_sources": [], "budget_gpu_minutes": 0,
    }
    base.update(changes)
    return base

def kaggle_plan(**changes):
    values = dict(
        run_id="reviewed-gpu-test", variant="Physical",
        estimated_cpu_minutes=120, requires_cuda=True,
        gpu_speedup_expected=True, estimated_gpu_minutes=30,
        gpu_strategy="seed_shard", seeds=[111, 222],
        workload="research/kaggle_workloads/dual_t4_smoke.py",
        diagnostic=None, budget_gpu_minutes=30)
    values.update(changes)
    return proposal(**values)


@pytest.mark.parametrize("name", FAMILIES)
def test_all_six_families_route_small_no_fit_to_github(name):
    d = decide(proposal(variant=name))
    assert d["backend"] == "github_actions"
    assert d["state"] == "READY_NO_FIT"
    assert d["approved_to_train"] is False


def test_gpu_never_routes_to_cpu_or_p100():
    d = decide(kaggle_plan())
    assert d["backend"] == "kaggle_t4x2"
    assert d["accelerator"] == GPU_ACCELERATOR == "NvidiaTeslaT4"
    assert d["gpu_devices_required"] == 2
    assert d["approved_to_train"] is False


def test_long_cpu_job_without_justified_gpu_does_not_fake_gpu_need():
    d = decide(proposal(estimated_cpu_minutes=150))
    assert d["backend"] == "github_actions"
    assert d["state"] == "BLOCKED_EVIDENCE_OR_DESIGN"


def test_long_job_with_evidenced_gpu_speedup_selects_kaggle():
    d = decide(kaggle_plan(requires_cuda=False))
    assert d["backend"] == "kaggle_t4x2"


@pytest.mark.parametrize("change", [
    {"gpu_strategy": None},
    {"gpu_strategy": "single_gpu"},
    {"budget_gpu_minutes": 20},
    {"seeds": [111]},
    {"workload": None},
])
def test_gpu_missing_two_device_plan_or_budget_blocks(change):
    p = kaggle_plan()
    p.update(change)
    if p["gpu_strategy"] == "single_gpu":
        with pytest.raises(ResearchPlanError, match="two-GPU"):
            decide(p)
    else:
        assert decide(p)["state"] == "BLOCKED_EVIDENCE_OR_DESIGN"


def test_training_on_github_cpu_needs_signed_source_and_replay_evidence():
    d = decide(proposal(kind="training", seeds=[100, 101]))
    assert d["backend"] == "github_actions"
    assert d["approved_to_train"] is False
    assert d["state"] == "BLOCKED_EVIDENCE_OR_DESIGN"
    assert any("signed" in b for b in d["blockers"])


def test_gpu_training_cannot_succeed_on_manifest_claim_alone():
    p = kaggle_plan()
    p.update({
        "kind": "training", "real_source_sha256": "a" * 64,
        "study_manifest": "research/mathematical-engineering/verified/FAIR_COMPARISON.json",
        "dataset_sources": ["original-owner/source-dataset"],
    })
    d = decide(p)
    assert d["backend"] == "kaggle_t4x2"
    assert d["approved_to_train"] is False
    assert d["state"] == "BLOCKED_EVIDENCE_OR_DESIGN"
    assert "independent signed prefit reward/source/ACL evidence required at execution" in d["blockers"]


@pytest.mark.parametrize("change", [
    {"run_id": "../outside"},
    {"run_id": "foo token"},
    {"issue_number": True},
    {"estimated_cpu_minutes": 0},
    {"gpu_speedup_expected": "yes"},
    {"seeds": [1,1]},
    {"study_manifest": "../FAIR_COMPARISON.json"},
    {"workload": "../../anything.py"},
    {"dataset_sources": ["https://evil.example/data"]},
    {"variant": "unverified"},
])
def test_untrusted_proposal_values_fail_before_execution(change):
    with pytest.raises(ResearchPlanError):
        decide(proposal(**change))


def test_phone_issue_comment_is_compact_and_explicitly_unqualified():
    s = phone_markdown(decide(kaggle_plan()))
    assert "Kaggle: **2× T4**" in s
    assert "<!-- tac-osm-phone-lab-status -->" in s
    assert "not model training approval" in s


def test_repo_proposal_path_must_be_reviewed_and_normalized(tmp_path):
    folder = tmp_path / "research/phone_queue"
    folder.mkdir(parents=True)
    p = folder / "test.json"
    p.write_text(json.dumps(proposal()))
    result = load_proposal(Path("research/phone_queue/test.json"), tmp_path)
    assert result["run_id"] == "reviewed-study-01"
    with pytest.raises(ResearchPlanError):
        load_proposal(Path("../outside.json"), tmp_path)


def test_gpu_staging_is_private_has_no_internet_and_enforces_t4_pair(tmp_path):
    p = package(Path("research/phone_queue/kaggle-dual-t4-smoke.json"),
                kaggle_user="researcher", destination=tmp_path / "stage")
    assert p["status"] == "STAGED_UNSUBMITTED"
    assert p["gpu_count_required"] == 2
    output = tmp_path / "stage"
    meta = json.loads((output / "kernel-metadata.json").read_text())
    cfg = json.loads((output / "config.json").read_text())
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["enable_internet"] == "false"
    assert meta["is_private"] == "true"
    assert meta["enable_gpu"] == "true"
    assert meta["title"].lower().replace(" ", "-") == meta["id"].split("/")[1]
    assert cfg["seeds"] == [101, 102]
    assert "real_t4_pair" in (output / "entrypoint.py").read_text()
    assert p["outcome_qualified"] is False


def test_kaggle_staging_rejects_unavailable_gpu_workload(tmp_path):
    missing = tmp_path / "stage"
    with pytest.raises(ResearchPlanError):
        package(Path("research/phone_queue/phone-lab-diagnostics.json"),
                kaggle_user="operator", destination=missing)


def test_phone_scan_reports_both_queues_and_never_runs_gpu(tmp_path):
    p = subprocess.run([sys.executable, str(ROOT / "scripts/phone_lab.py"),
                        "scan", "--out", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    status = json.loads((tmp_path / "progress.json").read_text())
    registered = [json.loads(path.read_text())["run_id"]
                  for path in (ROOT / "research/phone_queue").glob("*.json")]
    assert len(registered) == len(set(registered))
    assert status["total"] == len(registered)
    assert {row["run_id"] for row in status["runs"]} == set(registered)
    data = {r["run_id"]: r for r in status["runs"]}
    assert data["phone-lab-diagnostics"]["backend"] == "github_actions"
    assert data["dual-t4-readiness"]["backend"] == "kaggle_t4x2"
    assert "Kaggle 2× T4" in (tmp_path / "progress.md").read_text()


def test_phone_dispatch_workflow_requires_external_gpu_enablement_and_secrets():
    workflow = (ROOT / ".github/workflows/phone-kaggle-dispatch.yml").read_text()
    for mandatory in (
        "environment: kaggle-research", "PHONE_LAB_GPU_EXECUTION_ENABLED",
        "KAGGLE_API_TOKEN", "KAGGLE_USERNAME",
        "validate_research_parity.py", "--phase", "preflight",
        "TAC_OSM_PREFIT_ATTESTATION_JSON", "APPROVE_T4_X2",
        "NvidiaTeslaT4",
    ):
        if mandatory == "NvidiaTeslaT4":
            # referenced in command implementation, never a UI hardcoded alternative
            stage = (ROOT / "scripts/stage_kaggle_research.py").read_text()
            assert mandatory in stage or "GPU_ACCELERATOR" in stage
        else:
            assert mandatory in workflow


def test_phone_schedule_and_report_are_phone_first_and_nofit():
    workflow = (ROOT / ".github/workflows/phone-research-control.yml").read_text()
    assert "schedule:" in workflow and "workflow_dispatch:" in workflow
    assert "scripts/phone_lab.py scan" in workflow
    assert "issues: write" in workflow
    assert "research/phone_queue" in workflow
    assert "run-safe-diagnostics" in workflow
    monitor = (ROOT / ".github/workflows/phone-kaggle-monitor.yml").read_text()
    assert "kaggle" in monitor
    assert "--post-issue" in monitor
    assert "phone-kaggle-status" not in monitor


def test_gpu_runtime_checks_real_cuda_devices_before_workload():
    text = (ROOT / "scripts/kaggle_dual_t4_entry.py").read_text()
    assert "device_count()" in text
    assert 'count != 2' in text
    assert '"T4" not in name.upper()' in text
    assert 'CUDA_VISIBLE_DEVICES=str(gpu_id)' in text
    assert "COMPLETED_UNQUALIFIED_REQUIRES_INDEPENDENT_EVALUATION" in text


def test_github_actions_expressions_are_not_literal_escaped_strings():
    for name in ('phone-research-control.yml', 'phone-kaggle-dispatch.yml',
                 'phone-kaggle-monitor.yml'):
        content = (ROOT / '.github/workflows' / name).read_text()
        assert "\\${{" not in content, name
        assert "${{ github.token }}" in content
        assert "${{ github.repository }}" in content
    remote = (ROOT / '.github/workflows/phone-kaggle-dispatch.yml').read_text()
    assert "${{ secrets.KAGGLE_API_TOKEN }}" in remote
    assert "${{ vars.KAGGLE_USERNAME }}" in remote
