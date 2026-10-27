"""No-fit lightweight research lab software falsifiers, never capability evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.research_lab import (  # noqa: E402
    SCHEMA, LabError, create_receipt, read_receipts, render_dashboard, summary,
    write_receipt,
)
from tac_osm.research_lab_integrations import PROVIDERS, plan


def source(tmp_path: Path):
    p = tmp_path / "untrained-fixture.json"
    if not p.exists():
        p.write_text('{"source_provenance":"constructed_fixture"}')
    return p


def receipt(tmp_path, **overrides):
    values = dict(run_id="ci-run-001", experiment="S1OBS", variant="System 1",
                  stage="fixture", evidence="constructed_fixture",
                  source=source(tmp_path), repo_root=ROOT,
                  metrics={"mean_abs_error": 0.30, "latency_ms": 3.2},
                  parameters={"parameters": 100000, "tokens_seen": 20000},
                  seed=101)
    values.update(overrides)
    return create_receipt(**values)


def test_existing_report_is_hashed_with_no_capability_upgrade(tmp_path):
    source_file = source(tmp_path)
    r = receipt(tmp_path)
    assert r["schema"] == SCHEMA
    assert r["source_report"]["sha256"] == hashlib.sha256(source_file.read_bytes()).hexdigest()
    assert r["scientific_qualification"] == "NOT_EVALUATED_BY_LAB"
    assert r["capability_claim_authorized"] is False
    assert r["comparison_parity_verified"] is False
    assert r["reward_integrity_verified"] is False
    assert r["checkpoint_file"] is None and r["dataset_file"] is None


def test_immutable_per_run_record_is_idempotent_but_conflict_fails(tmp_path):
    r = receipt(tmp_path)
    root = tmp_path / "runs"
    a = write_receipt(r, root)
    assert a == write_receipt(r, root)
    altered = {**r, "metrics": {"mean_abs_error": 0.01}}
    with pytest.raises(LabError, match="different evidence"):
        write_receipt(altered, root)
    assert json.loads(a.read_text()) == r
    assert summary(root)["scientifically_qualified_by_ledger"] == 0


def test_datasets_and_checkpoints_are_only_hashed_never_copied(tmp_path):
    data = tmp_path / "authorized-data.csv"
    ckpt = tmp_path / "trained-weights.pt"
    data.write_bytes(b"data")
    ckpt.write_bytes(b"model-state")
    r = receipt(tmp_path, dataset=data, checkpoint=ckpt)
    assert r["dataset_file"]["sha256"] == hashlib.sha256(b"data").hexdigest()
    assert r["checkpoint_file"]["sha256"] == hashlib.sha256(b"model-state").hexdigest()
    assert data.read_bytes() == b"data"
    assert ckpt.read_bytes() == b"model-state"
    assert "data" not in json.dumps(r["metrics"])


def test_modified_source_gets_distinct_hash_but_same_run_id_cannot_be_overwritten(tmp_path):
    path = source(tmp_path)
    root = tmp_path / "runs"
    initial = receipt(tmp_path)
    write_receipt(initial, root)
    path.write_text('{"source_provenance":"another fixture"}')
    with pytest.raises(LabError, match="different evidence"):
        write_receipt(receipt(tmp_path), root)


@pytest.mark.parametrize("value", [
    float("nan"), float("inf"), float("-inf"), "0.3", True, {"private": 1}
])
def test_metrics_are_finite_numeric_scalars_only(tmp_path, value):
    with pytest.raises(LabError):
        receipt(tmp_path, metrics={"accuracy": value})


@pytest.mark.parametrize("field,wrong", [
    ("run_id", "../escape"),
    ("experiment", "experiment with spaces"),
    ("stage", "qualified_reward"),
    ("evidence", "real_world_proven"),
    ("variant", ""),
    ("seed", "not_a_seed"),
    ("parameters", {"api_key": "secret"}),
])
def test_malformed_or_self_promoting_metadata_is_rejected(tmp_path, field, wrong):
    with pytest.raises(LabError):
        receipt(tmp_path, **{field: wrong})


def test_uncertified_source_report_cannot_certify_scientific_evidence(tmp_path):
    r = receipt(tmp_path, stage="source_audit", evidence="source_only",
                metrics={"score": 1.0})
    assert r["capability_claim_authorized"] is False
    assert r["scientific_qualification"] == "NOT_EVALUATED_BY_LAB"


def test_loaded_ledger_rejects_self_promotion(tmp_path):
    root = tmp_path / "runs"
    path = write_receipt(receipt(tmp_path), root)
    forged = json.loads(path.read_text())
    forged["capability_claim_authorized"] = True
    path.write_text(json.dumps(forged))
    with pytest.raises(LabError, match="capability qualification"):
        read_receipts(root)


def test_render_dashboard_escapes_untrusted_labels_and_marks_limits(tmp_path):
    root = tmp_path / "runs"
    r = receipt(tmp_path, variant="<script>alert(1)</script>")
    write_receipt(r, root)
    html_path = render_dashboard(root, tmp_path / "dash.html")
    content = html_path.read_text()
    assert "<script>" not in content
    assert "&lt;script&gt;" in content
    assert "no real-world capability" in content
    assert "default-src 'none'" in content


def test_run_series_dashboard_contains_metric_sparkline(tmp_path):
    root = tmp_path / "runs"
    for i in range(3):
        write_receipt(receipt(tmp_path, run_id=f"r{i}", metrics={"error": float(i)}), root)
    body = render_dashboard(root, tmp_path / "dashboard.html").read_text()
    assert "<polyline" in body
    assert "not comparable by default" in body


def test_version_families_are_free_text_not_six_hardcoded_models(tmp_path):
    r = receipt(tmp_path, variant="Experimental-ConceptSwarm-v99")
    assert r["variant"] == "Experimental-ConceptSwarm-v99"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_all_optional_provider_recipes_are_disabled_without_network(provider):
    r = plan(provider)
    assert r["provider"] == provider
    assert r["enabled"] is False
    assert r["executed"] is False
    assert r["external_data_sent"] is False
    assert r["manual_opt_in_required"] is True
    assert r["reference"].startswith("https://")


def test_vendor_tracker_export_contains_only_scalar_metadata_and_digests(tmp_path):
    r = receipt(tmp_path)
    for name in ("mlflow", "wandb", "comet"):
        safe = plan(name, receipt=r)["safe_export"]
        assert safe["source_report_sha256"] == r["source_report"]["sha256"]
        assert safe["metrics"]["latency_ms"] == 3.2
        assert "checkpoint_file" not in safe
        assert "data" not in safe


def test_slurm_adapter_only_builds_reviewable_bounded_job_plan():
    item = plan("slurm", command=["python", "scripts/fit.py", "--seed", "101"],
                limit_minutes=60, cpus=2, gpus=1)
    assert "sbatch" in item["example_sbatch"].lower()
    assert "--gres=gpu:1" in item["example_sbatch"]
    assert not item["executed"]


@pytest.mark.parametrize("resources", [
    {"gpus": 17}, {"cpus": 0}, {"limit_minutes": 10000}
])
def test_gpu_and_cluster_resources_cannot_be_unbounded(resources):
    with pytest.raises(LabError):
        plan("slurm", **resources)


def test_cli_records_existing_file_and_creates_html_without_running_training(tmp_path):
    src = source(tmp_path)
    dest = tmp_path / "runs"
    cmd = [sys.executable, str(ROOT / "scripts/lab.py"),
           "record", "--run-id", "fixture-001", "--experiment", "DIAGKEY",
           "--variant", "Experimental", "--stage", "fixture",
           "--evidence", "constructed_fixture", "--source", str(src),
           "--metric", "error=0.01", "--out", str(dest)]
    v = subprocess.run(cmd, capture_output=True, text=True)
    assert v.returncode == 0, v.stderr
    report = json.loads(v.stdout)
    assert report["status"] == "RECORDED_NOT_QUALIFIED"
    assert (dest / "fixture-001/receipt.json").exists()
    dash = tmp_path / "dashboard.html"
    v = subprocess.run(
        [sys.executable, str(ROOT / "scripts/lab.py"), "report",
         "--runs", str(dest), "--output", str(dash)],
        capture_output=True, text=True)
    assert v.returncode == 0, v.stderr
    assert "DIAGKEY" in dash.read_text()


def test_cli_rejects_nonfinite_metric_and_unknown_provider(tmp_path):
    src = source(tmp_path)
    v = subprocess.run([
        sys.executable, str(ROOT / "scripts/lab.py"), "record",
        "--run-id", "fixture", "--experiment", "DIAGKEY", "--variant", "S1",
        "--stage", "fixture", "--evidence", "constructed_fixture",
        "--source", str(src), "--metric", "loss=nan",
        "--out", str(tmp_path / "runs")],
        capture_output=True, text=True)
    assert v.returncode == 2
    v = subprocess.run([sys.executable, str(ROOT / "scripts/lab.py"),
                        "plan", "--provider", "paid-cloud-without-consent"],
                       capture_output=True, text=True)
    assert v.returncode == 2


def test_existing_automatic_ci_can_capture_without_qualifying_fixture():
    for name in ("research-anticheat.yml", "research-observatory.yml"):
        t = (ROOT / ".github/workflows" / name).read_text()
        assert "python scripts/lab.py record" in t
        assert "--evidence constructed_fixture" in t
        assert "--stage fixture" in t
        assert "python scripts/lab.py report" in t
        assert "ci-artifacts/lab/dashboard.html" in t
        assert "permissions:\n  contents: read" in t
