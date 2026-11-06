"""Constructed Kaggle API-access diagnoses; no network, GPU, or private data."""
from __future__ import annotations

import json
from types import SimpleNamespace
import pytest
from scripts import kaggle_smoke_access_diagnostic as diag


@pytest.mark.parametrize("message,code", [
    ("Permission 'kernels.get' was denied", "KAGGLE_API_ACCESS_DENIED_NOT_GPU_FAILURE"),
    ("404 Not Found", "KAGGLE_KERNEL_REFERENCE_OR_VERSION_UNAVAILABLE"),
    ("429 Too Many Requests", "KAGGLE_API_RATE_LIMITED"),
    ("Transient backend error", "KAGGLE_OUTPUT_ACCESS_UNAVAILABLE"),
])
def test_errors_do_not_implicate_gpu_hardware(message, code):
    assert diag.classify_error(message) == code


@pytest.mark.parametrize("listing,expect", [
    ("ref,title\neweewee2/tacosm-dual-t4-readiness,TAC-OSM dual-t4-readiness\n",
     {"expected_handle_listed": True, "title_derived_handle_listed": False,
      "similar_handle_listed": False}),
    ("ref,title\neweewee2/tacosm-dual-t4-readiness-copy,TAC-OSM dual-t4-readiness\n",
     {"expected_handle_listed": False, "title_derived_handle_listed": False,
      "similar_handle_listed": True}),
    ("ref,title\neweewee2/tac-osm-dual-t4-readiness,TAC-OSM dual-t4-readiness\n",
     {"expected_handle_listed": False, "title_derived_handle_listed": True,
      "similar_handle_listed": True}),
])
def test_only_boolean_indicators_for_owned_kernel_listing(listing, expect):
    assert diag.listing_indicators(listing, "eweewee2",
                                   "eweewee2/tacosm-dual-t4-readiness") == expect


def test_denied_output_checks_read_only_list_but_does_not_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fake-not-real")
    commands = []

    def runner(cmd, **kwargs):
        commands.append(cmd)
        assert kwargs["capture_output"] is True
        if cmd[1:3] == ["kernels", "output"]:
            return SimpleNamespace(returncode=1, stdout="",
                                   stderr="Permission 'kernels.get' was denied")
        assert cmd[1:3] == ["kernels", "list"]
        return SimpleNamespace(returncode=0,
                               stdout="ref,title\neweewee2/tacosm-dual-t4-readiness,Smoke\n",
                               stderr="")

    ok, report = diag.inspect("eweewee2", tmp_path, runner=runner)
    assert ok is False
    assert report["state"] == "KAGGLE_API_ACCESS_DENIED_NOT_GPU_FAILURE"
    assert report["expected_handle_listed"] is True
    assert report["remote_execution_verified"] is False
    assert len(commands) == 2
    assert all(c[:2] == ["kaggle", "kernels"] for c in commands)
    assert not any("push" in c or "run" in c for c in commands)
    assert "fake-not-real" not in json.dumps(report)



def test_title_derived_handle_is_only_read_after_authenticated_listing(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fake-not-real")
    commands = []

    def runner(cmd, **kwargs):
        commands.append(cmd)
        assert kwargs["capture_output"] is True
        if cmd[1:3] == ["kernels", "list"]:
            return SimpleNamespace(returncode=0,
                 stdout="ref,title\neweewee2/tac-osm-dual-t4-readiness,TAC-OSM dual-t4-readiness\n",
                 stderr="")
        if cmd[1:3] == ["kernels", "output"]:
            if cmd[3] == "eweewee2/tacosm-dual-t4-readiness":
                return SimpleNamespace(returncode=1, stdout="",
                                       stderr="Permission 'kernels.get' was denied")
            assert cmd[3] == "eweewee2/tac-osm-dual-t4-readiness"
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        pytest.fail("unexpected external command")

    ok, report = diag.inspect("eweewee2", tmp_path, runner=runner)
    assert ok is True
    assert report["resolved_via_title_derived_slug"] is True
    assert report["saved_receipts_still_require_offline_audit"] is True
    assert len(commands) == 3
    assert not any("push" in c or "run" in c for c in commands)
    assert "fake-not-real" not in json.dumps(report)


def test_success_only_means_download_still_needs_receipt_audit(monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fake-not-real")
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    ok, report = diag.inspect("eweewee2", tmp_path, runner=runner)
    assert ok
    assert report["saved_receipts_still_require_offline_audit"] is True
    assert len(calls) == 1


def test_unavailable_remote_writes_fail_closed_report(monkeypatch, tmp_path):
    monkeypatch.setattr(diag, "inspect", lambda *args: (
        False, {"state": "KAGGLE_API_ACCESS_DENIED_NOT_GPU_FAILURE",
                "remote_execution_verified": False}))
    report = tmp_path / "status" / "verification.json"
    assert diag.main(["--username", "eweewee2", "--output-dir", str(tmp_path),
                      "--report", str(report)]) == 2
    saved = json.loads(report.read_text())
    assert saved["status"] == "INCOMPLETE_NOT_VERIFIED"
    assert saved["scientific_qualification"] == "NONE"
    assert saved["remote_execution_verified"] is False


def test_invalid_username_rejected_without_external_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fake-not-real")
    with pytest.raises(ValueError, match="invalid Kaggle account"):
        diag.inspect("../foo", tmp_path, runner=lambda *a, **kw: pytest.fail("no API"))
