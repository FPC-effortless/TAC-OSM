"""No-fit Kaggle monitoring scope and submission-evidence regressions."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from scripts import kaggle_status_fallback as fallback
from scripts import monitor_phone_kaggle as monitor
from tac_osm.phone_research_ops import load_proposal
from pathlib import Path


def test_accepted_git_dispatch_filters_failed_and_untrusted_ids(monkeypatch):
    def request(method, url, token):
        assert method == "GET" and token == "fake-token"
        assert url.endswith("phone-kaggle-dispatch.yml/runs?per_page=25")
        return {"workflow_runs": [
            {"id": 99, "conclusion": "failure", "event": "workflow_dispatch"},
            {"id": "bad", "conclusion": "success", "event": "workflow_dispatch"},
            {"id": 32, "conclusion": "success", "event": "schedule"},
            {"id": 42, "conclusion": "success", "event": "workflow_dispatch",
             "created_at": "2026-10-29T08:04:56Z"},
        ]}
    monkeypatch.setattr(fallback, "github_request", request)
    out = fallback.accepted_submission("FPC-effortless/TAC-OSM", "fake-token")
    assert out["run_id"] == 42
    assert out["url"].endswith("/actions/runs/42")


def test_no_token_monitor_preserves_submission_and_never_calls_kaggle(monkeypatch, tmp_path):
    monkeypatch.setenv("GH_TOKEN", "fake-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "FPC-effortless/TAC-OSM")
    monkeypatch.setattr(fallback, "accepted_submission", lambda *a:
                        {"run_id": 42, "url": "https://github.com/FPC-effortless/TAC-OSM/actions/runs/42",
                         "created_at": "2026-10-29T08:04:56Z"})
    monkeypatch.setattr(fallback, "update_issue", lambda *a: calls.append(a))
    calls = []
    assert fallback.main(["--out", str(tmp_path), "--post-issue"]) == 0
    report = json.loads((tmp_path / "submission-status.json").read_text())
    assert report["accepted_github_submission"]["run_id"] == 42
    assert not report["remote_execution_verified"]
    assert not report["observed_two_gpu_execution"]
    assert len(calls) == 1 and "UNVERIFIED" in calls[0][2]
    assert "no remote job submitted" not in calls[0][2].lower()


@pytest.mark.parametrize("rc,stderr,state", [
    (0, "", "REMOTE_STATUS_AVAILABLE_NOT_QUALIFIED"),
    (1, "Permission 'kernels.get' was denied", "REMOTE_ACCESS_DENIED_NOT_PROOF_OF_ABSENCE"),
    (1, "404 unavailable", "REMOTE_STATUS_UNAVAILABLE_NOT_PROOF_OF_ABSENCE"),
])
def test_kaggle_status_never_promotes_execution(monkeypatch, rc, stderr, state):
    monkeypatch.setattr(monitor.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=rc, stdout="", stderr=stderr))
    prop = load_proposal(Path("research/phone_queue/kaggle-dual-t4-smoke.json"),
                         monitor.ROOT)
    result = monitor.inspect(prop, "researcher")
    assert result["state"] == state
    assert result["observed_two_gpu_execution"] is False
    assert result["checkpoint_replay_verified"] is False
