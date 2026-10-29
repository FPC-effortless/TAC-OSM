#!/usr/bin/env python3
"""Phone status without Kaggle token: GitHub submission evidence only.

Read-only Actions metadata and one updated issue comment. Never contact
Kaggle, download notebook files, submit work or claim dual-GPU execution.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phone_lab import github_request, update_issue  # noqa: E402


def accepted_submission(repo: str, token: str) -> dict | None:
    """A successful GitHub dispatch is only an accepted Kaggle API push."""
    if not repo or "/" not in repo or not token:
        raise ValueError("GitHub repository and Actions token required")
    url = ("https://api.github.com/repos/" + repo
           + "/actions/workflows/phone-kaggle-dispatch.yml/runs?per_page=25")
    value = github_request("GET", url, token)
    runs = value.get("workflow_runs") if isinstance(value, dict) else None
    if not isinstance(runs, list):
        raise ValueError("unexpected GitHub Actions response")
    for run in runs:
        if not isinstance(run, dict):
            continue
        run_id = run.get("id")
        if (run.get("event") == "workflow_dispatch"
                and run.get("conclusion") == "success"
                and type(run_id) is int and run_id > 0):
            return {
                "run_id": run_id,
                "url": f"https://github.com/{repo}/actions/runs/{run_id}",
                "created_at": str(run.get("created_at") or ""),
            }
    return None


def render(submission: dict | None) -> tuple[dict, str]:
    report = {
        "schema": "tac-osm.phone-kaggle-status.v1",
        "state": "MONITOR_TOKEN_NOT_AVAILABLE",
        "accepted_github_submission": submission,
        "remote_execution_verified": False,
        "observed_two_gpu_execution": False,
        "scientific_qualification": "NONE",
    }
    latest = (
        f"**Last accepted GitHub submission:** [run {submission['run_id']}]"
        f"({submission['url']}) ({submission['created_at']}).\n\n"
        if submission else
        "**No successful dispatch in the last 25 checked workflow runs.**\n\n"
    )
    body = (
        "<!-- tac-osm-phone-lab-status -->\n"
        "### Kaggle dual-T4 research status\n\n"
        "**Monitor credential: UNAVAILABLE TO THIS SCHEDULED JOB.**\n\n"
        + latest
        + "**Kaggle remote execution: UNVERIFIED.** This is not evidence that "
        "the private notebook is absent or that the two T4 workers ran.\n\n"
        "GitHub environment-scoped secrets are not exposed to jobs outside "
        "that environment. To enable unattended remote status polling, the "
        "owner may configure a repository-scoped `KAGGLE_API_TOKEN` "
        "(wider permission scope). Otherwise inspect Kaggle manually. "
        "No GPU work was submitted or quota consumed by this monitor.\n"
    )
    return report, body


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ci-artifacts/kaggle-monitor"))
    parser.add_argument("--post-issue", action="store_true")
    args = parser.parse_args(argv)
    token, repo = os.environ.get("GH_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        print('{"state":"ERROR_NO_GITHUB_CREDENTIALS"}', file=sys.stderr)
        return 2
    try:
        report, markdown = render(accepted_submission(repo, token))
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "submission-status.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
        if args.post_issue:
            update_issue(repo, 473, markdown, token)
        print(json.dumps(report, sort_keys=True))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(json.dumps({"state": "STATUS_CHECK_FAILED", "reason": str(exc)[:200]}),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
