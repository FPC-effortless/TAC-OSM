#!/usr/bin/env python3
"""Mobile GitHub Issue research lab: plan -> status -> optional no-fit diagnostics.

No public issue text ever becomes shell arguments. Proposals must be reviewed,
checked-in JSON. Scheduled CI does not train, spend Kaggle quota or open TEST.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.phone_research_ops import (  # noqa: E402
    ResearchPlanError, decide, load_proposal, phone_markdown,
)

SUITES = {
    "lab": ["tests/test_research_lab.py"],
    "anticheat": ["tests/test_research_anticheat.py", "tests/test_robot_anticheat.py",
                  "tests/test_research_reward_integrity.py"],
    "observatory": ["tests/test_research_observatory.py", "tests/test_research_probe_agnostic.py"],
    "scientific": ["tests/test_research_parity_gate.py", "tests/test_reward_runner_integration.py",
                   "tests/test_adaptive_token_study_nofit.py"],
}
MARKER = "<!-- tac-osm-phone-lab-status -->"


def github_request(method: str, url: str, token: str, payload=None):
    if not url.startswith("https://api.github.com/repos/"):
        raise ResearchPlanError("untrusted GitHub API URL")
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "tac-osm-phone-research-lab",
    })
    with urllib.request.urlopen(req, timeout=15) as res:
        return json.loads(res.read().decode())


def update_issue(repo: str, issue: int, markdown: str, token: str) -> None:
    """Update one existing bot status comment rather than spamming a phone."""
    if not token or not repo or "/" not in repo or not repo.replace("/", "").replace("-", "").replace("_", "").isalnum():
        raise ResearchPlanError("authenticated repository issue updater required")
    base = "https://api.github.com/repos/" + repo
    comments = github_request("GET", f"{base}/issues/{issue}/comments?per_page=100", token)
    if not isinstance(comments, list):
        raise ResearchPlanError("invalid GitHub issue comments result")
    bot = next((x for x in comments if
                isinstance(x, dict) and
                MARKER in str(x.get("body", "")) and
                x.get("user", {}).get("type") == "Bot"), None)
    if bot is None:
        github_request("POST", f"{base}/issues/{issue}/comments", token, {"body": markdown})
    elif bot.get("body") != markdown:
        github_request("PATCH", f"{base}/issues/comments/{int(bot['id'])}", token, {"body": markdown})


def all_proposals() -> list[tuple[str, dict]]:
    rows = []
    for path in sorted((ROOT / "research" / "phone_queue").glob("*.json")):
        relative = path.relative_to(ROOT).as_posix()
        try:
            rows.append((relative, decide(load_proposal(Path(relative), ROOT))))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            rows.append((relative, {"state": "INVALID_PROPOSAL", "reason": type(exc).__name__}))
    return rows


def run_suite(diagnostic: str, artifact: Path) -> dict:
    """Autonomous cheap source-control diagnostics only; not scientific fits."""
    if diagnostic not in SUITES:
        raise ResearchPlanError("unknown reviewed diagnostic suite")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pytest", "-q", *SUITES[diagnostic]]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    receipt = {
        "schema": "tac-osm.phone-no-fit-result.v1",
        "suite": diagnostic,
        "status": "PASS_FIXTURE_ONLY" if result.returncode == 0 else "FAILED_NO_FIT",
        "exit_code": result.returncode,
        "qualification": "software_diagnostic_only_no_real_world_claim",
        "logs_tail": (result.stdout + "\n" + result.stderr)[-2500:],
    }
    artifact.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("plan")
    one.add_argument("proposal")
    scan = sub.add_parser("scan")
    scan.add_argument("--out", type=Path, default=Path("ci-artifacts/phone-lab"))
    scan.add_argument("--post-issues", action="store_true")
    scan.add_argument("--run-safe-diagnostics", action="store_true")
    args = p.parse_args(argv)
    try:
        if args.cmd == "plan":
            proposal = load_proposal(Path(args.proposal), ROOT)
            result = decide(proposal)
            print(json.dumps(result, indent=2, sort_keys=True))
            print(phone_markdown(result), file=sys.stderr)
            return 0
        rows = all_proposals()
        out = args.out
        out.mkdir(parents=True, exist_ok=True)
        report = {"schema": "tac-osm.phone-progress.v1",
                  "total": len(rows), "runs": [{"path": path, **decision}
                                               for path, decision in rows]}
        if args.run_safe_diagnostics:
            for path, row in rows[:3]:
                if row.get("state") != "READY_NO_FIT" or row.get("backend") != "github_actions":
                    continue
                prop = load_proposal(Path(path), ROOT)
                result = run_suite(prop["diagnostic"], out / (row["run_id"] + "-diagnostic.json"))
                row["diagnostic_status"] = result["status"]
                report["runs"] = [{"path": a, **b} for a, b in rows]
        (out / "progress.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        lines = ["# Phone-first research progress",
                 "", f"**Tracked runs:** {len(rows)}", "",
                 "| Run | Model | Stage | Compute | Progress |",
                 "| --- | --- | --- | --- | --- |"]
        for path, row in rows:
            if row.get("state") == "INVALID_PROPOSAL":
                lines.append(f"| {path} | — | — | — | INVALID |")
                continue
            route = "Kaggle 2× T4" if row["backend"] == "kaggle_t4x2" else "GitHub CPU"
            status = row.get("diagnostic_status", row["state"])
            lines.append(f"| {row['run_id']} | {row['variant']} | {row['kind']} | {route} | {status} |")
            if args.post_issues and row.get("backend") == "github_actions":
                # Remote notebook status is owned by the Kaggle monitor;
                # don't overwrite it with a generic "ready to plan" message.
                token, repo = os.environ.get("GH_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
                if not token or not repo:
                    raise ResearchPlanError("GitHub comment mode needs Actions token/repository")
                comment = phone_markdown(row)
                if row.get("diagnostic_status"):
                    comment += "\n**Last source-only diagnostic:** " + row["diagnostic_status"] + "\n"
                update_issue(repo, row["issue_number"], comment, token)
        (out / "progress.md").write_text("\n".join(lines) + "\n")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as file:
                file.write("\n".join(lines) + "\n")
        print(json.dumps({"status": "REPORT_CREATED_NO_GPU_SUBMISSION",
                          "count": len(rows), "path": str(out / "progress.md")}))
        return 0
    except (ResearchPlanError, ValueError, OSError, TypeError, KeyError,
            urllib.error.URLError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)[:300]}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
