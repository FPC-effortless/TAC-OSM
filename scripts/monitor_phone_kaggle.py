#!/usr/bin/env python3
"""Read-only Kaggle execution monitoring -> one compact phone Issue comment.

A remote completed status is NOT model/evaluator scientific qualification;
original source and independently recomputed goals remain separate gates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tac_osm.phone_research_ops import decide, load_proposal, phone_markdown
from phone_lab import update_issue  # noqa: E402

OWNER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


def inspect(proposal: dict, username: str) -> dict:
    plan = decide(proposal)
    if plan["backend"] != "kaggle_t4x2":
        raise ValueError("Kaggle monitoring requested for CPU research")
    if not OWNER.fullmatch(username):
        raise ValueError("invalid Kaggle username")
    slug = ("tacosm-" + proposal["run_id"].lower()
            .replace("_", "-").replace(".", "-"))[:68].strip("-")
    handle = username + "/" + slug
    try:
        r = subprocess.run(["kaggle", "kernels", "status", handle],
                           capture_output=True, text=True, timeout=30)
        evidence = (r.stdout + "\n" + r.stderr)[-800:]
        if r.returncode == 0:
            state = "REMOTE_STATUS_AVAILABLE_NOT_QUALIFIED"
        elif "Permission 'kernels.get' was denied" in evidence or "403" in evidence:
            state = "REMOTE_ACCESS_DENIED_NOT_PROOF_OF_ABSENCE"
        else:
            state = "REMOTE_STATUS_UNAVAILABLE_NOT_PROOF_OF_ABSENCE"
    except (OSError, subprocess.TimeoutExpired):
        evidence = "CLI missing, unavailable or exceeded query deadline"
        state = "STATUS_UNAVAILABLE"
    return {"schema": "tac-osm.phone-kaggle-monitor.v1",
            "run_id": proposal["run_id"], "issue_number": proposal["issue_number"],
            "backend": "kaggle_t4x2", "required_accelerator": "NvidiaTeslaT4",
            "kernel": handle, "state": state,
            "remote_status_excerpt": evidence,
            "observed_two_gpu_execution": False,
            "real_world_qualified": False,
            "checkpoint_replay_verified": False}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--proposal", type=Path, required=True)
    p.add_argument("--username", required=True)
    p.add_argument("--out", type=Path, default=Path("ci-artifacts/kaggle-monitor"))
    p.add_argument("--post-issue", action="store_true")
    args = p.parse_args(argv)
    try:
        prop = load_proposal(args.proposal, ROOT)
        status = inspect(prop, args.username)
        args.out.mkdir(exist_ok=True, parents=True)
        (args.out / (prop["run_id"] + ".json")).write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n")
        if args.post_issue:
            token, repo = os.environ.get("GH_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
            if not token or not repo:
                raise ValueError("GitHub read-only status updater requires issue comment auth")
            decision = decide(prop)
            markdown = phone_markdown(decision)
            markdown += ("\n**Remote Kaggle status:** " + status["state"]
                         + "\n\n**Kernel:** https://www.kaggle.com/code/" + status["kernel"]
                         + "\n\nRemote logs do not establish both GPUs were used, "
                           "a safe world model, or independently verified reward.\n")
            update_issue(repo, prop["issue_number"], markdown, token)
        print(json.dumps(status, sort_keys=True))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)[:200]}),
              file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
