#!/usr/bin/env python3
"""Read-only Kaggle saved-output retrieval with explicit API-access diagnosis.

Never submits notebooks, requests an accelerator, downloads non-smoke output,
or claims remote execution from mere kernel discoverability. Intended for
the protected phone-lab smoke audit workflow.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

EXPECTED_SLUG = "tacosm-dual-t4-readiness"
# The historical notebook title is "TAC-OSM dual-t4-readiness", which
# Kaggle may expose under a title-derived slug different from metadata.id.
TITLE_DERIVED_SLUG = "tac-osm-dual-t4-readiness"
SMOKE_PATTERN = r"(^|/)(phone-research-runtime|seed-101|seed-102)\.json$"
OWNER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
HANDLE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}/[a-z0-9][a-z0-9_-]{0,90}$")


def classify_error(stderr: str) -> str:
    lowered = stderr.lower()
    if "kernels.get" in lowered and ("denied" in lowered or "permission" in lowered):
        return "KAGGLE_API_ACCESS_DENIED_NOT_GPU_FAILURE"
    if "not found" in lowered or "404" in lowered:
        return "KAGGLE_KERNEL_REFERENCE_OR_VERSION_UNAVAILABLE"
    if "429" in lowered or "rate limit" in lowered:
        return "KAGGLE_API_RATE_LIMITED"
    return "KAGGLE_OUTPUT_ACCESS_UNAVAILABLE"


def listing_indicators(stdout: str, owner: str, expected: str) -> dict:
    """Only share flags, never private notebook names or CSV from the account."""
    exact = False
    similar = False
    title_derived = False
    for row in csv.reader(io.StringIO(stdout)):
        valid_refs = [cell.strip().lower() for cell in row
                      if HANDLE.fullmatch(cell.strip().lower())]
        for ref in valid_refs:
            if not ref.startswith(owner + "/"):
                continue
            if ref == expected:
                exact = True
            elif ref == owner + "/" + TITLE_DERIVED_SLUG:
                title_derived = True
                similar = True
            elif "tacosm" in ref or "dual-t4-readiness" in ref:
                similar = True
    return {"expected_handle_listed": exact,
            "title_derived_handle_listed": title_derived,
            "similar_handle_listed": similar}


def inspect(username: str, directory: Path, runner=subprocess.run) -> tuple[bool, dict]:
    if not OWNER.fullmatch(username):
        raise ValueError("invalid Kaggle account name")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise ValueError("Kaggle token unavailable in protected job")
    handle = username + "/" + EXPECTED_SLUG
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle", "kernels", "output", handle,
        "-p", str(directory), "--file-pattern", SMOKE_PATTERN, "--quiet",
    ]
    try:
        result = runner(command, capture_output=True, text=True, timeout=65)
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0:
        return True, {"state": "OUTPUT_DOWNLOAD_COMMAND_SUCCEEDED",
                      "saved_receipts_still_require_offline_audit": True}
    classification = (classify_error(result.stderr) if result is not None
                      else "KAGGLE_CLI_FAILED_OR_TIMED_OUT")
    listed = {"expected_handle_listed": None,
              "title_derived_handle_listed": None,
              "similar_handle_listed": None}
    try:
        res = runner(["kaggle", "kernels", "list", "--mine", "--csv",
                      "--page-size", "100", "--sort-by", "dateRun"],
                     capture_output=True, text=True, timeout=30)
        if res.returncode == 0:
            listed = listing_indicators(res.stdout, username, handle)
            listing_state = "LIST_QUERY_SUCCEEDED_NOT_PROOF_OF_NONEXISTENCE"
        else:
            listing_state = "LIST_QUERY_FAILED"
    except (OSError, subprocess.TimeoutExpired):
        listing_state = "LIST_QUERY_FAILED_OR_TIMED_OUT"

    # Only inspect the historically known title-derived identifier if
    # the authenticated account listing confirms that exact owned handle.
    # This is a SECOND READ of existing output, never a new Kaggle version.
    if listed["title_derived_handle_listed"] is True:
        alternate = username + "/" + TITLE_DERIVED_SLUG
        try:
            second = runner([
                "kaggle", "kernels", "output", alternate, "-p", str(directory),
                "--file-pattern", SMOKE_PATTERN, "--quiet"],
                capture_output=True, text=True, timeout=65)
        except (OSError, subprocess.TimeoutExpired):
            second = None
        if second is not None and second.returncode == 0:
            return True, {
                "state": "TITLE_DERIVED_OUTPUT_DOWNLOAD_SUCCEEDED",
                "resolved_via_title_derived_slug": True,
                "saved_receipts_still_require_offline_audit": True,
            }
        alternate_state = (classify_error(second.stderr) if second is not None
                           else "KAGGLE_CLI_FAILED_OR_TIMED_OUT")
    else:
        alternate_state = "NOT_ATTEMPTED_NO_CONFIRMED_ALTERNATE_HANDLE"

    return False, {
        "state": classification, "kaggle_listing_state": listing_state,
        "title_derived_access_state": alternate_state,
        **listed, "saved_receipts_still_require_offline_audit": True,
        "remote_execution_verified": False,
        "self_reported_two_worker_execution": False,
        "independent_remote_attestation": False,
        "scientific_qualification": "NONE",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        ok, summary = inspect(args.username, args.output_dir)
    except (ValueError, OSError) as exc:
        ok = False
        summary = {"state": "CONFIGURATION_OR_FILESYSTEM_FAILURE",
                   "reason": type(exc).__name__,
                   "remote_execution_verified": False,
                   "scientific_qualification": "NONE"}
    if not ok:
        report = {"schema": "tac-osm.kaggle-no-fit-receipt-audit.v1",
                  "status": "INCOMPLETE_NOT_VERIFIED",
                  **summary,
                  "remote_execution_verified": False,
                  "self_reported_two_worker_execution": False,
                  "independent_remote_attestation": False,
                  "scientific_qualification": "NONE"}
        args.report.parent.mkdir(exist_ok=True, parents=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
