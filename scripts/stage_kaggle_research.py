#!/usr/bin/env python3
"""Stage a source-reviewed Kaggle T4×2 notebook; NO upload without --submit.

The GitHub workflow uses a protected environment approval before passing
--submit and Kaggle credentials. The stage step is always offline/safe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.phone_research_ops import (  # noqa: E402
    GPU_ACCELERATOR, ResearchPlanError, decide, load_proposal,
)


def package(proposal_path: Path, *, kaggle_user: str, destination: Path) -> dict:
    proposal = load_proposal(proposal_path, ROOT)
    decision = decide(proposal)
    if decision["backend"] != "kaggle_t4x2":
        raise ResearchPlanError("Kaggle route was not selected by compute policy")
    # Signed evidence blockers are resolved ONLY in the protected execution
    # workflow. The package step is unqualified metadata preparation.
    concrete = [x for x in decision["blockers"] if not x.startswith(
        ("independent signed prefit", "independent evaluator and protected evaluation"))]
    if concrete:
        raise ResearchPlanError("unsafe/incomplete GPU plan: " + "; ".join(concrete))
    if not isinstance(kaggle_user, str) or not kaggle_user or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in kaggle_user):
        raise ResearchPlanError("Kaggle account name missing/invalid")
    job = ROOT / proposal["workload"]
    if job.is_symlink() or not job.is_file() or not job.resolve().is_relative_to(ROOT):
        raise ResearchPlanError("reviewed Kaggle workload missing or symlinked")
    harness = ROOT / "scripts/kaggle_dual_t4_entry.py"
    if not harness.is_file():
        raise ResearchPlanError("missing dual-T4 runtime guard")
    identifier = "tacosm-" + proposal["run_id"].lower().replace("_", "-").replace(".", "-")
    identifier = identifier[:68].strip("-")
    if not identifier or len(identifier) < 4:
        raise ResearchPlanError("invalid Kaggle kernel slug")
    handle = kaggle_user + "/" + identifier
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(job, destination / "workload.py")
    shutil.copyfile(harness, destination / "entrypoint.py")
    digest = hashlib.sha256(job.read_bytes()).hexdigest()
    config = {
        "run_id": proposal["run_id"], "gpu_strategy": proposal["gpu_strategy"],
        "seeds": proposal["seeds"], "accelerator": GPU_ACCELERATOR,
        "workload_sha256": digest, "decision_digest": decision["source_contract_fingerprint"],
        "scientific_qualification": "NONE_PRE_AND_POST_INDEPENDENT_GATES_REQUIRED",
    }
    # Kaggle links each kernel's URL slug to its title. Keep metadata.id
    # and the title-derived slug consistent for ALL future submissions.
    # Do not resubmit or rename historical private kernels during a status check.
    metadata = {
        "id": handle, "title": "TACOSM " + proposal["run_id"],
        "code_file": "entrypoint.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": proposal["dataset_sources"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    (destination / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    report = {"schema": "tac-osm.phone-kaggle-package.v1",
              "status": "STAGED_UNSUBMITTED", "kaggle_kernel": handle,
              "accelerator": "NvidiaTeslaT4",
              "gpu_count_required": 2,
              "estimated_gpu_minutes": proposal["estimated_gpu_minutes"],
              "workload_sha256": digest, "proposal_sha256": decision["source_contract_fingerprint"],
              "outcome_qualified": False, "path": str(destination)}
    (destination / "staging-receipt.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--kaggle-user", required=True)
    parser.add_argument("--out", type=Path, default=Path("ci-artifacts/kaggle-stage"))
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = package(args.proposal, kaggle_user=args.kaggle_user, destination=args.out)
        if args.submit:
            # Source and publisher verification MUST have been run by the
            # separately permissioned Actions execution job before this call.
            cmd = ["kaggle", "kernels", "push", "-p", str(args.out),
                   "--accelerator", GPU_ACCELERATOR,
                   "--timeout", str(min(report["estimated_gpu_minutes"] * 60, 14400))]
            completed = subprocess.run(cmd, check=False, timeout=150,
                                       capture_output=True, text=True)
            if completed.returncode:
                raise ResearchPlanError("Kaggle submission failed or quota exhausted; no accelerator fallback: " +
                                        completed.stderr[-500:])
            report["status"] = "SUBMITTED_AWAITING_KAGGLE_STATUS"
        print(json.dumps(report, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"status": "BLOCKED_NO_KAGGLE_FALLBACK",
                          "reason": str(exc)[:400]}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
