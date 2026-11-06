"""Optional vendor/cluster integration PLANS. Zero network and no job execution.

The standard JSON ledger remains canonical. This module maps its safe metadata
to tool choices without requiring MLflow/W&B/Comet/DVC/HF/K8s/Slurm/Ray.
No tool commands are executed; no checkpoint/dataset content is uploaded.
"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any
from .research_lab import LabError, SCHEMA, _name

TRACKERS = ("mlflow", "wandb", "comet")
STORAGE = ("dvc", "huggingface")
COMPUTE = ("local", "docker", "slurm", "ray", "kubernetes")
COLLAB = ("github", "jupyter", "colab", "notion", "confluence", "slack", "discord")
PROVIDERS = TRACKERS + STORAGE + COMPUTE + COLLAB
_DOC = {
    "mlflow": "https://mlflow.org/docs/latest/ml/tracking/",
    "wandb": "https://docs.wandb.ai/",
    "comet": "https://www.comet.com/docs/",
    "dvc": "https://dvc.org/doc",
    "huggingface": "https://huggingface.co/docs/hub/",
    "local": "https://docs.python.org/3/library/subprocess.html",
    "docker": "https://docs.docker.com/",
    "slurm": "https://slurm.schedmd.com/documentation.html",
    "ray": "https://docs.ray.io/",
    "kubernetes": "https://kubernetes.io/docs/",
    "github": "https://docs.github.com/en/actions",
    "jupyter": "https://jupyterlab.readthedocs.io/",
    "colab": "https://colab.research.google.com/",
    "notion": "https://developers.notion.com/",
    "confluence": "https://developer.atlassian.com/cloud/confluence/",
    "slack": "https://api.slack.com/messaging/webhooks",
    "discord": "https://discord.com/developers/docs/resources/webhook",
}


def plan(provider: str, *, receipt: dict | None = None,
         command: list[str] | None = None,
         limit_minutes: int = 30, cpus: int = 1,
         gpus: int = 0) -> dict[str, Any]:
    """A bounded no-side-effects setup recipe, not an active integration."""
    if provider not in PROVIDERS:
        raise LabError("unsupported optional provider")
    if (type(limit_minutes) is not int or not 1 <= limit_minutes <= 120
            or type(cpus) is not int or not 1 <= cpus <= 16
            or type(gpus) is not int or not 0 <= gpus <= 2):
        raise LabError("bounded manual job resource reservation required")
    if command is not None and (not isinstance(command, list) or
            not 1 <= len(command) <= 40 or
            any(not isinstance(x, str) or not x or len(x) > 300 or
                "\n" in x or "\r" in x for x in command)):
        raise LabError("command needs short explicit argument vector")
    if receipt is not None and (not isinstance(receipt, dict) or
                                receipt.get("schema") != SCHEMA or
                                receipt.get("capability_claim_authorized") is not False):
        raise LabError("only an unqualified verified-schema lab receipt can be mapped")
    base = {
        "schema": "tac-osm.lab-provider-plan.v1",
        "provider": provider, "reference": _DOC[provider],
        "enabled": False, "executed": False, "external_data_sent": False,
        "default_workflow": "GitHub Actions + existing Python + offline lab ledger",
        "manual_opt_in_required": True,
    }
    if provider in TRACKERS:
        base["category"] = "experiment_tracking"
        base["safe_export"] = (
            {"experiment": receipt["experiment"], "run_id": receipt["run_id"],
             "variant": receipt["variant"], "params": receipt["parameters"],
             "metrics": receipt["metrics"],
             "source_report_sha256": receipt["source_report"]["sha256"],
             "qualification": "NOT_EVALUATED_BY_LAB"}
            if receipt else None)
        base["instructions"] = (
            "Install selected SDK in a separate optional environment. "
            "Only publish explicit scalar metrics/parameters; never upload "
            "raw source records, heldout labels or checkpoints automatically.")
    elif provider in STORAGE:
        base["category"] = "data_model_versioning"
        base["instructions"] = (
            "Opt in to DVC with an operator-owned remote, or a PRIVATE Hugging Face "
            "model/dataset repository. Do not export protected publisher data or "
            "checkpoints absent license and privacy review. Run digests and "
            "licenses must remain in the canonical local lab record.")
    elif provider in COMPUTE:
        base["category"] = "compute"
        base["limits"] = {"minutes": limit_minutes, "cpus": cpus, "gpus": gpus}
        base["command_argv"] = command
        if provider == "local":
            base["instructions"] = "Use the existing Python script directly after source/preflight gates."
        elif provider == "docker":
            base["instructions"] = (
                "Choose a pinned image digest and mount only authorized source "
                "files. Mount evaluator data and signing keys NEVER.")
            base["example"] = {
                "image": "REPLACE_WITH_PINNED_IMAGE_SHA256",
                "network": "none", "read_only_root_fs": True,
                "memory": "SET_EXPLICIT_LIMIT",
                "command_argv": command,
            }
        elif provider == "slurm":
            # This text is only a reviewable proposal, NOT submitted.
            commands = shlex.join(command) if command else "<REPLACE_WITH_AUTHORIZED_COMMAND>"
            base["example_sbatch"] = (
                "#!/bin/bash\n#SBATCH --job-name=tac-osm-lab\n"
                f"#SBATCH --time={limit_minutes}\n#SBATCH --cpus-per-task={cpus}\n"
                + (f"#SBATCH --gres=gpu:{gpus}\n" if gpus else "")
                + "set -euo pipefail\n"
                + "# Run only after independent training preflight approval\n"
                + commands + "\n")
        elif provider == "kubernetes":
            base["instructions"] = ("REVIEW ONLY: use a namespace with ResourceQuota, "
                                    "read-only source volumes, zero evaluator credentials, "
                                    "disabled service account token, no hostPath, and NetworkPolicy.")
        elif provider == "ray":
            base["instructions"] = ("Optional local Ray trial orchestration only after "
                                    "preregistered hyperparameter budget approval. "
                                    "No Anyscale cloud billing or cluster creation.")
    else:
        base["category"] = "collaboration"
        base["instructions"] = {
            "github": "Use existing PRs, CI check summaries, Issues and artifact downloads.",
            "jupyter": "Explore local preapproved FIT/DEV reports read-only in JupyterLab.",
            "colab": "Mount only sharable sanitized receipts; avoid protected sources or passwords.",
            "notion": "Manual page link to canonical repository research note, never substitute for Git.",
            "confluence": "Manual synced paper/experiment index with GitHub as source of truth.",
            "slack": "Optional GitHub Actions completion notification after manual webhook setup.",
            "discord": "Optional GitHub Actions completion notification after manual webhook setup.",
        }[provider]
    return base


def read_one_receipt(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise LabError("not an existing lab receipt")
    if payload.get("capability_claim_authorized") is not False:
        raise LabError("record self-promotes capability")
    return payload
