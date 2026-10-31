"""Phone-first TAC-OSM research job decisions and trackable approval gates.

Pure Python stdlib. Does not train, fetch secrets, invent source evidence, or
launch Kaggle. Decisions are deterministic/replayable; agent recommendations
cannot override the frozen science/trust policy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

SCHEMA = "tac-osm.phone-research.v1"
FAMILIES = ("System 1", "Enterprise", "Temporal", "Physical", "Edge-Lite", "Counterfactual")
KINDS = ("no_fit", "source_audit", "training", "evaluation", "replay")
GPU_ACCELERATOR = "NvidiaTeslaT4"  # Kaggle CLI: exactly T4 x2
SEED_STRATEGIES = ("seed_shard", "ddp")
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,79}$")
SHA = re.compile(r"^[0-9a-f]{64}$")
KAGGLE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,38}/[a-z0-9][a-z0-9-]{1,69}$")
REG_PATH = re.compile(r"^research/mathematical-engineering/[a-z0-9_-]+/FAIR_COMPARISON\.json$")
WORKLOAD = re.compile(r"^research/kaggle_workloads/[a-z0-9_-]+\.py$")
DIAGNOSTICS = ("lab", "anticheat", "observatory", "scientific", "langmath095", "langmath098", "langtrain099")
CPU_JOB_LIMIT_MIN = 45
GPU_JOB_LIMIT_MIN = 240


class ResearchPlanError(ValueError):
    pass


def _int(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ResearchPlanError(f"{name} must be integer {low}..{high}")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ResearchPlanError(f"{name} needs independently traceable sha256")
    return value


def load_proposal(path: Path, root: Path) -> dict:
    """Only use a reviewable repo-side JSON proposal. No user-supplied shell."""
    root, path = Path(root).resolve(), Path(path)
    relative = path.as_posix()
    if not re.fullmatch(r"research/phone_queue/[a-z0-9_-]+\.json", relative):
        raise ResearchPlanError("run proposal must be versioned under research/phone_queue/")
    located = root / path
    if located.is_symlink() or not located.is_file() or not located.resolve().is_relative_to(root):
        raise ResearchPlanError("missing or escaped repository-owned proposal")
    return json.loads(located.read_text(encoding="utf-8"))


def decide(proposal: dict) -> dict:
    if not isinstance(proposal, dict) or proposal.get("schema") != SCHEMA:
        raise ResearchPlanError("invalid research proposal schema")
    required = {"schema", "run_id", "variant", "kind", "issue_number",
                "estimated_cpu_minutes", "requires_cuda", "gpu_speedup_expected",
                "estimated_gpu_minutes", "gpu_strategy", "seeds", "real_source_sha256",
                "study_manifest", "workload", "diagnostic", "dataset_sources",
                "budget_gpu_minutes"}
    if set(proposal) not in (required, required | {"execution_level"}):
        raise ResearchPlanError("exact proposal contract required; missing or unknown fields")
    from tac_osm.research_execution import execution_level, INTERNAL
    level = execution_level(proposal)
    run_id = proposal["run_id"]
    if not isinstance(run_id, str) or not ID.fullmatch(run_id):
        raise ResearchPlanError("unsafe run ID")
    variant, kind = proposal["variant"], proposal["kind"]
    if variant not in FAMILIES or kind not in KINDS:
        raise ResearchPlanError("unknown family or research kind")
    issue = _int(proposal["issue_number"], "issue_number", 1, 1_000_000_000)
    cpu = _int(proposal["estimated_cpu_minutes"], "estimated_cpu_minutes", 1, 10000)
    gpu = _int(proposal["estimated_gpu_minutes"], "estimated_gpu_minutes", 0, GPU_JOB_LIMIT_MIN)
    budget = _int(proposal["budget_gpu_minutes"], "budget_gpu_minutes", 0, GPU_JOB_LIMIT_MIN)
    if type(proposal["requires_cuda"]) is not bool or type(proposal["gpu_speedup_expected"]) is not bool:
        raise ResearchPlanError("CUDA and speedup declarations must be boolean")
    strategy = proposal["gpu_strategy"]
    if strategy is not None and strategy not in SEED_STRATEGIES:
        raise ResearchPlanError("unknown two-GPU execution contract")
    seeds = proposal["seeds"]
    if not isinstance(seeds, list) or len(seeds) > 12 or any(type(x) is not int or
                                  x < 0 or x > 2_147_483_647 for x in seeds):
        raise ResearchPlanError("invalid frozen seed set")
    if len(set(seeds)) != len(seeds):
        raise ResearchPlanError("duplicate seeds")
    diagnostic = proposal["diagnostic"]
    workload = proposal["workload"]
    study = proposal["study_manifest"]
    source = proposal["real_source_sha256"]
    sources = proposal["dataset_sources"]
    if not isinstance(sources, list) or len(sources) > 5 or any(
            not isinstance(s, str) or not KAGGLE_ID.fullmatch(s) for s in sources):
        raise ResearchPlanError("Kaggle dataset sources must be named, reviewed slugs")
    if source is not None:
        _digest(source, "real_source_sha256")
    if study is not None and (not isinstance(study, str) or not REG_PATH.fullmatch(study)):
        raise ResearchPlanError("invalid immutable fair-comparison registration path")
    if workload is not None and (not isinstance(workload, str) or not WORKLOAD.fullmatch(workload)):
        raise ResearchPlanError("workload must be a dedicated, reviewed Kaggle Python script")
    if diagnostic is not None and diagnostic not in DIAGNOSTICS:
        raise ResearchPlanError("only reviewed diagnostic suites may auto-run on GitHub")
    training = kind in ("training", "evaluation")
    # Agent chooses based on task requirements, not a claimed GPU preference:
    wants_gpu = proposal["requires_cuda"] or (
        proposal["gpu_speedup_expected"] and cpu > CPU_JOB_LIMIT_MIN)
    backend = "kaggle_t4x2" if wants_gpu else "github_actions"
    reasons = []
    blockers = []
    if wants_gpu:
        reasons.append("declared CUDA requirement or long job with justified GPU acceleration")
        if gpu == 0 or gpu > budget:
            blockers.append("missing or insufficient explicit GPU-minute budget")
        if strategy not in SEED_STRATEGIES:
            blockers.append("dual T4 execution strategy missing")
        if not workload:
            blockers.append("Kaggle script not source-registered")
        if not sources and training:
            blockers.append("independent authorized Kaggle dataset reference absent")
        if strategy == "seed_shard" and len(seeds) < 2:
            blockers.append("seed sharding requires at least two distinct seeds")
        if strategy == "ddp" and not seeds:
            blockers.append("DDP needs frozen seed")
    else:
        reasons.append("CPU or diagnostic workload fits small GitHub runner budget")
        if cpu > CPU_JOB_LIMIT_MIN:
            blockers.append("estimated CPU workload exceeds GitHub job budget; redesign plan")
        if kind in ("no_fit", "source_audit") and diagnostic not in DIAGNOSTICS:
            blockers.append("approved no-fit diagnostic suite required for auto-execution")
    if training:
        if source is None or study is None:
            blockers.append("source-authorized real data and preregistered scientific plan required")
        if not seeds:
            blockers.append("trained comparison/replay requires frozen seeds")
        # The source-controlled proposal itself NEVER qualifies evidence.
        if level != INTERNAL:
            blockers.append("independent signed prefit reward/source/ACL evidence required at execution")
        if kind == "evaluation" and level != INTERNAL:
            blockers.append("independent evaluator and protected evaluation authorization required")
    if kind == "replay" and not seeds:
        blockers.append("checkpoint replay requires frozen seeds and verifiable checkpoint")
    state = ("BLOCKED_EVIDENCE_OR_DESIGN" if blockers else
             "READY_NO_FIT" if kind in ("no_fit", "source_audit") else
             "READY_INTERNAL_RESEARCH_PLAN" if level == INTERNAL else
             "AWAITING_OPERATOR_APPROVAL")
    return {
        "schema": "tac-osm.compute-decision.v1",
        "run_id": run_id, "issue_number": issue, "variant": variant,
        "kind": kind, "backend": backend,
        "accelerator": GPU_ACCELERATOR if wants_gpu else None,
        "gpu_devices_required": 2 if wants_gpu else 0,
        "cpu_estimate_minutes": cpu, "gpu_budget_minutes": budget,
        "state": state, "reasons": reasons, "blockers": blockers,
        "execution_level": level,
        "approved_to_train": False, "approved_to_open_test": False,
        "automatic_submissions": False,
        "source_contract_fingerprint": hashlib.sha256(
            json.dumps(proposal, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def phone_markdown(decision: dict) -> str:
    def clean(value: object) -> str:
        # IDs and labels in validated proposal; still guard Markdown control.
        return str(value).replace("|", " ").replace("\n", " ")[:120]
    backend = "Kaggle: **2× T4**" if decision["backend"] == "kaggle_t4x2" else "GitHub Actions CPU"
    tasks = decision["blockers"] or ["No planning blockers; normal source/gate checks still apply"]
    return (
        "<!-- tac-osm-phone-lab-status -->\n"
        "### TAC-OSM research lab status\n\n"
        f"**Run:** {clean(decision['run_id'])} · **Model:** {clean(decision['variant'])}\n\n"
        f"**Compute decision:** {backend} · **Stage:** {clean(decision['kind'])}\n\n"
        f"**Progress:** {clean(decision['state'])}\n\n"
        + "\n".join(f"- [ ] {clean(item)}" for item in tasks)
        + "\n\n*This is a planning receipt, not model training approval or scientific validation.*\n"
    )
