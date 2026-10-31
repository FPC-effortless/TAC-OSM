"""Freeze data/order/model declarations for review, without authorizing a fit.

The emitted drafts are not FAIR_COMPARISON.json: T4 compute, mathematical/probe
coverage and independent signed evidence still have to qualify before fitting.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from source import canonical, digest, verify_package

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (10301, 10302, 10303)
SEQUENCE = 256
BATCH = 32
MILESTONES = (2560, 12800, 25600)


def source_blocks(receipt):
    blocks = []
    for row in sorted(receipt["identity"]["files"], key=lambda r: r["path"]):
        if row["split"] == "FIT":
            blocks.extend((row["path"], offset)
                          for offset in range(0, row["bytes"]-SEQUENCE, SEQUENCE))
    if len(blocks) < BATCH:
        raise ValueError("Insufficient independent complete FIT sequences")
    return blocks


def sample_stream(blocks, seed):
    rng = random.Random(seed)
    while True:
        shuffled = blocks.copy(); rng.shuffle(shuffled)
        yield from shuffled


def order_receipts(receipt, milestones=MILESTONES):
    blocks = source_blocks(receipt)
    result = {}
    for seed in SEEDS:
        stream = sample_stream(blocks, seed)
        h = hashlib.sha256(canonical({"source": receipt["source_sha256"], "seed": seed,
                "sequence": SEQUENCE, "batch": BATCH, "order": "whole-FIT-chunk-permutations.v1"}))
        rows = []
        for step in range(1, max(milestones)+1):
            for _ in range(BATCH):
                name, offset = next(stream)
                h.update(canonical([name, offset]) + b"\n")
            if step in milestones:
                rows.append({"steps": step, "examples_seen": step*BATCH,
                    "input_and_target_bytes_seen_each": step*BATCH*SEQUENCE,
                    "sample_order_sha256": h.hexdigest()})
        result[str(seed)] = rows
    return {"algorithm": "Python-3.12 random.Random(seed), shuffle original sorted complete FIT chunks each epoch",
            "file_boundaries_preserved": True, "carry_between_unrelated_chunks": False,
            "unique_complete_fit_chunks": len(blocks), "sequence_length": SEQUENCE,
            "batch_size": BATCH, "milestones": result}


def prepare(receipt_path, archive_path, resource_path, out):
    receipt, _ = verify_package(receipt_path, archive_path)
    resources = json.loads(Path(resource_path).read_text())
    if resources["source_sha256"] != receipt["source_sha256"] or resources["optimizer_updates"] != 0:
        raise ValueError("Original-input resource evidence does not match source or zero-update scope")
    candidates = json.loads((ROOT / "research/langcap103_results/PROFILE_RUN_41219856043.json").read_text())
    measured = {(r["target"], r["configuration"]["arm"]): r for r in resources["profiles"]}
    order = order_receipts(receipt)
    sys.path.insert(0, str(ROOT))
    from scripts.validate_research_parity import check_manifest, ParityError
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    drafted = []
    for cohort in candidates["cohorts"]:
        target = cohort["candidate_target"]
        models = []
        for original in cohort["arms"]:
            key = (target, original["configuration"]["arm"])
            observed = measured[key]
            if observed["configuration"] != original["configuration"] or any(
                    observed[k] != original[k] for k in ("trainable_parameters", "deployed_parameters")):
                raise ValueError("Resource-probed model differs from frozen candidate")
            for seed in SEEDS:
                models.append({"variant": key[1], "seed": seed,
                    "trainable_parameters": original["trainable_parameters"],
                    "deployed_parameters": original["deployed_parameters"],
                    "configuration": original["configuration"]})
        manifest = {"schema_version": "tac-osm.parity.v1",
            "study_id": f"LANGCAP103_{target//1_000_000}M_CODE_LEARNABILITY",
            "study_type": "single_model_training", "registered_before_first_fit": True,
            "model_scale_policy": "declared-target-v2",
            "parameter_scale": {"target_trainable_parameters": target,
                "target_deployed_parameters": target, "relative_tolerance": .05,
                "registered_before_first_fit": True,
                "rationale": "Larger-model code learnability; no architecture-superiority claim from this independent-fit cohort"},
            "real_world_source": {"name": "Original CPython 3.13.0 source byte records",
                "sha256": receipt["source_sha256"], "observed_real_outcomes": True,
                "outcome_semantics": "Original next source byte; not actions, rewards, economic outcomes or causal policy utility",
                "license": "Publisher PSF-2.0 and retained path-specific original notices; independent permission audit pending",
                "publisher_commit": receipt["identity"]["publisher_commit"]},
            "models": models, "no_architecture_superiority_claim": True,
            "registration_status": "REVIEW_DRAFT_NOT_A_COMPLETE_AUTHORIZED_TRAINING_CONTRACT",
            "training_enabled": False, "sample_order_contract": order,
            "input_contract": "Original raw bytes, same FIT file chunks/order across arms; reset between unrelated chunks",
            "source_split_contract": receipt["stats"],
            "optimizer_proposal": {"name": "AdamW", "base_lr": .0003, "betas": [.9, .95],
                "eps": 1e-8, "weight_decay": .1, "precision": "float32",
                "status": "PROPOSED_NOT_TUNED_OR_RESOURCE_AUTHORIZED"},
            "compute": {"full_training_flops": None, "t4_throughput": None, "approved_gpu_minutes": 0},
            "interpretation": "Learnability and loss curves only. Comparative superiority needs the full separately qualified architecture-comparison contract."}
        # Exercise the real shared signed gate on numeric/source declarations.
        # Never supply fixture keys, create signatures, or claim this negative is readiness.
        try:
            check_manifest(manifest, "preflight", reward_attestation=None,
                           trusted_reward_public_key_hex=None)
        except ParityError as exc:
            reason = str(exc)
            if "independent reward/outcome/access runner gate" not in reason:
                raise
        else:
            raise ValueError("Unsigned real-source training declaration was unexpectedly authorized")
        name = f"REGISTRATION_DRAFT_{target//1_000_000}M.json"
        (out / name).write_text(json.dumps(manifest, indent=2) + "\n")
        drafted.append({"path": name, "sha256": digest((out/name).read_bytes()),
            "numeric_declarations": "ACCEPTED_BY_SHARED_GATE_BEFORE_SIGNATURE_CHECK",
            "unsigned_prefit": "REJECTED_AS_REQUIRED", "actual_gate_reason": reason})
    readiness = {"schema": "langcap103.training-preparation.v1",
        "status": "BLOCKED_TRAINING_PREREQUISITES_REMAIN", "training_enabled": False,
        "source_sha256": receipt["source_sha256"], "registration_drafts": drafted,
        "completed": ["publisher original bytes/commit/tree/blob/archive verification",
            "module-group FIT/DEV and exact duplicate exclusions",
            "actual 1M/10M parameter counts and original-FIT finite backward graphs",
            "deterministic FIT-only source order and exact byte-exposure milestones",
            "real shared gate rejects absence of independent signatures"],
        "blockers": ["independent source/goal/reward/ACL prefit evidence and operator-pinned verifier key not supplied to this audit",
            "source-equivalent PNDS Lean/Python and architecture-probe qualification OPEN",
            "complete training FLOPs, actual two-T4 resource pilot and permitted quota OPEN",
            "reviewed optimizer/tuning/selection/evaluation contract and final FAIR_COMPARISON registration OPEN"],
        "scope_limits": ["public same-project DEV, no protected independent TEST or repository-transfer result",
            "repeated byte presentations are not unique training data; adequacy must be judged from learning curves",
            "partial profiler FLOPs never establish full resource parity"]}
    (out / "TRAINING_READINESS.json").write_text(json.dumps(readiness, indent=2) + "\n")
    print(json.dumps({"status": readiness["status"], "drafts": len(drafted),
                      "unsigned_gate": "REJECTED_AS_REQUIRED", "training_enabled": False}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt", required=True, type=Path)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--resources", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(); prepare(a.receipt, a.corpus, a.resources, a.out)
