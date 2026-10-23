"""Independent clean-checkout verification of original ME062 trained Git bytes.

Does not download heldout data, refit, infer real MAE, or certify complete FLOPs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from scripts.me062_prefit_original_fit import model_for, SEEDS, CONFIG
from scripts.validate_research_parity import check_manifest

ROOT = Path("research/replay-bundles/ME062")
RESEARCH = Path("research/mathematical-engineering/me062")


def sha(path):
    stream = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            stream.update(block)
    return stream.hexdigest()


def verify():
    plan = json.loads((ROOT / "FAIR_COMPARISON_FINAL.json").read_text())
    status = check_manifest(plan, "final")
    if status["status"] != "PASS":
        raise RuntimeError("postfit numerical parity mismatch")
    report = json.loads((ROOT / "TRAINING_RESULT.json").read_text())
    paths = report["checkpoints"]
    required = {(seed, architecture) for seed in SEEDS for architecture in CONFIG}
    seen = {(r["seed"], r["arm"]) for r in paths}
    if seen != required or len(paths) != len(required):
        raise RuntimeError("required original checkpoint inventory incomplete")
    verified = []
    for r in paths:
        path = Path(r["path"])
        if not path.is_relative_to(ROOT) or path.is_symlink():
            raise RuntimeError("invalid checkpoint path")
        if not path.exists() or sha(path) != r["checkpoint_sha256"]:
            raise RuntimeError("missing or mutated checkpoint original bytes")
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        if not checkpoint["training_complete"] or checkpoint["update_count"] != 600:
            raise RuntimeError("checkpoint is not a completed FIT-only model")
        if checkpoint["scoring_status"] != "NOT_EVALUATED_HOLDOUT_UNOPENED":
            raise RuntimeError("unfrozen source observation boundary")
        if checkpoint["seed"] != r["seed"] or checkpoint["architecture"] != r["arm"]:
            raise RuntimeError("wrong seed or architecture")
        if not isinstance(checkpoint["optimizer"], dict) or (
                "state" not in checkpoint["optimizer"]):
            raise RuntimeError("missing optimizer state")
        if checkpoint.get("torch_rng_state") is None or checkpoint.get("numpy_rng_state") is None:
            raise RuntimeError("missing training RNG receipts")
        identified = {k: np.asarray(v, dtype=np.float32)
                      for k, v in checkpoint["identified"].items()}
        model = model_for(r["arm"], identified)
        model.load_state_dict(checkpoint["model"], strict=True)
        if sum(p.numel() for p in model.parameters()) != (
                next(row["trainable_parameters"] for row in plan["arms"]
                    if row["architecture"] == r["arm"] and row["seed"] == r["seed"])):
            raise RuntimeError("incorrect restored parameter count")
        verified.append({"seed": r["seed"], "arm": r["arm"], "sha256": sha(path),
                         "bytes": path.stat().st_size,
                         "reloaded": True})
    receipt = {
        "study": "ME062_SELECTIVE_RELIABILITY_REAL_FIT_DEVELOPMENT",
        "execution": "separate clean GitHub Actions job, exact Git branch checkout",
        "checkpoint_byte_hash_and_torch_restore": "PASS",
        "checkpoints_restored": len(verified),
        "scoring_replay": "NOT_RUN",
        "true_full_hardware_flops": "NOT_CERTIFIED",
        "heldout_evaluation": "NOT_RUN",
        "training_data_observations_retrieved_by_verifier": 0,
        "original_checkpoints": verified,
        "model_replay": "PASS",
        "full_evaluation_replay": "BLOCKED",
        "scientific_superiority": "UNKNOWN",
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    path = ROOT / "CLEAN_CHECKOUT_MODEL_REPLAY.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print("ME062_INDEPENDENT_GIT_CHECKPOINT_LOAD_PASS", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if not args.check: raise SystemExit("pass --check")
    torch.set_num_threads(1)
    verify()
