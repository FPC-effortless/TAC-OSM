"""ME062 genuine original FIT-only registered training; never opens CAL/DEV/TEST.

On every successful seed/arm fit, original checkpoint bytes and optimizer/RNG
states go to the Git tracked research/replay-bundles/ME062/ directory.
The GitHub workflow must commit these bytes BEFORE any heldout scoring.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path
import numpy as np
import torch

from scripts.me060_train_trajectory import (
    prepare, batch, schedules, digest, HISTORY, HORIZON, BATCH, UPDATES,
)
from scripts.me057_train_fused_anchor_robot import sha_file, write_json
from scripts.me062_prefit_original_fit import (
    SEEDS, CONFIG, FIT_SHA256, model_for, objective, profile_once,
)
from scripts.validate_research_parity import check_manifest

ROOT = Path("research/mathematical-engineering/me062")
OUT = Path("research/replay-bundles/ME062")
REGISTRATION = ROOT / "FAIR_COMPARISON.json"


from tac_osm.parameter_budget import budgeted_optimizer, checked_optimizer_step, save_training_checkpoint

def fit():
    torch.set_num_threads(1)
    if not (OUT / "UNTRAINED_STORAGE_PROBE.pt").exists():
        raise RuntimeError("Git-backed checkpoint byte storage proof missing: stop before first fit")
    plan = json.loads(REGISTRATION.read_text())
    if plan.get("study_type") != "architecture_comparison" or (
            check_manifest(plan, "preflight")["status"] != "PASS"):
        raise RuntimeError("registered matched research preflight is not certified")
    measured = json.loads((ROOT / "PREFLIGHT_MEASURED.json").read_text())
    if measured["real_source_fit_sha256"] != FIT_SHA256 or measured["optimizer_steps"] != 0:
        raise RuntimeError("original FIT-only prefit provenance changed")
    if measured["parameter_relative_gap"] >= .01 or measured["operator_flop_relative_gap"] >= .01:
        raise RuntimeError("compulsory matched resource preflight failed")
    split, metadata, identified, source, source_hash = prepare(
        Path(__import__("os").environ["ME062_PUBLISHER_CACHE"]))
    if source_hash != plan["real_world_source"]["sha256"]:
        raise RuntimeError("real FIT publisher episode hash mismatch")
    if digest(split["FIT"]) != measured["fit_partition_digest"]:
        raise RuntimeError("registered original FIT split changed")
    if digest(split["DEV"]) != measured["dev_partition_digest"]:
        raise RuntimeError("historically exposed DEV cohort metadata changed")
    if digest(split["CAL"]) != measured["cal_partition_digest"]:
        raise RuntimeError("CAL cohort metadata changed")

    ckpts = OUT / "checkpoints"
    ckpts.mkdir(parents=True, exist_ok=True)
    if any(ckpts.iterdir()):
        raise RuntimeError("refuse to retrain/overwrite any previous original checkpoint")
    records = []
    # The real trajectory sampler is fixed and identical for both models
    # within each seed, and hashes must match source-only prefit receipt.
    for seed in SEEDS:
        picks = schedules(source, seed)
        if digest(picks) != measured["schedule_sha256"][str(seed)]:
            raise RuntimeError(f"original training window order changed: {seed}")
        for name in CONFIG:
            row = next(r for r in plan["arms"] if r["architecture"] == name and r["seed"] == seed)
            if digest(picks) != row["training_sample_order_sha256"]:
                raise RuntimeError("per-arm original real FIT schedule changed")
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = model_for(name, identified)
            active = sum(p.numel() for p in model.parameters())
            if active != row["trainable_parameters"]:
                raise RuntimeError("actual model width/parameter budget changed")
            optimizer = budgeted_optimizer(model, torch.optim.AdamW, model.parameters(), lr=.001, weight_decay=.0001)
            loss_trace = []
            began = time.process_time()
            for step, pick in enumerate(picks, 1):
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model, batch(source, pick))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite real FIT loss {name}/{seed}/{step}")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(gradient):
                    raise RuntimeError(f"nonfinite real FIT gradient {name}/{seed}/{step}")
                checked_optimizer_step(optimizer)
                if step == 1 or step % 100 == 0:
                    loss_trace.append({"update": step, "loss": float(loss.detach())})
                    print(json.dumps({"event": "ME062_AUTHENTIC_FIT", "seed": seed,
                         "arm": name, **loss_trace[-1]}), flush=True)
            cpu = time.process_time() - began
            sampled = batch(source, picks[-1])
            flop_receipt = profile_once(model, sampled)
            total_flops = flop_receipt["profiled_forward_backward_flops"] * UPDATES
            path = ckpts / f"{name}-seed{seed}.pt"
            save_training_checkpoint({
                "training_complete": True,
                "experiment": plan["study_id"],
                "architecture": name, "seed": seed,
                "update_count": UPDATES,
                "training_source_sha256": source_hash,
                "fit_order_sha256": digest(picks),
                "identified": {key: value.tolist() for key, value in identified.items()},
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "loss_trace": loss_trace,
                "model_config": list(CONFIG[name]),
                "scoring_status": "NOT_EVALUATED_HOLDOUT_UNOPENED",
            }, path)
            measured_row = {
                "actual_steps": UPDATES, "actual_examples": UPDATES * BATCH,
                "actual_tokens": UPDATES * BATCH * (HISTORY + HORIZON),
                "training_flops": total_flops,
                "optimizer_cpu_seconds": cpu,
                "peak_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                "flops_profiler": "same torch.profiler original FIT supported-operator forward/backward profile on final real minibatch; uncounted elementwise and optimizer ops explicitly excluded",
                "checkpoint_sha256": sha_file(path),
                "actual_sample_order_sha256": digest(picks),
                "evaluation_split_sha256": row["test_split_sha256"],
            }
            row["measured"] = measured_row
            records.append({"seed": seed, "arm": name,
                "path": str(path), "bytes": path.stat().st_size,
                "checkpoint_sha256": measured_row["checkpoint_sha256"],
                "measured": measured_row})
            write_json(OUT / "CHECKPOINTS_FROZEN_PARTIAL.json", {
                "trained_model_count": len(records), "expected_count": len(SEEDS)*len(CONFIG),
                "holdout_trajectory_files_opened": 0,
                "training_only": True, "records": records,
            })
            print(json.dumps({"event": "ME062_BYTES_ON_DISK_NOT_YET_GIT",
                              "seed": seed, "arm": name, "bytes": path.stat().st_size,
                              "sha256": measured_row["checkpoint_sha256"]}), flush=True)

    final = check_manifest(plan, "final")
    if final["status"] != "PASS":
        raise RuntimeError("postfit registered fairness parity failed; preserve failed model bytes")
    write_json(OUT / "FAIR_COMPARISON_FINAL.json", plan)
    write_json(OUT / "TRAINING_RESULT.json", {
        "study": plan["study_id"],
        "status": "ALL_FITS_PERFORMED_NOT_YET_REPOSITORY_PRESERVED",
        "checkpoints": records,
        "frozen_before_cal_dev_test": True,
        "original_fit_only": True,
        "fit_episodes": len(source),
        "source_sha256": source_hash,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "full_training_flops_certified": False,
        "independent_test": False,
        "source_test_trajectories_opened": 0,
        "cal_trajectories_opened": 0,
        "dev_trajectories_opened": 0,
        "transfer_trajectories_opened": 0,
        "comparison_validation": final,
        "model_replay": "PENDING_INDEPENDENT_CLEAN_CHECKOUT",
        "actual_forecast_mae": None,
    })
    print("ME062_REAL_FIT_FINISHED_HOLDOUT_STILL_UNOPENED_GIT_PRESERVATION_PENDING",
          flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--fit", action="store_true")
    args = cli.parse_args()
    if not args.fit:
        raise SystemExit("--fit is required for a registered authentic FIT-only training run")
    fit()
