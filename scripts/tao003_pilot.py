#!/usr/bin/env python3
"""TAOPILOT003: real publisher-byte five-arm fixed-LR optimizer screen.

Train all five arms from identical initial weights and *identical* original
publisher source sequence. Each original model/optimizer/RNG checkpoint must
be committed in Git before the separate clean-checkout DEV scorer is run.

Exploratory internal research only; no architecture/optimizer superiority claim.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT), str(ROOT / "src"),
    str(ROOT / "research/mathematical-engineering/langmem086"),
    str(ROOT / "research/mathematical-engineering/langbase024"),
]
import numpy as np
import torch
from torch.nn import functional as F
from compact_tac import CompactDualClippedTAC086
from scripts import langmem086_run as original
from scripts.tao002_prefit import ARMS, ROUTE, SOURCE_SHA, MODEL_SHA, SIZE
from tac_osm.optimizer_tao import build_tao
from scripts.validate_research_parity import check_manifest
from byte_next_reference import read_sha_source

REG = ROOT / "research/mathematical-engineering/taopilot003"
PREFIT = ROOT / "research/mathematical-engineering/tao002/ORIGINAL_PREFIT_RECEIPT.json"
SEED = 1701
STEPS = 32
BATCH = 4
LENGTH = 256
LR = 0.001
WD = 0.01
TARGETS = STEPS * BATCH * LENGTH

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def model_hash(model) -> str:
    return original.weight_hash(model)

def frozen():
    cfg = json.loads((REG / "FAIR_COMPARISON.json").read_text())
    before = check_manifest(cfg, "preflight")
    if before["status"] != "PASS_PREFIT_SINGLE_MODEL":
        raise ValueError("registered single-model pilot preflight did not pass")
    if cfg["budget"]["steps"] != STEPS or cfg["budget"]["targets_per_arm"] != TARGETS:
        raise ValueError("pilot updates/tokens registration differs")
    if {(r["name"], r["seed"]) for r in cfg["models"]} != {(arm, SEED) for arm in ARMS}:
        raise ValueError("pilot incomplete original arms")
    if any(r["trainable_parameters"] != SIZE or r["deployed_parameters"] != SIZE for r in cfg["models"]):
        raise ValueError("registered 100521p size does not match")
    if digest((ROOT / "research/mathematical-engineering/langmem086/compact_tac.py").read_bytes()) != MODEL_SHA:
        raise ValueError("original model source changed")
    if cfg["real_world_source"]["sha256"] != SOURCE_SHA:
        raise ValueError("FIT source registration changed")
    receipt = json.loads(PREFIT.read_text())
    if receipt["fit_performed"] is not False or receipt["optimizer_steps"] != 0:
        raise ValueError("original source preflight not no-fit")
    entry = next(r for r in receipt["samples"]["frozen_sample_schedules"] if r["seed"] == SEED)
    if entry["steps"] != 160 or entry["targets"] != 163840:
        raise ValueError("original frozen schedule changed")
    return cfg, entry

def optimizer_for(arm, model):
    if arm == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, foreach=False)
    return build_tao(model, orthogonal_names=ROUTE, mode=arm, lr=LR, weight_decay=WD)

def original_schedule(docs, frozen_entry):
    if original.schedule(docs, SEED) != frozen_entry:
        raise ValueError("original full 160-step publisher source schedule differs")
    rows = list(itertools.islice(original.batches(docs, SEED), STEPS))
    if len(rows) != STEPS:
        raise ValueError("incomplete original publisher schedule")
    order = hashlib.sha256()
    data = hashlib.sha256()
    for _, row in rows:
        for owner, start, raw in row:
            order.update(f"{owner}:{start};".encode())
            data.update(raw)
    return rows, {"steps": STEPS, "batch": BATCH, "context": LENGTH, "targets": TARGETS,
                  "sample_order_sha256": order.hexdigest(), "sample_bytes_sha256": data.hexdigest()}

def _git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

def train(out: Path):
    if out.exists():
        raise FileExistsError("original trained checkpoints already exist; no overwrite")
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only pilot; unexpected accelerator")
    torch.set_num_threads(2)
    cfg, entry = frozen()
    docs, source = original.fit_sources086()
    if source != SOURCE_SHA or len(docs) != 14:
        raise ValueError("original authenticated FIT corpus differs")
    batches, schedule = original_schedule(docs, entry)
    all_initial = set()
    full = {}
    started = time.process_time()
    # Critically: no DEV source is loaded in train() at any point.
    for arm in ARMS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        model = CompactDualClippedTAC086()
        if model.counts()["trainable"] != SIZE or model.counts()["deployed"] != SIZE:
            raise ValueError("actual live model size changed")
        optimizer = optimizer_for(arm, model)
        initial = model_hash(model)
        all_initial.add(initial)
        fixture = torch.tensor([list(next(iter(docs.values()))[:16])], dtype=torch.long)
        future = fixture.clone()
        future[:, 8:] = (future[:, 8:] + 113) % 256
        with torch.no_grad():
            clean, _ = model(fixture, owners=("FIT-poison",))
            changed, _ = model(future, owners=("FIT-poison",))
        if not torch.equal(clean[:, :8], changed[:, :8]):
            raise ValueError("actual FIT source future-suffix poison failed")
        trace, opt_time = [], 0.0
        counts = torch.zeros(256, dtype=torch.int64)
        wall = time.perf_counter()
        for step, row in batches:
            x = torch.tensor([list(raw[:-1]) for _, _, raw in row], dtype=torch.long)
            y = torch.tensor([list(raw[1:]) for _, _, raw in row], dtype=torch.long)
            counts += torch.bincount(y.reshape(-1), minlength=256)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x, owners=tuple(f"FIT-row-{i}" for i in range(BATCH)))
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            loss.backward()
            if not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError(f"nonfinite/incomplete actual model gradient: {arm}/{step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            t0 = time.perf_counter()
            optimizer.step()
            opt_time += time.perf_counter() - t0
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise ValueError(f"nonfinite model weight after update: {arm}/{step}")
            trace.append(float(loss.detach()))
            if step % 8 == 0:
                print(json.dumps({"phase": "FIT", "arm": arm, "step": step,
                                  "fit_nll": trace[-1]}), flush=True)
        cpu = time.perf_counter() - wall
        if len(trace) != STEPS or int(counts.sum()) != TARGETS or model_hash(model) == initial:
            raise ValueError("fit did not execute registered source update count")
        for state in optimizer.state.values():
            t = state.get("step", None)
            if t is None or int(t) != STEPS:
                raise ValueError("optimizer step-counter mismatch")
        with torch.no_grad():
            logits, _ = model(fixture, owners=("cold-check",))
        arm_out = out / arm
        arm_out.mkdir(parents=True, exist_ok=False)
        rng_np = np.random.get_state()
        record = {
            "study": "TAOPILOT003", "seed": SEED, "arm": arm,
            "steps": STEPS, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": (rng_np[0], rng_np[1].tolist(), rng_np[2], rng_np[3], rng_np[4]),
            "python_rng": random.getstate(),
            "source_hash": source, "original_model_source_hash": MODEL_SHA,
            "sample_receipt": schedule, "initial_weight_hash": initial,
            "final_weight_hash": model_hash(model), "fit_trace": trace,
            "fit_byte_counts": counts.tolist(),
            "cold_fixture": fixture, "cold_logits": logits,
        }
        path = arm_out / "MODEL.pt"
        torch.save(record, path)
        sha = digest(path.read_bytes())
        # Immediate clean model/optimizer cold-reload before any DEV access.
        cp = torch.load(path, map_location="cpu", weights_only=True)
        cold = CompactDualClippedTAC086()
        cold.load_state_dict(cp["model"], strict=True)
        co = optimizer_for(arm, cold)
        co.load_state_dict(cp["optimizer"])
        if any(int(x["step"]) != STEPS for x in co.state.values()):
            raise ValueError("cold optimizer state does not restore")
        with torch.no_grad():
            fresh, _ = cold(cp["cold_fixture"], owners=("cold-check",))
        if not torch.equal(fresh, cp["cold_logits"]) or model_hash(cold) != cp["final_weight_hash"]:
            raise ValueError("cold model/logits replay differs")
        receipt = {
            "study": "TAOPILOT003", "status": "FIT_SEALED_NOT_YET_INDEPENDENTLY_REPLAYED",
            "seed": SEED, "arm": arm, "trained_steps": STEPS, "targets_seen": TARGETS,
            "original_fit_sha256": source, "source_schedule": schedule,
            "model_source_sha256": MODEL_SHA,
            "trainable_parameters": SIZE, "deployed_parameters": SIZE,
            "orthogonal_parameter_names": list(ROUTE) if arm != "adamw" else [],
            "initial_weight_hash": initial,
            "trained_model_hash": cp["final_weight_hash"], "checkpoint_sha256": sha,
            "cold_logits_bitwise": True, "cold_optimizer_restored": True,
            "cpu_fit_wall_seconds": cpu, "optimizer_wall_seconds": opt_time,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "fit_nll_first": trace[0], "fit_nll_last": trace[-1],
            "fit_trace": trace, "actual_fit_future_poison": "PASS",
            "dev_read_before_seal": False, "protected_TEST_accessed": False, "gpu_minutes": 0,
            "research_tier": "single-seed early learning pilot, zero per-arm LR trials",
            "runner_git_commit": _git_sha(),
        }
        (arm_out / "FIT_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        full[arm] = {"checkpoint_sha256": sha, "trained_model_hash": cp["final_weight_hash"],
                     "cpu_wall_seconds": cpu, "optimizer_wall_seconds": opt_time,
                     "fit_last_nll": trace[-1]}
        print(json.dumps({"phase": "SEALED", "arm": arm, "checkpoint_sha256": sha}), flush=True)
    if len(all_initial) != 1:
        raise ValueError("initial weights differed between optimizer arms")
    (out / "FIT_MATRIX.json").write_text(json.dumps({
        "study": "TAOPILOT003", "status": "SEED1701_ALL_FIVE_ORIGINAL_CHECKPOINTS_SEALED",
        "seed": SEED, "initial_model_weight_hash": next(iter(all_initial)),
        "registered_original_source_sha256": source, "frozen_sample_receipt": schedule,
        "fit_start_to_all_checkpoints_wall_seconds": time.process_time() - started,
        "arms": full, "dev_accessed": False, "not_independently_qualified": True,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"phase": "ALL_FIVE_SEALED", "arms": list(full)}), flush=True)

@torch.no_grad()
def independent_numeric_replay(current, frozen):
    """Existing repo cross-host logit tolerance; never forgive altered weights."""
    if current.shape != frozen.shape or not bool(torch.isfinite(current).all()) or not bool(torch.isfinite(frozen).all()):
        raise ValueError("invalid independent logit shape/finite status")
    delta = (current - frozen).abs()
    report = {"bitwise": bool(torch.equal(current, frozen)),
              "max_abs_difference": float(delta.max()),
              "different_entries": int((current != frozen).sum()),
              "same_top1": bool(torch.equal(current.argmax(-1), frozen.argmax(-1))),
              "numerically_equivalent": bool(torch.allclose(current, frozen, atol=3e-5, rtol=1e-4)),
              "numeric_atol": 3e-5, "numeric_rtol": 1e-4}
    if not report["same_top1"] or not report["numerically_equivalent"]:
        raise ValueError("independent checkpoint logits exceed pre-existing cross-host tolerance")
    return report

@torch.no_grad()
def evaluate(directory: Path):
    """Separate checked-out runner: validate ALL Git-owned originals before DEV."""
    torch.set_num_threads(2)
    cfg, _ = frozen()
    original_manifest = json.loads((directory / "FIT_MATRIX.json").read_text())
    if set(original_manifest["arms"]) != set(ARMS):
        raise ValueError("full original five-arm checkpoint set absent")
    previous_diag = json.loads((directory / "INDEPENDENT_REPLAY_DIAG.json").read_text())
    if previous_diag.get("phase") != "NO_DEV_CROSS_HOST_REPLAY_DIAGNOSTIC" or previous_diag.get("dev_read") is not False:
        raise ValueError("prior independent replay diagnosis missing/untrusted")
    models = {}
    independent_replay = {}
    for arm in ARMS:
        root = directory / arm
        receipt = json.loads((root / "FIT_RECEIPT.json").read_text())
        path = root / "MODEL.pt"
        if digest(path.read_bytes()) != receipt["checkpoint_sha256"]:
            raise ValueError("Git-backed checkpoint SHA mismatched")
        cp = torch.load(path, map_location="cpu", weights_only=True)
        if cp["source_hash"] != SOURCE_SHA or cp["original_model_source_hash"] != MODEL_SHA:
            raise ValueError("trained checkpoint original-source identity drift")
        if cp["arm"] != arm or cp["seed"] != SEED or cp["steps"] != STEPS:
            raise ValueError("wrong trained model arm/seed/update count")
        if cp["sample_receipt"] != original_manifest["frozen_sample_receipt"]:
            raise ValueError("source sample parity differs between arms")
        model = CompactDualClippedTAC086().eval()
        model.load_state_dict(cp["model"], strict=True)
        optimizer = optimizer_for(arm, model)
        optimizer.load_state_dict(cp["optimizer"])
        if any(int(x["step"]) != STEPS for x in optimizer.state.values()):
            raise ValueError("independent optimizer step replay invalid")
        logits, _ = model(cp["cold_fixture"], owners=("cold-check",))
        if model_hash(model) != cp["final_weight_hash"]:
            raise ValueError("independent clean-checkout exact trained weight hash differs")
        if previous_diag["results"][arm]["exact_weight_sha"] is not True or previous_diag["results"][arm]["optimizer_steps"] != [STEPS]:
            raise ValueError("prior independent exact model/optimizer diagnostic failed")
        independent_replay[arm] = independent_numeric_replay(logits, cp["cold_logits"])
        models[arm] = model
    # Read registered *already exposed* DEV original source only after all
    # checkpoints have been Git-committed and validated in this new checkout.
    source_receipt = json.loads((ROOT / "research/mathematical-engineering/language_source023/SOURCE_RECEIPT.json").read_text())
    original_dev = [e for e in source_receipt["files"] if e["role"] == "DEV"]
    if len(original_dev) != 4:
        raise ValueError("expected exactly original four DEV publisher files")
    docs = {r["path"]: read_sha_source(ROOT, r["path"], r) for r in original_dev}
    target_hash = hashlib.sha256()
    for name, raw in sorted(docs.items()):
        if len(raw) < 257:
            raise ValueError("DEV source too short")
        target_hash.update(name.encode() + b":" + raw[1:257])
    metrics = {}
    torch.set_num_threads(2)
    for arm, model in models.items():
        nll, top1, brier, n = 0.0, 0, 0.0, 0
        for name, raw in sorted(docs.items()):
            tokens = torch.tensor([list(raw[:256])], dtype=torch.long)
            actual = torch.tensor([list(raw[1:257])], dtype=torch.long)
            logits, _ = model(tokens, owners=(name,))
            logp = logits.log_softmax(-1)
            probs = logp.exp()
            py = probs.gather(-1, actual.unsqueeze(-1)).squeeze(-1)
            nll += float((-logp.gather(-1, actual.unsqueeze(-1)).squeeze(-1)).sum())
            top1 += int((logits.argmax(-1) == actual).sum())
            brier += float((probs.square().sum(-1) - 2 * py + 1).sum())
            n += actual.numel()
        if n != 1024:
            raise ValueError("registered DEV target count differs")
        metrics[arm] = {"dev_nll": nll / n, "dev_top1_accuracy": top1 / n,
                        "dev_multiclass_brier": brier / n, "num_real_dev_targets": n,
                        "checkpoint_sha256": original_manifest["arms"][arm]["checkpoint_sha256"],
                        "train_optimizer_only_wall_seconds": original_manifest["arms"][arm]["optimizer_wall_seconds"],
                        "train_full_loop_wall_seconds": original_manifest["arms"][arm]["cpu_wall_seconds"]}
    result = {
        "study": "TAOPILOT003", "tier": "INTERNAL_EXPOSED_DEV_SINGLE_SEED_FIXED_LR",
        "qualification": "NO_ARCHITECTURE_OR_OPTIMIZER_SUPERIORITY_CLAIM",
        "seed": SEED, "train_updates": STEPS, "train_tokens_per_arm": TARGETS,
        "same_source_sample_receipt": original_manifest["frozen_sample_receipt"],
        "dev_target_sha256": target_hash.hexdigest(), "dev_owners": sorted(docs),
        "dev_targets_per_arm": 1024, "metrics": metrics,
        "independent_git_checkout_exact_weight_and_optimizer_replay": True,
        "independent_git_checkout_numeric_logits_replay": independent_replay,
        "all_arms_bitwise_logits": all(x["bitwise"] for x in independent_replay.values()),
        "historical_independent_bitwise_replay_failure_preserved": True,
        "numeric_replay_precision_gate": "EXISTING_3E-5_ABS_1E-4_REL_NOT_BITWISE_PROOF",
        "protected_TEST": "NOT_ACCESSED", "gpu_minutes": 0,
        "remaining": ["Only one seed", "Only one LR per optimizer",
                      "Previously exposed source DEV", "No paired seed uncertainty",
                      "Per-optimizer FLOPs are not matched", "No full PNDS formalization"],
    }
    output = directory / "DEV_RESULTS.json"
    if output.exists():
        raise FileExistsError("first read-only DEV results immutable")
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    # Final measured checkpoint identifiers are attached only after all five
    # actual Git-backed source model/replay identities have been verified.
    final_manifest = json.loads((REG / "FAIR_COMPARISON.json").read_text())
    for row in final_manifest["models"]:
        sha = original_manifest["arms"][row["name"]]["checkpoint_sha256"]
        row["measured"] = {"trainable_parameters": SIZE, "deployed_parameters": SIZE,
                           "checkpoint_sha256": sha}
    final_gate = check_manifest(final_manifest, "final")
    if final_gate["status"] != "PASS_SINGLE_MODEL":
        raise ValueError("postfit measured parameter/checkpoint gate failed")
    (directory / "FINAL_COMPARISON.json").write_text(json.dumps(final_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"phase": "DEV_SCORED", "metrics": metrics,
                      "target_sha256": result["dev_target_sha256"]}), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("--directory", required=True, type=Path)
    a = parser.parse_args()
    if a.phase == "train":
        train(a.directory)
    else:
        evaluate(a.directory)

if __name__ == "__main__":
    main()
