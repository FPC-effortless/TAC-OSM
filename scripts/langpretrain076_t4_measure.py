#!/usr/bin/env python3
"""LANGPRETRAIN076: bounded genuine-publisher FIT GPU training-step timing.

Not a Kaggle dispatcher. Run only in reviewed, protected dual-T4 seed-shard
harness with original repository source available offline and approved quota.
No TEST, DEV, downloads, unreviewed datasets or architectural superiority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = "LANGPRETRAIN076"
SOURCE = "60403a5409ff2c3f3b07dd2ca91a7a3e096839c7"
SOURCE_SHA = "dd5aabf90646be409e3f18ecf763dbcefc84b6575e12ff54f4a7195f92ff150e"
CONTEXT_SHA = "c88f3f7c1b628cb733a4a58dcd168a75411d8bb3d80e382068593713e887510a"
UPPER_GIT_BLOB = "b882f7c447d560b0156924e86633867babfc50d7"
SEEDS = (1701, 1702)
WARMUP_STEPS = 8
TIMED_STEPS = 32
BATCH = 8
SEQUENCE_BYTES = 128
TRAINING_BYTES_PER_SEED = (WARMUP_STEPS + TIMED_STEPS) * BATCH * SEQUENCE_BYTES
TIMED_BYTES_PER_SEED = TIMED_STEPS * BATCH * SEQUENCE_BYTES
PARAMETERS = 499623
MAX_WORKER_SECONDS = 600


class ThroughputGateError(ValueError):
    pass


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def preflight_source(root: Path) -> None:
    base = root / "research/mathematical-engineering"
    if digest_file(base / "context026/context_transition.py") != CONTEXT_SHA:
        raise ThroughputGateError("actual original CONTEXT026 model SHA mismatch")
    if git_blob(base / "langscale074/upper_reallocation.py") != UPPER_GIT_BLOB:
        raise ThroughputGateError("actual original h48 model blob mismatch")
    if not (root / "research/source-bundles/LANGUAGE-SOURCE023").is_dir():
        raise ThroughputGateError("original source-bundles missing")


def timed_rate(tokens: int, elapsed_seconds: float) -> float:
    if type(tokens) is not int or tokens <= 0:
        raise ThroughputGateError("positive original training byte tokens required")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ThroughputGateError("positive finite synchronized wall time required")
    return tokens / elapsed_seconds


def run(seed: int, device: str, output: Path) -> dict:
    if seed not in SEEDS or device != "cuda:0":
        raise ThroughputGateError("frozen two-seed / masked T4 worker device required")
    slot = os.environ.get("TAC_OSM_PHYSICAL_T4_SLOT")
    if slot not in ("0", "1") or int(slot) != SEEDS.index(seed):
        raise ThroughputGateError("physical T4 seed shard mapping missing or wrong")
    if output.exists() or output.with_suffix(".pt").exists():
        raise ThroughputGateError("refuse to overwrite original throughput/checkpoint evidence")
    import numpy as np
    import torch
    from torch.nn import functional as F

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ThroughputGateError("one masked worker-local CUDA device required")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name.upper():
        raise ThroughputGateError("actual NVIDIA T4 required; no fallback")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "research/mathematical-engineering/langscale074"))
    from upper_reallocation import ReallocatedUpperTAC074
    from scripts.langpretrain076_fit_only import fit_sources
    from scripts.langlayer035_sampling import iter_batches, schedule_receipt
    preflight_source(ROOT)
    fit = fit_sources(ROOT)  # FIT bytes only; never load DEV or TEST source bytes
    expected = schedule_receipt(fit, seed, WARMUP_STEPS + TIMED_STEPS)
    if expected["next_byte_targets_seen"] != TRAINING_BYTES_PER_SEED:
        raise ThroughputGateError("original source exposure/sampler altered")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ReallocatedUpperTAC074(tier="h48").to(device).train()
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    deployed = sum(p.numel() for p in model.parameters())
    if active != PARAMETERS or deployed != PARAMETERS:
        raise ThroughputGateError("500K h48 live parameter scale altered")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    sample_digest = hashlib.sha256()
    timed_start = None
    full_start = time.perf_counter()
    last_input = None
    losses = []
    torch.cuda.reset_peak_memory_stats(0)
    for step, source_rows in iter_batches(fit, seed, steps=WARMUP_STEPS + TIMED_STEPS):
        if step == WARMUP_STEPS:
            torch.cuda.synchronize(0)
            timed_start = time.perf_counter()
        rows = []
        for owner,offset,raw in source_rows:
            if raw != fit[owner][offset:offset + 129]:
                raise ThroughputGateError("FIT source owner/next-byte contamination")
            sample_digest.update(f"{owner}:{offset};".encode())
            rows.append(list(raw))
        batch = torch.tensor(rows, dtype=torch.long, device=device)
        last_input = batch[:,:-1]
        logits = model(last_input)
        if logits.shape != (BATCH, SEQUENCE_BYTES, 256):
            raise ThroughputGateError("original 256-byte logits geometry changed")
        loss = F.cross_entropy(logits.float().reshape(-1,256), batch[:,1:].reshape(-1))
        if not bool(torch.isfinite(loss).item()):
            raise ThroughputGateError("nonfinite genuine next-byte loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0 and any(p.grad is None or not bool(torch.isfinite(p.grad).all())
                             for p in model.parameters() if p.requires_grad):
            raise ThroughputGateError("missing/nonfinite training gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in (0, WARMUP_STEPS - 1, WARMUP_STEPS + TIMED_STEPS - 1):
            losses.append((step + 1, float(loss.detach().item())))
        if time.perf_counter() - full_start > MAX_WORKER_SECONDS:
            raise ThroughputGateError("bounded T4 worker wall budget exceeded")
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - timed_start
    total_seconds = time.perf_counter() - full_start
    if sample_digest.hexdigest() != expected["sample_order_sha256"]:
        raise ThroughputGateError("frozen source sampling schedule mismatch")
    peak_memory = torch.cuda.max_memory_allocated(0)
    checkpoint = output.with_suffix(".pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "study": STUDY, "seed": seed, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(0),
        "rng_python": random.getstate(),
        "rng_numpy": {"name":np.random.get_state()[0],
                      "keys":np.random.get_state()[1].tolist(),
                      "pos":int(np.random.get_state()[2])},
        "source_sha256": SOURCE_SHA, "sampler_sha256": sample_digest.hexdigest(),
        "trainable_parameters": active, "steps": WARMUP_STEPS + TIMED_STEPS,
    }, checkpoint)
    checkpoint_sha = digest_file(checkpoint)
    snapshot = torch.load(checkpoint, map_location=device, weights_only=True)
    replica = ReallocatedUpperTAC074(tier="h48").to(device).eval()
    replica.load_state_dict(snapshot["model"], strict=True)
    model.eval()
    with torch.no_grad():
        original = model(last_input)
        replay = replica(last_input)
        torch.testing.assert_close(replay, original, atol=1e-6, rtol=1e-6)
    result = {
        "schema": "tac-osm.langpretrain076.t4-throughput.v1",
        "study": STUDY, "execution_level": "internal_research",
        "evidence": "ORIGINAL_PUBLISHER_FIT_GPU_TRAIN_STEPS_NOT_QUALITY_OR_TEST",
        "seed": seed, "physical_slot": int(slot), "device_name": name,
        "trainable_parameters": active, "deployed_parameters": deployed,
        "model_tier": "h48", "optimizer": "AdamW lr=.001 FP32 grad_clip1",
        "batch": BATCH, "context_bytes": SEQUENCE_BYTES,
        "warmup_updates": WARMUP_STEPS, "measured_updates": TIMED_STEPS,
        "training_target_bytes": TRAINING_BYTES_PER_SEED,
        "measured_target_bytes": TIMED_BYTES_PER_SEED,
        "measured_synced_wall_seconds": elapsed,
        "measured_byte_tokens_per_second": timed_rate(TIMED_BYTES_PER_SEED, elapsed),
        "full_run_wall_seconds": total_seconds,
        "peak_cuda_tensor_allocated_bytes": int(peak_memory),
        "source_sha256": SOURCE_SHA,
        "original_publisher_commit": SOURCE,
        "sample_order_sha256": sample_digest.hexdigest(),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_cold_reload_logit_parity": "PASS_ATOL1E-6_RTOL1E-6_FIT_ONLY",
        "sample_losses_fully_exposed_FIT": losses,
        "scientific_qualification": "NOT_INDEPENDENT",
        "protected_TEST": "NOT_ACCESSED", "DEV": "NOT_ACCESSED",
        "paid_cloud_spend_authorized_usd": 0,
        "kaggle_free_quota_assumed_not_verified": True,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--device", required=True, choices=["cuda:0"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps({"status":"T4_GPU_TRAINING_MICROBENCHMARK_REQUESTED",
                          "seed":args.seed,"budget_seconds":MAX_WORKER_SECONDS}))
        result = run(args.seed, args.device, args.output)
    except (ThroughputGateError, OSError, RuntimeError, ImportError, AssertionError) as exc:
        parser.exit(2, f"BLOCKED_OR_FAILED_NO_RESULT: {type(exc).__name__}: {exc}\n")
    print(json.dumps({"status":"ACTUAL_T4_TIMING", "seed":args.seed,
                      "byte_tokens_per_second":result["measured_byte_tokens_per_second"]}))


if __name__ == "__main__":
    main()
