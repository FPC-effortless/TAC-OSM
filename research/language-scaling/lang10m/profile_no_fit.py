"""Bounded no-fit LANG10M001 throughput/memory profile on two verified T4s.

Engineering test on synthetic token IDs ONLY. No optimizer, fitting, DEV or
TEST, and no claim of language intelligence. Must explicitly launch on
already-authorized Kaggle quota; this file never submits a job.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def physical_t4s():
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    lines = subprocess.check_output(command, text=True, timeout=10).strip().splitlines()
    rows = [[part.strip() for part in line.split(",")] for line in lines]
    if len(rows) != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError("two physical CUDA devices required; no fallback")
    if len({r[0] for r in rows}) != 2 or any("T4" not in r[1] for r in rows):
        raise RuntimeError("expected two distinct physical NVIDIA Tesla T4 GPUs")
    if any(len(r) < 3 or int(r[2]) < 14000 for r in rows):
        raise RuntimeError("GPU memory not physically consistent with dual T4")
    return rows


def load_model():
    spec = importlib.util.spec_from_file_location("lang10m_profile_model", HERE / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TACLanguage10M001


def step(device, batch, context, model_type):
    torch.manual_seed(1234)
    torch.cuda.set_device(device)
    model = model_type().to(device).train()
    if sum(p.numel() for p in model.parameters()) != 10_001_188:
        raise RuntimeError("parameter mismatch")
    x = torch.randint(0, 4096, (batch, context), device=device)
    y = torch.randint(0, 4096, (batch, context), device=device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, 4096), y.reshape(-1))
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if not torch.isfinite(loss):
        raise ValueError("nonfinite no-fit loss")
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("missing or nonfinite no-fit gradients")
    output = {
        "elapsed_seconds": elapsed,
        "tokens_per_second_forward_backward": batch * context / elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "loss_on_random_fixture_not_capability": float(loss.detach().cpu()),
    }
    del model, x, y, logits, loss
    torch.cuda.empty_cache()
    return output


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--batch", type=int, default=2)
    cli.add_argument("--context", type=int, default=128)
    args = cli.parse_args()
    if args.batch < 1 or args.batch > 8 or args.context not in (128, 256, 512, 1024):
        raise ValueError("unregistered or unsafe prefit profile shape")
    rows = physical_t4s()
    model_type = load_model()
    receipt = {
        "study": "LANG10M001",
        "status": "STARTED",
        "evidence_level": "constructed_no_fit",
        "training_optimizer_steps": 0,
        "gpu_allocations_started_by_script": 0,
        "physical_t4s": rows,
        "batch": args.batch,
        "context": args.context,
        "torch_version": torch.__version__,
        "model_sha256": hashlib.sha256((HERE / "model.py").read_bytes()).hexdigest(),
        "results": [],
    }
    failed = False
    for index in range(2):
        device = torch.device(f"cuda:{index}")
        try:
            value = step(device, args.batch, args.context, model_type)
            receipt["results"].append({"device": index, "status": "PASS", **value})
        except (RuntimeError, ValueError) as exc:
            failed = True
            receipt["results"].append({
                "device": index, "status": "FAIL",
                "exception": type(exc).__name__, "message": str(exc)[:512],
            })
    receipt["status"] = "FAIL" if failed else "PASS_CONSTRUCTED_PREFIT"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
