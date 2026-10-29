#!/usr/bin/env python3
"""LANGPROFILE058: frozen checkpoint, no-fit decoder ATen operator CPU audit.

This profiles unchanged LANGSTREAM043 _advance and LANGDECODE057.step, not
new neural architecture. Profiler alters runtimes; report its *self-CPU*
operator accounting separately from uninstrumented wall clocks. Python
overhead cannot be inferred by subtraction. Genuine FIT-only source
prefixes and original checkpoint require strict provenance checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"),
                str(ROOT / "research/mathematical-engineering/langstream043")]
from incremental import IncrementalTAC043, SHAPES
from scripts.langdecode057_validation import BatchedValidationTAC057
from scripts.langdecode057_benchmark import exact_state, parity
from scripts.langrecomp052_replay import load_original
from scripts.langlayer035_sampling import fit_sources

REPETITIONS = 16
WARMUPS = 2
TRIALS = 7
PROFILE_REPETITIONS = 12
STAGES = ("step_with_guards", "transition_only", "validate_only")


def operator_group(name: str) -> str:
    """Mutually exclusive categories of *self* ATen CPU time (not stages)."""
    if not name.startswith("aten::"):
        return "not_aten"
    op = name[6:]
    if "linalg_solve" in op or "cholesky" in op or op in ("_lu_solve", "linalg_lu_factor_ex"):
        return "small_matrix_factorization_and_solve"
    if op in ("mm", "bmm", "addmm", "matmul", "linear", "einsum", "mv", "addmv"):
        return "matrix_projections_and_products"
    if op in ("cat", "stack", "empty", "empty_strided", "empty_like", "clone",
              "copy_", "contiguous", "new_empty", "new_zeros", "zeros", "ones"):
        return "tensor_allocation_and_materialization"
    if ("softmax" in op or "layer_norm" in op or op in
            ("tanh", "sigmoid", "silu", "linalg_vector_norm", "norm")):
        return "nonlinearities_and_normalization"
    return "other_aten_ops"


def summarize_events(events: list, top_n: int = 30) -> dict:
    if not events or top_n < 1:
        raise ValueError("missing profiler events")
    rows = []
    for event in events:
        key = str(event.key)
        us = float(event.self_cpu_time_total)
        count = int(event.count)
        if not math.isfinite(us) or us < 0 or count < 1:
            raise ValueError("nonfinite/invalid profiler event")
        rows.append({"operator": key, "self_cpu_us": us, "calls": count,
                     "self_cpu_memory_bytes": int(event.self_cpu_memory_usage)})
    aten = [row for row in rows if row["operator"].startswith("aten::")]
    if not aten:
        raise ValueError("no ATen operations observed")
    total = sum(row["self_cpu_us"] for row in aten)
    if total <= 0:
        raise ValueError("invalid total ATen self CPU")
    buckets = {}
    for row in aten:
        group = operator_group(row["operator"])
        buckets[group] = buckets.get(group, 0.0) + row["self_cpu_us"]
    result = {
        "aten_self_cpu_us": total,
        "all_profiled_self_cpu_us": sum(row["self_cpu_us"] for row in rows),
        "aten_calls": sum(row["calls"] for row in aten),
        "aten_operator_count": len(aten),
        "category_self_cpu_share": {
            group: {"self_cpu_us": cost, "share_of_aten_self_cpu": cost / total}
            for group, cost in sorted(buckets.items())},
        "top_aten_operators": sorted(aten, key=lambda r: (-r["self_cpu_us"], r["operator"]))[:top_n],
    }
    if abs(sum(r["share_of_aten_self_cpu"] for r in result["category_self_cpu_share"].values()) - 1) > 1e-9:
        raise ValueError("operator accounting does not add up")
    return result


def wall_samples(callable_stage) -> dict:
    for _ in range(WARMUPS):
        callable_stage()
    measurements = []
    for _ in range(TRIALS):
        for _ in range(REPETITIONS):
            tick = time.perf_counter_ns()
            callable_stage()
            measurements.append((time.perf_counter_ns() - tick) / 1e6)
    measurements.sort()
    return {
        "warmups": WARMUPS, "repeats": TRIALS * REPETITIONS,
        "median_ms": statistics.median(measurements),
        "p95_ms": measurements[math.ceil(.95 * len(measurements)) - 1],
        "raw_ms": measurements,
    }


def profile_stage(callable_stage) -> dict:
    # Do not use these instrumented times to claim decoder latency.
    with profile(activities=[ProfilerActivity.CPU],
                 record_shapes=True, profile_memory=True,
                 with_stack=False) as recorded:
        for _ in range(PROFILE_REPETITIONS):
            callable_stage()
    return {"profile_calls": PROFILE_REPETITIONS,
            "instrumentation": "torch.profiler CPU record_shapes/profile_memory enabled",
            **summarize_events(recorded.key_averages(group_by_input_shape=False))}


def exact_repeated_state(state, state_after) -> None:
    exact_state(state, state_after)


def run(checkpoint_dir: Path) -> dict:
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only diagnostic")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    fit = fit_sources()
    model, receipt = load_original(checkpoint_dir, fit)
    engines = {"original": IncrementalTAC043(model, "cap_all"),
               "batched": BatchedValidationTAC057(model, "cap_all")}
    for method in ("_advance", "step", "prefill", "save_state", "load_state"):
        if getattr(BatchedValidationTAC057, method) is not getattr(IncrementalTAC043, method):
            raise ValueError("source transition changed: " + method)
    owner = next(x for x in sorted(fit) if len(fit[x]) >= 544)
    original_weight_hash = engines["original"]._weight_hash()
    output = []
    with torch.inference_mode():
        for n in (128, 512):
            raw = fit[owner][:n]
            tokens = torch.tensor([list(raw)], dtype=torch.long)
            generated, equivalence = parity(engines, tokens, (owner,))
            _, state = engines["batched"].prefill(tokens, owners=(owner,))
            if state.tensor_bytes() != 3904 or len(generated) != 16:
                raise ValueError("unexpected prompt state or greedy continuation")
            next_token = generated[0].clone()
            mask = torch.ones_like(next_token, dtype=torch.bool)
            # Bypass validation only within the isolated transition-stage diagnostic.
            step_logits, step_state = engines["batched"].step(next_token, state, owners=(owner,))
            math_logits, math_state = engines["batched"]._advance(next_token, state, mask)
            if not torch.equal(step_logits, math_logits):
                raise ValueError("step and raw transition changed logits")
            exact_state(step_state, math_state)
            original_state = {name: value.clone() for name, value in state.tensors.items()}
            calls = {
                "step_with_guards": lambda: engines["batched"].step(next_token, state, owners=(owner,)),
                "transition_only": lambda: engines["batched"]._advance(next_token, state, mask),
                "validate_only": lambda: engines["batched"].validate(state, (owner,)),
            }
            result = {}
            # Wall clocks never collected inside an active profiler.
            for name in STAGES:
                result[name] = {"wall_clock": wall_samples(calls[name]),
                                "operator_profile": profile_stage(calls[name])}
            if any(not torch.equal(state.tensors[name], original_state[name]) for name in SHAPES):
                raise ValueError("input state mutated during read-only profiling")
            if engines["original"]._weight_hash() != original_weight_hash:
                raise ValueError("model changed during profiling")
            output.append({
                "prompt_bytes": n, "source_prefix_sha256": hashlib.sha256(raw).hexdigest(),
                "greedy_bytes_sha256": hashlib.sha256(bytes(int(t[0]) for t in generated)).hexdigest(),
                "original_comparison": equivalence,
                "step_transition_bitwise_equal": True,
                "persistent_state_bytes": state.tensor_bytes(),
                "stages": result,
            })
    return {
        "study": "LANGPROFILE058",
        "status": "ORIGINAL_CHECKPOINT_FIT_ONLY_NO_FIT_INTERNAL_OPERATOR_PROFILE",
        "original_checkpoint_receipt": receipt,
        "torch_version": torch.__version__, "host": platform.platform(),
        "source_partition": "FIT_ONLY_INFERENCE", "source_owner": owner,
        "reference_training_updates": 320, "new_training_updates": 0,
        "gpu_minutes": 0, "protected_TEST": "NOT_ACCESSED",
        "unchanged_original_transition": True, "model_parameters": 499623,
        "profile_protocol": {
            "host": "GitHub hosted CPU, FP32 eager, batch1, 2 intraop and 1 interop PyTorch threads",
            "prompt_bytes": [128, 512], "profile_calls": PROFILE_REPETITIONS,
            "wall_per_stage": "2 warmup calls then 7*16 unprofiled calls on the SAME frozen state",
            "interpreting_aten": "self CPU per ATen op inside profiler; NOT total wall time or exact model-component attribution",
            "repeated_state": "same valid state at same clock, source bytes/labels not read inside timed step",
        },
        "results": output,
        "independently_qualified": False, "new_model_quality": "NOT_MEASURED",
        "whole_process_memory": "NOT_MEASURED",
        "torch_profiler_overhead": "INSTRUMENTED_TIMES_NOT_PRODUCTION_SPEED",
        "limits": [
            "one historical checkpoint and previously exposed authentic publisher FIT owner",
            "top ATen operators are not identical to architectural component latency",
            "profile instrumentation distorts wall latency and memory allocation timings",
            "no CUDA, mobile CPU, peak RAM, energy, DEV/TEST or neural quality gain",
            "unaltered source-level transition and PNDS partial math record only",
            "prior LANGDECODE057 p95 repeatability failure remains open",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tac-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("cannot overwrite previous numerical evidence")
    data = run(args.tac_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "study": data["study"], "gpu_minutes": 0,
        "results": [
            {"prompt": row["prompt_bytes"],
             "stages": {name: {
                 "wall_median_ms": stage["wall_clock"]["median_ms"],
                 "wall_p95_ms": stage["wall_clock"]["p95_ms"],
                 "top_ops": [{"name": op["operator"], "self_cpu_us": round(op["self_cpu_us"], 2)}
                             for op in stage["operator_profile"]["top_aten_operators"][:10]],
                 "categories": {k: round(v["share_of_aten_self_cpu"], 4)
                                for k, v in stage["operator_profile"]["category_self_cpu_share"].items()},
             } for name, stage in row["stages"].items()}}
            for row in data["results"]]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
