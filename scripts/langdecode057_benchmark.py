#!/usr/bin/env python3
"""No-fit original-checkpoint CPU replay for batched finite-state validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.langdecode057_validation import BatchedValidationTAC057
from incremental import IncrementalTAC043, SHAPES
from candidate import controlled_position
from scripts.langrecomp052_replay import assert_equal, dist, load_original
from scripts.langlayer035_sampling import fit_sources


def exact_state(a, b):
    if (a.owners, a.clock, a.weights_sha256, a.position_mode, a.implementation_sha256) != (
            b.owners, b.clock, b.weights_sha256, b.position_mode, b.implementation_sha256):
        raise ValueError("state metadata differs")
    if any(not torch.equal(a.tensors[name], b.tensors[name]) for name in SHAPES):
        raise ValueError("state tensors differ")
    if a.tensor_bytes() != 3904 or b.tensor_bytes() != 3904:
        raise ValueError("unexpected persistent state size")


def parity(engines, tokens, owner):
    original, batched = engines.values()
    outputs = [engine.prefill(tokens, owners=owner) for engine in engines.values()]
    y, state = outputs[0]
    z, other = outputs[1]
    exact_state(state, other)
    if not torch.equal(y, z):
        raise ValueError("prefill logits differ")
    generated = []
    prefix = tokens
    with controlled_position(original.model, "cap_all"):
        error = assert_equal(y, original.model(prefix)[:, -1])
    for _ in range(16):
        token = y.argmax(-1)
        generated.append(token.clone())
        y, state = original.step(token, state, owners=owner)
        z, other = batched.step(token, other, owners=owner)
        if not torch.equal(y, z):
            raise ValueError("incremental logits differ")
        exact_state(state, other)
        prefix = torch.cat((prefix, token[:, None]), 1)
        with controlled_position(original.model, "cap_all"):
            error = max(error, assert_equal(y, original.model(prefix)[:, -1]))
    with tempfile.TemporaryDirectory() as folder:
        for i, (writer, reader) in enumerate(((original, batched), (batched, original))):
            path = Path(folder) / f"state-{i}.pt"
            writer.save_state(state, path)
            restored = reader.load_state(path, owners=owner)
            exact_state(state, restored)
            a, sa = writer.step(generated[-1], state, owners=owner)
            b, sb = reader.step(generated[-1], restored, owners=owner)
            if not torch.equal(a, b):
                raise ValueError("snapshot continuation differs")
            exact_state(sa, sb)
    return generated, {"bitwise_original_incremental_logits_and_all_states": True,
                       "full_prefix_max_abs_error": error,
                       "snapshot_bidirectional_bitwise_continuation": True,
                       "state_tensor_bytes": state.tensor_bytes()}


def trial(engine, tokens, owner, generated):
    tick = time.perf_counter_ns()
    _, state = engine.prefill(tokens, owners=owner)
    prefill = (time.perf_counter_ns() - tick) / 1e6
    steps = []
    for token in generated:
        tick = time.perf_counter_ns()
        _, state = engine.step(token, state, owners=owner)
        steps.append((time.perf_counter_ns() - tick) / 1e6)
    validation = []
    for _ in range(16):
        tick = time.perf_counter_ns()
        engine.validate(state, owner)
        validation.append((time.perf_counter_ns() - tick) / 1e6)
    return {"prefill_ms": prefill, "step_ms": steps, "validate_ms": validation,
            "state_bytes": state.tensor_bytes()}


def run(folder):
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only study")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    fit = fit_sources()
    model, receipt = load_original(folder, fit)
    setup = {}
    engines = {}
    for name, cls in (("original", IncrementalTAC043), ("batched", BatchedValidationTAC057)):
        tick = time.perf_counter_ns()
        engines[name] = cls(model, "cap_all")
        setup[name] = (time.perf_counter_ns() - tick) / 1e6
    for method in ("_advance", "step", "prefill", "save_state", "load_state"):
        if getattr(BatchedValidationTAC057, method) is not getattr(IncrementalTAC043, method):
            raise ValueError("unexpected transition/API change: " + method)
    original_weight_hash = engines["original"]._weight_hash()
    owner = (next(x for x in sorted(fit) if len(fit[x]) >= 544),)
    results = []
    with torch.inference_mode():
        for length in (128, 512):
            raw = fit[owner[0]][:length]
            tokens = torch.tensor([list(raw)], dtype=torch.long)
            generated, checks = parity(engines, tokens, owner)
            trials = {name: [] for name in engines}
            for i in range(9):
                order = ("original", "batched") if i % 2 == 0 else ("batched", "original")
                for name in order:
                    row = trial(engines[name], tokens, owner, generated)
                    if i >= 2:
                        trials[name].append(row)
            arms = {}
            for name, rows in trials.items():
                if any(row["state_bytes"] != 3904 for row in rows):
                    raise ValueError("state storage grew")
                arms[name] = {
                    "decode": dist([v for r in rows for v in r["step_ms"]]),
                    "prefill": dist([r["prefill_ms"] for r in rows]),
                    "validation": dist([v for r in rows for v in r["validate_ms"]]),
                    "total_prefill_decode": dist([r["prefill_ms"] + sum(r["step_ms"]) for r in rows]),
                }
            reduction = 1 - arms["batched"]["decode"]["median_ms"] / arms["original"]["decode"]["median_ms"]
            p95_ratio = arms["batched"]["decode"]["p95_ms"] / arms["original"]["decode"]["p95_ms"]
            results.append({"prompt_bytes": length, "source_sha256": hashlib.sha256(raw).hexdigest(),
                            "generated_sha256": hashlib.sha256(bytes(int(t[0]) for t in generated)).hexdigest(),
                            "checks": checks, "arms": arms, "median_reduction_fraction": reduction,
                            "p95_ratio": p95_ratio, "latency_gate": reduction >= .05 and p95_ratio <= 1.1})
    if engines["original"]._weight_hash() != original_weight_hash:
        raise ValueError("original weights changed")
    return {"study": "LANGDECODE057", "status": "INTERNALLY_MEASURED_NOT_INDEPENDENTLY_QUALIFIED",
            "acceptance_gate": all(r["latency_gate"] for r in results),
            "original_checkpoint_receipt": receipt, "source_owner": owner[0],
            "source_partition": "FIT_ONLY_INFERENCE", "training_updates": 0, "gpu_minutes": 0,
            "protected_TEST": "NOT_ACCESSED", "checkpoint_unchanged": True,
            "torch": torch.__version__, "platform": platform.platform(), "python": platform.python_version(),
            "settings": "CPU FP32 eager batch1 two threads one interop; two warmups seven repeats;16 bytes",
            "engine_setup_ms": setup, "results": results,
            "wrapper_sha256": hashlib.sha256((ROOT / "scripts/langdecode057_validation.py").read_bytes()).hexdigest(),
            "transition_sha256": hashlib.sha256((ROOT / "research/mathematical-engineering/langstream043/incremental.py").read_bytes()).hexdigest(),
            "limits": ["single host/checkpoint; previously exposed FIT owners; no language quality claim",
                       "3904-byte persistent payload excludes weights, object overhead and transient tensors",
                       "batch validation adds a transient 3904-byte concatenation per owner",
                       "GPU VRAM, mobile throughput, gradients and universal IEEE/AD/source bridge unmeasured/OPEN"],
            "checkpoint_promotion": False, "independently_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tac-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse to overwrite evidence")
    report = run(args.tac_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"study": report["study"], "acceptance_gate": report["acceptance_gate"],
                      "results": [{"prompt": r["prompt_bytes"], "median_reduction_fraction": r["median_reduction_fraction"],
                                   "p95_ratio": r["p95_ratio"], "checks": r["checks"]} for r in report["results"]]}))


if __name__ == "__main__":
    main()
