#!/usr/bin/env python3
"""No-fit 256-step ME111 streaming parity: ORIGINAL PyTorch vs frozen ONNX CPU.

Two independent source-owner streams, missing masks, irregular source hours and
byte-exact FP32 state snapshots. Constructed inputs are software diagnostics,
never authentic sensor performance or evidence of 256-event useful memory.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import time

ORIGINAL_SHA = "337b882e9a10631dc85700846ef2b15ca8908699499668186ad3a248dc108449"
ONNX_SHA = "dd65519a9e4f3388966538af53119b56e9292b46a9209d63ad4927b87f0a176b"
INPUT_NAMES = (
    "values", "observed", "ages", "clock", "owners", "hours",
    "source_sum", "source_mass", "evidence_sum", "count", "memory", "last_hour"
)
OUTPUT_NAMES = (
    "log_location", "left_scale", "right_scale",
    "next_source_sum", "next_source_mass", "next_evidence_sum",
    "next_count", "next_memory", "next_last_hour", "chronology_ok"
)
STATE_INPUT_NAMES = (
    "source_sum", "source_mass", "evidence_sum", "count", "memory", "last_hour"
)
STATE_OUTPUT_NAMES = (
    "next_source_sum", "next_source_mass", "next_evidence_sum",
    "next_count", "next_memory", "next_last_hour"
)
FLOAT_OUTPUTS = tuple(name for name in OUTPUT_NAMES if name not in
                      ("next_count", "next_last_hour", "chronology_ok"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def snapshot_roundtrip(state: dict, np) -> dict:
    """Simulate a process restart, preserving precisely the deployed FP32 bits."""
    b = BytesIO()
    np.savez(b, **{k: state[k] for k in STATE_INPUT_NAMES})
    b.seek(0)
    with np.load(b, allow_pickle=False) as loaded:
        restored = {k: loaded[k].copy() for k in STATE_INPUT_NAMES}
    for k in STATE_INPUT_NAMES:
        np.testing.assert_array_equal(restored[k], state[k])
        if restored[k].dtype != state[k].dtype:
            raise AssertionError("persisted tensor dtype changed: " + k)
    return restored


def event(np, owner: int, step: int, source_hour: int):
    rng = np.random.default_rng(111001 + owner * 100000 + step)
    values = np.clip(rng.normal(0.05, 0.7, size=(1, 1, 13)), -1.8, 1.8).astype("float32")
    observed = (rng.random((1, 1, 13)) > (0.1 if owner == 0 else 0.55))
    if step % 17 == 0:
        observed[:] = False
    if step % 29 == 0 and step % 17 != 0:
        observed[:, :, :6] = False
    values = np.where(observed, values, 0.0).astype("float32")
    ages = rng.random((1, 1, 13)).astype("float32")
    clock = np.array([[
        [np.sin(step / 24), np.cos(step / 24),
         np.sin(step / 168), np.cos(step / 168),
         np.sin(step / 12), np.cos(step / 12)]
    ]], dtype="float32")
    return {
        "values": values,
        "observed": observed.astype("bool"),
        "ages": ages,
        "clock": clock,
        "owners": np.array([owner], dtype="int64"),
        "hours": np.array([[source_hour]], dtype="int64"),
    }


def check(source_root: Path, model_path: Path):
    import numpy as np
    import torch
    import onnxruntime as ort
    sys.path[:0] = [str(source_root / "src"), str(source_root)]
    from tac_osm.variants.conditional_belief_sensor_world import ConditionalBeliefSensorWorld

    original = (source_root / "research/replay-bundles/ME111/conditional_belief"
                / "seed-111001/final.pt")
    if digest(original) != ORIGINAL_SHA or digest(model_path) != ONNX_SHA:
        raise RuntimeError("original checkpoint or exported graph bytes changed")
    checkpoint = torch.load(original, map_location="cpu", weights_only=False)
    if checkpoint.get("seed") != 111001 or checkpoint.get("steps") != 1536:
        raise RuntimeError("checkpoint identity or exact training stage mismatch")
    torch.set_num_threads(1)
    native = ConditionalBeliefSensorWorld(torch.ones(6, dtype=torch.float32))
    native.load_state_dict(checkpoint["model"], strict=True)
    native.eval().requires_grad_(False)
    if sum(p.numel() for p in native.parameters()) != 104663:
        raise RuntimeError("original neural parameter count changed")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    if set(x.name for x in session.get_inputs()) != set(INPUT_NAMES):
        raise RuntimeError("ONNX input/state schema mismatch")
    if set(x.name for x in session.get_outputs()) != set(OUTPUT_NAMES):
        raise RuntimeError("ONNX output/state schema mismatch")

    start = time.monotonic()
    max_error = {name: 0.0 for name in FLOAT_OUTPUTS}
    count_all_missing = 0
    count_missing_targets = 0
    count_irregular = 0
    snapshot_count = 0
    checked = 0
    owner_results = []

    with torch.inference_mode():
        for owner in (0, 5):
            owners = torch.tensor([owner], dtype=torch.int64)
            native_state = replace(native.initial_state(owners),
                                   last_hour=torch.tensor([99], dtype=torch.int64))
            ort_state = {
                "source_sum": np.zeros((1, 13), dtype="float32"),
                "source_mass": np.zeros((1, 13), dtype="float32"),
                "evidence_sum": np.zeros((1, 62), dtype="float32"),
                "count": np.zeros((1, 1), dtype="int64"),
                "memory": np.zeros((1, 8, 8), dtype="float32"),
                "last_hour": np.array([99], dtype="int64")
            }
            hour = 99
            observed_updates = 0
            for step in range(256):
                delta = 3 if step % 11 == 0 else 1
                if delta > 1:
                    count_irregular += 1
                hour += delta
                inp = event(np, owner, step, hour)
                if not inp["observed"].any():
                    count_all_missing += 1
                if not inp["observed"][:, :, :6].all():
                    count_missing_targets += 1
                observed_updates += int(inp["observed"].any())

                expected_forecast, native_state = native(
                    torch.from_numpy(inp["values"]),
                    torch.from_numpy(inp["observed"]),
                    torch.from_numpy(inp["ages"]),
                    torch.from_numpy(inp["clock"]),
                    torch.from_numpy(inp["owners"]),
                    torch.from_numpy(inp["hours"]),
                    state=native_state
                )
                expected = {
                    "log_location": expected_forecast.log_location.detach().numpy(),
                    "left_scale": expected_forecast.left_scale.detach().numpy(),
                    "right_scale": expected_forecast.right_scale.detach().numpy(),
                    "next_source_sum": native_state.source_sum.detach().numpy(),
                    "next_source_mass": native_state.source_mass.detach().numpy(),
                    "next_evidence_sum": native_state.evidence_sum.detach().numpy(),
                    "next_count": native_state.count.detach().numpy(),
                    "next_memory": native_state.memory.detach().numpy(),
                    "next_last_hour": native_state.last_hour.detach().numpy(),
                    "chronology_ok": np.array([True], dtype=bool),
                }
                got_values = session.run(
                    list(OUTPUT_NAMES), {**inp, **ort_state}
                )
                got = dict(zip(OUTPUT_NAMES, got_values))
                for name in OUTPUT_NAMES:
                    if got[name].shape != expected[name].shape:
                        raise AssertionError(f"shape mismatch {owner=} {step=} {name=}: "
                                             f"{got[name].shape} != {expected[name].shape}")
                    if name in FLOAT_OUTPUTS:
                        if not np.isfinite(got[name]).all():
                            raise AssertionError(f"nonfinite ONNX {owner=} {step=} {name=}")
                        error = float(np.max(np.abs(got[name] - expected[name])))
                        max_error[name] = max(max_error[name], error)
                        np.testing.assert_allclose(
                            got[name], expected[name], rtol=2e-4, atol=2e-5,
                            err_msg=f"independent 256-event native/ONNX drift {owner=} {step=} {name=}"
                        )
                    else:
                        np.testing.assert_array_equal(
                            got[name], expected[name],
                            err_msg=f"int/bool state mismatch {owner=} {step=} {name=}"
                        )
                ort_state = {
                    name: np.array(got[output_name], copy=True)
                    for name, output_name in zip(STATE_INPUT_NAMES, STATE_OUTPUT_NAMES)
                }
                if step in (31, 63, 127, 191, 255):
                    ort_state = snapshot_roundtrip(ort_state, np)
                    snapshot_count += 1
                checked += 1
            assert int(ort_state["count"][0, 0]) == observed_updates
            assert int(ort_state["last_hour"][0]) == hour
            owner_results.append({
                "owner": owner,
                "constructed_steps": 256,
                "observed_event_count": observed_updates,
                "final_hour": hour,
                "final_state_restart_roundtrip_count": 5
            })

    return {
        "status": "PASS_256_EVENT_LONG_STREAM_HOST_CPU_PARITY",
        "scientific_value_from_constructed_inputs": "NONE",
        "original_me111_real_sensor_gate": "FAIL_MIXED_TRADEOFF",
        "training_performed": False,
        "original_checkpoint_sha256": ORIGINAL_SHA,
        "onnx_sha256": ONNX_SHA,
        "actual_trainable_parameters": 104663,
        "source_owner_streams_checked": 2,
        "steps_per_owner": 256,
        "total_independent_reference_steps": checked,
        "all_masked_events_checked": count_all_missing,
        "missing_target_events_checked": count_missing_targets,
        "non_unit_hour_intervals_checked": count_irregular,
        "lossless_binary_state_restart_roundtrips": snapshot_count,
        "max_abs_error_by_output": max_error,
        "per_owner": owner_results,
        "host_cpu_seconds_for_test": round(time.monotonic() - start, 5),
        "android_256_step_replay": "NOT_RUN",
        "physical_phone": "NOT_RUN",
        "real_world_long_memory_advantage": "NOT_ESTABLISHED",
        "real_world_independent_scientific_qualification": "NOT_QUALIFIED"
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--onnx", required=True, type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    args = p.parse_args()
    if args.receipt.exists():
        p.error("refusing to overwrite prior diagnostic evidence")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        receipt = check(args.source_root.resolve(), args.onnx.resolve())
    except Exception as exc:
        receipt = {
            "status": "FAIL_256_EVENT_HOST_CPU_PARITY",
            "reason_type": type(exc).__name__,
            "reason": str(exc)[:3000],
            "training_performed": False,
            "original_me111_real_sensor_gate": "FAIL_MIXED_TRADEOFF",
            "physical_phone": "NOT_RUN"
        }
    args.receipt.write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "PASS_256_EVENT_LONG_STREAM_HOST_CPU_PARITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
