#!/usr/bin/env python3
"""No-fit feasibility probe: original ME111 104663-param streaming TAC -> ONNX CPU.

This is NOT an Android device run, independent evaluation, or proof that
PyTorch input guards were exported. No sensor source/label data are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODEL = ROOT / "research/replay-bundles/ME111/conditional_belief/seed-111001/final.pt"
EXPECTED_SHA256 = "337b882e9a10631dc85700846ef2b15ca8908699499668186ad3a248dc108449"
PARAMETERS = 104663


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probe(output_dir):
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from torch import nn
    import tac_osm.variants.entity_sensor_world as entity
    from tac_osm.variants.entity_sensor_world import EntityEvidenceState
    from tac_osm.variants.conditional_belief_sensor_world import ConditionalBeliefSensorWorld

    if not MODEL.is_file() or sha(MODEL) != EXPECTED_SHA256:
        raise RuntimeError("original ME111 trained model bytes unavailable or hash mismatch")
    cp = torch.load(MODEL, map_location="cpu", weights_only=False)
    if cp.get("seed") != 111001 or cp.get("steps") != 1536:
        raise RuntimeError("checkpoint seed/training-step identity mismatch")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = ConditionalBeliefSensorWorld(torch.ones(6, dtype=torch.float32))
    model.load_state_dict(cp["model"], strict=True)
    model.eval()
    model.requires_grad_(False)
    num_parameters = sum(p.numel() for p in model.parameters())
    if num_parameters != PARAMETERS:
        raise RuntimeError(f"actual trained parameter count changed: {num_parameters}")

    class OneStep(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, values, observed, ages, clock, owners, hours,
                    source_sum, source_mass, evidence_sum, count, memory, last_hour):
            state = EntityEvidenceState(source_sum, source_mass, evidence_sum,
                      count, memory, owners, last_hour, id(self.m), self.m.versions())
            forecast, next_state = self.m(
                values, observed, ages, clock, owners, hours, state=state)
            chronology_ok = hours[:, 0] > last_hour
            return (forecast.log_location, forecast.left_scale, forecast.right_scale,
                    next_state.source_sum, next_state.source_mass,
                    next_state.evidence_sum, next_state.count,
                    next_state.memory, next_state.last_hour, chronology_ok)

    adapter = OneStep(model).eval()
    names = ["values", "observed", "ages", "clock", "owners", "hours",
             "source_sum", "source_mass", "evidence_sum", "count", "memory", "last_hour"]
    output_names = ["log_location", "left_scale", "right_scale",
                    "next_source_sum", "next_source_mass", "next_evidence_sum",
                    "next_count", "next_memory", "next_last_hour", "chronology_ok"]
    x = [
        torch.zeros(1, 1, 13), torch.ones(1, 1, 13, dtype=torch.bool),
        torch.zeros(1, 1, 13), torch.zeros(1, 1, 6),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([[100]], dtype=torch.int64),
        torch.zeros(1, 13), torch.zeros(1, 13), torch.zeros(1, 62),
        torch.zeros(1, 1, dtype=torch.int64),
        torch.zeros(1, 8, 8),
        torch.tensor([99], dtype=torch.int64)
    ]
    with torch.no_grad():
        original_native = adapter(*x)

    # Export-only T=1 exact-algebra specialization. The original source
    # functions compute (T-1).bit_length() from a symbolic shape and cannot
    # be traced. Do not modify trained weights or original research source.
    # Never reuse this adapter for T>1 or to bypass runtime input checks.
    def diagonal_one_step(a, b, initial):
        return a * initial[:, None, :] + b

    def associative_one_step(key, value, retention, strength, writable):
        keep = torch.where(writable, retention, torch.ones_like(retention))
        beta = torch.where(writable, strength, torch.zeros_like(strength))
        eye = torch.eye(8, dtype=key.dtype, device=key.device).view(1, 1, 8, 8)
        outer = key.unsqueeze(-1) * key.unsqueeze(-2)
        P = keep.unsqueeze(-1) * (eye - beta.unsqueeze(-1) * outer)
        Q = beta.unsqueeze(-1) * value.unsqueeze(-1) * key.unsqueeze(-2)
        return P, Q

    def memory_one_step(P, Q, initial):
        return initial[:, None, :, :] @ P + Q

    entity.diagonal_affine_prefix = diagonal_one_step
    entity.associative_affine_schedule = associative_one_step
    entity.work_efficient_affine_prefix = memory_one_step
    with torch.no_grad():
        specialized_native = adapter(*x)
    for original, specialized in zip(original_native, specialized_native):
        torch.testing.assert_close(original, specialized, rtol=1e-6, atol=1e-6)
    path = output_dir / "ME111_ONE_STEP_EXPLORATORY.onnx"
    # Legacy ONNX tracer has no aten::expm1 translation for opset18.
    # Exact real identity expm1(x)=exp(x)-1; float32 roundoff is assessed
    # independently against original torch in the runtime parity loop.
    from torch.onnx import register_custom_op_symbolic
    def expm1_onnx(graph, x):
        one = graph.op("Constant", value_t=torch.tensor(1.0, dtype=torch.float32))
        return graph.op("Sub", graph.op("Exp", x), one)
    register_custom_op_symbolic("aten::expm1", expm1_onnx, 18)
    torch.onnx.export(
        adapter, tuple(x), str(path), export_params=True,
        opset_version=18, do_constant_folding=True,
        input_names=names, output_names=output_names, dynamo=False
    )
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    exported_inputs = [i.name for i in session.get_inputs()]
    if set(exported_inputs) != set(names):
        raise RuntimeError(f"required streaming state inputs omitted: {sorted(set(names) - set(exported_inputs))}")
    output_set = {v.name for v in session.get_outputs()}
    if set(output_names) != output_set:
        raise RuntimeError(f"streaming output state omitted: {sorted(set(output_names) - output_set)}")

    abs_error = 0.0
    rel_error = 0.0
    all_output_shapes = {}
    with torch.no_grad():
        for step in range(3):
            if step:
                x[5] = torch.tensor([[100 + step]], dtype=torch.int64)
                x[0] = torch.full((1, 1, 13), 0.05 * step, dtype=torch.float32)
                x[6:10] = [v.clone() for v in native[3:7]]
                x[10:12] = [v.clone() for v in native[7:9]]
            native = adapter(*x)
            actual = session.run(output_names, {
                name: t.detach().numpy() for name, t in zip(names, x) if name in exported_inputs
            })
            for name, expected, got in zip(output_names, native, actual):
                target = expected.detach().numpy()
                if name in ("next_count", "next_last_hour", "chronology_ok"):
                    np.testing.assert_array_equal(got, target)
                else:
                    np.testing.assert_allclose(got, target, rtol=2e-4, atol=2e-5)
                    abs_error = max(abs_error, float(np.max(np.abs(got - target))))
                    rel_error = max(rel_error, float(np.max(np.abs(got - target) / (np.abs(target) + 1e-6))))
                all_output_shapes[name] = list(got.shape)
                if name == "chronology_ok" and not bool(np.all(got)):
                    raise RuntimeError("streaming chronology check failed")
    return {
        "status": "HOST_ONNX_CPU_EXPLORATORY_PASS",
        "original_checkpoint_sha256": EXPECTED_SHA256,
        "original_checkpoint_seed": 111001,
        "actual_trainable_parameter_count": num_parameters,
        "training_performed": False,
        "operator_backend": "ONNX Runtime CPU on GitHub runner, not Android",
        "opset": 18,
        "expm1_operator": "EXPORT_ONLY_EXP_MINUS_ONE_WITH_FP32_PARITY_CHECK",
        "fixed_shape_B": 1, "fixed_shape_T": 1,
        "export_only_single_step_kernel_specialization": "CHECKED_AGAINST_ORIGINAL_T1_PYTORCH",
        "streaming_steps_parity_checked": 3,
        "max_abs_error": abs_error, "max_reference_relative_error": rel_error,
        "per_output_shapes": all_output_shapes,
        "onnx_model_bytes": path.stat().st_size,
        "onnx_model_sha256": sha(path),
        "original_source_scientific_gate": "FAIL_MIXED_TRADEOFF_ME111",
        "torch_python_validation_guards_exported": "NOT_VERIFIED_REQUIRE_LOCAL_SDK_VALIDATION",
        "physical_android": "NOT_RUN", "airplane_mode": "NOT_RUN",
        "peak_ram": "NOT_RUN", "p50_p95_latency": "NOT_RUN",
        "energy": "NOT_RUN", "model_delivery": "NOT_RELEASED"
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.out_dir / "FEASIBILITY.json"
    if receipt.exists():
        raise SystemExit("refuse to replace existing export feasibility evidence")
    try:
        result = probe(args.out_dir)
    except Exception as error:
        result = {
            "status": "ONNX_EXPORT_OR_PARITY_BLOCKED",
            "reason_type": type(error).__name__, "reason": str(error)[:2000],
            "checkpoint": "ME111 seed111001 original only", "trained": False,
            "android_physical": "NOT_RUN", "original_scientific_gate": "FAIL"
        }
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "HOST_ONNX_CPU_EXPLORATORY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
