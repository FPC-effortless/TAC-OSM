#!/usr/bin/env python3
"""Frozen operator audit. No model construction, optimizer, corpus or checkpoints."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tac_osm.csl125_r2 import selective_affine_scan
from tac_osm.csl125_r2_fast_scan import bounded_cumprod_scan
from tac_osm.csl125_r2_math_reference import selective_serial

CASES = ("alternating", "constructed_cancellation", "near_empty", "long_tau_tiny_gate",
         "closed_after_write", "signed_zero_hold", "stationary")
CHUNKS = (1, 7, 32, 64)
OBJECTIVES = ("linear_state", "normalized_read", "normalized_read_source_default")
NAMES = ("z", "carry", "rate", "gate")


def fixture(name):
    if name not in CASES:
        raise ValueError("unregistered fixture")
    z = torch.ones((1, 65, 2, 2, 2), dtype=torch.float32)
    z[:, :, 0] *= 1e-8 if name == "near_empty" else .25
    z[:, :, 1] *= 1 - (1e-8 if name == "near_empty" else .25)
    z[:, 1::2, :, :, 0] *= -1
    carry = torch.tensor([[[[.1, .2], [-.1, .2]], [[.3, .4], [-.3, .4]]]])
    rate = torch.tensor([[.5, 1/32768], [.125, .03125]])
    gate = torch.ones((1, 65, 2))
    if name == "constructed_cancellation":
        rate.fill_(.5)
        z[..., 0] = -z[..., 1]
        carry[..., 1] = z[:, 0, ..., 1]
        carry[..., 0] = math.expm1(.5) * carry[..., 1]
        gate[:, 1:] = 0
    elif name == "near_empty":
        carry.fill_(1e-12)
        carry[..., 0] *= -1
        gate[:, :, 0] = 1e-5
    elif name == "long_tau_tiny_gate":
        rate.fill_(1/32768)
        gate.fill_(1e-8)
    elif name == "closed_after_write":
        gate[:, 1:] = 0
    elif name == "signed_zero_hold":
        carry[..., 0] = -0.0
        gate.zero_()
    elif name == "stationary":
        z[..., 0] = z[..., 1]
        carry.zero_()
    return z, carry, rate, gate


def serial_torch(z, carry, rate, gate, _chunk):
    state, outputs = carry, []
    for t in range(z.shape[1]):
        rg = rate.unsqueeze(0) * gate[:, t].unsqueeze(-1)
        beta, eta = torch.exp(-rg), -torch.expm1(-rg)
        state = beta.unsqueeze(-1) * state + eta.unsqueeze(-1) * z[:, t]
        outputs.append(state)
    return torch.stack(outputs, 1), state


def decimal_serial(args):
    z, carry, rate, gate = [v.detach().numpy() for v in args]
    out = np.empty(z.shape, dtype=np.float64)
    with localcontext() as context:
        context.prec = 80
        for r in range(2):
            for e in range(2):
                state = [Decimal.from_float(float(v)) for v in carry[0, r, e]]
                rr = Decimal.from_float(float(rate[r, e]))
                for t in range(65):
                    beta = (-rr * Decimal.from_float(float(gate[0, t, r]))).exp()
                    eta = 1 - beta
                    state = [beta * s + eta * Decimal.from_float(float(v))
                             for s, v in zip(state, z[0, t, r, e])]
                    out[0, t, r, e] = [float(s) for s in state]
    return torch.from_numpy(out)


def objective(out, name):
    if name not in OBJECTIVES:
        raise ValueError("unregistered objective")
    epsilon = 1e-8 if name == "normalized_read_source_default" else 1e-6
    value = out if name == "linear_state" else out[..., 0] / (out[..., 1] + epsilon)
    probe = ((torch.arange(value.numel(), device=value.device).reshape(value.shape) % 7) - 3).to(value.dtype) / 3
    return (value * probe).mean()


def evaluate(function, args, chunk, loss, dtype):
    leaves = [a.detach().to(dtype).clone().requires_grad_(True) for a in args]
    out, end = function(*leaves, chunk)
    scalar = objective(out, loss)
    gradients = torch.autograd.grad(scalar, leaves, allow_unused=False)
    if any(g.shape != a.shape for g, a in zip(gradients, leaves)):
        raise ValueError("gradient schema changed")
    return out.detach(), end.detach(), tuple(g.detach() for g in gradients)


def difference(actual, reference, atol, rtol):
    if actual.shape != reference.shape or actual.numel() == 0:
        return {"status": "FAIL_SHAPE_OR_EMPTY", "max_abs": None, "max_scaled": None}
    a, b = actual.double(), reference.double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"status": "FAIL_NONFINITE", "max_abs": None, "max_scaled": None}
    error = (a - b).abs()
    scaled = error / (atol + rtol * b.abs())
    worst = int(scaled.flatten().argmax())
    return {"status": "PASS" if bool((scaled <= 1).all()) else "FAIL_TOLERANCE",
            "max_abs": float(error.max()), "max_scaled": float(scaled.max()),
            "worst_scaled_flat_index": worst, "actual_at_worst_scaled": float(a.flatten()[worst]),
            "reference_at_worst_scaled": float(b.flatten()[worst]), "reference_max_abs": float(b.abs().max())}


def audit():
    if torch.cuda.is_initialized():
        raise ValueError("GPU not authorized")
    torch.set_num_threads(1)
    protocol = json.loads((ROOT / "research/csl_scan_audit_20261106/PROTOCOL.json").read_text())
    amendment = json.loads((ROOT / "research/csl_scan_audit_20261106/SOURCE_EPSILON_AMENDMENT.json").read_text())
    if hashlib.sha256((ROOT / amendment["source_path"]).read_bytes()).hexdigest() != amendment["source_sha256"]:
        raise ValueError("source default-epsilon implementation changed")
    for p, digest in protocol["source_sha256"].items():
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen operator source changed: " + p)
    tolerance = protocol["tolerances"]
    fa, fr = tolerance["forward_atol"], tolerance["forward_rtol"]
    ga, gr = tolerance["gradient_atol"], tolerance["gradient_rtol"]
    rows = []
    for case in CASES:
        args = fixture(case)
        independent = decimal_serial(args)
        numpy_out, _ = selective_serial(*(a.numpy() for a in args))
        for chunk in CHUNKS:
            for loss in OBJECTIVES:
                ref = evaluate(serial_torch, args, chunk, loss, torch.float64)
                stable = evaluate(selective_affine_scan, args, chunk, loss, torch.float32)
                fast = evaluate(bounded_cumprod_scan, args, chunk, loss, torch.float32)
                comparisons = {"serial_vs_decimal": difference(ref[0], independent, 1e-12, 1e-12),
                               "numpy_vs_decimal": difference(torch.from_numpy(numpy_out), independent, 1e-12, 1e-12)}
                for name, values in (("stable", stable), ("candidate", fast)):
                    comparisons[name + "_forward"] = difference(values[0], independent, fa, fr)
                    comparisons[name + "_final"] = difference(values[1], independent[:, -1], fa, fr)
                    for arg, got, expected in zip(NAMES, values[2], ref[2]):
                        comparisons[name + "_grad_" + arg] = difference(got, expected, ga, gr)
                for arg, got, expected in zip(NAMES, fast[2], stable[2]):
                    comparisons["candidate_vs_stable_grad_" + arg] = difference(got, expected, ga, gr)
                # Poison only later writes/gates; no outcome labels exist.
                poisoned = [a.clone() for a in args]
                poisoned[0][:, 32:, ..., 0] *= -1
                poisoned[3][:, 32:] = 1 - poisoned[3][:, 32:]
                for name, fn, result in (("stable", selective_affine_scan, stable),
                                         ("candidate", bounded_cumprod_scan, fast)):
                    changed = fn(*poisoned, chunk)[0]
                    comparisons[name + "_future_poison"] = difference(result[0][:, :32], changed[:, :32], fa, fr)
                failed = [k for k, v in comparisons.items() if v["status"] != "PASS"]
                rows.append({"case": case, "chunk": chunk, "objective": loss,
                             "status": "FAIL_NUMERICAL_GATE" if failed else "PASS_BOUNDED_FIXTURE",
                             "failed_checks": failed, "checks": comparisons,
                             "candidate_signed_zero_bit_identity": (bool(torch.equal(torch.signbit(fast[1]), torch.signbit(args[1])))
                                                                    if case == "signed_zero_hold" else "NOT_APPLICABLE"),
                             "stable_signed_zero_bit_identity": (bool(torch.equal(torch.signbit(stable[1]), torch.signbit(args[1])))
                                                                 if case == "signed_zero_hold" else "NOT_APPLICABLE")})
    failed = [r for r in rows if r["failed_checks"]]
    return {"study": "CSL_SCAN_AUDIT_20261106", "audit_status": "COMPLETE_WITH_FAILURES" if failed else "COMPLETE_BOUNDED_CHECKS_PASS",
            "case_rows": len(rows), "failed_rows": len(failed), "rows": rows,
            "source_sha256": protocol["source_sha256"], "tolerances": tolerance,
            "source_epsilon_amendment": amendment,
            "torch": torch.__version__, "threads": 1, "precision": "actual source FP32 versus independent FP64/Decimal80",
            "models_constructed": 0, "model_parameters_allocated": 0, "optimizer_updates": 0,
            "GPU_minutes": 0, "protected_TEST": False, "independently_qualified": False,
            "kernel_promotion_authorized": False, "production_source_modified": False,
            "full_model_parameter_gradient_parity": "NOT_RUN",
            "PNDS_full_source_IEEE_AD_reassociation": "OPEN",
            "CUDA_compiler_AMP_mobile": "NOT_RUN",
            "physical_device": {k: "NOT_RUN" for k in ("target", "operator_support", "numerical_agreement", "p50_p95", "RAM", "energy", "airplane_mode")},
            "scope": "Constructed operator falsifiers only; not learned-memory, corpus or device performance evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing audit")
    with patch.object(torch.nn.Module, "__init__", side_effect=RuntimeError("model construction prohibited")):
        result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("audit_status", "case_rows", "failed_rows", "models_constructed", "kernel_promotion_authorized")}))


if __name__ == "__main__":
    main()
