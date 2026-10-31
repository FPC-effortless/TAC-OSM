"""Read-only TRAIN-prefix recurrence/gradient checks on six original final fits.

This diagnostic never invokes an optimizer or opens DEV/TEST. Its fixed tolerances
are finite-fixture engineering criteria, not a global precision or capability proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from research.langcache104 import model as source_model
from research.langdepth106.evaluate import audit_final_checkpoints
from research.langdepth106.source import verify_training_stage
from research.langdepth106.train import load_protocol, parameter_digest, sha256_file

ATOL = 3e-5
RTOL = 3e-4
PREFIX_LENGTH = 129


def whole_model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compare(left, right):
    if left.shape != right.shape:
        return {"pass": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite:
        return {"pass": False, "finite": False, "max_absolute_error": None,
                "max_tolerance_ratio": None}
    difference = (left.detach().double()-right.detach().double()).abs()
    allowed = ATOL+RTOL*right.detach().double().abs()
    return {"pass": bool((difference <= allowed).all()), "finite": True,
            "max_absolute_error": float(difference.max()) if difference.numel() else 0.,
            "max_tolerance_ratio": float((difference/allowed).max()) if difference.numel() else 0.}


def state_tensors(state):
    return {f"layer.{index}.{name}": value for index, layer in enumerate(state.layers)
            for name, value in layer.tensors().items()}


def normalized_bank_observation(args, result):
    """Check trained offers and a separate serial FP32 cone monitor from zero."""
    q, keys, values, a, w, initial_memory, initial_mass = args
    reads, memory, mass, evidence = result
    inputs_finite = all(bool(torch.isfinite(x).all()) for x in args)
    if not inputs_finite:
        return {"pass": False, "inputs_finite": False}
    zero_initial = bool((initial_memory == 0).all() and (initial_mass == 0).all())
    conditions = {
        "valid_zero_initial": zero_initial,
        "keys_queries_nonnegative": bool((keys >= 0).all() and (q >= 0).all()),
        "values_absolute_at_most_one": bool((values.abs() <= 1).all()),
        "retention_in_unit_interval": bool(((a >= 0) & (a <= 1)).all()),
        "write_nonnegative_and_at_most_one_minus_retention": bool(((w >= 0) & (w <= 1-a)).all()),
    }
    monitored_memory, monitored_mass = initial_memory.clone(), initial_mass.clone()
    max_serial_cone_excess = 0.
    min_serial_mass, max_serial_mass = 0., 0.
    for time in range(a.shape[1]):
        rank = keys.shape[-2]
        write = values[:, time].transpose(-1, -2) @ keys[:, time]
        monitored_memory = a[:, time, :, None, None]*monitored_memory + w[:, time, :, None, None]/rank*write
        monitored_mass = a[:, time, :, None]*monitored_mass + w[:, time, :, None]/rank*keys[:, time].sum(-2)
        max_serial_cone_excess = max(max_serial_cone_excess,
            float((monitored_memory.abs()-monitored_mass[..., None, :]).max()))
        min_serial_mass = min(min_serial_mass, float(monitored_mass.min()))
        max_serial_mass = max(max_serial_mass, float(monitored_mass.max()))
    final_cone_excess = float((memory.abs()-mass[..., None, :]).max())
    read_max = float(reads.abs().max())
    observed_finite = all(bool(torch.isfinite(x).all()) for x in result)
    passed = (all(conditions.values()) and observed_finite and
              max_serial_cone_excess <= ATOL and min_serial_mass >= -ATOL and
              max_serial_mass <= 1+ATOL and final_cone_excess <= ATOL and
              bool((mass >= -ATOL).all() and (mass <= 1+ATOL).all()) and read_max <= 1+ATOL)
    return {"pass": passed, "conditions": conditions,
            "retention_min": float(a.min()), "retention_max": float(a.max()),
            "write_min": float(w.min()), "write_max": float(w.max()),
            "retention_zero_count": int((a == 0).sum()), "retention_one_count": int((a == 1).sum()),
            "final_cone_excess": max(0., final_cone_excess),
            "serial_monitor_all_prefixes_cone_excess": max_serial_cone_excess,
            "serial_monitor_mass_min": min_serial_mass, "serial_monitor_mass_max": max_serial_mass,
            "normalized_read_max_abs": read_max, "evidence_min": float(evidence.min()),
            "evidence_max": float(evidence.max()),
            "serial_monitor_scope": "separate FP32 recurrence from actual trained offers; actual chunk final state and all reads checked separately"}


def rotary_observation(args, trajectory):
    a, theta, offer, initial = args
    finite = all(bool(torch.isfinite(x).all()) for x in (*args, trajectory))
    if not finite:
        return {"pass": False, "finite": False}
    envelope = initial.double().norm(dim=-1)
    envelopes = []
    for time in range(offer.shape[1]):
        retention = a[:, time, :, 0].double()
        envelope = retention*envelope + (1-retention)*offer[:, time].double().norm(dim=-1)
        envelopes.append(envelope)
    envelopes = torch.stack(envelopes, 1)
    norms = trajectory.double().norm(dim=-1)
    violation = float((norms-envelopes).max())
    norm_bound = math.sqrt(offer.shape[-1])
    valid = bool((initial == 0).all() and (offer.abs() <= 1).all() and ((a >= 0) & (a <= 1)).all())
    return {"pass": valid and violation <= ATOL and float(norms.max()) <= norm_bound+ATOL,
            "valid_zero_initial_and_bounded_offers": valid,
            "retention_min": float(a.min()), "retention_max": float(a.max()),
            "theta_min": float(theta.min()), "theta_max": float(theta.max()),
            "maximum_trajectory_l2": float(norms.max()), "sqrt_value_dimension_bound": norm_bound,
            "maximum_norm_envelope_excess": max(0., violation),
            "envelope_scope": "FP64 triangle-inequality envelope from actual represented learned coefficients and offers"}


def inspect_inference_state(model, tokens, owners):
    banks, fields = [], []
    bank_function, field_function = source_model.associative_read, source_model.rotary_scan
    def bank_wrapper(*args, **kwargs):
        result = bank_function(*args, **kwargs)
        banks.append({"layer": len(banks)//2, "bank": "fast" if len(banks)%2 == 0 else "slow",
                      **normalized_bank_observation(args, result)})
        return result
    def field_wrapper(*args, **kwargs):
        result = field_function(*args, **kwargs)
        fields.append({"layer": len(fields), **rotary_observation(args, result)})
        return result
    source_model.associative_read, source_model.rotary_scan = bank_wrapper, field_wrapper
    try:
        with torch.no_grad():
            _, state = model(tokens, owners=owners, mode="chunk")
    finally:
        source_model.associative_read, source_model.rotary_scan = bank_function, field_function
    storage_rows, unique = {}, {}
    for name, value in state_tensors(state).items():
        storage = value.untyped_storage()
        logical = value.numel()*value.element_size()
        physical = storage.nbytes()
        unique[(str(value.device), storage.data_ptr())] = physical
        storage_rows[name] = {"logical_tensor_bytes": logical, "underlying_storage_bytes": physical,
                              "compact": logical == physical}
    physical = sum(unique.values())
    compact = all(row["compact"] for row in storage_rows.values()) and physical == state.state_bytes()
    return {"pass": compact and all(row["pass"] for row in banks+fields),
            "bank_observations": banks, "rotary_observations": fields,
            "inference_state": {"clock": state.clock, "logical_tensor_bytes": state.state_bytes(),
                                "physical_unique_storage_bytes": physical, "compact": compact,
                                "per_tensor": storage_rows,
                                "metadata_python_overhead": "EXCLUDED_UNMEASURED"}}


def verify_case(item, tokens, targets):
    model, arm, seed = item["model"], item["arm"], item["seed"]
    model.eval()
    owners = (f"trained-TRAIN-parity-{arm}-{seed}",)
    before, before_all = parameter_digest(model), whole_model_digest(model)
    chunk, chunk_state = model(tokens, owners=owners, mode="chunk")
    recurrent, recurrent_state = model(tokens, owners=owners, mode="recurrent")
    logits = compare(chunk, recurrent)
    metadata_equal = (chunk_state.owners == recurrent_state.owners and
                      chunk_state.clock == recurrent_state.clock and
                      chunk_state.config_signature == recurrent_state.config_signature)
    states = {name: compare(value, state_tensors(recurrent_state)[name])
              for name, value in state_tensors(chunk_state).items()}
    loss_chunk = F.cross_entropy(chunk.flatten(0, 1), targets.flatten())
    loss_recurrent = F.cross_entropy(recurrent.flatten(0, 1), targets.flatten())
    parameters = dict(model.named_parameters())
    chunk_gradients = torch.autograd.grad(loss_chunk, tuple(parameters.values()), allow_unused=True)
    recurrent_gradients = torch.autograd.grad(loss_recurrent, tuple(parameters.values()), allow_unused=True)
    gradients = {}
    for name, left, right in zip(parameters, chunk_gradients, recurrent_gradients):
        connected = left is not None and right is not None
        gradients[name] = {"connected": connected,
                           **(compare(left, right) if connected else {"pass": False})}
    inference = inspect_inference_state(model, tokens, owners)
    unchanged = before == parameter_digest(model) and before_all == whole_model_digest(model)
    passed = (logits["pass"] and metadata_equal and all(row["pass"] for row in states.values()) and
              all(row["pass"] for row in gradients.values()) and inference["pass"] and unchanged)
    return {"arm": arm, "seed": seed, "pass": passed,
            "checkpoint_integrity": item["checkpoint_integrity"],
            "parameter_sha256_before": before, "model_state_sha256_before": before_all,
            "weights_and_buffers_unchanged": unchanged,
            "logits": logits, "state_metadata_equal": metadata_equal,
            "all_carried_tensors": states, "all_named_parameter_gradients": gradients,
            "named_parameter_tensors_compared": len(gradients),
            "fixed_TRAIN_fixture_cross_entropy": {"chunk": float(loss_chunk.detach()),
                                                    "recurrent": float(loss_recurrent.detach())},
            "inference_invariants_and_storage": inference}


def run(*, run_dir, train_bin, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("refuse overwrite of original trained-parity evidence")
    protocol, protocol_sha = load_protocol()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    report = {"schema": "langdepth106.trained-TRAIN-serial-parity.v1",
              "status": "STARTED", "protocol_sha256": protocol_sha,
              "tolerance": {"atol": ATOL, "rtol": RTOL,
                            "pointwise_rule": "abs(chunk-recurrent) <= atol+rtol*abs(recurrent)",
                            "bound_violation_absolute_tolerance": ATOL,
                            "selected_before_reading_results": True},
              "fixture": {"batch": 1, "observed_input_bytes": PREFIX_LENGTH,
                          "raw_TRAIN_prefix_bytes": PREFIX_LENGTH+1},
              "optimizer_updates": 0, "dev_opened": False, "test_opened": False,
              "hardware": "CPU", "dtype": "float32", "torch_version": str(torch.__version__),
              "cpu_threads": 2, "source_files_sha256": protocol["implementation"]["source_files_sha256"],
              "verification_code_sha256": sha256_file(Path(__file__)), "cases": [],
              "scope": ["Six original final exploratory checkpoints; one fixed authentic TRAIN prefix per checkpoint",
                  "Chunk/recurrent paths share projections, local attention, residual/FFN and output implementation",
                  "Checks affine associative and scaled-rotation execution consistency under actually trained weights",
                  "No optimizer, held-out source access, new quality benchmark or capability/architecture-superiority claim",
                  "Finite FP32 checks and bounds do not prove all inputs, long horizons, mixed precision or GPU behavior"]}
    try:
        train_bin = Path(train_bin).resolve()
        if train_bin.name != "train.bin":
            raise ValueError("explicit isolated TRAIN input required")
        receipt = verify_training_stage(train_bin.parent)
        report["train_sha256"] = receipt["partitions"]["train"]["sha256"]
        if report["train_sha256"] != protocol["source"]["splits"]["TRAIN"]["sha256"]:
            raise ValueError("TRAIN identity differs from frozen protocol")
        with train_bin.open("rb") as stream:
            raw = stream.read(PREFIX_LENGTH+1)
        if len(raw) != PREFIX_LENGTH+1:
            raise ValueError("TRAIN prefix is incomplete")
        report["fixture"]["prefix_sha256"] = hashlib.sha256(raw).hexdigest()
        tokens = torch.tensor(list(raw[:-1]), dtype=torch.long)[None]
        targets = torch.tensor(list(raw[1:]), dtype=torch.long)[None]
        models = audit_final_checkpoints(Path(run_dir), protocol, protocol_sha)
        for item in models:
            try:
                case = verify_case(item, tokens, targets)
            except Exception as error:
                case = {"arm": item["arm"], "seed": item["seed"], "pass": False,
                        "error": {"type": type(error).__name__, "message": str(error)}}
            report["cases"].append(case)
        report["status"] = "PASS_FINITE_TRAINED_TRAIN_FIXTURE_ONLY" if (
            len(report["cases"]) == 6 and all(row["pass"] for row in report["cases"])) else "FAILED_PRESERVED_NO_TOLERANCE_CHANGE"
    except Exception as error:
        report["status"] = "FAILED_PRESERVED_NO_TOLERANCE_CHANGE"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]),
        "passed": sum(bool(row["pass"]) for row in report["cases"]), "output": str(output)}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(run_dir=args.run_dir, train_bin=args.train_bin, output=args.output)
    if result["status"] != "PASS_FINITE_TRAINED_TRAIN_FIXTURE_ONLY":
        raise SystemExit(1)
