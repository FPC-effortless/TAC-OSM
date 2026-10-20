#!/usr/bin/env python3
"""ME018: identical native TAC arithmetic with scripted recurrent execution.

Real KuaiRec original FIT-only data; immutable three ME016 engineered TAC
checkpoints. No retraining, model selection, new future sources or model claims.
Compare Python eager vs TorchScript execution with exact active parameters.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from scripts.me014b_exact_action_memory import (
    FIT_CUT, HISTORY, VOCAB, PIN_MD5, batch, build_examples, parameters, read_early,
)
from scripts.me016_native_slow_identity import EngineeredAgent, SEEDS, SOURCE_SHA
from scripts.me017_frozen_real_credit import TRAIN_SHA

ARM = "native_chrono_anchor"
MODEL_PARAMS = 135050
SAMPLE = 768
BATCH = 128
REPEATS = 7
INNER = 8


class NativeComputeKernel(nn.Module):
    """Exact native ME015 equations for trusted, prevalidated tensor batches.

    No third-party kernel, change to learned model, SSM, new parameters, or
    detached slow state. Distinguish input validation from numerical compute.
    """
    def __init__(self, original: EngineeredAgent):
        super().__init__()
        core = original.core
        self.emb = copy.deepcopy(core.emb)
        self.enc = copy.deepcopy(core.enc)
        self.transition = copy.deepcopy(core.transition)
        self.gate = copy.deepcopy(core.gate)
        self.slow = copy.deepcopy(core.slow)
        self.slow_gate = copy.deepcopy(core.slow_gate)
        self.head = copy.deepcopy(core.head)

    def forward(
        self, items: torch.Tensor, metadata: torch.Tensor,
        action: torch.Tensor, duration: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat((self.emb(items), metadata), dim=-1)
        z = torch.zeros((items.size(0), 32), device=x.device, dtype=x.dtype)
        s = torch.zeros_like(z)
        for t in range(32):
            u = x[:, t]
            q = torch.tanh(self.enc(u) + self.transition(z))
            gf = torch.sigmoid(self.gate(torch.cat((u, z), dim=-1)))
            z = (1.0 - gf) * z + gf * q
            context = torch.cat((z, s), dim=-1)
            gs = torch.sigmoid(self.slow_gate(context))
            s = (1.0 - gs) * s + gs * torch.tanh(self.slow(context))
        logits = self.head(torch.cat((z, s, self.emb(action), duration), dim=-1))
        count = metadata[:, :, 1].sum(dim=-1, keepdim=True)
        prior = (count + 1.0) / float(HISTORY + 2)
        return logits + torch.logit(prior)


def validate_inputs(values):
    items, meta, act, dur = values
    if items.ndim != 2 or items.shape[1] != HISTORY:
        raise ValueError("wrong real historical event window")
    if meta.shape != (len(items), HISTORY, 3):
        raise ValueError("wrong causal historical feedback")
    if act.shape != (len(items),) or dur.shape != (len(items), 1):
        raise ValueError("wrong current offered action")
    if not bool(((meta[:, :, 1] >= 0) & (meta[:, :, 1] <= 1)).all()):
        raise ValueError("invalid observed feedback")
    if int(items.min()) < 0 or int(items.max()) >= VOCAB or int(act.min()) < 0 or int(act.max()) >= VOCAB:
        raise ValueError("unknown video in source")
    if not torch.isfinite(meta).all() or not torch.isfinite(dur).all():
        raise ValueError("nonfinite authentic source features")


def fetch_original_models(archive):
    root = Path(archive)
    report = root / "results.json"
    if hashlib.sha256(report.read_bytes()).hexdigest() != TRAIN_SHA:
        raise ValueError("original ME016 600-update real weights ledger mismatch")
    manifest = json.loads(report.read_text())
    if (manifest["study"] != "ME016_NATIVE_SLOW_IDENTITY_CREDIT_DIAGNOSIS"
            or manifest["source"]["zip_sha256"] != SOURCE_SHA
            or manifest["real_train_examples"] != 68825
            or manifest["fixed_optimizer_updates"] != 600):
        raise ValueError("wrong genuine trained source/budget")
    result = []
    for row in manifest["models"]:
        if row["model"] != ARM:
            continue
        path = root / row["checkpoint"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["checkpoint_sha256"]:
            raise ValueError("trained model SHA changed before real data read")
        c = torch.load(path, weights_only=True, map_location="cpu")
        if (c["kind"], c["seed"], c["steps"], c["source_sha256"]) != (
                ARM, row["seed"], 600, SOURCE_SHA):
            raise ValueError("wrong original immutable checkpoint")
        model = EngineeredAgent(ARM)
        model.load_state_dict(c["state_dict"], strict=True)
        if parameters(model) != MODEL_PARAMS:
            raise ValueError("wrong active trainables")
        model.eval()
        result.append((row, model))
    if {r["seed"] for r, _ in result} != set(SEEDS):
        raise ValueError("not all three original seeded TAC checkpoints")
    return result


def compare_gradients(eager, scripted, values, target):
    eager.zero_grad(set_to_none=True)
    scripted.zero_grad(set_to_none=True)
    x = eager(*values)
    y = scripted(*values)
    if not torch.allclose(x, y, atol=3e-6, rtol=3e-6):
        raise AssertionError("scripted native logits not equal to exact eager native")
    F.binary_cross_entropy_with_logits(x, target).backward()
    F.binary_cross_entropy_with_logits(y, target).backward()
    eg = dict(eager.named_parameters())
    sg = dict(scripted.named_parameters())
    if eg.keys() != sg.keys():
        raise AssertionError("scripted native parameter names altered")
    maxdiff = 0.0
    for name in eg:
        a, b = eg[name].grad, sg[name].grad
        if a is None or b is None or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ArithmeticError("missing or nonfinite native gradient " + name)
        if not torch.allclose(a, b, atol=1e-5, rtol=1e-4):
            raise AssertionError("scripted backward diverged at " + name)
        maxdiff = max(maxdiff, float((a-b).abs().max()))
    return {"backprop_gradient_parity_pass": True, "max_gradient_absolute_difference": maxdiff}


def timed(func, rounds=REPEATS, inner=INNER):
    durations = []
    for trial in range(rounds):
        start = time.perf_counter()
        for _ in range(inner):
            func()
        durations.append((time.perf_counter()-start)/inner)
    return statistics.median(durations), durations


def measure_one(row, original, data, ids):
    values = batch(data, ids)
    target = data["y"][ids]
    validate_inputs(values)
    eager = NativeComputeKernel(original).eval()
    scripted = torch.jit.script(copy.deepcopy(eager)).eval()
    if sum(x.numel() for x in eager.parameters()) != MODEL_PARAMS:
        raise AssertionError("new compute path changed TAC capacity")
    if sum(x.numel() for x in scripted.parameters()) != MODEL_PARAMS:
        raise AssertionError("scripted compute path changed TAC capacity")
    with torch.no_grad():
        legacy = original(*values)
        baseline = eager(*values)
        jit_output = scripted(*values)
        if not torch.allclose(legacy, baseline, atol=3e-6, rtol=3e-6):
            raise AssertionError("candidate not equal to verified original native inference")
        if not torch.allclose(legacy, jit_output, atol=3e-6, rtol=3e-6):
            raise AssertionError("candidate script not equal to original TAC")
    gradient = compare_gradients(eager, scripted, values, target)
    eager.eval()
    scripted.eval()
    # Warm both paths then benchmark identical real input/output work; no CUDA,
    # no optimizer updates, same tensor CPU placement, no artificial sleeps.
    with torch.inference_mode():
        for _ in range(4):
            eager(*values)
            scripted(*values)
        # Paired balanced timed order to reduce systematic drift.
        def e(): return eager(*values)
        def s(): return scripted(*values)
        e_median, e_times = timed(e)
        s_median, s_times = timed(s)
        p = torch.sigmoid(jit_output).reshape(-1)
        brier = float(((p-target.reshape(-1))**2).mean())
    # Differentiate exact speed of core against legacy class input-validation overhead.
    with torch.inference_mode():
        old_median, old_times = timed(lambda: original(*values))
    def train_microstep(module):
        module.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(module(*values), target).backward()
    # Warmups do not mutate weights. The measured step has backward, no optimizer.
    for _ in range(2):
        train_microstep(eager)
        train_microstep(scripted)
    eager_train, eager_train_times = timed(lambda: train_microstep(eager), rounds=5, inner=3)
    jit_train, jit_train_times = timed(lambda: train_microstep(scripted), rounds=5, inner=3)
    return {
        "original_kind": ARM, "seed": row["seed"],
        "original_checkpoint_sha256": row["checkpoint_sha256"],
        "real_fit_diagnostic_rows": len(ids),
        "native_active_trainable_parameters": MODEL_PARAMS,
        "original_train_seconds": row["actual_training_seconds"],
        "same_genuine_features_and_targets": True,
        "new_model_training_updates": 0,
        "script_compile_excluded_from_steady_state_timing": True,
        "exact_original_forward_parity_pass": True,
        "max_logit_abs_error": float((legacy-jit_output).abs().max()),
        "diagnostic_real_fit_brier_unchanged": brier,
        **gradient,
        "infer_median_seconds_per_batch": {
            "legacy_validating_eager": old_median,
            "same_core_eager": e_median,
            "same_core_torchscript": s_median,
        },
        "forward_backward_no_optimizer_median_seconds_per_batch": {
            "same_core_eager": eager_train,
            "same_core_torchscript": jit_train,
        },
        "speed_ratio_eager_core_over_torchscript_infer": e_median/s_median,
        "speed_ratio_eager_core_over_torchscript_gradient": eager_train/jit_train,
        "raw_timing": {
            "legacy": old_times, "eager": e_times, "script": s_times,
            "eager_gradient": eager_train_times, "script_gradient": jit_train_times,
        },
    }


def run(original, output):
    torch.set_num_threads(1)
    models = fetch_original_models(original)  # all original hashes before source read
    people, source = read_early()
    if source["zip_md5"] != PIN_MD5 or source["zip_sha256"] != SOURCE_SHA:
        raise ValueError("publisher original source changed")
    if source["future_test_labels_parsed"]:
        raise ValueError("unopened future data should not be consumed")
    fit = build_examples(people, 0, FIT_CUT)
    if len(fit["y"]) != 68825 or any(ts >= FIT_CUT for ts in fit["timestamps"]):
        raise ValueError("original fit source window changed")
    indices = torch.linspace(0, len(fit["y"])-1, steps=SAMPLE, dtype=torch.long).unique()
    if len(indices) != SAMPLE:
        raise ValueError("incorrect registered source subset")
    # Pre-registered first 128 genuine windows of fixed 768-window subset.
    sample_ids = indices[:BATCH]
    report = {
        "study": "ME018_NATIVE_IDENTICAL_COMPUTE_REAL_FIT_ONLY",
        "source_sha256": SOURCE_SHA,
        "original_training_run": 39412071932,
        "all_original_checkpoint_hashes_verified_before_data": True,
        "model_weight_changes": 0,
        "source_future_or_independent_test_parsed": False,
        "scientific_world_model_claim": False,
        "timing_is_hardware_specific": True,
        "training_compute_match_with_gru_claimed": False,
        "models": [],
    }
    for row, model in models:
        result = measure_one(row, model, fit, sample_ids)
        report["models"].append(result)
        print(json.dumps({"seed": row["seed"],
              "infer_script_speed_ratio": result["speed_ratio_eager_core_over_torchscript_infer"],
              "gradient_script_speed_ratio": result["speed_ratio_eager_core_over_torchscript_gradient"],
              "original_logits_equal": True}, sort_keys=True),flush=True)
    out = Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"status": "ME018_REAL_DATA_IDENTICAL_NATIVE_INFERENCE_COMPUTE",
         "three_immutable_trained_models": len(report["models"]),
         "new_training": False, "new_future_test_used": False}))
    return report


def preflight():
    torch.set_num_threads(1)
    torch.manual_seed(18018)
    source = EngineeredAgent(ARM)
    with torch.no_grad():
        source.core.head.weight.normal_(std=.02)
    eager = NativeComputeKernel(source)
    scripted = torch.jit.script(copy.deepcopy(eager))
    items = torch.randint(VOCAB, (4,HISTORY))
    metadata = torch.rand(4,HISTORY,3)
    metadata[:,:,1] = (metadata[:,:,1]>.5).float()
    action = torch.randint(VOCAB,(4,))
    duration = torch.rand(4,1)
    values = (items, metadata, action, duration)
    validate_inputs(values)
    assert torch.allclose(source(*values),eager(*values),atol=3e-6,rtol=3e-6)
    y = torch.rand(4,1).round()
    grad = compare_gradients(eager, scripted, values, y)
    assert grad["backprop_gradient_parity_pass"]
    assert parameters(eager) == parameters(source) == MODEL_PARAMS
    print(json.dumps({"status":"ME018_NUMERICAL_ONLY_TORCHSCRIPT_PARITY_PASS",
        "native_equal_parameters": MODEL_PARAMS,
        "predictive_performance_claim": False,
        "gradient_absolute_maxdiff":grad["max_gradient_absolute_difference"]}))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",default="results/me016")
    p.add_argument("--out",default="results/me018/native_kernel_real_fit.json")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    preflight() if a.self_test else run(a.original,a.out)
