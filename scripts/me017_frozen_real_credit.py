#!/usr/bin/env python3
"""ME017: frozen, genuine-FIT gradient and memory-credit audit.

Reuses the original 12 hash-verified ME016 checkpoints. Nothing is trained.
Original FIT-only labels measure gradients and counterfactual *state ablations*
(diagnostic perturbations only, NOT real outcomes from an intervention).
Never reopens the exposed user1/user2/user3 future cohorts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.me014b_exact_action_memory import (
    FIT_CUT, HISTORY, VOCAB, batch, build_examples, parameters, read_early,
)
from scripts.me016_native_slow_identity import (
    ARMS, EngineeredAgent, SEEDS, SOURCE_SHA, PIN_MD5,
)

EXPECTED_ARMS = (
    "native_chrono_anchor", "native_identity_anchor",
    "gru_anchor", "gru_original",
)
TRAIN_SHA = "6870e254b65b74e8a55a242a3528aa478bf7bd08c6812ce92bf32a87fd4ba416"
SAMPLE = 384
MINIBATCH = 96


def load_frozen(original):
    root = Path(original)
    manifest_path = root / "results.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != TRAIN_SHA:
        raise ValueError("original ME016 training manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (manifest["study"] != "ME016_NATIVE_SLOW_IDENTITY_CREDIT_DIAGNOSIS"
            or manifest["source"]["zip_sha256"] != SOURCE_SHA
            or len(manifest["models"]) != 12
            or manifest["fixed_optimizer_updates"] != 600
            or manifest["real_train_examples"] != 68825
            or not manifest["all_prior_user1_user2_and_future_user3_labels_excluded"]):
        raise ValueError("wrong trained real source/protocol")
    models = []
    for info in manifest["models"]:
        path = root / info["checkpoint"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["checkpoint_sha256"]:
            raise ValueError("checkpoint changed BEFORE real data is parsed")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if (ck["kind"], ck["seed"], ck["steps"], ck["source_sha256"]) != (
                info["model"], info["seed"], 600, SOURCE_SHA):
            raise ValueError("wrong frozen model identity")
        net = EngineeredAgent(info["model"])
        net.load_state_dict(ck["state_dict"], strict=True)
        if parameters(net) != 135050 or info["trainable_params"] != 135050:
            raise ValueError("parameter equality not verified")
        net.eval()
        models.append((info, net))
    if {(v["model"], v["seed"]) for v, _ in models} != {
            (a, s) for a in EXPECTED_ARMS for s in SEEDS}:
        raise ValueError("incomplete original 12-model matrix")
    return models


def replay(net, items, metadata, action, duration):
    """Exact trained forward, with separate differentiable history input leaf.

    Partial derivatives wrt x measure neural history sensitivity; the explicit
    analytical user-history logit bypasses x and is reported separately.
    """
    core = net.core
    x = torch.cat((core.emb(items), metadata), -1).detach().requires_grad_(True)
    z_states, slow_states = [], []
    fast_gate, slow_gate = [], []
    if core.arm == "gru":
        outputs, _ = core.recur(x)
        state = outputs[:, -1]
        base = core.head(torch.cat((state, core.emb(action), duration), -1))
        refine = torch.cat((state[:, :8], core.emb(action)), -1)
        base = base + torch.tanh(core.action_refine(refine)).mean(-1, keepdim=True)
        fast_only = slow_only = None
    else:
        z = x.new_zeros((len(items), 32))
        s = z.clone()
        for t in range(HISTORY):
            u = x[:, t]
            proposal = torch.tanh(core.enc(u) + core.transition(z))
            gf = torch.sigmoid(core.gate(torch.cat((u, z), -1)))
            z = (1 - gf) * z + gf * proposal
            context = torch.cat((z, s), -1)
            gs = torch.sigmoid(core.slow_gate(context))
            candidate = core.slow(context)
            if net.kind == "native_identity_anchor":
                candidate = candidate + z
            s = (1 - gs) * s + gs * torch.tanh(candidate)
            z_states.append(z)
            slow_states.append(s)
            fast_gate.append(gf)
            slow_gate.append(gs)
        action_features = torch.cat((core.emb(action), duration), -1)
        base = core.head(torch.cat((z, s, action_features), -1))
        fast_only = core.head(torch.cat((z, torch.zeros_like(s), action_features), -1))
        slow_only = core.head(torch.cat((torch.zeros_like(z), s, action_features), -1))
    if net.anchored:
        count = metadata[:, :, 1].sum(-1, keepdim=True)
        prior = (count + 1) / (HISTORY + 2)
        anchor = torch.logit(prior)
    else:
        anchor = torch.zeros_like(base)
    return {
        "logit": base + anchor, "x": x, "z": z_states, "s": slow_states,
        "fast_gate": fast_gate, "slow_gate": slow_gate,
        "fast_only": None if fast_only is None else fast_only + anchor,
        "slow_only": None if slow_only is None else slow_only + anchor,
        "anchor": anchor,
    }


def rms_time_grad(grad):
    # Shape B,T,D. Per-time norm gives a direct differentiable sensitivity
    # measure; this is NOT a causal information-value or memory-utility metric.
    return torch.linalg.vector_norm(grad, dim=-1).mean(0)


def analyse_one(model, record, indices):
    all_x, all_z, all_s = [], [], []
    full, no_slow, no_fast, gatef, gates = [], [], [], [], []
    y_all = []
    started = time.perf_counter()
    for start in range(0, len(indices), MINIBATCH):
        ids = indices[start:start + MINIBATCH]
        items, metadata, action, duration = batch(record, ids)
        y = record["y"][ids]
        out = replay(model, items, metadata, action, duration)
        with torch.no_grad():
            reference = model(items, metadata, action, duration)
        if not torch.allclose(out["logit"], reference, atol=2e-6, rtol=2e-6):
            raise AssertionError("instrumented forward diverges from frozen production model")
        gradients = torch.autograd.grad(
            out["logit"].sum(),
            [out["x"]] + out["z"] + out["s"],
            allow_unused=True, retain_graph=False,
        )
        gx = gradients[0]
        if gx is None or not torch.isfinite(gx).all():
            raise AssertionError("nonfinite or missing real-input history gradients")
        all_x.append(rms_time_grad(gx).detach())
        nstates = len(out["z"])
        if nstates:
            zgrads = gradients[1:1 + nstates]
            sgrads = gradients[1 + nstates:]
            if any(v is None or not torch.isfinite(v).all() for v in zgrads + sgrads):
                raise AssertionError("missing native fast/slow state credit gradients")
            all_z.append(torch.stack(
                [torch.linalg.vector_norm(g, dim=-1).mean() for g in zgrads]))
            all_s.append(torch.stack(
                [torch.linalg.vector_norm(g, dim=-1).mean() for g in sgrads]))
            gatef.append(torch.stack(out["fast_gate"]).mean().detach())
            gates.append(torch.stack(out["slow_gate"]).mean().detach())
            no_slow.append(torch.sigmoid(out["fast_only"]).detach())
            no_fast.append(torch.sigmoid(out["slow_only"]).detach())
        full.append(torch.sigmoid(out["logit"]).detach())
        y_all.append(y.detach())
    p = torch.cat(full).reshape(-1)
    y = torch.cat(y_all).reshape(-1)
    def brier(v):
        return float((v.reshape(-1) - y).square().mean())
    mean_x = torch.stack(all_x).mean(0)
    if not torch.isfinite(mean_x).all():
        raise ArithmeticError("unstable native neural input credit")
    result = {
        "observed_real_fit_windows": len(indices),
        "observed_true_Brier_training_diagnostic": brier(p),
        "input_logit_gradient_norm_at_lags": {
            "oldest_event": float(mean_x[0]),
            "quarter": float(mean_x[7]),
            "middle": float(mean_x[15]),
            "newest_event": float(mean_x[-1]),
        },
        "oldest_newest_input_gradient_ratio": float(
            mean_x[0] / mean_x[-1].clamp_min(1e-12)),
        "input_logit_gradient_curve_32": [float(v) for v in mean_x],
        "instrumented_forward_matches_frozen_model": True,
        "inference_and_gradient_wall_seconds": round(time.perf_counter()-started, 3),
    }
    if no_slow:
        ns, nf = torch.cat(no_slow), torch.cat(no_fast)
        mean_z = torch.stack(all_z).mean(0)
        mean_s = torch.stack(all_s).mean(0)
        result.update({
            "native_actual_mean_fast_gate": float(torch.stack(gatef).mean()),
            "native_actual_mean_slow_gate": float(torch.stack(gates).mean()),
            "fast_state_logit_gradient_curve_32": [float(v) for v in mean_z],
            "slow_state_logit_gradient_curve_32": [float(v) for v in mean_s],
            "slow_oldest_newest_gradient_ratio": float(
                mean_s[0]/mean_s[-1].clamp_min(1e-12)),
            "fast_oldest_newest_gradient_ratio": float(
                mean_z[0]/mean_z[-1].clamp_min(1e-12)),
            "slow_zeroed_Brier_change_on_real_fit_labels_DIAGNOSTIC_ONLY":
                brier(ns)-brier(p),
            "fast_zeroed_Brier_change_on_real_fit_labels_DIAGNOSTIC_ONLY":
                brier(nf)-brier(p),
            "slow_zeroed_mean_abs_probability_change": float((ns-p).abs().mean()),
            "fast_zeroed_mean_abs_probability_change": float((nf-p).abs().mean()),
        })
    return result


def run(original, output):
    # Security and provenance: digest all 12 ORIGINAL trained artifacts BEFORE
    # opening even the real early training events.
    torch.set_num_threads(1)
    frozen = load_frozen(original)
    users, source = read_early()
    if (source["zip_sha256"] != SOURCE_SHA
            or source["zip_md5"] != PIN_MD5
            or source["future_test_labels_parsed"]):
        raise ValueError("source signature/boundary changed")
    real_fit = build_examples(users, 0, FIT_CUT)
    if len(real_fit["y"]) != 68825 or any(
            uid % 16 != 0 for uid in real_fit["users"]):
        raise ValueError("unexpected authentic FIT cohort")
    if any(t >= FIT_CUT for t in real_fit["timestamps"]):
        raise ValueError("later source observations in real FIT")
    indices = torch.linspace(
        0, len(real_fit["y"])-1, steps=SAMPLE, dtype=torch.long
    ).unique()
    if len(indices) != SAMPLE:
        raise ValueError("failed fixed deterministic real source subset")
    report = {
        "study": "ME017_FROZEN_REAL_FIT_MEMORY_CREDIT_AUDIT",
        "source_sha256": SOURCE_SHA,
        "original_training_report_sha256": TRAIN_SHA,
        "original_training_run": 39412071932,
        "original_checkpoint_hashes_checked_before_real_data": True,
        "genuine_real_early_fit_only": True,
        "historical_dev_used": False,
        "users_1_2_3_and_future_test_scored": False,
        "model_training_updates": 0,
        "historical_fit_is_not_independent_success_evidence": True,
        "memory_knockout_is_diagnostic_not_real_intervention": True,
        "all_models_parameters": 135050,
        "models": [],
    }
    for info, model in frozen:
        result = analyse_one(model, real_fit, indices)
        report["models"].append({
            "arm": info["model"], "seed": info["seed"],
            "original_checkpoint_sha256": info["checkpoint_sha256"],
            "original_train_seconds": info["actual_training_seconds"],
            "diagnostics": result,
        })
        print(json.dumps({
            "arm": info["model"], "seed": info["seed"],
            "grad_old_new": result["oldest_newest_input_gradient_ratio"],
            "slow_zero_Brier_delta": result.get(
                "slow_zeroed_Brier_change_on_real_fit_labels_DIAGNOSTIC_ONLY"),
        }, sort_keys=True), flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ME017_FROZEN_REAL_CREDIT_DIAGNOSTICS",
        "all_12_matched_frozen_models": len(report["models"]),
        "no_new_model_fits": True, "no_new_future_source_exposure": True},sort_keys=True))
    return report


def preflight():
    # Synthetic/random inputs permitted exclusively for numerical software tests.
    torch.set_num_threads(1)
    torch.manual_seed(17017)
    ids = torch.randint(VOCAB,(4,HISTORY))
    meta = torch.rand(4,HISTORY,3)
    meta[:,:,1] = (meta[:,:,1] > 0.5).float()
    action = torch.randint(VOCAB,(4,))
    duration = torch.rand(4,1)
    for kind in EXPECTED_ARMS:
        net = EngineeredAgent(kind)
        check = replay(net,ids,meta,action,duration)
        true = net(ids,meta,action,duration)
        assert torch.allclose(check["logit"],true,atol=2e-6,rtol=2e-6)
        grad = torch.autograd.grad(check["logit"].sum(),check["x"])[0]
        assert grad.shape==(4,HISTORY,11)
        assert torch.isfinite(grad).all()
        assert parameters(net) == 135050
    print(json.dumps({"status":"ME017_NUMERICAL_ONLY_PREFLIGHT_PASS",
        "proves_real_world_performance":False,"models":4}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--original",default="results/me016")
    parser.add_argument("--out",default="results/me017/real_fit_credit_audit.json")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    preflight() if args.self_test else run(args.original,args.out)
