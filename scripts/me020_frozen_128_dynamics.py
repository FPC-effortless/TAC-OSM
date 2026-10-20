#!/usr/bin/env python3
"""ME020: true real-FIT frozen native TAC recurrent Jacobian, gates and linear decodability.

Never train the original predictors or open previously exposed new-user outcome cohorts.
Source: original publisher-authenticated KuaiRec, uid%16==0, FIT timestamp <2020-08-01.
Frozen neural weights from original ME019 run 39443325909 are SHA-checked FIRST.
Small artificial numeric preflight is exclusively a Jacobian software unit test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.me014b_exact_action_memory import (
    FIT_CUT, PIN_MD5, batch, parameters, read_early
)
from scripts.me019_authentic_history import (
    ARMS, LENGTHS, PARAMS, SEEDS, SOURCE_SHA, FIT_MAX,
    make_model, original_128_windows, view
)

ORIGINAL_SHA = "2a10ae3205e866ed792f5ad145e3c3c775e95c39de675f421184de44e56455b6"
ORIGINAL_RUN = 39443325909
DEPTHS = (8, 32, 64, 96, 128)
OLD_TARGET = 8
# The original source-only sampled ME019 FIT has 10,776 authentic windows.
PROBE_LIMIT = 1200
GRAD_WINDOWS = 72
JAC_WINDOWS = 12
BATCH_SIZE = 12
RIDGE_LAMBDA = 1.0


def restore(original):
    root = Path(original)
    p = root / "results.json"
    if hashlib.sha256(p.read_bytes()).hexdigest() != ORIGINAL_SHA:
        raise ValueError("ME019 original raw result changed BEFORE touching source")
    ledger = json.loads(p.read_text())
    if (ledger["study"] != "ME019_GENUINE_128_VS_32_PAST_HISTORY_ORIGINAL_FIT_ONLY"
            or ledger["real_original_data_sha256"] != SOURCE_SHA
            or ledger["train_sampled_windows"] != 10776
            or ledger["optimizer_updates_per_arm"] != 300
            or ledger["exact_all_arms_trainable_params"] != PARAMS
            or len(ledger["models"]) != 18
            or ledger["user1_2_3_4_outcome_cohorts_opened"]):
        raise ValueError("not exact original real training manifest")
    # Verify EVERY original weight file before reading ANY genuine observed history.
    retained = {}
    for row in ledger["models"]:
        cp = root / row["trained_weights"]
        if hashlib.sha256(cp.read_bytes()).hexdigest() != row["checkpoint_sha256"]:
            raise ValueError("ME019 original individual checkpoint SHA mismatch")
        key = (row["arm"], row["history"], row["seed"])
        ck = torch.load(cp, map_location="cpu", weights_only=True)
        if (ck["arm"], ck["history"], ck["seed"], ck["updates"], ck["source_sha256"]) != (
                *key, 300, SOURCE_SHA):
            raise ValueError("checkpoint identity mismatch")
        # Register all 18 original digests, instantiate ONLY registered diagnostic arms.
        retained[key] = {"info": row, "weights": ck["state_dict"]}
    expected = {(a, h, s) for a in ARMS for h in LENGTHS for s in SEEDS}
    if set(retained) != expected:
        raise ValueError("incomplete original 18-model registry")
    selected = {}
    for (arm, length, seed), item in retained.items():
        if (arm == "native_chrono_anchor" or
                (arm == "gru_original" and length == 128)):
            net = make_model(arm, length, script_native=False)
            net.load_state_dict(item["weights"], strict=True)
            if parameters(net) != PARAMS:
                raise ValueError("original active parameter mismatch")
            net.eval()
            selected[(arm, length, seed)] = net
    return selected, ledger


def native_step(net, u, z, s, jacobian=False):
    """Exact coupled 64-differentiable-state native transition, not slow-only."""
    proposal_fast = torch.tanh(net.enc(u) + net.transition(z))
    gate_fast = torch.sigmoid(net.gate(torch.cat((u, z), dim=-1)))
    next_fast = (1.0-gate_fast)*z + gate_fast*proposal_fast
    context = torch.cat((next_fast, s), dim=-1)
    proposal_slow = torch.tanh(net.slow(context))
    gate_slow = torch.sigmoid(net.slow_gate(context))
    next_slow = (1.0-gate_slow)*s + gate_slow*proposal_slow
    if not jacobian:
        return next_fast, next_slow, gate_fast, gate_slow, None
    # Let A = d znew / d z, B = d snew / d znew, C = d snew / d s.
    # The actual coupled recurrent Jacobian is [[A,0],[B@A,C]], not C alone.
    eye = torch.eye(32, dtype=z.dtype, device=z.device).unsqueeze(0)
    A = ((1.0-gate_fast).unsqueeze(-1)*eye
         + (gate_fast*(1.0-proposal_fast.square())).unsqueeze(-1)
               * net.transition.weight.unsqueeze(0)
         + ((proposal_fast-z)*gate_fast*(1.0-gate_fast)).unsqueeze(-1)
               * net.gate.weight[:, 11:].unsqueeze(0))
    factor_s = gate_slow*(1.0-proposal_slow.square())
    factor_g = (proposal_slow-s)*gate_slow*(1.0-gate_slow)
    B = (factor_s.unsqueeze(-1)*net.slow.weight[:, :32].unsqueeze(0)
         + factor_g.unsqueeze(-1)*net.slow_gate.weight[:, :32].unsqueeze(0))
    C = ((1.0-gate_slow).unsqueeze(-1)*eye
         + factor_s.unsqueeze(-1)*net.slow.weight[:, 32:].unsqueeze(0)
         + factor_g.unsqueeze(-1)*net.slow_gate.weight[:, 32:].unsqueeze(0))
    zeros = torch.zeros_like(A)
    J = torch.cat([torch.cat([A, zeros], dim=-1),
                   torch.cat([torch.bmm(B,A), C], dim=-1)], dim=1)
    return next_fast, next_slow, gate_fast, gate_slow, J


def capture(net, values, full_states=False):
    """Replay exact frozen neural model; capture native states at registered depths."""
    items, meta, action, duration = values
    x = torch.cat((net.emb(items), meta), dim=-1)
    if hasattr(net, "transition"):
        z = x.new_zeros(len(items), 32)
        s = z.clone()
        snapshots = {}
        gates_f, gates_s = [], []
        norm_f, norm_s = [], []
        for t in range(items.shape[1]):
            z, s, gf, gs, _ = native_step(net, x[:,t], z, s)
            gates_f.append(gf)
            gates_s.append(gs)
            norm_f.append(z.norm(dim=1))
            norm_s.append(s.norm(dim=1))
            if (t+1) in DEPTHS:
                snapshots[t+1] = torch.cat((z, s), -1)
        base = net.head(torch.cat((z, s, net.emb(action), duration), dim=-1))
        anchor = torch.logit((meta[:,-32:,1].sum(-1,keepdim=True)+1)/34.0)
        logits = base + anchor
        return {"logits": logits, "x": x, "states": snapshots,
                "fast_gate": torch.stack(gates_f,1),
                "slow_gate": torch.stack(gates_s,1),
                "fast_norm": torch.stack(norm_f,1),
                "slow_norm": torch.stack(norm_s,1)}
    outputs = net.recur(x)[0]
    z = outputs[:,-1]
    logit = net.head(torch.cat((z,net.emb(action),duration),-1))
    refine = torch.cat((z[:,:8],net.emb(action)),-1)
    logit = logit + torch.tanh(net.action_refine(refine)).mean(-1,keepdim=True)
    if net.anchored:
        logit = logit + torch.logit((meta[:,-32:,1].sum(-1,keepdim=True)+1)/34.0)
    snapshots = {d:outputs[:,d-1] for d in DEPTHS}
    return {"logits":logit, "x":x, "states":snapshots}


def actual_input_gradient(net, values):
    items, meta, action, duration = values
    # Detached history leaf isolates neural recurrence from separate analytic prior,
    # as in ME017. Current target is NEVER an input.
    x = torch.cat((net.emb(items),meta),-1).detach().requires_grad_(True)
    if hasattr(net,"transition"):
        z=x.new_zeros(len(x),32); s=z.clone()
        for t in range(items.shape[1]):
            z,s,_,_,_=native_step(net,x[:,t],z,s)
        logit=net.head(torch.cat((z,s,net.emb(action),duration),-1))
    else:
        state=net.recur(x)[0][:,-1]
        logit=net.head(torch.cat((state,net.emb(action),duration),-1))
        refine=torch.cat((state[:,:8],net.emb(action)),-1)
        logit=logit + torch.tanh(net.action_refine(refine)).mean(-1,keepdim=True)
    gradient=torch.autograd.grad(logit.sum(),x)[0]
    if gradient.shape != x.shape or not torch.isfinite(gradient).all():
        raise ArithmeticError("nonfinite or missing true observed input-logit gradients")
    return gradient.detach().norm(dim=-1)


def gate_summary(arr):
    # Exact factual real-window numerical gate values, no proxies for overwrite.
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(unbiased=False)),
        "q01": float(torch.quantile(arr.reshape(-1),.01)),
        "q10": float(torch.quantile(arr.reshape(-1),.10)),
        "q50": float(torch.quantile(arr.reshape(-1),.50)),
        "q90": float(torch.quantile(arr.reshape(-1),.90)),
        "q99": float(torch.quantile(arr.reshape(-1),.99)),
        "fraction_gate_below_point01": float((arr<.01).float().mean()),
        "fraction_gate_above_point99": float((arr>.99).float().mean()),
    }


def frozen_gradient_and_gates(net, d):
    # All results on a frozen real original FIT subset, not independent performance.
    grad_accum=[]
    all_fast=[];all_slow=[];norm_fast=[];norm_slow=[]
    for st in range(0,GRAD_WINDOWS,BATCH_SIZE):
        ids=slice(st,min(GRAD_WINDOWS,st+BATCH_SIZE))
        values=batch(d,ids)
        with torch.no_grad():
            out=capture(net,values)
            exact=net(*values)
            if not torch.allclose(exact,out["logits"],atol=3e-6,rtol=3e-6):
                raise ValueError("native/GRU frozen original forward parity failed")
            if "fast_gate" in out:
                all_fast.append(out["fast_gate"])
                all_slow.append(out["slow_gate"])
                norm_fast.append(out["fast_norm"])
                norm_slow.append(out["slow_norm"])
        grad_accum.append(actual_input_gradient(net,values))
    grads=torch.cat(grad_accum,0)
    curve=grads.mean(0)
    result={
        "genuine_real_fit_windows":GRAD_WINDOWS,
        "frozen_original_forward_equal":True,
        "neural_only_input_gradient_curve":curve.tolist(),
        "mean_oldest8_neural_logit_grad":float(curve[:8].mean()),
        "mean_newest8_neural_logit_grad":float(curve[-8:].mean()),
        "oldest8_to_newest8_gradient_ratio":
            float(curve[:8].mean()/curve[-8:].mean().clamp_min(1e-12)),
    }
    if all_fast:
        gf=torch.cat(all_fast,0); gs=torch.cat(all_slow,0)
        result["gate_by_history_blocks"]=[]
        for i in range(0,gf.shape[1],32):
            result["gate_by_history_blocks"].append({
                "events_1based":[i+1,i+32],
                "fast":gate_summary(gf[:,i:i+32]),
                "slow":gate_summary(gs[:,i:i+32]),
                "mean_fast_norm":float(torch.cat(norm_fast,0)[:,i:i+32].mean()),
                "mean_slow_norm":float(torch.cat(norm_slow,0)[:,i:i+32].mean()),
            })
        result["fast_gate_global"]=gate_summary(gf)
        result["slow_gate_global"]=gate_summary(gs)
        # Diagonal direct carry-only factor, *not* true full derivative retention.
        result["last_to_first_direct_gate_only_fast_survival_log"] = float(
            torch.log1p(-gf.clamp(max=1-1e-8)).sum(dim=1).mean())
        result["last_to_first_direct_gate_only_slow_survival_log"] = float(
            torch.log1p(-gs.clamp(max=1-1e-8)).sum(dim=1).mean())
    return result


def jacobian_spectra(net, data):
    # Same first JAC_WINDOWS actual past contexts across every original model.
    d64 = torch.float64
    items, meta, action, duration = batch(data,slice(0,JAC_WINDOWS))
    with torch.no_grad():
        xx=torch.cat((net.emb(items),meta),-1).to(d64)
        # Original weights stay frozen and numerically unchanged in the research
        # model; a float64 copy is used solely to avoid product underflow.
        double_net = make_model("native_chrono_anchor",items.shape[1],script_native=False)
        double_net.load_state_dict(net.state_dict(),strict=True)
        double_net.double().eval()
        z=xx.new_zeros(JAC_WINDOWS,32)
        s=z.clone()
        product=torch.eye(64,dtype=d64).unsqueeze(0).repeat(JAC_WINDOWS,1,1)
        depths=[d for d in DEPTHS if d<=items.shape[1]]
        by_depth={}
        for t in range(items.shape[1]):
            z,s,gf,gs,J=native_step(double_net,xx[:,t],z,s,jacobian=True)
            product=torch.bmm(J,product)
            if t+1 in depths:
                sig_loc=torch.linalg.svdvals(J)
                sig_prod=torch.linalg.svdvals(product)
                by_depth[str(t+1)]={
                    "max_local_64d_jacobian_spectral_norm_mean":
                      float(sig_loc[:,0].mean()),
                    "max_cumulative_true_64d_jacobian_spectral_norm_mean":
                      float(sig_prod[:,0].mean()),
                    "min_cumulative_true_64d_singular_value_mean":
                      float(sig_prod[:,-1].mean()),
                    "log10_max_cumulative_spectral_norm_mean":
                      float(torch.log10(sig_prod[:,0].clamp_min(1e-300)).mean()),
                    "log10_min_cumulative_singular_value_mean":
                      float(torch.log10(sig_prod[:,-1].clamp_min(1e-300)).mean()),
                }
        return {
            "real_jacobian_windows":JAC_WINDOWS,
            "full_coupled_state_dimension":64,
            "uses_analytic_full_coupled_jacobian_not_slow_only":True,
            "actual_J128_times_through_J1_not_product_of_individual_norms":True,
            "float64_only_for_diagnostic_precision_not_training":True,
            "chronological_cumulative_jacobian_spectrum":by_depth
        }


def ridge_fit_predict(Xtrain,ytrain,Xtest):
    Xtrain=np.asarray(Xtrain,dtype=np.float64)
    ytrain=np.asarray(ytrain,dtype=np.float64).reshape(-1)
    Xtest=np.asarray(Xtest,dtype=np.float64)
    if Xtrain.ndim==1: Xtrain=Xtrain[:,None]
    if Xtest.ndim==1: Xtest=Xtest[:,None]
    mean=Xtrain.mean(0)
    std=Xtrain.std(0).clip(min=1e-3)
    tr=np.column_stack([np.ones(len(Xtrain)),(Xtrain-mean)/std])
    te=np.column_stack([np.ones(len(Xtest)),(Xtest-mean)/std])
    reg=np.eye(tr.shape[1])*RIDGE_LAMBDA
    reg[0,0]=0.0
    w=np.linalg.solve(tr.T@tr+reg,tr.T@ytrain)
    return te@w


def probe_task(fit, models, original_indices):
    """TRAIN/TEST probe split by REAL user identity, all within original FIT.

    The frozen original neural weights are NEVER changed. Probe target is original
    oldest 8 real watch responses. Second regression probes conditional old96
    relevance to current factual response, with recent32 user rate baseline.
    """
    ids=original_indices
    uid=np.asarray([fit["users"][int(v)] for v in ids],dtype=np.int64)
    train=(uid%5 != 4)
    test=~train
    if train.sum()<400 or test.sum()<90 or len(np.unique(uid[test]))<6:
        raise ValueError("not enough authentic disjoint FIT probe user identities")
    sel=ids.long()
    d={key:(v[sel] if torch.is_tensor(v) else v) for key,v in fit.items()
       if key in ("items","metadata","action","duration","y")}
    inputs=d["metadata"][:,:,1].numpy()
    early=inputs[:,:OLD_TARGET].mean(axis=1)
    recent=inputs[:,-32:].mean(axis=1)
    old96=inputs[:,:96].mean(axis=1)
    future_y=d["y"].flatten().numpy()
    # genuine FIT heldout users, old-history information test:
    p32=ridge_fit_predict(recent[train],future_y[train],recent[test])
    p_all=ridge_fit_predict(
        np.stack([recent[train],old96[train]],axis=1),future_y[train],
        np.stack([recent[test],old96[test]],axis=1))
    def mse(pred,target):return float(np.mean((np.asarray(pred)-target)**2))
    rel={
        "original_real_fit_probe_train_windows":int(train.sum()),
        "original_real_fit_probe_test_windows":int(test.sum()),
        "disjoint_train_test_user_ids":True,
        "unique_test_users":int(len(np.unique(uid[test]))),
        "future_outcome_Brier_recent32_ridge":mse(p32,future_y[test]),
        "future_outcome_Brier_recent32_plus_old96_ridge":mse(p_all,future_y[test]),
        "future_outcome_Brier_old96_increment":mse(p_all,future_y[test])-mse(p32,future_y[test]),
        "no_new_future_or_independent_cohort":True,
        "probe_is_exploratory_not_policy":True,
    }
    # Exactly same old8 target, real user partitions and ridge hyperparameters
    # for all frozen native and GRU models, depths 8,32,64,96,128.
    detail=[]
    for (arm,length,seed),net in models.items():
        if length != 128:
            continue
        snapshots={depth:[] for depth in DEPTHS}
        with torch.no_grad():
            for start in range(0,len(sel),128):
                batch_ids=slice(start,min(len(sel),start+128))
                record=capture(net,batch(d,batch_ids))
                for depth in DEPTHS:
                    snapshots[depth].append(record["states"][depth].cpu())
        for depth in DEPTHS:
            features=torch.cat(snapshots[depth]).numpy()
            predictions=ridge_fit_predict(features[train],early[train],features[test])
            base=np.full(int(test.sum()),float(early[train].mean()))
            last32_reference=(recent if depth>=32 else inputs[:,:depth].mean(1))
            control=ridge_fit_predict(last32_reference[train],
                                      early[train],last32_reference[test])
            detail.append({
                "arm":arm,"history":length,"seed":seed,"snapshot_depth":depth,
                "frozen_state_dimension":features.shape[1],
                "decoding_target":"mean genuine oldest8 observed full_watch in original H128 context",
                "trained_probe":"user-disjoint frozen-state ridge_only",
                "ridge_lambda":RIDGE_LAMBDA,
                "old8_state_probe_outofuser_MSE":mse(predictions,early[test]),
                "old8_constant_baseline_MSE":mse(base,early[test]),
                "old8_observable_rate_control_MSE":mse(control,early[test]),
                "state_vs_constant_delta_MSE":mse(predictions,early[test])-mse(base,early[test]),
                "original_model_weight_updates":0
            })
    return rel,detail


def run(original,output):
    torch.set_num_threads(1)
    started=time.perf_counter()
    # EXACT trained checkpoint ledger and 18 SHA verifications BEFORE data.
    models,ledger=restore(original)
    people,src=read_early()
    if (src["zip_sha256"]!=SOURCE_SHA or src["zip_md5"]!=PIN_MD5
            or src["future_test_labels_parsed"]):
        raise ValueError("changed genuine KuaiRec publisher source")
    fit=original_128_windows(people,0,FIT_CUT,FIT_MAX)
    if len(fit["y"])!=10776 or any(u%16!=0 for u in fit["users"]):
        raise ValueError("actual original genuine FIT changed")
    if any(t>=FIT_CUT for t in fit["timestamps"]):
        raise ValueError("future date contaminates original FIT")
    indices=torch.linspace(0,len(fit["y"])-1,PROBE_LIMIT,dtype=torch.long).unique()
    if len(indices)!=PROBE_LIMIT:
        raise ValueError("nonunique source-only frozen probe sample")
    d={key:(v[indices] if torch.is_tensor(v) else v) for key,v in fit.items()
        if key in ("items","metadata","action","duration","y")}
    short=view(d,32)
    report={
        "study":"ME020_FROZEN_TRUE_REAL_128_JACOBIAN_GATES_DECODABILITY",
        "original_training_run":ORIGINAL_RUN,
        "original_training_manifest_sha256":ORIGINAL_SHA,
        "all_18_original_checkpoint_hashes_verified_before_source":True,
        "original_source_SHA256":SOURCE_SHA,
        "actual_genuine_user0_early_FIT_only":True,
        "historically_opened_development_used":False,
        "user1_2_3_4_or_future_outcomes_reopened":False,
        "original_optimizer_updates":0,
        "new_native_architecture_created":False,
        "independent_predictive_or_commercial_claim":False,
        "all_trainable_original_params":PARAMS,
        "grad_real_windows":GRAD_WINDOWS,
        "jacobian_real_windows":JAC_WINDOWS,
        "probe_real_windows":len(indices),
        "models":[],
    }
    for key,net in models.items():
        arm,length,seed=key
        source=short if length==32 else d
        tele=frozen_gradient_and_gates(net,source)
        row={"arm":arm,"history":length,"seed":seed,
             "original_checkpoint_sha256":
                next(m["checkpoint_sha256"] for m in ledger["models"]
                    if (m["arm"],m["history"],m["seed"])==key),
             "telemetry":tele}
        if arm=="native_chrono_anchor":
            row["full_true_jacobian"]=jacobian_spectra(net,source)
        report["models"].append(row)
        print(json.dumps({"arm":arm,"history":length,"seed":seed,
            "grad_oldest8_newest8":tele["oldest8_to_newest8_gradient_ratio"],
            "slow_mean":tele.get("slow_gate_global",{}).get("mean")},sort_keys=True),flush=True)
    # Probe fitted only on existing original FIT observation rows/users;
    # this does not fit or select any of the frozen 18 actual predictors.
    rel, dec=probe_task(fit,models,indices)
    report["frozen_observational_old_history_relevance"]=rel
    report["frozen_linear_state_decodability"]=dec
    report["original_model_unchanged"]=True
    report["diagnostic_seconds"]=round(time.perf_counter()-started,2)
    out=Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"ME020_FROZEN_GENUINE_FIT_MATH_AUDIT_COMPLETED",
        "models":len(report["models"]),"decoders":len(dec),
        "user_future_source_opened":False,"weights_changed":False},sort_keys=True))
    return report


def preflight():
    torch.set_num_threads(1)
    torch.manual_seed(20020)
    net=make_model("native_chrono_anchor",128,script_native=False)
    u=torch.randn(1,11,dtype=torch.float64)
    z=torch.randn(1,32,dtype=torch.float64).tanh()
    s=torch.randn(1,32,dtype=torch.float64).tanh()
    net.double()
    zn,sn,gf,gs,analytic=native_step(net,u,z,s,jacobian=True)
    def scalar_native(state):
        zz,ss,_,_,_=native_step(net,u,state[:32].reshape(1,32),
                               state[32:].reshape(1,32))
        return torch.cat((zz,ss),-1).flatten()
    exact=torch.autograd.functional.jacobian(scalar_native,torch.cat((z,s),-1).flatten())
    err=float((exact-analytic[0]).abs().max())
    if not torch.allclose(exact,analytic[0],atol=5e-9,rtol=5e-8):
        raise AssertionError("analytic FULL coupled Jacobian incorrect")
    assert analytic.shape==(1,64,64)
    assert parameters(net)==PARAMS
    x=np.random.default_rng(0).normal(size=(60,4))
    y=x[:,0]*0.05+x[:,1]*0.1
    zpred=ridge_fit_predict(x[:40],y[:40],x[40:])
    assert zpred.shape==(20,) and np.isfinite(zpred).all()
    print(json.dumps({"status":"ME020_SYNTHETIC_ONLY_NUMERICAL_PREFLIGHT",
        "full_64_state_autodiff_jacobian_matches_analytic":True,
        "max_jacobian_absolute_error":err,
        "real_model_performance_claim":False},sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",default="results/me019")
    p.add_argument("--out",default="results/me020/frozen_genuine_fit_128_dynamics.json")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    preflight() if a.self_test else run(a.original,a.out)
