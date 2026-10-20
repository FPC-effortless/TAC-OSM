#!/usr/bin/env python3
"""ME019: genuine 32 vs 128 prior logged events with exact-capacity native TAC/GRUs.

Training targets/actions are IDENTICAL across all six arms for a seed.
Only the amount of real *causally past* context changes (intended intervention).
The Aug01-Aug14 DEV has previously been inspected. NO independent success claim.
Reserved later-time/new-user cohorts MUST NOT be opened by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from scripts.me014b_exact_action_memory import (
    ActionMemory, FIT_CUT, DEV_CUT, VOCAB, EMB, PIN_MD5,
    batch, binary_scores, parameters, read_early,
)
from scripts.me016_native_slow_identity import EngineeredAgent, SOURCE_SHA, SEEDS

HISTORY_LONG = 128
HISTORY_SHORT = 32
ARMS = ("native_chrono_anchor", "gru_original", "gru_anchor")
LENGTHS = (32, 128)
PARAMS = 135050
UPDATES = 300
BATCH = 128
FIT_MAX = 12000
DEV_MAX = 6000


class NativeExtended(nn.Module):
    """Unaltered native ME015 gated fast/slow recurrence at variable history T."""
    def __init__(self, history: int):
        super().__init__()
        core = EngineeredAgent("native_chrono_anchor").core
        self.history = history
        self.emb = core.emb
        self.enc = core.enc
        self.transition = core.transition
        self.gate = core.gate
        self.slow = core.slow
        self.slow_gate = core.slow_gate
        self.head = core.head

    def forward(self, items: torch.Tensor, metadata: torch.Tensor,
                action: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.emb(items), metadata), -1)
        z = torch.zeros((items.size(0), 32), device=x.device, dtype=x.dtype)
        s = torch.zeros_like(z)
        for t in range(self.history):
            u = x[:, t]
            proposal = torch.tanh(self.enc(u) + self.transition(z))
            g = torch.sigmoid(self.gate(torch.cat((u, z), -1)))
            z = (1.0-g)*z + g*proposal
            ctx = torch.cat((z, s), -1)
            gs = torch.sigmoid(self.slow_gate(ctx))
            s = (1.0-gs)*s + gs*torch.tanh(self.slow(ctx))
        raw = self.head(torch.cat((z, s, self.emb(action), duration), -1))
        # Prior only uses causally available newest 32, even for 128 history.
        prior = (metadata[:, -32:, 1].sum(-1, keepdim=True) + 1.0)/34.0
        return raw + torch.logit(prior)


class GRUExtended(nn.Module):
    """Same original 42-cell GRU and active 68-param refine control."""
    def __init__(self, history: int, anchored: bool):
        super().__init__()
        core = EngineeredAgent("gru_anchor" if anchored else "gru_original").core
        self.history = history
        self.anchored = anchored
        self.emb = core.emb
        self.recur = core.recur
        self.head = core.head
        self.action_refine = core.action_refine

    def forward(self, items: torch.Tensor, metadata: torch.Tensor,
                action: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.emb(items), metadata), -1)
        state = self.recur(x)[0][:, -1]
        logit = self.head(torch.cat((state, self.emb(action), duration), -1))
        refinement = torch.cat((state[:, :8], self.emb(action)), -1)
        logit = logit + torch.tanh(self.action_refine(refinement)).mean(-1, keepdim=True)
        if self.anchored:
            prior = (metadata[:, -32:, 1].sum(-1, keepdim=True)+1.0)/34.0
            logit = logit + torch.logit(prior)
        return logit


def make_model(arm: str, history: int, script_native: bool = True):
    if arm == "native_chrono_anchor":
        model = NativeExtended(history)
        if script_native:
            model = torch.jit.script(model)
    elif arm in ("gru_original", "gru_anchor"):
        model = GRUExtended(history, arm == "gru_anchor")
    else:
        raise ValueError("unregistered model arm")
    if parameters(model) != PARAMS:
        raise AssertionError("strict active parameter equality")
    return model


def original_128_windows(users, begin: float, end: float, limit: int):
    """Two-pass deterministic source-only striding; build only eligible real anchors.

    Exact same 128 original historical event records for all arms. At index 96,
    reset the temporal-gap feature to zero: then H32 is the identical suffix
    of H128 and has NO feature that refers to an event outside its own history.
    Current target is NEVER included among history features.
    """
    n = sum(sum(begin <= records[i][0] < end for i in range(HISTORY_LONG, len(records)))
            for records in users.values())
    if n < 1500:
        raise ValueError("not enough genuine H128 causal windows; cannot extrapolate")
    stride = max(1, math.ceil(n/limit))
    rows = []
    candidate = 0
    for uid, records in sorted(users.items()):
        for i in range(HISTORY_LONG, len(records)):
            ts, current, duration, label = records[i]
            if not begin <= ts < end:
                continue
            selected = candidate % stride == 0
            candidate += 1
            if not selected:
                continue
            prior = records[i-HISTORY_LONG:i]
            items = [event[1] for event in prior]
            metadata = []
            for j, event in enumerate(prior):
                gap_seconds = 0.0 if j in (0, HISTORY_LONG-HISTORY_SHORT) else max(
                    0.0, event[0]-prior[j-1][0])
                gap = min(8.0, max(0.0, math.log1p(gap_seconds/3600)))/8.0
                metadata.append([event[2], event[3], gap])
            rows.append((uid, ts, items, metadata, current, duration, label))
    if candidate != n or not rows:
        raise ValueError("source source-window counts changed across passes")
    records = {
        "items": torch.tensor([r[2] for r in rows], dtype=torch.long),
        "metadata": torch.tensor([r[3] for r in rows], dtype=torch.float32),
        "action": torch.tensor([r[4] for r in rows], dtype=torch.long),
        "duration": torch.tensor([[r[5]] for r in rows], dtype=torch.float32),
        "y": torch.tensor([[r[6]] for r in rows], dtype=torch.float32),
        "users": [r[0] for r in rows],
        "timestamps": [r[1] for r in rows],
        "eligible_target_count": n,
        "sampling_stride": stride,
    }
    if len(records["y"]) > limit:
        raise ValueError("invalid size ceiling")
    return records


def view(data, history):
    if history not in LENGTHS:
        raise ValueError("unregistered real history window")
    if history == HISTORY_LONG:
        return data
    return {**data, "items": data["items"][:, -32:].contiguous(),
            "metadata": data["metadata"][:, -32:, :].contiguous()}


def validate_pair(short, long):
    if short["items"].shape[1] != HISTORY_SHORT or long["items"].shape[1] != HISTORY_LONG:
        raise AssertionError("history length is not correct")
    if not torch.equal(short["items"], long["items"][:, -32:]):
        raise AssertionError("different real recent actions across histories")
    if not torch.equal(short["metadata"], long["metadata"][:, -32:, :]):
        raise AssertionError("different real recent feedback across histories")
    for key in ("action", "duration", "y"):
        if not torch.equal(short[key], long[key]):
            raise AssertionError("label/current action mismatch between history arms")
    if not ((long["metadata"][:, :, 1] == 0)|(long["metadata"][:, :, 1] == 1)).all():
        raise AssertionError("not genuine binary past feedback")
    if min(short["items"].flatten()) < 0 or max(long["items"].flatten()) >= VOCAB:
        raise AssertionError("illegal genuine item")
    if any(u % 16 != 0 for u in long["users"]):
        raise AssertionError("train cohort contaminates new users")
    return True


def eval_brier(model, data):
    model.eval()
    probs = []
    with torch.inference_mode():
        for i in range(0, len(data["y"]), 512):
            idx = slice(i, min(len(data["y"]), i+512))
            probs.append(torch.sigmoid(model(*batch(data, idx))).flatten().cpu())
    return binary_scores(torch.cat(probs).numpy(), data["y"].numpy())


def train_one(arm, history, seed, train, dev, root):
    torch.manual_seed(seed)
    eager_model = make_model(arm, history, script_native=False)
    model = (torch.jit.script(eager_model) if arm == "native_chrono_anchor"
             else eager_model)
    if parameters(model) != PARAMS:
        raise AssertionError("active parameter count changed in scripted model")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    sampler = torch.Generator().manual_seed(seed+300)
    start = time.perf_counter()
    trace = []
    for step in range(1, UPDATES+1):
        # Reset generator for each arm => exact same real window indices per seed.
        ids = torch.randint(len(train["y"]), (BATCH,), generator=sampler)
        logits = model(*batch(train, ids))
        loss = F.binary_cross_entropy_with_logits(logits, train["y"][ids])
        if not bool(torch.isfinite(loss)):
            raise ArithmeticError("nonfinite authentic loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(grad)):
            raise ArithmeticError("nonfinite actual recurrent gradients")
        optimizer.step()
        if step in (1, UPDATES//2, UPDATES):
            trace.append({"step":step,"fit_bce":float(loss.detach()),
                          "global_grad_norm":float(grad.detach())})
    train_s = time.perf_counter() - start
    dev_result = eval_brier(model, dev)
    checkpoint = root/"checkpoints"/f"{arm}-h{history}-seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arm":arm,"history":history,"seed":seed,
                "updates":UPDATES,"source_sha256":SOURCE_SHA,
                "state_dict":model.state_dict()}, checkpoint)
    return {
        "arm":arm,"seed":seed,"history":history,"active_params":PARAMS,
        "train_windows":len(train["y"]),"dev_windows":len(dev["y"]),
        "optimizer_updates":UPDATES,"batch_size":BATCH,
        "total_target_window_draws":UPDATES*BATCH,
        "train_seconds":round(train_s,3),
        "dev_historical_inspected_score":dev_result,
        "train_trace":trace,
        "trained_weights":str(checkpoint.relative_to(root)),
        "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_compute_matched_across_architectures":False,
        "source_future_labels_opened":False,
    }


def run(root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    users,source=read_early()
    if source["zip_sha256"] != SOURCE_SHA or source["zip_md5"] != PIN_MD5:
        raise ValueError("authentic published source signature mismatched")
    if source["future_test_labels_parsed"] or not source["later_rows_skipped_before_label_parsing"]:
        raise ValueError("original non-exposure rule violated")
    # Data anchors are fixed independently of model scores.
    train_long=original_128_windows(users,0,FIT_CUT,FIT_MAX)
    dev_long=original_128_windows(users,FIT_CUT,DEV_CUT,DEV_MAX)
    train_short=view(train_long,32)
    dev_short=view(dev_long,32)
    validate_pair(train_short,train_long)
    validate_pair(dev_short,dev_long)
    if min(train_long["timestamps"]) >= FIT_CUT or max(train_long["timestamps"]) >= FIT_CUT:
        raise ValueError("FIT chrono boundary")
    if min(dev_long["timestamps"]) < FIT_CUT or max(dev_long["timestamps"]) >= DEV_CUT:
        raise ValueError("DEV chrono boundary")
    prior = dev_long["metadata"][:,-32:,1].mean(-1)
    prior_brier = binary_scores(prior.numpy(),dev_long["y"].numpy())
    result = {
        "study":"ME019_GENUINE_128_VS_32_PAST_HISTORY_ORIGINAL_FIT_ONLY",
        "real_original_data_sha256":SOURCE_SHA,"train_sampled_windows":len(train_long["y"]),
        "development_sampled_windows":len(dev_long["y"]),
        "actual_eligible_fit_128":train_long["eligible_target_count"],
        "actual_eligible_dev_128":dev_long["eligible_target_count"],
        "fit_stride":train_long["sampling_stride"],
        "dev_stride":dev_long["sampling_stride"],
        "only_original_early_train_user_mod16":0,
        "original_labels_after_aug14_parsed":False,
        "user1_2_3_4_outcome_cohorts_opened":False,
        "dev_already_inspected_in_prior_research":True,
        "same_real_training_targets_and_actions_all_arms":True,
        "history_length_different_by_design":True,
        "same_last32_exact_inputs_in_both_history_lengths":True,
        "baseline_real_prior32_dev":prior_brier,
        "exact_all_arms_trainable_params":PARAMS,
        "optimizer_updates_per_arm":UPDATES,
        "training_wall_clock_budget_equal":False,
        "head_to_head_fully_compute_fair":False,
        "success_promoted":False,"models":[],
    }
    for seed in SEEDS:
        for history in LENGTHS:
            tr = train_short if history==32 else train_long
            dv = dev_short if history==32 else dev_long
            for arm in ARMS:
                row=train_one(arm,history,seed,tr,dv,root)
                result["models"].append(row)
                print(json.dumps({"arm":arm,"history":history,"seed":seed,
                    "dev_brier":row["dev_historical_inspected_score"]["brier"],
                    "train_seconds":row["train_seconds"]},sort_keys=True),flush=True)
    if len(result["models"]) != len(SEEDS)*len(LENGTHS)*len(ARMS):
        raise AssertionError("incomplete preregistered matrix")
    (root/"results.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"ME019_ORIGINAL_REAL_EARLY_HISTORY_FITS_COMPLETE",
        "models":len(result["models"]),"reserved_cohorts_opened":False,
        "independent_scientific_success":False},sort_keys=True))
    return result


def preflight():
    # Constructed data ONLY proves mathematical/software consistency, no model accuracy.
    torch.manual_seed(19019)
    torch.set_num_threads(1)
    ids=torch.randint(VOCAB,(3,128))
    meta=torch.rand(3,128,3)
    meta[:,:,1]=(meta[:,:,1]>.5).float()
    meta[:,96,2]=0
    action=torch.randint(VOCAB,(3,))
    duration=torch.rand(3,1)
    n_params={}
    for arm in ARMS:
        for history in LENGTHS:
            torch.manual_seed(123)
            model=make_model(arm,history,script_native=False)
            n_params[f"{arm}_{history}"]=parameters(model)
            logits=model(ids[:,-history:],meta[:,-history:],action,duration)
            if logits.shape!=(3,1) or not torch.isfinite(logits).all():
                raise AssertionError("nonnative computation")
            if history==32:
                torch.manual_seed(123)
                orig=EngineeredAgent(arm)
                expected=orig(ids[:,-32:],meta[:,-32:],action,duration)
                if not torch.allclose(logits,expected,atol=3e-6,rtol=3e-6):
                    raise AssertionError("new 32-step math != established trained architecture")
            y=torch.rand(3,1).round()
            F.binary_cross_entropy_with_logits(logits,y).backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise AssertionError("invalid gradient")
            if arm=="native_chrono_anchor":
                torch.manual_seed(123)
                scripted=make_model(arm,history,script_native=True)
                if not torch.allclose(scripted(ids[:,-history:],
                        meta[:,-history:],action,duration),logits,atol=3e-6,rtol=3e-6):
                    raise AssertionError("scripted native context regression")
    if set(n_params.values()) != {PARAMS}:
        raise AssertionError("all active capacity must be exactly equal")
    print(json.dumps({"status":"ME019_REAL_ONLY_CONTEXT_EXTENSION_MATH_PREFLIGHT",
        "all_active_parameter_counts":n_params,
        "original_32step_equivalence":True,
        "full_history_128_contains_exact_recent_32":True,
        "model_accuracy_evidence":False},sort_keys=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="results/me019")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    preflight() if args.self_test else run(args.out)
