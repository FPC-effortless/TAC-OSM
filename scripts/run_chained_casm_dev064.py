#!/usr/bin/env python3
"""DEV064 chained-control benchmark; training teacher is never a deployed policy.

Research preflight/pilot only: no TEST seed is accessible through this script.
See research/dev064-chained-casm/PROTOCOL.md before changing any setting.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TRAIN_SEED, VAL_SEED, TEST_SEED = 640641, 640642, 640643
TRAIN_N, VAL_N = 4096, 1024
STEPS, BATCH, LR = 1024, 128, 0.001
ARMS = ("casm", "no_casm", "gru", "attention", "mlp")
WIDTH = {"casm": 32, "no_casm": 48, "gru": 32, "attention": 16, "mlp": 48}


@dataclass(frozen=True)
class World:
    parent: tuple[int, int, int]
    costs: tuple[float, float, float]
    success: tuple[float, float, float]
    collateral: tuple[float, float, float]
    penalty: float

    @property
    def adjacency(self):
        result = np.zeros((3, 3), dtype=np.float32)
        for child, parent in enumerate(self.parent):
            if parent >= 0:
                result[parent, child] = 1.0
        return result


def world(rng):
    # Fresh permutations prevent stable node-index identities.
    order = list(rng.permutation(3))
    parent = [-1] * 3
    for i in range(1, 3):
        parent[order[i]] = order[i - 1]
    return World(tuple(parent),
                 tuple(float(v) for v in rng.uniform(0.4, 1.6, 3)),
                 tuple(float(v) for v in rng.uniform(0.72, 0.96, 3)),
                 tuple(float(v) for v in rng.uniform(0.10, 0.55, 3)),
                 float(rng.uniform(5.0, 9.0)))


def transition(w, state, action):
    """Return exact probabilities over the physical successor states."""
    state = tuple(state)
    if action == 0:
        return ((1.0, state),)
    target = action - 1
    parent = w.parent[target]
    if parent >= 0 and state[parent]:
        return ((1.0, state),)  # Blocked repair still costs money.
    children = [i for i, p in enumerate(w.parent) if p == target]
    child = children[0] if children else None
    results = {}
    def add(prob, after):
        if prob > 0:
            key = tuple(after)
            results[key] = results.get(key, 0.0) + prob
    if state[target]:
        repaired = list(state)
        repaired[target] = 0
        p = w.success[target]
        add(1 - p, state)
        if child is not None and not repaired[child]:
            harmed = repaired.copy()
            harmed[child] = 1
            add(p * w.collateral[target], harmed)
            add(p * (1 - w.collateral[target]), repaired)
        else:
            add(p, repaired)
    else:
        # Unnecessary intervention may damage an already-healthy child.
        if child is not None and not state[child]:
            harmed = list(state)
            harmed[child] = 1
            add(w.collateral[target], harmed)
            add(1 - w.collateral[target], state)
        else:
            add(1.0, state)
    assert abs(sum(results.values()) - 1.0) < 1e-8
    return tuple((prob, next_state) for next_state, prob in sorted(results.items()))


def oracle(w, state, remaining):
    """Finite-horizon public information reference, forbidden at neural inference."""
    @functools.lru_cache(None)
    def q(states, h):
        stopping = w.penalty * sum(states)
        if h == 0:
            return (stopping,) + (float("inf"),) * 3
        values = [stopping]
        for action in range(1, 4):
            v = w.costs[action - 1]
            v += sum(p * min(q(next_state, h - 1))
                     for p, next_state in transition(w, states, action))
            values.append(v)
        return tuple(values)
    return q(tuple(state), int(remaining))


def features(w, state, remaining):
    return np.concatenate([
        np.asarray(state, dtype=np.float32),
        w.adjacency.flatten(),
        np.asarray(w.costs, dtype=np.float32) / 2,
        np.asarray(w.success, dtype=np.float32),
        np.asarray(w.collateral, dtype=np.float32),
        np.asarray([remaining / 4], dtype=np.float32),
    ]).astype(np.float32)


def generate(seed, count, *, purpose):
    if purpose not in ("train", "validation"):
        raise PermissionError("TEST is sealed: no test generation in preflight/pilot")
    if seed == TEST_SEED:
        raise PermissionError("Reserved test namespace is sealed")
    rng = np.random.default_rng(seed)
    X = np.empty((count, 22), np.float32)
    Y = np.empty((count, 4), np.float32)
    greedy_gap = []
    examples = []
    for i in range(count):
        w = world(rng)
        state = tuple(int(z) for z in rng.integers(0, 2, 3))
        if not any(state):
            state = tuple(int(j == int(rng.integers(0, 3))) for j in range(3))
        remaining = int(rng.integers(2, 5))
        target = oracle(w, state, remaining)
        greedy = [w.penalty * sum(state)]
        for a in range(1, 4):
            greedy.append(w.costs[a - 1] + sum(
                p * w.penalty * sum(s) for p, s in transition(w, state, a)))
        greedy_action = int(np.argmin(greedy))
        greedy_gap.append(target[greedy_action] - min(target))
        X[i], Y[i] = features(w, state, remaining), np.asarray(target, np.float32)
        if i < 6:
            examples.append(dict(parent=w.parent, state=state, budget=remaining,
                                 oracle_action=int(np.argmin(target)),
                                 greedy_action=greedy_action))
    return torch.from_numpy(X), torch.from_numpy(Y), {
        "count": count, "seed": seed, "purpose": purpose,
        "oracle_vs_one_step_greedy_expected_regret": float(np.mean(greedy_gap)),
        "oracle_vs_greedy_nonzero_fraction": float(np.mean(np.array(greedy_gap) > 1e-6)),
        "examples": examples,
        "x_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
        "y_sha256": hashlib.sha256(Y.tobytes()).hexdigest(),
    }


def node_inputs(x):
    state = x[:, :3, None]
    edges = x[:, 3:12].reshape(-1, 3, 3)
    costs = x[:, 12:15, None]
    rates = x[:, 15:18, None]
    harm = x[:, 18:21, None]
    inbound = edges.transpose(1, 2)
    outgoing = edges
    time = x[:, 21:22, None].expand(-1, 3, -1)
    z = torch.cat((state, costs, rates, harm, inbound, outgoing, time), -1)
    assert z.shape[-1] == 11
    return z, edges


class CASM(nn.Module):
    """Two rounds of learned operator composition with supplied public adjacency."""
    def __init__(self, width):
        super().__init__()
        self.inp = nn.Linear(11, width)
        self.ops = nn.ModuleList(nn.Sequential(nn.Linear(width, width),
                                               nn.Tanh()) for _ in range(3))
        self.selector = nn.Linear(width, 3)
        self.head = nn.Sequential(nn.Linear(width * 3, width), nn.ReLU(),
                                  nn.Linear(width, 4))

    def forward(self, x):
        z, graph = node_inputs(x)
        h = torch.tanh(self.inp(z))
        for _ in range(2):
            messages = torch.bmm(graph.transpose(1, 2), h)
            sources = h + messages
            mixture = self.selector(h.mean(1)).softmax(-1)
            choices = torch.stack([op(sources) for op in self.ops], 1)
            h = h + (choices * mixture[:, :, None, None]).sum(1)
        return self.head(h.flatten(1))


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(22, width), nn.ReLU(),
                               nn.Linear(width, width), nn.ReLU(),
                               nn.Linear(width, 4))
    def forward(self, x):
        return self.m(x)


class NoCASM(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.local = nn.Linear(11, width)
        self.refine = nn.Sequential(nn.Linear(width * 3, width), nn.ReLU(),
                                    nn.Linear(width, width), nn.ReLU())
        self.head = nn.Linear(width, 4)
    def forward(self, x):
        z, _ = node_inputs(x)
        h = torch.tanh(self.local(z)).flatten(1)
        return self.head(self.refine(h))


class GRU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gru = nn.GRU(11, width, batch_first=True)
        self.head = nn.Linear(width, 4)
    def forward(self, x):
        z, _ = node_inputs(x)
        _, h = self.gru(z)
        return self.head(h[0])


class Attention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(11, width)
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(width * 3, width), nn.ReLU(),
                                  nn.Linear(width, 4))
    def forward(self, x):
        z, _ = node_inputs(x)
        z = self.proj(z)
        z = z + self.attn(z, z, z, need_weights=False)[0]
        return self.head(z.flatten(1))


def model(arm):
    return {"casm": CASM, "no_casm": NoCASM,
            "gru": GRU, "attention": Attention, "mlp": MLP}[arm](WIDTH[arm])


def losses(pred, true):
    return F.smooth_l1_loss(pred, true / 10) + F.cross_entropy(-pred, true.argmin(-1))


def fit(arm, seed, train, val, steps):
    torch.manual_seed(seed)
    m = model(arm).train()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 1_000_000)
    xs, qs = train
    for step in range(steps):
        indices = torch.randint(0, len(xs), (BATCH,), generator=gen)
        out = m(xs[indices])
        loss = losses(out, qs[indices])
        if not torch.isfinite(loss):
            raise ArithmeticError(f"nonfinite {arm} seed={seed} step={step}")
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    vx, vq = val
    with torch.inference_mode():
        pred = m(vx)
        actions = pred.argmin(-1)
        selected_regret = (vq.gather(1, actions[:, None]).squeeze(1) -
                           vq.min(-1).values)
        metrics = {
            "arm": arm, "seed": seed, "steps": steps,
            "parameters": sum(p.numel() for p in m.parameters()),
            "val_action_accuracy": float((actions == vq.argmin(-1)).float().mean()),
            "val_expected_regret": float(selected_regret.mean()),
            "val_fault_free_action_rate": None,
            "val_loss": float(losses(pred, vq)),
        }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--output", default="results/dev064-chained")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(640641)
    np.random.seed(640641)
    train_x, train_q, train_info = generate(TRAIN_SEED, TRAIN_N, purpose="train")
    val_x, val_q, val_info = generate(VAL_SEED, VAL_N, purpose="validation")
    # This is a diagnostic on disjoint validation data only; no TEST generator exists here.
    params = {arm: sum(p.numel() for p in model(arm).parameters()) for arm in ARMS}
    summary = {
        "study": "DEV064-CHAINED-CASM",
        "stage": args.stage, "status": "PREFLIGHT_ONLY" if args.stage == "smoke" else "VALIDATION_PILOTS_ONLY",
        "test_opened": False, "training": train_info, "validation": val_info,
        "parameters": params, "fits": [],
    }
    # Pilot fits are exploratory entry screening; never choose a test winner here.
    seeds = (640650,) if args.stage == "smoke" else (640650, 640651)
    steps = 8 if args.stage == "smoke" else STEPS
    for seed in seeds:
        for arm in ARMS:
            record = fit(arm, seed, (train_x, train_q), (val_x, val_q), steps)
            summary["fits"].append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    target = Path(args.output) / (args.stage + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"recorded={target} test_opened=False", flush=True)


if __name__ == "__main__":
    main()
