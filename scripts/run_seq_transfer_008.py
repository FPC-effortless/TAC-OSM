#!/usr/bin/env python3
"""SEQ008: randomized update timing / noncancelling relational state pilots.

Exploratory validation ONLY. Reserved TEST seeds are inaccessible in this module.
Uses the SEQ006 scratch recurrent controls, not pretrained DEV061.5 or neural CASM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_seq_transfer_006 import (
    ARMS, BATCH, LR, MAX_ENTITIES, STEPS, TASKS, StreamingAgent, evaluate,
)

TRAIN_SEED = 780801
VALIDATION_SEEDS = (780802, 780812)
RESERVED_TEST_SEEDS = (780803, 780813, 780823)
PILOT_SEEDS = (780820, 780821)
LABELS_PER_TASK = 100  # 600 balanced examples per validation stratum
OPERATIONS = (0, 1, 2, 3)  # SET0, SET1, TOGGLE, NOOP
ALLOWED_PURPOSES = {
    "train": (TRAIN_SEED,),
    "validation": VALIDATION_SEEDS,
}


def replay(ids, ops, length):
    """Independent deterministic reference state transition; never an input."""
    state = np.zeros(MAX_ENTITIES, dtype=np.float32)
    history = np.zeros((length, MAX_ENTITIES), dtype=np.float32)
    for t in range(length):
        e, operation = int(ids[t]), int(ops[t])
        if operation == 0:
            state[e] = 0
        elif operation == 1:
            state[e] = 1
        elif operation == 2:
            state[e] = 1 - state[e]
        elif operation != 3:
            raise ValueError("Unknown operation")
        history[t] = state
    return history


def _sample(rng, length, n_entities, kind):
    """Choose operations and target-event times before calculating their answer.

    No answer bit is inserted into an observation. All target events are placed
    at random unique times before a per-example cutoff, followed by irrelevant
    interleaved traffic. This does NOT test alias discovery or partial observability.
    """
    qa = int(rng.integers(n_entities))
    qb = qa
    if kind == 2:
        qb = int(rng.choice([i for i in range(n_entities) if i != qa]))
    targets = {qa, qb}
    distractors = [e for e in range(n_entities) if e not in targets]
    assert distractors
    cutoff = int(rng.integers(max(16, length // 2), min(length - 1, (length * 4) // 5) + 1))
    ids = rng.choice(distractors, size=length).astype(np.int64)
    ops = rng.choice(OPERATIONS, size=length, p=(.25, .25, .30, .20)).astype(np.int64)
    injected = []
    if kind == 0:
        count = int(rng.integers(3, 8))
        first_bit = int(rng.integers(2))
        assignments = [(qa, first_bit), (qa, 1 - first_bit)]
        assignments += [(qa, int(rng.integers(3))) for _ in range(count - 2)]
    elif kind == 1:
        count = int(rng.integers(4, 9))
        assignments = [(qa, int(rng.integers(2)))] + [(qa, 2)] * (count - 1)
    elif kind == 2:
        # Opposite toggle parities guarantee a noncancelling joint relation.
        odd = int(rng.choice((1, 3, 5)))
        even = int(rng.choice((2, 4)))
        n_a, n_b = (odd, even) if int(rng.integers(2)) else (even, odd)
        assignments = ([(qa, int(rng.integers(2)))] + [(qa, 2)] * n_a +
                       [(qb, int(rng.integers(2)))] + [(qb, 2)] * n_b)
    else:
        raise ValueError(kind)
    positions = sorted(int(v) for v in rng.choice(cutoff, size=len(assignments), replace=False))
    if kind == 2:
        # Each entity's first occurrence is SET and subsequent occurrences
        # are TOGGLE, regardless of interleaving between entities.
        counts = {qa: 0, qb: 0}
        bits = {qa: int(rng.integers(2)), qb: int(rng.integers(2))}
        assignments = []
        order = [qa] * (n_a + 1) + [qb] * (n_b + 1)
        rng.shuffle(order)
        for e in order:
            assignments.append((e, bits[e] if counts[e] == 0 else 2))
            counts[e] += 1
    for t, (e, op) in zip(positions, assignments):
        ids[t], ops[t] = e, op
        injected.append((t, e, op))
    truth = replay(ids, ops, length)
    answer = int(truth[-1, qa]) if kind != 2 else int(truth[-1, qa]) ^ int(truth[-1, qb])
    assert any(int(ids[t]) in targets for t in range(cutoff))
    assert all(int(ids[t]) not in targets for t in range(cutoff, length))
    # Both label classes arise from the outcome of the event stream, not
    # through a desired-label event injected at fixed positional offsets.
    return ids, ops, truth, qa, qb, answer, {
        "cutoff": cutoff, "injected": injected,
        "last_target_lag": length - 1 - max(t for t in range(length) if ids[t] in targets),
    }


def generate(seed, n, lengths=(32, 64), entities=4, *, purpose):
    if purpose not in ALLOWED_PURPOSES or seed not in ALLOWED_PURPOSES[purpose]:
        raise PermissionError("Only registered TRAIN/VALIDATION sources are callable")
    if seed in RESERVED_TEST_SEEDS:
        raise PermissionError("TEST is reserved, not exposed in exploratory pilot")
    if n % 6 != 0 or n < 6:
        raise ValueError("n must be a positive multiple of 6 (balanced tasks/labels)")
    if entities < 4 or entities > MAX_ENTITIES or not lengths or min(lengths) < 32:
        raise ValueError("Unsupported source shape")
    rng = np.random.default_rng(seed)
    max_len = max(lengths)
    ids = np.zeros((n, max_len), dtype=np.int64)
    ops = np.full((n, max_len), 3, dtype=np.int64)
    active = np.zeros((n, max_len), dtype=np.float32)
    history = np.zeros((n, max_len, MAX_ENTITIES), dtype=np.float32)
    query = np.zeros((n, 2), dtype=np.int64)
    task = np.zeros(n, dtype=np.int64)
    y = np.zeros(n, dtype=np.float32)
    lags, attempts = [], 0
    index = 0
    for kind in range(3):
        for required_label in (0, 1):
            accepted = 0
            while accepted < n // 6:
                attempts += 1
                length = int(rng.choice(lengths))
                e, o, h, qa, qb, answer, meta = _sample(rng, length, entities, kind)
                if answer != required_label:
                    continue
                ids[index, :length], ops[index, :length] = e, o
                active[index, :length] = 1
                history[index, :length] = h
                if length < max_len:
                    history[index, length:] = h[-1]
                query[index], task[index], y[index] = (qa, qb), kind, answer
                lags.append(meta["last_target_lag"])
                index += 1
                accepted += 1
    permutation = rng.permutation(n)
    arrays = [arr[permutation] for arr in (ids, ops, active, history, query, task, y)]
    ids, ops, active, history, query, task, y = arrays
    h = hashlib.sha256()
    for a in (ids, ops, active, query, task, y):
        h.update(a.tobytes())
    manifest = {
        "purpose": purpose, "seed": seed, "examples": n,
        "lengths": list(lengths), "entities": entities,
        "attempts": attempts, "sha256_inputs_labels": h.hexdigest(),
        "min_last_target_lag": int(min(lags)),
        "median_last_target_lag": float(np.median(lags)),
        "task_labels": {
            name: {"n": int(np.sum(task == j)),
                   "positive": int(np.sum((task == j) & (y == 1)))}
            for j, name in enumerate(TASKS)
        },
    }
    return tuple(torch.from_numpy(a) for a in arrays), manifest


def fit_final_only(arm, seed, training, validation, steps, root, *, save=False):
    torch.manual_seed(seed)
    model = StreamingAgent(arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sample_rng = torch.Generator().manual_seed(seed + 90_000)
    curves, start = [], time.monotonic()
    for step in range(steps):
        ix = torch.randint(0, len(training[0]), (BATCH,), generator=sample_rng)
        ids, ops, active, _truth, q, task, y = (a[ix] for a in training)
        model.train()
        logits, _aux, _times = model(ids, ops, active, q, task)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        if not bool(torch.isfinite(loss)):
            raise ArithmeticError(f"Non-finite loss {arm}/{seed}/{step}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step + 1 in (1, max(1, steps // 2), steps):
            curves.append({"step": step + 1, "final_loss": float(loss.detach())})
    scores = {name: evaluate(model, data) for name, data in validation.items()}
    row = {
        "arm": arm, "seed": seed, "steps": steps,
        "train_condition": "final_label_only",
        "parameters": sum(p.numel() for p in model.parameters()),
        "validation": scores, "curves": curves,
        "elapsed_cpu_host_wall_s": round(time.monotonic() - start, 3),
    }
    if save:
        file = root / "checkpoints" / f"{arm}-{seed}.pt"
        file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arm": arm, "seed": seed, "weights": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "row": row}, file)
        row["checkpoint"] = str(file.relative_to(root))
        row["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--output", default="results/seq008")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(args.output)
    if args.stage == "smoke":
        train_n, val_n, steps, seeds = 36, 36, 2, PILOT_SEEDS[:1]
    else:
        train_n, val_n, steps, seeds = 3000, 600, STEPS, PILOT_SEEDS
    training, mt = generate(TRAIN_SEED, train_n, (32, 64), 4, purpose="train")
    val64, m64 = generate(VALIDATION_SEEDS[0], val_n, (64,), 4, purpose="validation")
    val128, m128 = generate(VALIDATION_SEEDS[1], val_n, (128,), 4, purpose="validation")
    report = {
        "study": "SEQ-TRANSFER-008", "status": "EXPLORATORY_VALIDATION_ONLY",
        "test_opened": False, "test_evaluations": 0,
        "teacher_intermediate_states_consumed": False,
        "data": {"train": mt, "L64_E4": m64, "L128_E4": m128},
        "fits": [],
    }
    for seed in seeds:
        for arm in ARMS:
            row = fit_final_only(arm, seed, training,
                                 {"L64_E4": val64, "L128_E4": val128},
                                 steps, root, save=args.stage == "pilot")
            report["fits"].append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    target = root / (args.stage + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"saved={target}; test_opened=False", flush=True)


if __name__ == "__main__":
    main()
