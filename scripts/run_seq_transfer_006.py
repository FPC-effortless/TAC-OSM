#!/usr/bin/env python3
"""SEQ-TRANSFER-006: independent, TEST-sealed evolving-state pilots.

Task schema and entity routing are supplied; this is NOT the frozen DEV061.5 model.
Training uses exact intermediate-state labels, explicitly a symbolic teacher.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MAX_ENTITIES = 6
TRAIN_SEED, VALID_SEED, TEST_SEED = 690601, 690602, 690603
PILOT_SEEDS = (690610, 690611)
TASKS = ("late_overwrite", "repeated_toggle", "pair_xor")
ARMS = ("tac_gated", "routed_gru", "global_gru")
STEPS, BATCH, LR = 400, 48, .003
READS = 8
WIDTHS = {"tac_gated": 20, "routed_gru": 32, "global_gru": 40}


def _build_one(rng, length, count_entities, task_idx, desired):
    # Public IDs are randomized per sample; queried IDs are not given until readout.
    physical = rng.permutation(count_entities).astype(np.int64)
    a, b = 0, 1
    excluded = {a, b} if task_idx == 2 else {a}
    distractors = [i for i in range(count_entities) if i not in excluded]
    # Operations are SET0, SET1, TOGGLE, NOOP.
    entity = rng.choice(distractors, size=length).astype(np.int64)
    op = rng.choice(4, size=length, p=[.25, .25, .30, .20]).astype(np.int64)
    forced = {}
    if task_idx == 0:  # old fact contradicted by late SET
        forced[1] = (a, 1 - desired)
        forced[int(length * .55)] = (a, desired)
    elif task_idx == 1:  # a series of real parity-changing updates
        toggles = int(rng.integers(1, 5))
        forced[1] = (a, desired ^ (toggles & 1))
        places = np.linspace(max(3, int(length * .15)),
                             int(length * .57), toggles, dtype=int)
        for t in places:
            forced[int(t)] = (a, 2)
    else:  # pair XOR with two entities undergoing state updates
        a_initial = int(rng.integers(0, 2))
        forced[1] = (a, a_initial)
        forced[2] = (b, a_initial ^ desired)
        forced[int(length * .25)] = (a, 2)
        forced[int(length * .32)] = (b, 2)
        forced[int(length * .44)] = (a, 2)
        forced[int(length * .56)] = (b, 2)
    for t, (logical, operation) in forced.items():
        entity[t], op[t] = logical, operation
    last_query_write = max(forced)
    # Apply the public identity permutation to all records and final query.
    entity = physical[entity]
    qa, qb = int(physical[a]), int(physical[b if task_idx == 2 else a])
    state = np.zeros(MAX_ENTITIES, dtype=np.float32)
    teacher = np.zeros((length, MAX_ENTITIES), dtype=np.float32)
    for t, (e, operation) in enumerate(zip(entity, op)):
        if operation == 0:
            state[e] = 0.
        elif operation == 1:
            state[e] = 1.
        elif operation == 2:
            state[e] = 1. - state[e]
        teacher[t] = state
    answer = int(state[qa]) if task_idx != 2 else int(state[qa]) ^ int(state[qb])
    if answer != desired:
        raise AssertionError("Generator constructed wrong final state")
    return entity, op, teacher, qa, qb, answer, length - 1 - last_query_write


def generate(seed, n, lengths=(32, 64), entities=4, *, purpose):
    """No callable TEST generator exists in the research/pilot implementation."""
    allowed = {"train": {TRAIN_SEED}, "validation": {VALID_SEED, VALID_SEED + 10}}
    if purpose not in allowed or seed not in allowed[purpose] or seed == TEST_SEED:
        raise PermissionError("Unregistered source seed or sealed SEQ006 TEST")
    if entities < 3 or entities > MAX_ENTITIES or not lengths or min(lengths) < 24:
        raise ValueError("Unsupported source configuration")
    rng = np.random.default_rng(seed)
    maximum = max(lengths)
    ids = np.zeros((n, maximum), dtype=np.int64)
    ops = np.full((n, maximum), 3, dtype=np.int64)
    teacher = np.zeros((n, maximum, MAX_ENTITIES), dtype=np.float32)
    active = np.zeros((n, maximum), dtype=np.float32)
    q = np.zeros((n, 2), dtype=np.int64)
    task = np.zeros(n, dtype=np.int64)
    y = np.zeros(n, dtype=np.float32)
    lag = np.zeros(n, dtype=np.int64)
    for i in range(n):
        kind, desired = i % 3, (i // 3) % 2
        L = int(lengths[int(rng.integers(0, len(lengths)))])
        e, o, states, a, b, label, age = _build_one(rng, L, entities, kind, desired)
        ids[i, :L], ops[i, :L] = e, o
        teacher[i, :L] = states
        if L < maximum:
            teacher[i, L:] = states[-1]
        active[i, :L] = 1.
        q[i], task[i], y[i], lag[i] = (a, b), kind, label, age
    chunks = (ids.tobytes(), ops.tobytes(), q.tobytes(), y.tobytes(), active.tobytes())
    digest = hashlib.sha256(b"".join(chunks)).hexdigest()
    manifest = {"seed": seed, "purpose": purpose, "n": n,
                "lengths": list(lengths), "entity_count": entities,
                "sha256_public_inputs_and_labels": digest,
                "min_target_lag": int(lag.min()), "median_target_lag": float(np.median(lag)),
                "tasks": {name: {"count": int((task == k).sum()),
                                 "positive": int(((task == k) & (y == 1)).sum())}
                          for k, name in enumerate(TASKS)}}
    tensors = tuple(torch.from_numpy(x) for x in (ids, ops, active, teacher, q, task, y))
    return tensors, manifest


class StreamingAgent(nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        w = WIDTHS[arm]
        self.width = w
        self.operation = nn.Embedding(4, 12)
        self.cell = nn.GRUCell(12 + (MAX_ENTITIES if arm == "global_gru" else 0), w)
        if arm == "tac_gated":
            self.op_delta = nn.Sequential(nn.Linear(w + 12, w), nn.Tanh(),
                                          nn.Linear(w, w))
            self.update_gate = nn.Linear(w + 12, w)
        if arm == "global_gru":
            self.identity = nn.Embedding(MAX_ENTITIES, w)
        self.state_head = nn.Linear(w, 1)
        self.answer = nn.Sequential(nn.Linear(w * 2 + 3, w), nn.ReLU(),
                                    nn.Linear(w, 1))

    def forward(self, ids, ops, active, queries, tasks):
        B, L = ids.shape
        w = self.width
        if self.arm == "global_gru":
            h = self.operation.weight.new_zeros(B, w)
        else:
            bank = self.operation.weight.new_zeros(B, MAX_ENTITIES, w)
        checkpoints = set(int(t) for t in np.linspace(0, L - 1, READS))
        intermediate, used = [], []
        for t in range(L):
            e = ids[:, t]
            embedding = self.operation(ops[:, t])
            mask = active[:, t:t+1].bool()
            if self.arm == "global_gru":
                x = torch.cat((embedding, F.one_hot(e, MAX_ENTITIES).to(embedding.dtype)), -1)
                new = self.cell(x, h)
                h = torch.where(mask, new, h)
                state_bank = h[:, None, :] + self.identity.weight[None, :, :]
            else:
                gather_index = e[:, None, None].expand(-1, 1, w)
                previous = bank.gather(1, gather_index).squeeze(1)
                new = self.cell(embedding, previous)
                if self.arm == "tac_gated":
                    both = torch.cat((previous, embedding), -1)
                    gated = torch.sigmoid(self.update_gate(both))
                    new = new + gated * self.op_delta(both)
                new = torch.where(mask, new, previous)
                bank = bank.scatter(1, gather_index, new[:, None, :])
                state_bank = bank
            if t in checkpoints:
                intermediate.append(self.state_head(state_bank).squeeze(-1))
                used.append(t)
        if self.arm == "global_gru":
            state_bank = h[:, None, :] + self.identity.weight[None, :, :]
        j = queries[:, :, None].expand(-1, -1, w)
        chosen = state_bank.gather(1, j)
        task_vector = F.one_hot(tasks, len(TASKS)).to(chosen.dtype)
        features = torch.cat((chosen.flatten(1), task_vector), -1)
        return self.answer(features).squeeze(-1), torch.stack(intermediate, 1), used


def objective(model, batch):
    ids, ops, active, truth, q, tasks, labels = batch
    logits, intermediate, used = model(ids, ops, active, q, tasks)
    final_loss = F.binary_cross_entropy_with_logits(logits, labels)
    truth_at = truth[:, used]
    valid = active[:, used, None]
    aux = F.binary_cross_entropy_with_logits(intermediate, truth_at, reduction="none")
    aux = (aux * valid).sum() / valid.sum().clamp_min(1) / MAX_ENTITIES
    return final_loss + .25 * aux, final_loss, aux


@torch.inference_mode()
def evaluate(model, data):
    ids, ops, active, teacher, q, tasks, labels = data
    model.eval()
    predictions = []
    for low in range(0, len(ids), 96):
        high = low + 96
        scores, _, _ = model(ids[low:high], ops[low:high], active[low:high],
                             q[low:high], tasks[low:high])
        predictions.append((scores.sigmoid() >= .5).to(torch.float32))
    predictions = torch.cat(predictions)
    return {name: float((predictions[tasks == k] == labels[tasks == k])
                        .float().mean()) for k, name in enumerate(TASKS)}


def fit(arm, seed, train, validations, steps, root, *, save=False):
    torch.manual_seed(seed)
    agent = StreamingAgent(arm)
    opt = torch.optim.AdamW(agent.parameters(), lr=LR, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(seed + 90_000)
    progress = []
    start = time.monotonic()
    for step in range(steps):
        idx = torch.randint(0, len(train[0]), (BATCH,), generator=rng)
        batch = tuple(x[idx] for x in train)
        agent.train()
        loss, primary, aux = objective(agent, batch)
        if not bool(torch.isfinite(loss)):
            raise ArithmeticError(f"Nonfinite loss arm={arm} seed={seed} update={step}")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        if step in (0, steps // 2 - 1, steps - 1):
            progress.append({"update": step + 1, "loss": float(loss.detach()),
                             "final_loss": float(primary.detach()), "aux_loss": float(aux.detach())})
    scores = {name: evaluate(agent, dat) for name, dat in validations.items()}
    record = {"arm": arm, "seed": seed, "updates": steps,
              "parameters": sum(p.numel() for p in agent.parameters()),
              "validation": scores, "progress": progress,
              "cpu_elapsed_seconds": round(time.monotonic() - start, 3)}
    if save:
        destination = root / "checkpoints" / f"{arm}-{seed}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arm": arm, "seed": seed, "weights": agent.state_dict(),
                    "optimizer": opt.state_dict(), "record": record}, destination)
        record["checkpoint"] = str(destination.relative_to(root))
        record["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("smoke", "pilot"), default="smoke")
    p.add_argument("--output", default="results/seq006")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    np.random.seed(TRAIN_SEED)
    random.seed(TRAIN_SEED)
    root = Path(args.output)
    if args.stage == "smoke":
        training_n, validation_n, iterations, seeds = 36, 36, 2, (PILOT_SEEDS[0],)
    else:
        training_n, validation_n, iterations, seeds = 3000, 600, STEPS, PILOT_SEEDS
    train, train_manifest = generate(TRAIN_SEED, training_n, (32, 64), 4, purpose="train")
    v64, manifest64 = generate(VALID_SEED, validation_n, (64,), 4, purpose="validation")
    v128, manifest128 = generate(VALID_SEED + 10, validation_n, (128,), 4, purpose="validation")
    manifest = {"study": "SEQ-TRANSFER-006", "status": "SOFTWARE_SMOKE_ONLY"
                if args.stage == "smoke" else "VALIDATION_PILOTS_ONLY",
                "test_opened": False, "test_evaluations": 0,
                "training": train_manifest, "validation": {"L64_E4": manifest64,
                                                          "L128_E4": manifest128},
                "training_steps": iterations, "fits": []}
    for seed in seeds:
        for arm in ARMS:
            result = fit(arm, seed, train, {"L64_E4": v64, "L128_E4": v128},
                         iterations, root, save=args.stage == "pilot")
            manifest["fits"].append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    out = root / f"{args.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(f"wrote={out} test_opened=False", flush=True)


if __name__ == "__main__":
    main()
