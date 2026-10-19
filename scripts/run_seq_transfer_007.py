#!/usr/bin/env python3
"""SEQ007 validation-only auxiliary state-label ablation; no TEST path exists."""
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
    ARMS, STEPS, BATCH, LR, READS,
    TRAIN_SEED, VALID_SEED, TASKS, StreamingAgent, evaluate, generate,
)

PILOTS = (690810, 690811)
CONDITIONS = {"final_only": 0.0, "dense_teacher": 0.25}
EXPECTED = {
    "train": "d7df850483e85fdf3380c4d80c87f3ad3e62e0ee2ea5eca98a01442b65168baf",
    "L64_E4": "743f20cc77d07ed5f8baededcb8d18899a8a8bfd49c2ce69ffbe6929e7b419f8",
    "L128_E4": "77851fe3b5017ada28716d1dbe0121da630c34d0a9e780024b40aadaaa537e3b",
}


def data():
    tr, mt = generate(TRAIN_SEED, 3000, (32, 64), 4, purpose="train")
    v64, m64 = generate(VALID_SEED, 600, (64,), 4, purpose="validation")
    v128, m128 = generate(VALID_SEED + 10, 600, (128,), 4, purpose="validation")
    for key, report in (("train", mt), ("L64_E4", m64), ("L128_E4", m128)):
        if report["sha256_public_inputs_and_labels"] != EXPECTED[key]:
            raise AssertionError("SEQ006 source provenance changed " + key)
    return tr, {"L64_E4": v64, "L128_E4": v128}, {
        "train": mt, "L64_E4": m64, "L128_E4": m128,
    }


def train(arm, condition, weight, seed, training, validations,
          steps, output, *, persist):
    torch.manual_seed(seed)
    agent = StreamingAgent(arm)
    opt = torch.optim.AdamW(agent.parameters(), lr=LR, weight_decay=1e-4)
    stream = torch.Generator().manual_seed(seed + 90000)
    start = time.monotonic()
    curves = []
    for step in range(steps):
        sample = torch.randint(0, len(training[0]), (BATCH,), generator=stream)
        ids, ops, active, truth, query, tasks, labels = (a[sample] for a in training)
        agent.train()
        logits, aux_logits, positions = agent(ids, ops, active, query, tasks)
        final = F.binary_cross_entropy_with_logits(logits, labels)
        if weight:
            active_at = active[:, positions, None]
            aux = F.binary_cross_entropy_with_logits(
                aux_logits, truth[:, positions], reduction="none")
            aux = (aux * active_at).sum() / active_at.sum().clamp_min(1) / 6
        else:
            aux = final.detach().new_zeros(())
        loss = final + weight * aux
        if not bool(torch.isfinite(loss)):
            raise ArithmeticError(f"Nonfinite loss {arm}/{condition}/{seed}/{step}")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        if step + 1 in (1, steps // 2, steps):
            curves.append({"update": step+1, "train_loss": float(loss.detach()),
                           "final_loss": float(final.detach()),
                           "aux_loss": float(aux.detach())})
    validation = {k: evaluate(agent, v) for k, v in validations.items()}
    row = {
        "arm": arm, "condition": condition, "state_teacher_weight": weight,
        "seed": seed, "updates": steps, "validation": validation,
        "parameters": sum(p.numel() for p in agent.parameters()),
        "curves": curves, "elapsed_wall_seconds": round(time.monotonic()-start, 3),
    }
    if persist:
        checkpoint = output / "checkpoints" / f"{arm}-{condition}-{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arm": arm, "condition": condition, "seed": seed,
                    "weights": agent.state_dict(), "optimizer": opt.state_dict()}, checkpoint)
        row["checkpoint"] = str(checkpoint.relative_to(output))
        row["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("smoke", "pilot"), default="smoke")
    p.add_argument("--output", default="results/seq007")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path(args.output)
    training, validations, provenance = data()
    if args.stage == "smoke":
        seeds, steps = PILOTS[:1], 2
    else:
        seeds, steps = PILOTS, STEPS
    report = {
        "study": "SEQ-TRANSFER-007", "stage": args.stage,
        "status": "VALIDATION_ONLY_EXPLORATORY",
        "test_opened": False, "test_evaluations": 0,
        "provenance": provenance, "fits": [],
    }
    for seed in seeds:
        for arm in ARMS:
            for condition, weight in CONDITIONS.items():
                row = train(arm, condition, weight, seed, training,
                            validations, steps, output, persist=args.stage=="pilot")
                report["fits"].append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    path = output / f"{args.stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(f"saved={path} test_opened=False", flush=True)


if __name__ == "__main__":
    main()
