"""DEV-009 — controlled *development-only* correction for sparse indexer starvation.

In DEV-008 a gold candidate excluded by hard top-k received an extremely large
constant-masked loss with no gradient back to its first-stage index score.
We add matched first-index retrieval supervision to both compared arms and
only apply sparse final task CE when the target is actually retained. This
changes the training objective; it is NOT a fair score against DEV-008.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.benchmark_v1 import generate_task  # noqa: E402
from tac_osm.development.ced_hardened_router import QueryOnlyCED  # noqa: E402
from run_ced_hardened_dev import _hash_weights, _model_inputs, evaluate_loop  # noqa: E402

SEEDS = (43, 59, 71)
MODES = ("shared", "sparse")


def train(seed: int, mode: str, steps: int, n: int):
    torch.manual_seed(seed)
    model = QueryOnlyCED(mode=mode, topk=4)
    beginning = _hash_weights(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    model.train()
    missed, task_steps = 0, 0
    first_ce, final_ce = 0.0, 0.0
    for step in range(steps):
        task = generate_task(seed * 10_000 + step, dim=8,
                             n_candidates=n, relation="equality", validity="unique",
                             step=step)
        ref, mask, desc = _model_inputs(task)
        logits, _ = model(ref, mask, desc)
        gold = next(iter(task.acceptable_actions))  # training label only
        label = torch.tensor([gold], dtype=torch.long)
        first = model.last_first_index_scores
        if first is None or first.shape != (n,) or not first.requires_grad:
            raise AssertionError("first-stage index logits must have a gradient")
        index_loss = F.cross_entropy(first[None], label)
        # Applying masked final CE to missing gold supplies no useful indexer
        # gradient and produced >600,000 CE in DEV-008. Train the first index
        # until it selects gold; then train the final scorer on that subset.
        target_selected = (mode != "sparse" or bool((logits[gold] > -1e5).item()))
        if target_selected:
            decision_loss = F.cross_entropy(logits[None], label)
            task_steps += 1
        else:
            missed += 1
            decision_loss = torch.zeros_like(index_loss)
        loss = index_loss + decision_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        first_ce, final_ce = float(index_loss.detach()), float(decision_loss.detach())
    model.eval()
    return model, beginning, {
        "initial_hash":beginning,
        "final_hash":_hash_weights(model),
        "missing_gold_during_training":missed,
        "task_ce_training_steps":task_steps,
        "last_index_ce":first_ce,
        "last_task_ce":final_ce,
        "loss_description":"first-stage CE + final CE only when target accessible",
    }


@torch.no_grad()
def measure_index_recall(model: QueryOnlyCED, seed: int, n: int, trials: int):
    model.eval()
    hits = []
    for j in range(trials):
        task = generate_task(seed * 100_000 + j, n_candidates=n, dim=8,
                             relation="equality", validity="unique")
        ref, mask, desc = _model_inputs(task)
        model(ref, mask, desc)
        index = model.last_first_index_scores
        if index is None or index.shape != (n,):
            raise AssertionError("missing first-stage scores")
        gold = next(iter(task.acceptable_actions))
        hits.append(int(gold in index.topk(min(n, model.topk)).indices.tolist()))
    return {"candidates":n, "topk":model.topk, "trials":trials,
            "recall_at_k":sum(hits)/len(hits), "raw_hits":hits}


def run(output: Path, *, smoke: bool = False):
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[:1] if smoke else SEEDS
    steps = 8 if smoke else 120
    samples = 2 if smoke else 24
    files = ["src/tac_osm/development/ced_hardened_router.py",
             "scripts/run_ced_hardened_dev.py", "scripts/run_ced_index_gradient_dev.py"]
    report = {
        "study":"DEV-009", "classification":"DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "smoke":smoke,"training_steps":steps,"train_n":8,
        "episodes_per_cell":samples,
        "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        "records": [],
    }
    for seed in seeds:
        entry={"seed":seed,"arms":{}}
        for mode in MODES:
            model,start,info=train(seed,mode,steps,n=8)
            path=output/f"{mode}-seed{seed}.pt"
            torch.save(model.state_dict(),path)
            cells=[evaluate_loop(model,seed=seed+987,n=n,delay=delay,samples=samples)
                   for n in (8,32) for delay in (0,4)]
            recall=[measure_index_recall(model,seed+1234,n,trials=12 if smoke else 96)
                    for n in (8,32)]
            entry["arms"][mode]={
                **info,
                "param_count":sum(p.numel() for p in model.parameters()),
                "checkpoint":path.name,
                "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "cells":cells,
                "index_recall":recall,
            }
            print(f"seed {seed} {mode}: "+" ".join(
                f"N{c['candidates']} D{c['delay']}={c['accuracy_pre_repair']:.2%}"
                for c in cells)+f" index@4: {[round(x['recall_at_k'],3) for x in recall]}"
                f" missing-train={info['missing_gold_during_training']}",flush=True)
        assert entry["arms"]["shared"]["initial_hash"]==entry["arms"]["sparse"]["initial_hash"]
        report["records"].append(entry)
    (output/"report.json").write_text(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("artifacts/DEV-009"))
    args=parser.parse_args()
    run(args.output,smoke=args.smoke)
