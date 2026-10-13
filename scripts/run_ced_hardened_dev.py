"""DEV-008: small *development-only* trained CED routing experiment in HardenedLoop.

Not a learned PLM/CASM, DeepSeek replication, prospective held-out result,
world model, or end-to-end efficiency study. Uses synthetic unique-equality
tasks and deterministic temporal storage. Records PRE-REPAIR decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm import Query  # noqa: E402
from tac_osm.benchmark_v1 import generate_task  # noqa: E402
from tac_osm.hardened import HardenedLoop, TemporalBenchmark  # noqa: E402
from tac_osm.development.ced_hardened_router import (  # noqa: E402
    QueryOnlyCED, HardenedCEDRouter,
)

MODES = ("shared", "sparse")
SEEDS = (43, 59, 71)


def _hash_weights(model: nn.Module) -> str:
    return hashlib.sha256(b"".join(
        v.detach().cpu().contiguous().numpy().tobytes()
        for v in model.state_dict().values()
    )).hexdigest()


def _model_inputs(task):
    # Training uses a publicly visible non-persistent query. Gold action
    # is evaluator/supervisor-only and is NEVER passed to QueryOnlyCED.
    query = task.public()
    bits = tuple(int(x) for x in query.text.partition("\t")[0].split())
    return (torch.tensor(bits, dtype=torch.float32),
            torch.tensor(query.context, dtype=torch.float32),
            torch.tensor([c.descriptor for c in task.candidates], dtype=torch.float32))


def train_model(seed: int, mode: str, steps: int, candidates: int):
    torch.manual_seed(seed)
    model = QueryOnlyCED(mode=mode, topk=4)
    initial_hash = _hash_weights(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    model.train()
    last = 0.0
    for step in range(steps):
        task = generate_task(seed * 10_000 + step, dim=8,
                             n_candidates=candidates, relation="equality",
                             validity="unique", step=step)
        reference, marks, descriptors = _model_inputs(task)
        logits, _ = model(reference, marks, descriptors)
        gold = next(iter(task.acceptable_actions))  # supervisor only
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0),
                                                 torch.tensor([gold]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = float(loss.detach())
    model.eval()
    return model, initial_hash, last


def evaluate_loop(model: QueryOnlyCED, *, seed: int, n: int,
                  delay: int, samples: int):
    raw = []
    for replicate in range(samples):
        # Independent deterministic evaluation stream. Every test case
        # constructs a full HardenedLoop (including executor and verifier).
        benchmark = TemporalBenchmark(seed=seed * 100_000 + replicate,
                                      dim=8, n_candidates=n,
                                      filler_candidates=2)
        if delay:
            bits = tuple(((replicate * 5) >> (j % 4)) & 1 for j in range(8))
            task = benchmark.schedule_lookup_probe(write_step=0, delay=delay,
                         key="persistent:probe", value=bits)
        else:
            task = benchmark.schedule_task(0)
        decisions = []
        adapter = HardenedCEDRouter(model)
        def recorded_router(query, state, retained):
            decision = adapter(query, state, retained)
            decisions.append(decision)
            return decision
        loop = HardenedLoop(router=recorded_router, benchmark=benchmark,
                            repair=None, learning_enabled=False)
        for tick in range(delay + 1):
            loop.step(tick)
        choice = decisions[-1].selected  # no index -> original candidate index
        correct = choice in task.acceptable_actions
        assert loop.costs.executor_invocations == delay + 1
        assert loop.costs.verifier_checks == delay + 1
        assert adapter.last_public_source == ("persistent_state" if delay else "query")
        raw.append({
            "replicate": replicate, "selected": choice, "success_pre_repair": correct,
            "gold_actions_evaluator_only": sorted(task.acceptable_actions),
            "query_source": adapter.last_public_source,
            "candidate_count": adapter.last_cost.candidates,
            "logical_score_pairs": adapter.last_cost.total_score_pairs,
            "global_kv_scalars": adapter.last_cost.global_kv_scalars,
            "executions_including_fillers": loop.costs.executor_invocations,
            "verifications_including_fillers": loop.costs.verifier_checks,
        })
    return {"candidates": n, "delay": delay, "episodes": len(raw),
            "accuracy_pre_repair": sum(int(r["success_pre_repair"]) for r in raw) / len(raw),
            "raw": raw}


def run(steps: int, samples: int, seeds: tuple[int, ...], candidates: int,
        out: Path) -> dict:
    if not 1 <= steps <= 500 or not 1 <= samples <= 100 or candidates < 4:
        raise ValueError("invalid development budget")
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=True)
    files = ("src/tac_osm/development/ced_hardened_router.py",
             "scripts/run_ced_hardened_dev.py")
    result = {
        "study": "DEV-008", "classification": "DEVELOPMENT_SYNTHETIC_ONLY",
        "architecture": "TAC-OSM hardened loop, CED routing adapter",
        "exclusions": ("no learned persistent memory; no CASM task capability; "
                       "no parameter/FLOP-matched conventional Transformer; "
                       "no independent confirmatory holdout; no sublinear compute"),
        "steps": steps, "samples_per_cell": samples, "train_candidate_count": candidates,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in files}, "records": [],
    }
    for seed in seeds:
        record = {"seed": seed, "arms": {}}
        for mode in MODES:
            model, start_hash, last_loss = train_model(seed, mode, steps, candidates)
            path = out / f"{mode}-seed{seed}.pt"
            torch.save(model.state_dict(), path)
            checks = [evaluate_loop(model, seed=seed + 987, n=n,
                        delay=delay, samples=samples)
                      for n in (candidates, 32) for delay in (0, 4)]
            record["arms"][mode] = {
                "initial_weights_sha256": start_hash,
                "final_weights_sha256": _hash_weights(model),
                "checkpoint_file": path.name,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "last_training_ce": last_loss,
                "measurements": checks,
            }
            print(f"seed {seed} {mode}: " +
                  ", ".join(f"H{c['candidates']} D{c['delay']} "
                            f"{c['accuracy_pre_repair']:.2%}" for c in checks), flush=True)
        assert record["arms"]["shared"]["initial_weights_sha256"] == (
            record["arms"]["sparse"]["initial_weights_sha256"]), "initial weights differ"
        result["records"].append(record)
    (out / "report.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/DEV-008"))
    args = parser.parse_args()
    run(steps=8 if args.smoke else 120,
        samples=2 if args.smoke else 24,
        seeds=SEEDS[:1] if args.smoke else SEEDS,
        candidates=8, out=args.output)
