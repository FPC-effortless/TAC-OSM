#!/usr/bin/env python3
"""TACOSM-GRAPH-CASM-SELECTIVE-010.

Public executable program graphs are routed; exact topological execution is the
control. This separates executable-structure sufficiency from selective
candidate routing.
"""

import argparse
import hashlib
import json
import os
import random
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CASM_SOURCE_ROOT = Path(
    os.environ.get(
        "CASM_SOURCE_ROOT",
        str(ROOT / "third_party" / "cdl-attention-experiment"),
    )
)
if not CASM_SOURCE_ROOT.exists():
    raise RuntimeError("CASM generator source is missing")
sys.path.insert(0, str(CASM_SOURCE_ROOT))

from casm_v01.phase1_dag.generator import BooleanDAGGenerator, Episode


SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (32, 64, 128, 256, 512)
BUDGETS = (1, 2, 4, 8)
MAX_NODES = 10
INPUT_COUNT = 4
TRAIN_PROGRAMS = 256
ROUTER_STEPS = 1200
ROUTER_BATCH = 32
SUPPORT_ROWS = 4
VERIFIER_ROWS = 12
LATENT_DIM = 32
ROLE_HOLDOUT = ("NOT", "XOR")
FEATURE_DIM = 290


@dataclass(frozen=True)
class Task:
    task_id: str
    support: tuple[tuple[tuple[int, ...], int], ...]
    verify: tuple[tuple[tuple[int, ...], int], ...]
    target_index: int


@dataclass(frozen=True)
class Work:
    edge_ops: int
    node_ops: int

    @property
    def total(self) -> int:
        return self.edge_ops + self.node_ops


class Memory:
    def __init__(self):
        self.tasks = {}
        self.experience = {}

    def write(self, task_id, support):
        value = tuple(support)
        if task_id in self.tasks and self.tasks[task_id] != value:
            raise ValueError("task address collision")
        self.tasks[task_id] = value

    def read(self, task_id):
        return self.tasks[task_id]

    def commit(self, task_id, key):
        self.experience.setdefault(task_id, set()).add(key)

    def keys(self, task_id):
        return frozenset(self.experience.get(task_id, set()))


def truth_signature(ep: Episode) -> tuple[int, ...]:
    return tuple(int(ep.truth_table[k]) for k in sorted(ep.truth_table))


def structure_key(ep: Episode) -> tuple:
    return (
        tuple(
            (n.index, n.op.value, n.depth, n.arity)
            for n in ep.nodes[:ep.active_count]
        ),
        tuple(sorted((e.src, e.dst, e.port) for e in ep.true_edges)),
        tuple(ep.inputs),
        ep.output,
    )


def has_role_pair(ep: Episode, pair=ROLE_HOLDOUT) -> bool:
    return any(
        ep.nodes[e.src].op.value == pair[0]
        and ep.nodes[e.dst].op.value == pair[1]
        for e in ep.true_edges
    )


def generate(seed: int, count: int, *, exclude_role_pair=None,
             require_role_pair=False, exclude_structures=None, exclude_truths=None):
    gen = BooleanDAGGenerator(max_nodes=MAX_NODES, min_nodes=MAX_NODES, seed=seed)
    out = []
    seen_s = set()
    seen_t = set()
    attempts = 0
    while len(out) < count:
        attempts += 1
        if attempts > count * 20000:
            raise RuntimeError("unique program generation stalled")
        ep = gen._make(MAX_NODES, input_count=INPUT_COUNT)
        if exclude_role_pair and has_role_pair(ep, exclude_role_pair):
            continue
        if require_role_pair and not has_role_pair(ep):
            continue
        sk = structure_key(ep)
        ts = truth_signature(ep)
        if sk in seen_s or ts in seen_t:
            continue
        if exclude_structures and sk in exclude_structures:
            continue
        if exclude_truths and ts in exclude_truths:
            continue
        out.append(ep)
        seen_s.add(sk)
        seen_t.add(ts)
    return out


def build_manifest(seed: int, m_levels: Sequence[int], tasks_per_m: int,
                   *, role_holdout=False, exclude_structures=None, exclude_truths=None):
    total = len(m_levels) * tasks_per_m
    rows = generate(
        seed + (9003 if role_holdout else 9001),
        total,
        require_role_pair=role_holdout,
        exclude_structures=exclude_structures,
        exclude_truths=exclude_truths,
    )
    result = {}
    k = 0
    for m in m_levels:
        for i in range(tasks_per_m):
            result[(m, i)] = rows[k]
            k += 1
    return result

def make_task(seed: int, m: int, target: Episode, train_structures, train_truths, *,
              heldout=False):
    if structure_key(target) in train_structures or truth_signature(target) in train_truths:
        raise RuntimeError("target is not disjoint from training structures/truth tables")
    decoys = generate(
        seed + 7000 + m,
        m * 4,
        exclude_structures={structure_key(target)},
        exclude_truths={truth_signature(target)},
    )
    rng = random.Random(seed * 1000003 + m * 7919 + int(heldout))
    candidates = [target]
    for ep in decoys:
        if structure_key(ep) in train_structures or truth_signature(ep) in train_truths:
            continue
        if structure_key(ep) == structure_key(target):
            continue
        if truth_signature(ep) == truth_signature(target):
            continue
        candidates.append(ep)
        if len(candidates) == m:
            break
    if len(candidates) < m:
        raise RuntimeError("candidate pool construction failed")
    support_keys = rng.sample(sorted(target.truth_table), SUPPORT_ROWS)
    verify_keys = [k for k in sorted(target.truth_table) if k not in support_keys]
    support = tuple((tuple(k), int(target.truth_table[k])) for k in support_keys)
    verify = tuple((tuple(k), int(target.truth_table[k])) for k in verify_keys)
    rng.shuffle(candidates)
    target_index = next(i for i, ep in enumerate(candidates) if structure_key(ep) == structure_key(target))
    digest = hashlib.sha256(repr(support).encode()).hexdigest()[:16]
    return (
        Task(f"g10:{seed}:{m}:{int(heldout)}:{digest}", support, verify, target_index),
        candidates,
    )


def node_features(ep: Episode) -> list[float]:
    ops = ("INPUT", "NOT", "AND", "OR", "XOR")
    result = []
    for n in ep.nodes[:MAX_NODES]:
        one = [0.0] * len(ops)
        one[ops.index(n.op.value)] = 1.0
        result.extend(one)
        result.extend([
            n.depth / MAX_NODES,
            n.index / max(1, MAX_NODES - 1),
            n.arity / 2.0,
            float(n.index < ep.active_count),
        ])
    return result


def features(ep: Episode, include_wiring: bool) -> list[float]:
    out = node_features(ep)
    wiring = [0.0] * 200
    if include_wiring:
        for e in ep.true_edges:
            wiring[(e.dst * MAX_NODES + e.src) * 2 + e.port] = 1.0
    out.extend(wiring)
    return out


def query_vector(support) -> list[float]:
    return [float(x) for bits, y in support for x in (*bits, y)]


class Router(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.query = nn.Sequential(
            nn.Linear(SUPPORT_ROWS * (INPUT_COUNT + 1), 64),
            nn.GELU(),
            nn.Linear(64, LATENT_DIM),
        )
        self.graph = nn.Linear(FEATURE_DIM, LATENT_DIM)
        self.summary = nn.Linear(FEATURE_DIM, LATENT_DIM)

    def forward(self, q, c, representation):
        zq = nn.functional.normalize(self.query(q), dim=-1)
        layer = self.graph if representation == "graph" else self.summary
        zc = nn.functional.normalize(layer(c), dim=-1)
        return zq @ zc.T


def fit_router(seed: int, train_programs: Sequence[Episode], representation: str) -> Router:
    router = Router(seed + (101 if representation == "summary" else 0))
    opt = torch.optim.AdamW(router.parameters(), lr=1e-3)
    q_rows = []
    for ep in train_programs:
        rng = random.Random(seed * 3001 + ep.output * 101 + len(ep.true_edges))
        keys = rng.sample(sorted(ep.truth_table), SUPPORT_ROWS)
        q_rows.append(query_vector(tuple((k, ep.truth_table[k]) for k in keys)))
    q = torch.tensor(q_rows, dtype=torch.float32)
    c = torch.tensor(
        [features(ep, representation == "graph") for ep in train_programs],
        dtype=torch.float32,
    )
    rng = random.Random(seed + (77 if representation == "graph" else 88))
    router.train()
    for _ in range(ROUTER_STEPS):
        idx = rng.sample(range(len(train_programs)), ROUTER_BATCH)
        logits = router(q[idx], c[idx], representation) / 0.10
        labels = torch.arange(len(idx), dtype=torch.long)
        loss = nn.functional.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        opt.step()
    return router.eval()


def execute_exact(ep: Episode, bits: Sequence[int]) -> tuple[int, Work]:
    values = {i: int(v) for i, v in zip(ep.inputs, bits)}
    incoming = {}
    for e in ep.true_edges:
        incoming.setdefault(e.dst, {})[e.port] = e.src
    edge_ops = node_ops = 0
    for node in ep.nodes[:ep.active_count]:
        if node.op.value == "INPUT":
            continue
        args = []
        for port in range(node.arity):
            args.append(values[incoming[node.index][port]])
            edge_ops += 1
        if node.op.value == "NOT":
            values[node.index] = 1 - args[0]
        elif node.op.value == "AND":
            values[node.index] = args[0] & args[1]
        elif node.op.value == "OR":
            values[node.index] = args[0] | args[1]
        elif node.op.value == "XOR":
            values[node.index] = args[0] ^ args[1]
        node_ops += 1
    return int(values[ep.output]), Work(edge_ops, node_ops)


def executor_check(programs) -> dict:
    checked = 0
    failures = []
    for ep in programs:
        for bits, expected in ep.truth_table.items():
            got, _ = execute_exact(ep, bits)
            checked += 1
            if got != int(expected):
                failures.append({"bits": list(bits), "got": got, "expected": int(expected)})
                if len(failures) == 5:
                    return {"pass": False, "checked_rows": checked, "failures": failures}
    return {"pass": True, "checked_rows": checked, "failures": []}


def candidate_work(ep: Episode, verify_rows: int) -> int:
    _, w = execute_exact(ep, ep.input_values)
    return w.total * verify_rows


def evaluate(router: Router, representation: str, task: Task, candidates, memory: Memory):
    support = memory.read(task.task_id)
    c = torch.tensor(
        [features(ep, representation == "graph") for ep in candidates],
        dtype=torch.float32,
    )
    q = torch.tensor([query_vector(support)], dtype=torch.float32)
    with torch.no_grad():
        scores = router(q, c, representation)[0].tolist()
    bonus_keys = memory.keys(task.task_id)
    order = sorted(
        range(len(candidates)),
        key=lambda i: (-(scores[i] + (5.0 if structure_key(candidates[i]) in bonus_keys else 0.0)), i),
    )

    def execute(indices):
        start = time.perf_counter()
        ok_any = False
        total_work = 0
        for idx in indices:
            candidate = candidates[idx]
            ok = True
            for bits, expected in task.verify:
                got, work = execute_exact(candidate, bits)
                total_work += work.total
                ok = ok and got == int(expected)
            ok_any = ok_any or ok
        elapsed = (time.perf_counter() - start) * 1000.0
        return ok_any, total_work, elapsed

    exhaustive_ok, exhaustive_work, exhaustive_ms = execute(range(len(candidates)))
    budgets = {}
    for b in BUDGETS:
        selected = order[:b]
        ok, work, ms = execute(selected)
        adaptive_work = 0
        adaptive_ms = 0.0
        adaptive_ok = False
        used = 0
        for idx in selected:
            t0 = time.perf_counter()
            one_ok, one_work, one_ms = execute([idx])
            adaptive_work += one_work
            adaptive_ms += one_ms + (time.perf_counter() - t0) * 1000.0
            used += 1
            if one_ok:
                adaptive_ok = True
                break
        budgets[str(b)] = {
            "routing_recall": float(order.index(task.target_index) < b),
            "semantic_success": float(ok),
            "adaptive_semantic_success": float(adaptive_ok),
            "fixed_budget_executed_candidates": b,
            "adaptive_executed_candidates": used,
            "adaptive_execution_work_units": adaptive_work,
            "fixed_budget_execution_work_units": work,
            "adaptive_execution_work_fraction": adaptive_work / max(1, exhaustive_work),
            "fixed_budget_execution_work_fraction": work / max(1, exhaustive_work),
            "fixed_budget_execution_ms": ms,
            "adaptive_execution_ms": adaptive_ms,
        }
    return {
        "rank": order.index(task.target_index) + 1,
        "exhaustive": {
            "semantic_success": float(exhaustive_ok),
            "execution_work_units": exhaustive_work,
            "execution_ms": exhaustive_ms,
        },
        "budgets": budgets,
    }


def ci(values, seed, rounds=4000):
    if len(values) <= 1:
        x = float(values[0]) if values else 0.0
        return [x, x]
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return [float(draws[int(.025 * (rounds - 1))]), float(draws[int(.975 * (rounds - 1))])]


def aggregate(by_seed):
    out = {}
    for rep, seed_rows in by_seed.items():
        out[rep] = {}
        all_m = sorted({int(m) for rows in seed_rows.values() for m in rows})
        for m in all_m:
            seed_blocks = [
                seed_rows[str(s)][str(m)]
                for s in SEEDS
                if str(s) in seed_rows and str(m) in seed_rows[str(s)]
            ]
            trials = [t for block in seed_blocks for t in block["trials"]]
            n = len(trials)
            ex = sum(int(t["exhaustive"]["semantic_success"]) for t in trials) / max(1, n)
            row = {
                "trials": n,
                "exhaustive_success": ex,
                "exhaustive_ceiling_pass": ex >= .80,
                "budgets": {},
            }
            for b in BUDGETS:
                sem = sum(t["budgets"][str(b)]["semantic_success"] for t in trials) / max(1, n)
                adapt = sum(t["budgets"][str(b)]["adaptive_semantic_success"] for t in trials) / max(1, n)
                route = sum(t["budgets"][str(b)]["routing_recall"] for t in trials) / max(1, n)
                work = sum(t["budgets"][str(b)]["fixed_budget_execution_work_units"] for t in trials)
                awork = sum(t["budgets"][str(b)]["adaptive_execution_work_units"] for t in trials)
                exwork = sum(t["exhaustive"]["execution_work_units"] for t in trials)
                seed_sem = [
                    sum(t["budgets"][str(b)]["semantic_success"] for t in block["trials"])
                    / max(1, len(block["trials"]))
                    for block in seed_blocks
                ]
                seed_work = [
                    sum(t["budgets"][str(b)]["fixed_budget_execution_work_units"] for t in block["trials"])
                    / max(1, sum(t["exhaustive"]["execution_work_units"] for t in block["trials"]))
                    for block in seed_blocks
                ]
                retention = sem / ex if ex > 0 else None
                row["budgets"][str(b)] = {
                    "routing_recall": route,
                    "semantic_success": sem,
                    "adaptive_semantic_success": adapt,
                    "capability_retention_vs_exhaustive": retention,
                    "adaptive_capability_retention_vs_exhaustive": adapt / ex if ex > 0 else None,
                    "primary_eligible": bool(ex >= .80 and retention is not None and retention >= .80),
                    "execution_work_fraction": work / max(1, exwork),
                    "adaptive_execution_work_fraction": awork / max(1, exwork),
                    "seed_bootstrap_95ci_semantic_success": ci(seed_sem, 9100 + m * 17 + b),
                    "seed_bootstrap_95ci_execution_work_fraction": ci(seed_work, 9110 + m * 17 + b),
                }
            eligible = [(b, v) for b, v in row["budgets"].items() if v["primary_eligible"]]
            row["primary_endpoint"] = (
                {"eligible": True, "min_execution_work_fraction": min(v["execution_work_fraction"] for _, v in eligible)}
                if eligible else
                {"eligible": False, "reason": "exhaustive_ceiling_below_0.80" if ex < .80 else "no_budget_reached_capability_floor"}
            )
            out[rep][str(m)] = row
    return out


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", default="artifacts/TACOSM-GRAPH-CASM-SELECTIVE-010.json")
    args = p.parse_args()

    seeds = (0,) if args.smoke else SEEDS
    m_levels = (32, 64) if args.smoke else M_LEVELS
    tasks_per_m = 2 if args.smoke else 32
    results = {"graph": {}, "summary": {}}
    checks = {}
    output = ROOT / args.output

    for seed in seeds:
        training = generate(seed + 100, TRAIN_PROGRAMS)
        manifest = build_manifest(seed, m_levels, tasks_per_m)
        forbidden_s = {structure_key(ep) for ep in manifest.values()}
        forbidden_t = {truth_signature(ep) for ep in manifest.values()}
        checks[str(seed)] = executor_check(training[:32] + list(manifest.values()))
        if not checks[str(seed)]["pass"]:
            raise RuntimeError("exact graph executor check failed")
        train_s = {structure_key(ep) for ep in training}
        train_t = {truth_signature(ep) for ep in training}

        for rep in ("graph", "summary"):
            router = fit_router(seed, training, rep)
            results[rep][str(seed)] = {}
            for m in m_levels:
                trials = []
                for i in range(tasks_per_m):
                    task, pool = make_task(
                        seed * 100 + i, m, manifest[(m, i)],
                        train_s - forbidden_s, train_t - forbidden_t,
                    )
                    mem = Memory()
                    mem.write(task.task_id, task.support)
                    measured = evaluate(router, rep, task, pool, mem)
                    verified = measured["budgets"]["4"]["adaptive_semantic_success"] > 0
                    if verified:
                        mem.commit(task.task_id, structure_key(pool[measured["rank"] - 1]))
                    trials.append({
                        "task": {
                            "task_id": task.task_id,
                            "target_index": task.target_index,
                            "router_visible_examples": task.support,
                            "verifier_only_examples": task.verify,
                        },
                        **measured,
                    })
                results[rep][str(seed)][str(m)] = {"trials": trials}
                write_json(output, {
                    "experiment_id": "TACOSM-GRAPH-CASM-SELECTIVE-010",
                    "status": "checkpoint",
                    "provenance": {
                        "tacosm_commit": os.environ.get("GITHUB_SHA", "local"),
                        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
                        "python": sys.version,
                        "torch": torch.__version__,
                        "generator_commit": "c31554413301e3c9d3e6b3f8c8c6be572a74a748",
                    },
                    "executor_checks": checks,
                    "results": results,
                })

    final = {
        "experiment_id": "TACOSM-GRAPH-CASM-SELECTIVE-010",
        "status": "measured",
        "provenance": {
            "tacosm_commit": os.environ.get("GITHUB_SHA", "local"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "python": sys.version,
            "torch": torch.__version__,
            "generator_commit": "c31554413301e3c9d3e6b3f8c8c6be572a74a748",
        },
        "protocol": {
            "seeds": list(seeds),
            "M_levels": list(m_levels),
            "budgets": list(BUDGETS),
            "train_programs": TRAIN_PROGRAMS,
            "support_rows": SUPPORT_ROWS,
            "verifier_rows": VERIFIER_ROWS,
            "candidate_feature_dimension": FEATURE_DIM,
            "graph_wiring_public": True,
            "exact_graph_executor": True,
        },
        "executor_checks": checks,
        "results": results,
        "aggregate": aggregate(results),
        "scope": {
            "primary": "selective execution work conditional on public executable program graph",
            "asymptotic_claim": False,
            "hardware_speedup_claim": False,
            "language_semantics": False,
        },
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    write_json(output, final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
