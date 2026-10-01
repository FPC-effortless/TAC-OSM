#!/usr/bin/env python3
"""TACOSM-M2-CASM-SELECTIVE-009.

Frozen structural representation -> selective admission -> actual CASM-S
execution -> public-task outcome verification.

The runner imports the pinned external CASM-S source from CASM_SOURCE_ROOT.
No target ID, true wiring, or hidden target descriptor enters router inputs.
"""
from __future__ import annotations

import argparse
import json
import math
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
    os.environ.get("CASM_SOURCE_ROOT", str(ROOT / "third_party" / "cdl-attention-experiment"))
)
if not CASM_SOURCE_ROOT.exists():
    raise RuntimeError(f"CASM source root does not exist: {CASM_SOURCE_ROOT}")
sys.path.insert(0, str(CASM_SOURCE_ROOT))
sys.path.insert(0, str(ROOT / "src"))

from casm_v01.phase1_dag.generator import BooleanDAGGenerator, Episode as CASMEpisode
from tac_osm.casm_s_adapter import CASMSAdapter, WorkAccounting


SEEDS = (0, 1, 2, 3, 4)
MAX_NODES = 10
INPUT_COUNT = 4
TRAIN_PROGRAMS = 256
EXECUTOR_TRAIN_EPOCHS = 25
EXECUTOR_BATCH = 32
ROUTER_STEPS = 1200
ROUTER_BATCH = 32
SUPPORT_ROWS = 4
M_LEVELS = (32, 64, 128, 256, 512)
BUDGETS = (1, 2, 4, 8)
EVAL_TASKS_PER_SEED_M = 32
LATENT_DIM = 32
ROUTER_TEMP = 0.10
ROUTER_LR = 1e-3
ROLE_HOLDOUT = ("NOT", "XOR")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    examples: tuple[tuple[tuple[int, ...], int], ...]
    target_truth: tuple[int, ...]
    target_index: int


class PersistentTaskMemory:
    """Opaque task-addressed state plus verifier-gated experience."""

    def __init__(self) -> None:
        self._tasks: dict[str, tuple[tuple[tuple[int, ...], int], ...]] = {}
        self._experience: dict[str, set[tuple]] = {}

    def write_task(self, task_id: str, examples) -> None:
        value = tuple(examples)
        if task_id in self._tasks and self._tasks[task_id] != value:
            raise ValueError("task address collision with different content")
        self._tasks[task_id] = value

    def read_task(self, task_id: str):
        if task_id not in self._tasks:
            raise KeyError(task_id)
        return self._tasks[task_id]

    def commit_verified_experience(self, task_id: str, candidate_key: tuple) -> None:
        self._experience.setdefault(task_id, set()).add(candidate_key)

    def experience_keys(self, task_id: str) -> frozenset[tuple]:
        return frozenset(self._experience.get(task_id, set()))


def truth_signature(ep: CASMEpisode) -> tuple[int, ...]:
    return tuple(int(ep.truth_table[k]) for k in sorted(ep.truth_table))


def role_pair_present(ep: CASMEpisode, src: str, dst: str) -> bool:
    for e in ep.true_edges:
        if ep.nodes[e.src].op.value == src and ep.nodes[e.dst].op.value == dst:
            return True
    return False


def structural_key(ep: CASMEpisode) -> tuple:
    rows = []
    for node in ep.nodes[:ep.active_count]:
        rows.append((node.index, node.op.value, node.depth, node.arity))
    edges = tuple(sorted((e.src, e.dst, e.port) for e in ep.true_edges))
    return tuple(rows), edges, tuple(ep.inputs), ep.output


def generate_unique_programs(
    seed: int,
    count: int,
    *,
    exclude_role_pair: tuple[str, str] | None = None,
    include_role_pair: tuple[str, str] | None = None,
) -> list[CASMEpisode]:
    gen = BooleanDAGGenerator(max_nodes=MAX_NODES, min_nodes=MAX_NODES, seed=seed)
    rows: list[CASMEpisode] = []
    seen_structures: set[tuple] = set()
    seen_truths: set[tuple[int, ...]] = set()
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 20000:
            raise RuntimeError("unable to construct requested unique program set")
        ep = gen._make(MAX_NODES, input_count=INPUT_COUNT)
        pair = ROLE_HOLDOUT
        has_pair = role_pair_present(ep, *pair)
        if exclude_role_pair and has_pair:
            continue
        if include_role_pair and not has_pair:
            continue
        sk = structural_key(ep)
        ts = truth_signature(ep)
        if sk in seen_structures or ts in seen_truths:
            continue
        rows.append(ep)
        seen_structures.add(sk)
        seen_truths.add(ts)
    return rows


def sample_support(ep: CASMEpisode, rng: random.Random) -> tuple[tuple[tuple[int, ...], int], ...]:
    keys = sorted(ep.truth_table)
    selected = rng.sample(keys, SUPPORT_ROWS)
    return tuple((tuple(bits), int(ep.truth_table[bits])) for bits in selected)


def make_task_pool(
    seed: int,
    m: int,
    *,
    heldout: bool,
    train_structures: set[tuple],
) -> tuple[TaskSpec, list[CASMEpisode]]:
    if heldout:
        candidates = generate_unique_programs(
            seed + 7000 + m,
            m * 3,
            include_role_pair=ROLE_HOLDOUT,
        )
    else:
        candidates = generate_unique_programs(seed + 7000 + m, m * 3)
    rng = random.Random(seed * 1_000_003 + m * 7919 + (1 if heldout else 0))
    for ep in candidates:
        if structural_key(ep) in train_structures:
            continue
        support = sample_support(ep, rng)
        support_sig = tuple((tuple(bits), out) for bits, out in support)
        pool = [ep]
        for decoy in candidates:
            if structural_key(decoy) == structural_key(ep):
                continue
            if truth_signature(decoy) == truth_signature(ep):
                continue
            decoy_outputs = tuple(int(decoy.truth_table[bits]) for bits, _ in support)
            target_outputs = tuple(int(out) for _, out in support)
            if decoy_outputs == target_outputs:
                continue
            pool.append(decoy)
            if len(pool) == m:
                break
        if len(pool) < m:
            continue
        rng.shuffle(pool)
        target_idx = next(i for i, x in enumerate(pool) if structural_key(x) == structural_key(ep))
        task = TaskSpec(
            task_id=f"m2:{seed}:{m}:{int(heldout)}:{target_idx}",
            examples=support_sig,
            target_truth=truth_signature(ep),
            target_index=target_idx,
        )
        return task, pool
    raise RuntimeError(f"failed to construct a support-discriminative task for M={m}")


def make_executor_training_programs(seed: int, *, exclude_role_pair: tuple[str, str] | None = None) -> list[CASMEpisode]:
    gen = BooleanDAGGenerator(max_nodes=MAX_NODES, min_nodes=MAX_NODES, seed=seed + 111)
    train: list[CASMEpisode] = []
    seen: set[tuple] = set()
    while len(train) < TRAIN_PROGRAMS:
        ep = gen._make(MAX_NODES, input_count=INPUT_COUNT)
        sk = structural_key(ep)
        if sk in seen:
            continue
        if exclude_role_pair and role_pair_present(ep, *exclude_role_pair):
            continue
        train.append(ep)
        seen.add(sk)
    return train


def train_casm(seed: int) -> tuple[CASMSAdapter, dict, list[CASMEpisode]]:
    train = make_executor_training_programs(seed, exclude_role_pair=None)

    casm = CASMS(max_nodes=MAX_NODES, dim=32, temperature=2.0, seed=seed)
    device = torch.device("cpu")
    casm = casm.to(device)
    opt = torch.optim.AdamW(casm.parameters(), lr=2e-3)
    for _epoch in range(EXECUTOR_TRAIN_EPOCHS):
        order = list(range(len(train)))
        random.Random(seed * 101 + _epoch).shuffle(order)
        casm.train()
        for start in range(0, len(order), EXECUTOR_BATCH):
            batch = [train[i] for i in order[start : start + EXECUTOR_BATCH]]
            runtime = [list(ep.input_values) for ep in batch]
            x = torch.tensor(runtime, dtype=torch.float32, device=device)
            target = torch.tensor([ep.target for ep in batch], dtype=torch.float32, device=device)
            pred, _g, _meta = casm(batch, x)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(casm.parameters(), 1.0)
            opt.step()
    casm.eval()
    adapter = CASMSAdapter(casm, torch)
    return adapter, {
        "train_programs": len(train),
        "train_epochs": EXECUTOR_TRAIN_EPOCHS,
        "role_pair_excluded_from_executor_training": None,
        "train_structures": [structural_key(ep) for ep in train],
    }, train


def build_summary(ep: CASMEpisode) -> list[float]:
    out = [0.0] * 32
    op_names = ("NOT", "AND", "OR", "XOR")
    for node in ep.nodes[:ep.active_count]:
        if node.op.value in op_names:
            out[op_names.index(node.op.value)] += 1.0
    out[4] = float(ep.active_count)
    out[5] = float(len(ep.inputs))
    depths = [n.depth for n in ep.nodes[:ep.active_count]]
    out[6] = float(max(depths))
    out[7] = sum(depths) / max(1, len(depths))
    for d in range(min(10, MAX_NODES)):
        out[8 + d] = float(sum(int(n.depth == d) for n in ep.nodes[:ep.active_count]))
    out[18] = float(len(ep.candidate_edges))
    out[19] = float(len(ep.true_edges))
    out[20] = float(len(ep.true_edges)) / max(1, ep.active_count - len(ep.inputs))
    for i, node in enumerate(ep.nodes[:ep.active_count]):
        if node.op.value in op_names:
            out[21 + (i % 11)] += 1.0
    return out


class TwoTowerRouter(nn.Module):
    def __init__(self, candidate_dim: int, *, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.query = nn.Sequential(
            nn.Linear(SUPPORT_ROWS * (INPUT_COUNT + 1), 64),
            nn.GELU(),
            nn.Linear(64, LATENT_DIM),
        )
        self.candidate = nn.Linear(candidate_dim, LATENT_DIM)

    def forward(self, q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        zq = nn.functional.normalize(self.query(q), dim=-1)
        zc = nn.functional.normalize(self.candidate(c), dim=-1)
        return zq @ zc.T


def query_vector(examples: Sequence[tuple[tuple[int, ...], int]]) -> list[float]:
    out: list[float] = []
    for bits, y in examples:
        out.extend(float(b) for b in bits)
        out.append(float(y))
    return out


def fit_router(
    seed: int,
    adapter: CASMSAdapter,
    train_programs: Sequence[CASMEpisode],
    *,
    representation: str,
) -> TwoTowerRouter:
    if representation == "structural":
        with torch.no_grad():
            cand = adapter.encode_structure(train_programs).cpu()
        cand_dim = LATENT_DIM
        candidate_features = [cand[i] for i in range(len(train_programs))]
        def feature(ep, i):
            return candidate_features[i]
    elif representation == "summary":
        cand_dim = 32
        candidate_features = [torch.tensor(build_summary(ep), dtype=torch.float32) for ep in train_programs]
        def feature(ep, i):
            return candidate_features[i]
    else:
        raise ValueError(representation)

    router = TwoTowerRouter(cand_dim, seed=seed + (17 if representation == "summary" else 0))
    opt = torch.optim.AdamW(router.parameters(), lr=ROUTER_LR)
    rng = random.Random(seed * 65 + (1 if representation == "summary" else 0))
    target_query = [query_vector(sample_support(ep, random.Random(rng.randrange(1 << 30)))) for ep in train_programs]
    q_tensor = torch.tensor(target_query, dtype=torch.float32)
    c_tensor = torch.stack(candidate_features).float()
    n = len(train_programs)
    router.train()
    for step in range(ROUTER_STEPS):
        idx = rng.sample(range(n), ROUTER_BATCH)
        logits = router(q_tensor[idx], c_tensor[idx]) / ROUTER_TEMP
        labels = torch.arange(len(idx), dtype=torch.long)
        loss = nn.functional.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        opt.step()
    return router


def evaluate_router(
    router: TwoTowerRouter,
    representation: str,
    task: TaskSpec,
    candidates: Sequence[CASMEpisode],
    adapter: CASMSAdapter,
    memory: PersistentTaskMemory,
    *,
    use_experience: bool = False,
) -> dict:
    examples = memory.read_task(task.task_id)
    if representation == "structural":
        with torch.no_grad():
            c = adapter.encode_structure(candidates).cpu()
    else:
        c = torch.stack([torch.tensor(build_summary(ep), dtype=torch.float32) for ep in candidates])
    q = torch.tensor([query_vector(examples)], dtype=torch.float32)
    with torch.no_grad():
        scores = router(q, c)[0].tolist()
    experience_keys = memory.experience_keys(task.task_id) if use_experience else frozenset()
    order = sorted(
        range(len(candidates)),
        key=lambda i: (
            -(scores[i] + (5.0 if structural_key(candidates[i]) in experience_keys else 0.0)),
            i,
        ),
    )
    rank = order.index(task.target_index) + 1
    per_candidate_work = [adapter.work_for_episode(ep) for ep in candidates]

    def execute_indices(indices: Sequence[int]) -> tuple[bool, int, float, dict]:
        if not indices:
            return False, 0, 0.0, {
                "gate_evaluations": 0,
                "edge_message_operations": 0,
                "structural_node_operations": 0,
                "total_work_units": 0,
            }
        reps = []
        runtime = []
        for idx in indices:
            for bits, _out in examples:
                reps.append(candidates[idx])
                runtime.append(list(bits))
        t0 = time.perf_counter()
        outputs = adapter.execute_batch(reps, runtime)
        elapsed = (time.perf_counter() - t0) * 1000.0
        ok = True
        for j, (_idx) in enumerate(indices):
            start = j * SUPPORT_ROWS
            vals = outputs[start : start + SUPPORT_ROWS]
            expected = [float(y) for _, y in examples]
            ok = ok and all((float(v) >= 0.5) == (float(y) >= 0.5) for v, y in zip(vals, expected))
        work = WorkAccounting(0, 0, 0)
        for idx in indices:
            wa = per_candidate_work[idx]
            work = WorkAccounting(
                work.gate_evaluations + wa.gate_evaluations * SUPPORT_ROWS,
                work.edge_message_operations + wa.edge_message_operations * SUPPORT_ROWS,
                work.structural_node_operations + wa.structural_node_operations * SUPPORT_ROWS,
            )
        return ok, work.total, elapsed, {
            "gate_evaluations": work.gate_evaluations,
            "edge_message_operations": work.edge_message_operations,
            "structural_node_operations": work.structural_node_operations,
            "total_work_units": work.total,
        }

    exhaustive_indices = list(range(len(candidates)))
    ex_ok, ex_work, ex_ms, ex_breakdown = execute_indices(exhaustive_indices)

    budgets = {}
    for b in BUDGETS:
        selected = order[:b]
        ok, work, ms, breakdown = execute_indices(selected)
        adaptive_indices = []
        adaptive_ok = False
        adaptive_work = 0
        adaptive_ms = 0.0
        for idx in order[:b]:
            adaptive_indices.append(idx)
            t0 = time.perf_counter()
            one_ok, one_work, _one_ms, _one_breakdown = execute_indices([idx])
            adaptive_ms += (time.perf_counter() - t0) * 1000.0
            adaptive_work += one_work
            if one_ok:
                adaptive_ok = True
                break
        budgets[str(b)] = {
            "routing_recall": float(rank <= b),
            "semantic_success": float(ok),
            "adaptive_semantic_success": float(adaptive_ok),
            "fixed_budget_executed_candidates": len(selected),
            "adaptive_executed_candidates": len(adaptive_indices),
            "adaptive_verified_candidate_indices": [adaptive_indices[-1]] if adaptive_ok and adaptive_indices else [],
            "fixed_budget_execution_work_units": work,
            "fixed_budget_execution_work_fraction": work / max(1, ex_work),
            "adaptive_execution_work_units": adaptive_work,
            "adaptive_execution_work_fraction": adaptive_work / max(1, ex_work),
            "fixed_budget_execution_ms": ms,
            "adaptive_execution_ms": adaptive_ms,
            "breakdown": breakdown,
        }

    return {
        "rank": rank,
        "routing_recall": {str(b): float(rank <= b) for b in BUDGETS},
        "exhaustive": {
            "semantic_success": float(ex_ok),
            "execution_work_units": ex_work,
            "execution_ms": ex_ms,
            "work_breakdown": ex_breakdown,
        },
        "budgets": budgets,
    }


def aggregate(seed_results: dict) -> dict:
    out = {}
    for representation, by_m in seed_results.items():
        out[representation] = {}
        for m in M_LEVELS:
            rows = [seed_results[representation][str(s)][str(m)] for s in SEEDS]
            all_trials = []
            for r in rows:
                all_trials.extend(r["trials"])
            pooled_ex = sum(int(t["exhaustive"]["semantic_success"]) for t in all_trials)
            n = len(all_trials)
            ex_rate = pooled_ex / max(1, n)
            out[representation][str(m)] = {
                "trials": n,
                "exhaustive_success": ex_rate,
                "budgets": {},
            }
            for b in BUDGETS:
                sem = sum(t["budgets"][str(b)]["semantic_success"] for t in all_trials) / max(1, n)
                adapt = sum(t["budgets"][str(b)]["adaptive_semantic_success"] for t in all_trials) / max(1, n)
                route = sum(t["budgets"][str(b)]["routing_recall"] for t in all_trials) / max(1, n)
                work_frac = sum(t["budgets"][str(b)]["fixed_budget_execution_work_units"] for t in all_trials) / max(1, sum(t["exhaustive"]["execution_work_units"] for t in all_trials))
                adapt_work_frac = sum(t["budgets"][str(b)]["adaptive_execution_work_units"] for t in all_trials) / max(1, sum(t["exhaustive"]["execution_work_units"] for t in all_trials))
                out[representation][str(m)]["budgets"][str(b)] = {
                    "routing_recall": route,
                    "semantic_success": sem,
                    "adaptive_semantic_success": adapt,
                    "capability_retention_vs_exhaustive": sem / ex_rate if ex_rate > 0 else None,
                    "adaptive_capability_retention_vs_exhaustive": adapt / ex_rate if ex_rate > 0 else None,
                    "execution_work_fraction": work_frac,
                    "adaptive_execution_work_fraction": adapt_work_frac,
                }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="artifacts/TACOSM-M2-CASM-SELECTIVE-009.json")
    args = parser.parse_args()

    seeds = (0,) if args.smoke else SEEDS
    m_levels = (32, 64) if args.smoke else M_LEVELS
    tasks_per_m = 4 if args.smoke else EVAL_TASKS_PER_SEED_M

    all_results = {"structural": {}, "summary": {}}
    executor_meta = {}

    for seed in seeds:
        adapter, meta, executor_train = train_casm(seed)
        executor_meta[str(seed)] = meta
        primary_router_train = generate_unique_programs(seed + 2111, TRAIN_PROGRAMS)
        for rep in ("structural", "summary"):
            router = fit_router(seed, adapter, primary_router_train, representation=rep)
            all_results[rep][str(seed)] = {}
            train_structures = {structural_key(ep) for ep in role_holdout_router_train}
            for m in m_levels:
                trials = []
                for i in range(tasks_per_m):
                    task, pool = make_task_pool(
                        seed * 100 + i,
                        m,
                        heldout=False,
                        train_structures=train_structures,
                    )
                    trials.append({
                        "task": {
                            "target_index": task.target_index,
                            "examples": task.examples,
                        },
                        **evaluate_router(router, rep, task, pool, adapter),
                    })
                all_results[rep][str(seed)][str(m)] = {"trials": trials}
        # Secondary structural-holdout condition.
        role_holdout_router_train = generate_unique_programs(
            seed + 3111, TRAIN_PROGRAMS, exclude_role_pair=ROLE_HOLDOUT
        )
        primary_train_structures = {structural_key(ep) for ep in primary_router_train}
        for rep in ("structural", "summary"):
            router = fit_router(seed, adapter, role_holdout_router_train, representation=rep)
            all_results.setdefault("secondary_role_holdout", {}).setdefault(rep, {})[str(seed)] = {}
            for m in m_levels:
                trials = []
                for i in range(max(2, tasks_per_m // 2)):
                    task, pool = make_task_pool(
                        seed * 1000 + i,
                        m,
                        heldout=True,
                        train_structures=train_structures,
                    )
                    trials.append({
                        "task": {
                            "target_index": task.target_index,
                            "examples": task.examples,
                        },
                        **evaluate_router(router, rep, task, pool, adapter),
                    })
                all_results["secondary_role_holdout"][rep][str(seed)][str(m)] = {"trials": trials}

    result = {
        "experiment_id": "TACOSM-M2-CASM-SELECTIVE-009",
        "status": "measured",
        "external_executor_commit": "c31554413301e3c9d3e6b3f8c8c6be572a74a748",
        "protocol": {
            "seeds": list(seeds),
            "M_levels": list(m_levels),
            "budgets": list(BUDGETS),
            "support_rows": SUPPORT_ROWS,
            "executor_training_epochs": EXECUTOR_TRAIN_EPOCHS,
            "router_steps": ROUTER_STEPS,
            "capacity_matched_router": True,
            "target_id_hidden_from_router": True,
            "true_edges_hidden_from_router": True,
        },
        "executor_meta": executor_meta,
        "results": all_results,
        "aggregate": aggregate({
            rep: all_results[rep] for rep in ("structural", "summary")
        }) if not args.smoke else {},
        "scope": {
            "primary": "selective execution work versus capability using actual CASM-S",
            "routing_cost_is_separate": True,
            "asymptotic_claim": False,
            "language_semantics": False,
            "structural_holdout_is_secondary": True,
        },
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
