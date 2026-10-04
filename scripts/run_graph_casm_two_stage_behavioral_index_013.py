#!/usr/bin/env python3
"""G-CASM-013: exact behavioral index + cached late-interaction reranking.

This experiment separates three quantities:
1. exact finite-domain behavioral retrieval,
2. learned compatibility ranking after retrieval,
3. exact execution/verification after routing.

The candidate library is static within each seed and the behavioral index and
candidate latent cache are built once, offline, before query evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import os
import random
import resource
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_graph_casm_behavioral_compatibility_012 as g012
import run_graph_casm_selective_010 as g010
from tac_osm.behavioral_inverted_index import BehavioralInvertedIndex
from tac_osm.indexed_behavioral_reranker import CachedBehavioralReranker, RerankerMacProfile

EXPERIMENT_ID = "TACOSM-GRAPH-CASM-TWO-STAGE-BEHAVIORAL-INDEX-013"
SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (32, 64, 128, 256, 512)
BUDGETS = (1, 2, 4, 8)
SUPPORT_SIZES = (4, 8, 12)
PRIMARY_SUPPORT_SIZE = 4
TASKS_PER_M = 32
SMOKE_M_LEVELS = (32, 128)
SMOKE_TASKS_PER_M = 4
TRAIN_PROGRAMS = 256
LIBRARY_SIZE = 512
ROUTER_STEPS = 1200
FEATURE_DIM = g010.FEATURE_DIM
INPUT_COUNT = g010.INPUT_COUNT
INPUT_ROWS = tuple(itertools.product((0, 1), repeat=INPUT_COUNT))
GENERATOR_COMMIT = g012.GENERATOR_COMMIT

@dataclass(frozen=True)
class QueryTask:
    task_id: str
    target_index: int
    row_order: tuple[int, ...]

    def support(self, library: Sequence, support_size: int):
        target = library[self.target_index]
        return tuple(
            (INPUT_ROWS[i], int(target.truth_table[INPUT_ROWS[i]]))
            for i in self.row_order[:support_size]
        )

    def verify(self, library: Sequence, support_size: int):
        target = library[self.target_index]
        return tuple(
            (INPUT_ROWS[i], int(target.truth_table[INPUT_ROWS[i]]))
            for i in self.row_order[support_size:]
        )

def load_registered_contract():
    from tac_osm.contract import load_contract
    return load_contract(EXPERIMENT_ID)

def row_order(seed: int, m: int, task_i: int) -> tuple[int, ...]:
    rng = random.Random(seed * 1000003 + m * 7919 + task_i * 104729 + 1301)
    rows = list(range(len(INPUT_ROWS)))
    rng.shuffle(rows)
    return tuple(rows)

def build_query_task(seed: int, m: int, task_i: int, library: Sequence) -> QueryTask:
    rng = random.Random(seed * 3000001 + m * 10007 + task_i * 1009 + 17)
    target_index = rng.randrange(m)
    perm = row_order(seed, m, task_i)
    digest = hashlib.sha256(repr((seed, m, task_i, target_index, perm)).encode()).hexdigest()[:16]
    return QueryTask(
        task_id=f"g13:{seed}:{m}:{task_i}:{digest}",
        target_index=target_index,
        row_order=perm,
    )

def candidate_features(library: Sequence) -> torch.Tensor:
    return torch.tensor([g010.features(ep, True) for ep in library], dtype=torch.float32)

def fit_model(seed: int, training: Sequence):
    return g012.fit_behavioral_router(seed, training, "graph")

def verify_selected(
    candidates: Sequence,
    selected: Sequence[int],
    verify_rows,
    *,
    adaptive: bool,
) -> dict:
    ok_any = False
    work = 0
    candidates_used = 0
    rows_evaluated = 0
    for idx in selected:
        candidates_used += 1
        candidate_ok = True
        for bits, expected in verify_rows:
            got, w = g010.execute_exact(candidates[idx], bits)
            work += w.total
            rows_evaluated += 1
            if got != int(expected):
                candidate_ok = False
                if adaptive:
                    break
        if candidate_ok and verify_rows:
            ok_any = True
            if adaptive:
                break
    return {
        "success": float(ok_any),
        "work_units": int(work),
        "candidates_used": int(candidates_used),
        "rows_evaluated": int(rows_evaluated),
    }

def rank_cached(
    reranker: CachedBehavioralReranker,
    support,
    indices: Sequence[int],
) -> tuple[int, ...]:
    rows = torch.tensor(
        [[float(x) for x in (*bits, y)] for bits, y in support],
        dtype=torch.float32,
    )
    scores = reranker.score(indices, rows)
    order = sorted(range(len(indices)), key=lambda i: (-scores[i], indices[i]))
    return tuple(indices[i] for i in order)

def evaluate_arm(
    arm: str,
    reranker: CachedBehavioralReranker,
    index: BehavioralInvertedIndex,
    library: Sequence,
    task: QueryTask,
    support_size: int,
    seed: int,
) -> dict:
    m = int(task.task_id.split(":")[2])
    task_i = int(task.task_id.split(":")[3])
    prefix = library[:m]
    support = task.support(library, support_size)
    verify = task.verify(library, support_size)

    exact = g012.exact_support_consistent(prefix, support)
    retrieved, index_cost = index.retrieve(support, population_size=m)
    if retrieved != exact:
        raise RuntimeError("behavioral index is not extensionally exact")

    if arm == "exhaustive_behavioral":
        scored = tuple(range(m))
        order = rank_cached(reranker, support, scored)
    elif arm == "indexed_behavioral":
        scored = tuple(retrieved)
        order = rank_cached(reranker, support, scored)
    elif arm == "index_only":
        scored = tuple(retrieved)
        order = scored
    elif arm == "random":
        scored = tuple(range(m))
        order_list = list(scored)
        random.Random(seed * 2000003 + m + task_i * 97 + support_size).shuffle(order_list)
        order = tuple(order_list)
    else:
        raise ValueError(arm)

    rerank_macs = (
        reranker.profile.query_macs(support_size, len(scored))
        if arm in {"exhaustive_behavioral", "indexed_behavioral"}
        else 0
    )

    exhaustive = verify_selected(prefix, tuple(range(m)), verify, adaptive=False)
    if exhaustive["success"] != 1.0:
        raise RuntimeError("exact exhaustive verification failed")
    exhaustive_rows_expected = m * len(verify)
    if exhaustive["rows_evaluated"] != exhaustive_rows_expected:
        raise RuntimeError("exhaustive fixed-row accounting invariant failed")

    budgets = {}
    for b in BUDGETS:
        selected = tuple(order[:b])
        fixed = verify_selected(prefix, selected, verify, adaptive=False)
        adaptive = verify_selected(prefix, selected, verify, adaptive=True)
        expected_rows = len(selected) * len(verify)
        if fixed["rows_evaluated"] != expected_rows:
            raise RuntimeError("fixed-budget verification-row invariant failed")
        budgets[str(b)] = {
            "target_in_budget": float(task.target_index in selected),
            "routing_recall": float(task.target_index in selected),
            "semantic_success": fixed["success"],
            "adaptive_semantic_success": adaptive["success"],
            "fixed_candidates_used": fixed["candidates_used"],
            "adaptive_candidates_used": adaptive["candidates_used"],
            "fixed_rows_evaluated": fixed["rows_evaluated"],
            "adaptive_rows_evaluated": adaptive["rows_evaluated"],
            "expected_fixed_rows": expected_rows,
            "fixed_work_units": fixed["work_units"],
            "adaptive_work_units": adaptive["work_units"],
            "fixed_work_fraction": fixed["work_units"] / max(1, exhaustive["work_units"]),
            "adaptive_work_fraction": adaptive["work_units"] / max(1, exhaustive["work_units"]),
        }

    return {
        "target_index": task.target_index,
        "support_size": support_size,
        "support_consistent_count": len(retrieved),
        "index_exact": True,
        "candidates_scored": len(scored),
        "query_macs": int(rerank_macs),
        "index_posting_lookups": int(index_cost.posting_lookups),
        "index_bitmap_word_ops": int(index_cost.bitmap_and_word_ops),
        "index_result_size": int(index_cost.result_candidates),
        "target_rank": order.index(task.target_index) + 1 if task.target_index in order else None,
        "exhaustive_execution_work_units": int(exhaustive["work_units"]),
        "exhaustive_rows_evaluated": int(exhaustive["rows_evaluated"]),
        "budgets": budgets,
    }

def seed_bootstrap(values: Sequence[float], seed: int, rounds: int = 4000):
    if len(values) <= 1:
        x = float(values[0]) if values else 0.0
        return [x, x]
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(rounds)
    )
    return [
        float(draws[int(0.025 * (rounds - 1))]),
        float(draws[int(0.975 * (rounds - 1))]),
    ]

def summarize_arm(raw_arm, seeds: Sequence[int], m_levels: Sequence[int]):
    out = {}
    for m in m_levels:
        out[str(m)] = {}
        for support_size in SUPPORT_SIZES:
            blocks = [
                raw_arm[str(seed)][str(m)][str(support_size)]
                for seed in seeds
            ]
            flat = [t for block in blocks for t in block]
            row = {
                "trials": len(flat),
                "support_consistent_count_mean": statistics.fmean(t["support_consistent_count"] for t in flat),
                "index_exact_fraction": statistics.fmean(float(t["index_exact"]) for t in flat),
                "exhaustive_exact_success": 1.0,
            }
            for b in BUDGETS:
                semantic = statistics.fmean(t["budgets"][str(b)]["semantic_success"] for t in flat)
                adaptive = statistics.fmean(t["budgets"][str(b)]["adaptive_semantic_success"] for t in flat)
                route = statistics.fmean(t["budgets"][str(b)]["routing_recall"] for t in flat)
                work = statistics.fmean(t["budgets"][str(b)]["adaptive_work_fraction"] for t in flat)
                seed_sem = [
                    statistics.fmean(t["budgets"][str(b)]["semantic_success"] for t in block)
                    for block in blocks
                ]
                seed_work = [
                    statistics.fmean(t["budgets"][str(b)]["adaptive_work_fraction"] for t in block)
                    for block in blocks
                ]
                row["budgets"] = row.get("budgets", {})
                row["budgets"][str(b)] = {
                    "semantic_success": semantic,
                    "adaptive_semantic_success": adaptive,
                    "routing_recall": route,
                    "capability_retention_vs_exhaustive_exact": semantic,
                    "adaptive_capability_retention_vs_exhaustive_exact": adaptive,
                    "adaptive_execution_work_fraction": work,
                    "primary_eligible": semantic >= 0.80,
                    "seed_bootstrap_95ci_semantic_success": seed_bootstrap(seed_sem, 14000 + m * 31 + support_size * 7 + b),
                    "seed_bootstrap_95ci_adaptive_work_fraction": seed_bootstrap(seed_work, 14100 + m * 31 + support_size * 7 + b),
                }
            if support_size == PRIMARY_SUPPORT_SIZE:
                eligible = [
                    x["adaptive_execution_work_fraction"]
                    for x in row["budgets"].values()
                    if x["primary_eligible"]
                ]
                row["primary_endpoint"] = (
                    {"eligible": True, "min_adaptive_execution_work_fraction": min(eligible)}
                    if eligible else
                    {"eligible": False, "reason": "no_budget_reached_capability_floor"}
                )
            out[str(m)][str(support_size)] = row
    return out

def main(smoke: bool = False):
    contract = load_registered_contract()
    if smoke:
        seeds = (0,)
        m_levels = SMOKE_M_LEVELS
        tasks_per_m = SMOKE_TASKS_PER_M
    else:
        seeds = SEEDS
        m_levels = M_LEVELS
        tasks_per_m = TASKS_PER_M
        contract.require_levels(M_LEVELS)
        contract.require_seeds(SEEDS)
        contract.require_steps(ROUTER_STEPS)
        contract.require_eval_steps(TASKS_PER_M)
        contract.require_k_levels(SUPPORT_SIZES)
        contract.require_arms(["exhaustive_behavioral", "indexed_behavioral", "index_only", "random"])

    arms = ("exhaustive_behavioral", "indexed_behavioral", "index_only", "random")
    raw = {arm: {} for arm in arms}
    checks = {}
    profile = RerankerMacProfile()

    for seed in seeds:
        training = g010.generate(seed + 100, TRAIN_PROGRAMS)
        train_s = {g010.structure_key(ep) for ep in training}
        train_t = {g010.truth_signature(ep) for ep in training}
        library = g010.generate(
            seed + 5000,
            LIBRARY_SIZE,
            exclude_structures=train_s,
            exclude_truths=train_t,
        )
        library_s = {g010.structure_key(ep) for ep in library}
        library_t = {g010.truth_signature(ep) for ep in library}
        if len(library) != LIBRARY_SIZE:
            raise RuntimeError("evaluation library construction failed")
        if train_s & library_s or train_t & library_t:
            raise RuntimeError("train/evaluation disjointness gate failed")

        model = fit_model(seed, training)
        features = candidate_features(library)
        reranker = CachedBehavioralReranker(model, features)
        index = BehavioralInvertedIndex(library)

        checks[str(seed)] = {
            "train_programs": TRAIN_PROGRAMS,
            "eval_library_size": LIBRARY_SIZE,
            "train_eval_structure_disjoint": True,
            "train_eval_truth_disjoint": True,
            "candidate_cache_built_once": True,
            "behavior_index_built_once": True,
            "candidate_cache_build_macs": LIBRARY_SIZE * profile.candidate_encoder_per_candidate,
            "index_build_membership_operations": index.build_membership_operations,
            "index_posting_count": index.posting_count,
        }
        for arm in arms:
            raw[arm][str(seed)] = {}

        for m in m_levels:
            for support_size in SUPPORT_SIZES:
                for task_i in range(tasks_per_m):
                    task = build_query_task(seed, m, task_i, library)
                    if task.target_index >= m:
                        raise RuntimeError("target outside population prefix")
                    for arm in arms:
                        raw[arm][str(seed)].setdefault(str(m), {}).setdefault(str(support_size), []).append(
                            evaluate_arm(
                                arm, reranker, index, library, task, support_size, seed
                            )
                        )

    summary = {arm: summarize_arm(raw[arm], seeds, m_levels) for arm in arms}
    final = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "provenance": {
            "tacosm_commit": os.environ.get("GITHUB_SHA", "local"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "generator_commit": GENERATOR_COMMIT,
            "python": sys.version,
            "torch": torch.__version__,
        },
        "protocol": {
            "seeds": list(seeds),
            "M_levels": list(m_levels),
            "tasks_per_seed_M": tasks_per_m,
            "library_size": LIBRARY_SIZE,
            "support_sizes": list(SUPPORT_SIZES),
            "primary_support_size": PRIMARY_SUPPORT_SIZE,
            "budgets": list(BUDGETS),
            "train_programs": TRAIN_PROGRAMS,
            "router_steps": ROUTER_STEPS,
            "input_count": INPUT_COUNT,
            "input_domain_rows": len(INPUT_ROWS),
            "behavior_index_built_once_per_seed": True,
            "candidate_latent_cache_built_once_per_seed": True,
        },
        "cost_model": {
            "candidate_encoder_macs_per_candidate": profile.candidate_encoder_per_candidate,
            "row_encoder_macs_per_support_row": profile.row_encoder_per_support_row,
            "pair_head_macs_per_candidate_row": profile.pair_head_per_candidate_row,
            "cached_query_formula": "S*2368 + R*S*8256",
            "uncached_012_query_formula": "N*20608 + S*2368 + N*S*8256",
            "index_units": "posting lookups plus bitmap word operations",
            "units_not_combined": True,
        },
        "checks": checks,
        "summary": summary,
        "results": raw,
        "scope": {
            "primary": "capability-constrained execution after an exact public-behavior retrieval ceiling and cached behavioral reranking",
            "exact_index_is_finite_domain_ceiling": True,
            "input_width": INPUT_COUNT,
            "asymptotic_claim": False,
            "hardware_speedup_claim": False,
            "general_semantic_retrieval_claim": False,
        },
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    (out / f"{EXPERIMENT_ID}.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args().smoke)
