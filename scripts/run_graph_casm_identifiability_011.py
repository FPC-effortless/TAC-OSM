#!/usr/bin/env python3
"""TACOSM-GRAPH-CASM-IDENTIFIABILITY-011.

Audits whether the public support information uniquely identifies the target
inside the exact G-CASM candidate construction before learned routing is judged.
No model is trained. The behavior-index control is a deterministic semantic
addressing mechanism over public executable graphs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_graph_casm_selective_010 as g010

SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (32, 64, 128, 256, 512)
SUPPORT_SIZES = (4, 8, 12, 16)
TASKS_PER_M = 32
INPUT_ROWS = tuple(__import__("itertools").product((0, 1), repeat=4))
TRAIN_PROGRAMS = 256


def support_indices(seed: int, m: int, task_i: int) -> tuple[int, ...]:
    rng = random.Random(seed * 1000003 + m * 7919 + task_i * 104729 + 1101)
    rows = list(range(len(INPUT_ROWS)))
    rng.shuffle(rows)
    return tuple(rows)


def support_consistent(candidates, support):
    wanted = {tuple(bits): int(y) for bits, y in support}
    matches = []
    for idx, ep in enumerate(candidates):
        if all(int(ep.truth_table[bits]) == y for bits, y in wanted.items()):
            matches.append(idx)
    return matches


class BehaviorIndex:
    """Exact inverted index over executable candidate behavior."""

    def __init__(self, candidates):
        self.postings = {}
        self.build_work = 0
        for idx, ep in enumerate(candidates):
            for bits in INPUT_ROWS:
                y, work = g010.execute_exact(ep, bits)
                self.build_work += work.total
                self.postings.setdefault((bits, int(y)), []).append(idx)
        for key in self.postings:
            self.postings[key].sort()

    def query(self, support):
        posting_lists = [self.postings[(tuple(bits), int(y))] for bits, y in support]
        inspected = sum(len(x) for x in posting_lists)
        if not posting_lists:
            return [], inspected
        result = set(posting_lists[0])
        intersection_ops = 0
        for p in posting_lists[1:]:
            intersection_ops += min(len(result), len(p))
            result.intersection_update(p)
        return sorted(result), inspected + intersection_ops


def bootstrap(values: Sequence[float], seed: int = 20261003, rounds: int = 4000):
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(rounds)
    )
    return [draws[int(0.025 * rounds)], draws[int(0.975 * rounds) - 1]]


def run(smoke: bool):
    seeds = (0,) if smoke else SEEDS
    m_levels = (32, 128) if smoke else M_LEVELS
    tasks_per_m = 4 if smoke else TASKS_PER_M
    rows = []
    checks = []

    for seed in seeds:
        training = g010.generate(seed + 100, TRAIN_PROGRAMS)
        train_s = {g010.structure_key(ep) for ep in training}
        train_t = {g010.truth_signature(ep) for ep in training}
        manifest = g010.build_manifest(
            seed, m_levels, tasks_per_m,
            exclude_structures=train_s, exclude_truths=train_t,
        )
        for m in m_levels:
            for i in range(tasks_per_m):
                target = manifest[(m, i)]
                candidates_task, target_index, _old_support = g010.make_task(
                    seed * 100 + i, m, target, train_s, train_t
                )
                index = BehaviorIndex(candidates_task)
                if len(index.postings) < 2:
                    raise RuntimeError("behavior index unexpectedly degenerate")
                target_sig = g010.truth_signature(target)
                candidate_truths = [g010.truth_signature(ep) for ep in candidates_task]
                if len(candidate_truths) != len(set(candidate_truths)):
                    raise RuntimeError("candidate truth signatures are not unique")
                permutation = support_indices(seed, m, i)
                for support_size in SUPPORT_SIZES:
                    chosen = tuple(permutation[:support_size])
                    support = tuple((INPUT_ROWS[k], target.truth_table[INPUT_ROWS[k]]) for k in chosen)
                    scan = support_consistent(candidates_task, support)
                    indexed, index_work = index.query(support)
                    if scan != indexed:
                        raise RuntimeError("behavior index disagreement with exact scan")
                    if target_index not in scan:
                        raise RuntimeError("target missing from support-consistent set")
                    n_match = len(scan)
                    row = {
                        "seed": seed,
                        "M": m,
                        "task": i,
                        "support_size": support_size,
                        "support_consistent_count": n_match,
                        "target_unique": int(n_match == 1),
                        "expected_uniform_tie_success": 1.0 / n_match,
                        "support_bits": [list(INPUT_ROWS[k]) for k in chosen],
                        "support_labels": [int(target.truth_table[INPUT_ROWS[k]]) for k in chosen],
                        "index_candidate_count": len(indexed),
                        "index_posting_entries_scanned": index_work,
                        "index_build_execution_work": index.build_work,
                        "candidate_truth_signatures_unique": True,
                        "target_truth_signature": format(target_sig, "016b"),
                    }
                    rows.append(row)
        checks.append({
            "seed": seed,
            "candidate_truth_signatures_unique": True,
            "all_support_queries_agree_with_exact_scan": True,
        })

    summaries = {}
    for m in m_levels:
        for s in SUPPORT_SIZES:
            cell = [r for r in rows if r["M"] == m and r["support_size"] == s]
            counts = [r["support_consistent_count"] for r in cell]
            unique = [r["target_unique"] for r in cell]
            tie = [r["expected_uniform_tie_success"] for r in cell]
            work = [r["index_posting_entries_scanned"] for r in cell]
            build = [r["index_build_execution_work"] for r in cell]
            summaries[f"{m}:{s}"] = {
                "tasks": len(cell),
                "support_consistent_count_mean": statistics.fmean(counts),
                "support_consistent_count_median": statistics.median(counts),
                "support_consistent_count_max": max(counts),
                "target_unique_fraction": statistics.fmean(unique),
                "target_unique_seed_means": [
                    statistics.fmean(r["target_unique"] for r in cell if r["seed"] == seed)
                    for seed in seeds
                ],
                "target_unique_seed_bootstrap_ci95": bootstrap([
                    statistics.fmean(r["target_unique"] for r in cell if r["seed"] == seed)
                    for seed in seeds
                ], seed=20261003 + m * 17 + s),
                "expected_uniform_tie_success_mean": statistics.fmean(tie),
                "index_posting_entries_scanned_mean": statistics.fmean(work),
                "index_build_execution_work_mean": statistics.fmean(build),
            }

    payload = {
        "experiment_id": "TACOSM-GRAPH-CASM-IDENTIFIABILITY-011",
        "status": "measured",
        "provenance": {
            "tacosm_commit": os.environ.get("GITHUB_SHA", "local"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "generator_commit": "c31554413301e3c9d3e6b3f8c8c6be572a74a748",
        },
        "protocol": {
            "seeds": list(seeds),
            "M_levels": list(m_levels),
            "tasks_per_seed_M": tasks_per_m,
            "support_sizes": list(SUPPORT_SIZES),
            "input_rows": [list(x) for x in INPUT_ROWS],
        },
        "checks": checks,
        "summary": summaries,
        "task_rows": rows,
    }
    out = ROOT / ("artifacts/TACOSM-GRAPH-CASM-IDENTIFIABILITY-011.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
