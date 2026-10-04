#!/usr/bin/env python3
"""G-CASM-014: adaptive evidence acquisition ceiling.

The intervention changes only which observable behavior row is queried next.
It uses exact public candidate truth tables and the observations obtained so
far; it never receives target identity or target index.
"""

from __future__ import annotations

import argparse
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_graph_casm_selective_010 as g010
from tac_osm.adaptive_behavioral_probe import AdaptiveBehavioralProbeSelector

EXPERIMENT_ID = "TACOSM-GRAPH-CASM-ADAPTIVE-PROBE-014"
SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (32, 64, 128, 256, 512)
TASKS_PER_M = 32
PROBE_BUDGETS = (1, 2, 4, 6, 8)
PRIMARY_K = 4
CANDIDATE_BUDGETS = (1, 2, 4, 8)
PRIMARY_B = 8
LIBRARY_SIZE = 512
TRAIN_PROGRAMS = 256
INPUT_COUNT = g010.INPUT_COUNT
INPUT_ROWS = tuple(itertools.product((0, 1), repeat=INPUT_COUNT))
GENERATOR_COMMIT = "c31554413301e3c9d3e6b3f8c8c6be572a74a748"

@dataclass(frozen=True)
class QueryTask:
    task_id: str
    target_index: int

def load_registered_contract():
    from tac_osm.contract import load_contract
    return load_contract(EXPERIMENT_ID)

def row_order(seed: int, m: int, task_i: int) -> tuple[int, ...]:
    rng = random.Random(seed * 1000003 + m * 7919 + task_i * 104729 + 1301)
    rows = list(range(len(INPUT_ROWS)))
    rng.shuffle(rows)
    return tuple(rows)

def build_query_task(seed: int, m: int, task_i: int) -> QueryTask:
    rng = random.Random(seed * 3000001 + m * 10007 + task_i * 1009 + 17)
    target_index = rng.randrange(m)
    digest = hashlib.sha256(
        repr((seed, m, task_i, target_index)).encode()
    ).hexdigest()[:16]
    return QueryTask(
        task_id=f"g14:{seed}:{m}:{task_i}:{digest}",
        target_index=target_index,
    )

def filter_compatible(
    library: Sequence,
    indices: Sequence[int],
    row_index: int,
    output: int,
) -> tuple[int, ...]:
    y = int(output)
    return tuple(
        idx for idx in indices
        if int(library[idx].truth_table[INPUT_ROWS[row_index]]) == y
    )

def adaptive_trajectory(
    library: Sequence,
    task: QueryTask,
    max_steps: int,
    population_size: int,
) -> dict:
    selector = AdaptiveBehavioralProbeSelector()
    compatible = tuple(range(population_size))
    available = set(range(len(INPUT_ROWS)))
    chosen: list[int] = []
    counts: list[int] = []
    decisions: list[dict] = []

    for _ in range(max_steps):
        decision = selector.choose(
            library, compatible, available, row_keys=INPUT_ROWS
        )
        row = decision.row_index
        y = int(library[task.target_index].truth_table[INPUT_ROWS[row]])

        chosen.append(row)
        available.remove(row)
        compatible = selector.filter_compatible(
            library, compatible, row, y, row_keys=INPUT_ROWS
        )
        counts.append(len(compatible))
        decisions.append({
            "row_index": row,
            "output": y,
            "partition_zero": decision.partition_zero,
            "partition_one": decision.partition_one,
            "worst_case_remaining": decision.worst_case_remaining,
            "remaining_after_observation": len(compatible),
        })
        if not compatible:
            raise RuntimeError("target-identity-free update eliminated target")
        if len(compatible) == 1 and not available:
            break

    return {
        "probe_rows": chosen,
        "compatible_counts": counts,
        "decisions": decisions,
    }

def fixed_trajectory(
    library: Sequence,
    task: QueryTask,
    max_steps: int,
    seed: int,
    m: int,
    task_i: int,
) -> dict:
    order = row_order(seed, m, task_i)
    compatible = tuple(range(m))
    chosen: list[int] = []
    counts: list[int] = []
    for row in order[:max_steps]:
        y = int(library[task.target_index].truth_table[INPUT_ROWS[row]])
        compatible = filter_compatible(library, compatible, row, y)
        chosen.append(row)
        counts.append(len(compatible))
        if not compatible:
            raise RuntimeError("fixed update eliminated target")
    return {
        "probe_rows": chosen,
        "compatible_counts": counts,
        "decisions": [],
    }

def verify_selected(
    library: Sequence,
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
            got, w = g010.execute_exact(library[idx], bits)
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

def evaluate_strategy(
    strategy: str,
    library: Sequence,
    task: QueryTask,
    seed: int,
    m: int,
    task_i: int,
) -> dict:
    if strategy == "adaptive_minimax":
        traj = adaptive_trajectory(
            library, task, max(PROBE_BUDGETS), m
        )
    elif strategy == "fixed_random":
        traj = fixed_trajectory(library, task, max(PROBE_BUDGETS), seed, m, task_i)
    else:
        raise ValueError(strategy)

    out = {"target_index": task.target_index, "probe_budgets": {}}
    for k in PROBE_BUDGETS:
        probe_rows = tuple(traj["probe_rows"][:k])
        compatible = tuple(
            range(m)
        )
        for row in probe_rows:
            y = int(library[task.target_index].truth_table[INPUT_ROWS[row]])
            compatible = filter_compatible(library, compatible, row, y)

        verify_rows = tuple(
            (INPUT_ROWS[row], int(library[task.target_index].truth_table[INPUT_ROWS[row]]))
            for row in range(len(INPUT_ROWS))
            if row not in set(probe_rows)
        )
        exhaustive = verify_selected(
            library, tuple(range(m)), verify_rows, adaptive=False
        )
        if exhaustive["success"] != 1.0:
            raise RuntimeError("exhaustive verifier did not recover the target")
        if exhaustive["rows_evaluated"] != m * len(verify_rows):
            raise RuntimeError("exhaustive row-count invariant failed")

        candidate_order = tuple(sorted(compatible))
        budget_rows = {}
        for b in CANDIDATE_BUDGETS:
            selected = candidate_order[:b]
            fixed = verify_selected(
                library, selected, verify_rows, adaptive=False
            )
            adaptive = verify_selected(
                library, selected, verify_rows, adaptive=True
            )
            expected_rows = len(selected) * len(verify_rows)
            if fixed["rows_evaluated"] != expected_rows:
                raise RuntimeError("fixed-budget row-count invariant failed")
            budget_rows[str(b)] = {
                "target_in_budget": float(task.target_index in selected),
                "fixed_success": fixed["success"],
                "adaptive_success": adaptive["success"],
                "fixed_work_fraction": fixed["work_units"] / max(1, exhaustive["work_units"]),
                "adaptive_work_fraction": adaptive["work_units"] / max(1, exhaustive["work_units"]),
                "selected_candidates": len(selected),
            }

        out["probe_budgets"][str(k)] = {
            "probe_rows": list(probe_rows),
            "compatible_candidate_count": len(compatible),
            "unique_candidate": float(len(compatible) == 1),
            "target_rank_index_only": (
                candidate_order.index(task.target_index) + 1
                if task.target_index in candidate_order else None
            ),
            "verifier_rows": len(verify_rows),
            "budgets": budget_rows,
        }
    return out

def bootstrap_ci(values: Sequence[float], seed: int, rounds: int = 4000):
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        x = float(values[0])
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

def summarize(raw, seeds: Sequence[int]) -> dict:
    strategies = ("adaptive_minimax", "fixed_random")
    summary = {}
    for strategy in strategies:
        summary[strategy] = {}
        for m in M_LEVELS:
            summary[strategy][str(m)] = {}
            for k in PROBE_BUDGETS:
                flat = []
                for seed in seeds:
                    flat.extend(raw[strategy][str(seed)][str(m)][str(k)])
                row = {
                    "trials": len(flat),
                    "compatible_candidate_count_mean": statistics.fmean(
                        t["compatible_candidate_count"] for t in flat
                    ),
                    "unique_candidate_fraction": statistics.fmean(
                        t["unique_candidate"] for t in flat
                    ),
                    "target_rank_index_only_mean": statistics.fmean(
                        float(t["target_rank_index_only"])
                        for t in flat
                        if t["target_rank_index_only"] is not None
                    ),
                }
                for b in CANDIDATE_BUDGETS:
                    fixed = statistics.fmean(
                        t["budgets"][str(b)]["fixed_success"] for t in flat
                    )
                    adaptive = statistics.fmean(
                        t["budgets"][str(b)]["adaptive_success"] for t in flat
                    )
                    work = statistics.fmean(
                        t["budgets"][str(b)]["adaptive_work_fraction"] for t in flat
                    )
                    row.setdefault("budgets", {})[str(b)] = {
                        "fixed_success": fixed,
                        "adaptive_success": adaptive,
                        "adaptive_work_fraction": work,
                    }
                summary[strategy][str(m)][str(k)] = row

    primary = {}
    for m in M_LEVELS:
        seed_deltas = []
        seed_adaptive = []
        seed_fixed = []
        for seed in seeds:
            a = raw["adaptive_minimax"][str(seed)][str(m)][str(PRIMARY_K)]
            f = raw["fixed_random"][str(seed)][str(m)][str(PRIMARY_K)]
            am = statistics.fmean(
                t["budgets"][str(PRIMARY_B)]["fixed_success"] for t in a
            )
            fm = statistics.fmean(
                t["budgets"][str(PRIMARY_B)]["fixed_success"] for t in f
            )
            seed_adaptive.append(am)
            seed_fixed.append(fm)
            seed_deltas.append(am - fm)
        primary[str(m)] = {
            "seed_adaptive_success": seed_adaptive,
            "seed_fixed_success": seed_fixed,
            "seed_level_paired_delta": seed_deltas,
            "mean_paired_delta": statistics.fmean(seed_deltas),
            "seed_bootstrap_95ci": bootstrap_ci(seed_deltas, 81000 + m),
        }
    return {"by_strategy": summary, "primary": primary}

def main(smoke: bool = False):
    contract = load_registered_contract()
    if smoke:
        seeds = (0,)
        m_levels = (32, 128)
        tasks_per_m = 4
    else:
        seeds = SEEDS
        m_levels = M_LEVELS
        tasks_per_m = TASKS_PER_M
        contract.require_levels(M_LEVELS)
        contract.require_seeds(SEEDS)
        contract.require_steps(0)
        contract.require_eval_steps(TASKS_PER_M)
        contract.require_k_levels(PROBE_BUDGETS)
        contract.require_arms(["adaptive_minimax", "fixed_random"])

    raw = {
        "adaptive_minimax": {str(s): {} for s in seeds},
        "fixed_random": {str(s): {} for s in seeds},
    }
    checks = {}

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
            raise RuntimeError("train/eval disjointness gate failed")

        checks[str(seed)] = {
            "train_programs": TRAIN_PROGRAMS,
            "eval_library_size": LIBRARY_SIZE,
            "train_eval_structure_disjoint": True,
            "train_eval_truth_disjoint": True,
            "selector_receives_target_identity": False,
            "selector_receives_target_output_before_choice": False,
            "candidate_library_built_once": True,
            "generator_commit": GENERATOR_COMMIT,
        }
        for strategy in raw:
            raw[strategy][str(seed)] = {str(m): {} for m in m_levels}

        for m in m_levels:
            for task_i in range(tasks_per_m):
                task = build_query_task(seed, m, task_i)
                for strategy in raw:
                    result = evaluate_strategy(
                        strategy, library, task, seed, m, task_i
                    )
                    for k in PROBE_BUDGETS:
                        raw[strategy][str(seed)][str(m)].setdefault(str(k), []).append(
                            result["probe_budgets"][str(k)]
                        )
                    # Target must remain compatible for both strategies.
                    for k in PROBE_BUDGETS:
                        if result["probe_budgets"][str(k)]["compatible_candidate_count"] < 1:
                            raise RuntimeError("target was eliminated")

    final = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "provenance": {
            "tacosm_commit": os.environ.get("GITHUB_SHA", "local"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "generator_commit": GENERATOR_COMMIT,
            "python": sys.version,
        },
        "protocol": {
            "seeds": list(seeds),
            "M_levels": list(m_levels),
            "tasks_per_seed_M": tasks_per_m,
            "probe_budgets": list(PROBE_BUDGETS),
            "primary_probe_budget": PRIMARY_K,
            "candidate_budgets": list(CANDIDATE_BUDGETS),
            "primary_candidate_budget": PRIMARY_B,
            "input_count": INPUT_COUNT,
            "input_domain_rows": len(INPUT_ROWS),
            "library_size": LIBRARY_SIZE,
            "train_programs": TRAIN_PROGRAMS,
        },
        "checks": checks,
        "summary": summarize(raw, seeds),
        "results": raw,
        "scope": {
            "exact_selector_is_finite_domain_ceiling": True,
            "target_identity_blind": True,
            "asymptotic_claim": False,
            "learned_probe_policy_claim": False,
            "language_image_audio_claim": False,
        },
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    (out / f"{EXPERIMENT_ID}.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final["summary"], indent=2, sort_keys=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args().smoke)
