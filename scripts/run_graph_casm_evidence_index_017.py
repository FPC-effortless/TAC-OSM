#!/usr/bin/env python3
"""G-CASM-017: exact indexed structured-evidence selection.

The intervention changes only the candidate-side implementation of the
activation-trace probe selector. The indexed selector must reproduce the
exhaustive selector exactly while reducing its explicit scan work.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_graph_casm_structured_action_probe_015 as g15
from tac_osm.evidence_index import ExactEvidenceIndex
from tac_osm.structured_action_probe import choose_best_action


EXPERIMENT_ID = "TACOSM-GRAPH-CASM-EVIDENCE-INDEX-017"
SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (32, 64, 128, 256, 512)
TASKS_PER_M = 32
PRIMARY_M = 512
TRACE_WIDTH = g15.TRACE_READ_UNITS
INPUT_ROWS = g15.INPUT_ROWS
ACTIONS = tuple(range(len(INPUT_ROWS)))
REGISTERED_TASKS_PER_SEED = len(M_LEVELS) * TASKS_PER_M
GENERATOR_COMMIT = g15.GENERATOR_COMMIT


def load_registered_contract():
    from tac_osm.contract import load_contract
    return load_contract(EXPERIMENT_ID)


def build_index(cache, library_size: int):
    evidence_by_action = {}
    costs = {}
    for row in ACTIONS:
        evidence_by_action[row] = tuple(
            cache.trace[(idx, row)][0] for idx in range(library_size)
        )
        costs[row] = statistics.fmean(
            cache.trace[(idx, row)][1] + TRACE_WIDTH
            for idx in range(library_size)
        )
    index = ExactEvidenceIndex.build(
        evidence_by_action,
        evidence_width=TRACE_WIDTH,
    )
    return index, costs


def exhaustive_select(cache, m: int):
    scores = g15.action_scores(cache, tuple(range(m)), "activation_trace")
    chosen = choose_best_action(scores)
    return chosen, m * len(INPUT_ROWS) * TRACE_WIDTH


def indexed_select(index, costs, m: int):
    chosen, query_units = index.choose_best(
        index.full_bitmap(m),
        population_size=m,
        expected_cost_by_action={
            row: costs[row]
            for row in ACTIONS
        },
    )
    return chosen, query_units


def evaluate(cache, index, costs, m: int, task, cache_amortized: float, index_build_amortized: float):
    exhaustive, exhaustive_units = exhaustive_select(cache, m)
    indexed, indexed_units = indexed_select(index, costs, m)

    if int(indexed.action.parameters[0]) != int(exhaustive.action.parameters[0]):
        raise RuntimeError(
            "indexed selector changed the chosen probe row: "
            f"{indexed.action.parameters[0]} != {exhaustive.action.parameters[0]}"
        )

    if abs(
        indexed.information_gain_bits - exhaustive.information_gain_bits
    ) > 1e-12:
        raise RuntimeError("indexed information gain diverges from exhaustive")
    if abs(
        indexed.expected_remaining_candidates
        - exhaustive.expected_remaining_candidates
    ) > 1e-12:
        raise RuntimeError(
            "indexed expected remaining candidates diverge from exhaustive"
        )
    if abs(indexed.information_per_work - exhaustive.information_per_work) > 1e-12:
        raise RuntimeError("indexed information-per-cost diverges from exhaustive")

    row = int(exhaustive.action.parameters[0])
    env_work = cache.trace[(task.target_index, row)][1] + TRACE_WIDTH
    exhaustive_total_work = (
        exhaustive_units + float(exhaustive.expected_cost) + cache_amortized
    )
    indexed_total_work = (
        indexed_units
        + float(indexed.expected_cost)
        + cache_amortized
        + index_build_amortized
    )
    result = {
        "target_index": task.target_index,
        "selected_row": row,
        "information_gain_bits": exhaustive.information_gain_bits,
        "expected_remaining_candidates": exhaustive.expected_remaining_candidates,
        "expected_environment_work_units": float(exhaustive.expected_cost),
        "exhaustive_selector_work_units": exhaustive_units,
        "indexed_selector_work_units": indexed_units,
        "exhaustive_environment_information_per_work": exhaustive.information_per_work,
        "indexed_environment_information_per_work": indexed.information_per_work,
        "exhaustive_total_work_units": float(exhaustive_total_work),
        "indexed_total_work_units": float(indexed_total_work),
        "exhaustive_information_per_total_work": exhaustive.information_gain_bits / max(1.0, exhaustive_total_work),
        "indexed_information_per_total_work": exhaustive.information_gain_bits / max(1.0, indexed_total_work),
        "realized_target_environment_work_units": float(env_work),
        "selector_exact_match": True,
    }
    return result


def bootstrap_ci(values: Sequence[float], seed: int, rounds: int = 4000):
    if len(values) <= 1:
        x = float(values[0]) if values else 0.0
        return [x, x]
    rng = __import__("random").Random(seed)
    n = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(rounds)
    )
    return [
        float(draws[int(0.025 * (rounds - 1))]),
        float(draws[int(0.975 * (rounds - 1))]),
    ]


def summarize(raw, cache_meta, index_meta, seeds, m_levels):
    by_m = {}
    primary_seed_deltas = []

    for m in m_levels:
        ex = []
        ix = []
        exact_matches = []
        for seed in seeds:
            rows = raw[str(seed)][str(m)]
            ex.append(
                statistics.fmean(r["exhaustive_information_per_total_work"] for r in rows)
            )
            ix.append(
                statistics.fmean(r["indexed_information_per_total_work"] for r in rows)
            )
            exact_matches.extend(r["selector_exact_match"] for r in rows)

        by_m[str(m)] = {
            "trials": len(exact_matches),
            "exhaustive_information_per_total_work_mean": statistics.fmean(ex),
            "indexed_information_per_total_work_mean": statistics.fmean(ix),
            "mean_total_information_per_work_delta": statistics.fmean(ix[i] - ex[i] for i in range(len(seeds))),
            "selector_exact_action_agreement": statistics.fmean(
                float(x) for x in exact_matches
            ),
            "exhaustive_selector_work_units_mean": statistics.fmean(
                r["exhaustive_selector_work_units"]
                for seed in seeds
                for r in raw[str(seed)][str(m)]
            ),
            "indexed_selector_work_units_mean": statistics.fmean(
                r["indexed_selector_work_units"]
                for seed in seeds
                for r in raw[str(seed)][str(m)]
            ),
            "cache_work_amortized_mean": statistics.fmean(
                cache_meta[str(seed)] / REGISTERED_TASKS_PER_SEED
                for seed in seeds
            ),
            "index_build_work_amortized_mean": statistics.fmean(
                index_meta[str(seed)] / REGISTERED_TASKS_PER_SEED
                for seed in seeds
            ),
        }

    for seed in seeds:
        rows = raw[str(seed)][str(PRIMARY_M)]
        ex = statistics.fmean(
            r["exhaustive_information_per_work"] for r in rows
        )
        ix = statistics.fmean(
            r["indexed_information_per_work"] for r in rows
        )
        primary_seed_deltas.append(ix - ex)

    return {
        "by_m": by_m,
        "primary": {
            "M": PRIMARY_M,
            "seed_level_paired_delta": primary_seed_deltas,
            "mean_paired_delta": statistics.fmean(primary_seed_deltas),
            "seed_bootstrap_95ci": bootstrap_ci(
                primary_seed_deltas, 91700 + PRIMARY_M
            ),
        },
    }


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
        contract.require_steps(1)
        contract.require_eval_steps(TASKS_PER_M)
        contract.require_arms(["trace_exhaustive", "trace_indexed"])

    raw = {
        str(seed): {str(m): [] for m in m_levels}
        for seed in seeds
    }
    cache_meta = {}
    index_meta = {}

    for seed in seeds:
        training = g15.g010.generate(seed + 100, g15.TRAIN_PROGRAMS)
        train_s = {g15.g010.structure_key(ep) for ep in training}
        train_t = {g15.g010.truth_signature(ep) for ep in training}
        library = g15.g010.generate(
            seed + 5000,
            g15.LIBRARY_SIZE,
            exclude_structures=train_s,
            exclude_truths=train_t,
        )
        library_s = {g15.g010.structure_key(ep) for ep in library}
        library_t = {g15.g010.truth_signature(ep) for ep in library}
        if len(library) != g15.LIBRARY_SIZE:
            raise RuntimeError("evaluation library construction failed")
        if train_s & library_s or train_t & library_t:
            raise RuntimeError("train/eval disjointness gate failed")

        cache, cache_info = g15.build_cache(library)
        index, costs = build_index(cache, len(library))
        cache_meta[str(seed)] = cache_info["candidate_cache_trace_work_units"]
        index_meta[str(seed)] = (
            index.build_prediction_units + index.build_posting_writes
        )

        for m in m_levels:
            for task_i in range(tasks_per_m):
                task = g15.build_query_task(seed, m, task_i)
                cache_amortized = (
                    cache_info["candidate_cache_trace_work_units"]
                    / REGISTERED_TASKS_PER_SEED
                )
                index_build_amortized = (
                    index.build_prediction_units
                    + index.build_posting_writes
                ) / REGISTERED_TASKS_PER_SEED
                result = evaluate(
                    cache,
                    index,
                    costs,
                    m,
                    task,
                    cache_amortized,
                    index_build_amortized,
                )
                raw[str(seed)][str(m)].append(result)

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
            "input_rows": len(INPUT_ROWS),
            "trace_width": TRACE_WIDTH,
            "library_size": g15.LIBRARY_SIZE,
            "primary_M": PRIMARY_M,
            "cache_amortization_tasks_per_seed": REGISTERED_TASKS_PER_SEED,
            "exact_selector_required": True,
        },
        "checks": {
            str(seed): {
                "train_eval_structure_disjoint": True,
                "train_eval_truth_disjoint": True,
                "target_identity_used": False,
                "target_evidence_used_before_choice": False,
                "same_activation_trace_channel": True,
                "indexed_selector_exact_equivalence_enforced": True,
            }
            for seed in seeds
        },
        "summary": summarize(raw, cache_meta, index_meta, seeds, m_levels),
        "results": raw,
        "scope": {
            "exact_finite_domain_index_claim": True,
            "learned_probe_policy_claim": False,
            "asymptotic_claim": False,
            "language_image_audio_claim": False,
            "hardware_speedup_claim": False,
        },
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    (out / f"{EXPERIMENT_ID}.json").write_text(
        json.dumps(final, indent=2, sort_keys=True)
    )
    print(json.dumps(final["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    main(smoke=parser.parse_args().smoke)
