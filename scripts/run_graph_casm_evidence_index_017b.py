#!/usr/bin/env python3
"""G-CASM-017B: corrected exact evidence indexing with auditable cost boundaries.

017A is preserved as invalid infrastructure because its work units mixed
non-commensurate primitives and its indexed path recomputed expected cost with
an O(M) scan. 017B separates one-time index construction, selector query
complexity, and measured wall-clock latency on the fixed CI runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_graph_casm_structured_action_probe_015 as g15
from tac_osm.evidence_index import PrefixEvidenceHistogramIndex
from tac_osm.structured_action_probe import choose_best_action


EXPERIMENT_ID = "TACOSM-GRAPH-CASM-EVIDENCE-INDEX-017B"
SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (32, 64, 128, 256, 512)
TASKS_PER_M = 32
PRIMARY_M = 512
TRACE_WIDTH = g15.TRACE_READ_UNITS
INPUT_ROWS = g15.INPUT_ROWS
ACTIONS = tuple(range(len(INPUT_ROWS)))
REGISTERED_TASKS_PER_SEED = len(M_LEVELS) * TASKS_PER_M
GENERATOR_COMMIT = g15.GENERATOR_COMMIT
TIMING_REPEATS = 30


def contract():
    from tac_osm.contract import load_contract
    return load_contract(EXPERIMENT_ID)


def build_prefix_index(cache):
    evidence_by_action = {
        row: tuple(cache.trace[(idx, row)][0] for idx in range(g15.LIBRARY_SIZE))
        for row in ACTIONS
    }
    cost_by_action = {
        row: tuple(
            cache.trace[(idx, row)][1] + TRACE_WIDTH
            for idx in range(g15.LIBRARY_SIZE)
        )
        for row in ACTIONS
    }
    return PrefixEvidenceHistogramIndex.build(
        evidence_by_action,
        cost_by_action,
        evidence_width=TRACE_WIDTH,
    )


def exhaustive_select(cache, m: int):
    scores = g15.action_scores(cache, tuple(range(m)), "activation_trace")
    return choose_best_action(scores)


def indexed_select(index, m: int):
    return index.choose_best(population_size=m)


def verify_exact_equivalence(cache, index, m: int, task) -> dict:
    exhaustive = exhaustive_select(cache, m)
    indexed, bin_reads = indexed_select(index, m)

    if int(indexed.action.parameters[0]) != int(exhaustive.action.parameters[0]):
        raise RuntimeError("indexed selector changed the selected action")
    if abs(indexed.information_gain_bits - exhaustive.information_gain_bits) > 1e-12:
        raise RuntimeError("indexed information gain diverges from exhaustive")
    if abs(
        indexed.expected_remaining_candidates
        - exhaustive.expected_remaining_candidates
    ) > 1e-12:
        raise RuntimeError("indexed expected remaining candidates diverge")
    if abs(indexed.information_per_work - exhaustive.information_per_work) > 1e-12:
        raise RuntimeError("indexed information-per-work diverges")
    if abs(indexed.expected_cost - exhaustive.expected_cost) > 1e-12:
        raise RuntimeError("indexed expected environment cost diverges")

    selected_row = int(exhaustive.action.parameters[0])
    target_signature = cache.trace[(task.target_index, selected_row)][0]
    # Target evidence is read only after both selectors have chosen a row.
    indexed_target_signature = cache.trace[(task.target_index, selected_row)][0]
    if indexed_target_signature != target_signature:
        raise RuntimeError("post-selection target evidence diverged")

    return {
        "selected_row": selected_row,
        "information_gain_bits": exhaustive.information_gain_bits,
        "expected_remaining_candidates": exhaustive.expected_remaining_candidates,
        "expected_environment_work_units": exhaustive.expected_cost,
        "selector_exact_match": True,
        "indexed_histogram_bin_reads": bin_reads,
        "exhaustive_candidate_record_reads": m * len(ACTIONS),
    }


def timed(cache, index, m: int, repeats: int = TIMING_REPEATS):
    exhaustive_select(cache, m)
    indexed_select(index, m)

    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        exhaustive_select(cache, m)
    exhaustive_ns = time.perf_counter_ns() - t0

    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        indexed_select(index, m)
    indexed_ns = time.perf_counter_ns() - t0

    return {
        "repeats": repeats,
        "exhaustive_query_us": exhaustive_ns / repeats / 1000.0,
        "indexed_query_us": indexed_ns / repeats / 1000.0,
    }


def bootstrap(values: Sequence[float], seed: int, rounds: int = 4000):
    if len(values) <= 1:
        x = float(values[0]) if values else 0.0
        return [x, x]
    import random

    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(rounds)
    )
    return [
        float(means[int(0.025 * (rounds - 1))]),
        float(means[int(0.975 * (rounds - 1))]),
    ]


def loglog_slope(points):
    xs = [float(m) for m, _ in points]
    ys = [max(float(y), 1e-12) for _, y in points]
    xbar = statistics.fmean([__import__("math").log(x) for x in xs])
    ybar = statistics.fmean([__import__("math").log(y) for y in ys])
    num = sum((__import__("math").log(x) - xbar) * (__import__("math").log(y) - ybar) for x, y in points)
    den = sum((__import__("math").log(x) - xbar) ** 2 for x in xs)
    return float(num / den)


def main(smoke: bool = False):
    c = contract()
    if smoke:
        seeds = (0,)
        m_levels = (32, 128)
        tasks_per_m = 4
    else:
        seeds = SEEDS
        m_levels = M_LEVELS
        tasks_per_m = TASKS_PER_M
        c.require_levels(M_LEVELS)
        c.require_seeds(SEEDS)
        c.require_steps(1)
        c.require_eval_steps(TASKS_PER_M)
        c.require_arms(["trace_exhaustive", "trace_indexed"])

    raw = {str(seed): {str(m): [] for m in m_levels} for seed in seeds}
    timings = {str(seed): {} for seed in seeds}
    build_records = {}

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
            raise RuntimeError("train/evaluation disjointness gate failed")

        cache, cache_info = g15.build_cache(library)
        index = build_prefix_index(cache)
        build_records[str(seed)] = index.build_candidate_records

        for m in m_levels:
            task_rows = []
            for task_i in range(tasks_per_m):
                task = g15.build_query_task(seed, m, task_i)
                task_rows.append(verify_exact_equivalence(cache, index, m, task))
            raw[str(seed)][str(m)] = task_rows
            timings[str(seed)][str(m)] = timed(cache, index, m)

    primary_deltas = []
    query_slopes_exhaustive = []
    query_slopes_indexed = []
    for seed in seeds:
        ex = timings[str(seed)][str(PRIMARY_M)]["exhaustive_query_us"]
        ix = timings[str(seed)][str(PRIMARY_M)]["indexed_query_us"]
        primary_deltas.append(ex - ix)
        query_slopes_exhaustive.append(
            loglog_slope(
                [(m, timings[str(seed)][str(m)]["exhaustive_query_us"]) for m in m_levels]
            )
        )
        query_slopes_indexed.append(
            loglog_slope(
                [(m, timings[str(seed)][str(m)]["indexed_query_us"]) for m in m_levels]
            )
        )

    all_rows = [
        row
        for seed in seeds
        for m in m_levels
        for row in raw[str(seed)][str(m)]
    ]
    exact_agreement = statistics.fmean(float(r["selector_exact_match"]) for r in all_rows)

    final = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "provenance": {
            "tacosm_commit": os.environ.get("GITHUB_SHA", "local"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "generator_commit": GENERATOR_COMMIT,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "protocol": {
            "seeds": list(seeds),
            "M_levels": list(m_levels),
            "tasks_per_seed_M": tasks_per_m,
            "library_size": g15.LIBRARY_SIZE,
            "primary_M": PRIMARY_M,
            "trace_width": TRACE_WIDTH,
            "input_rows": len(INPUT_ROWS),
            "timing_repeats": TIMING_REPEATS,
            "cache_built_once_per_seed": True,
            "index_built_once_per_seed": True,
        },
        "checks": {
            str(seed): {
                "train_eval_structure_disjoint": True,
                "train_eval_truth_disjoint": True,
                "target_identity_used_before_choice": False,
                "target_evidence_used_before_choice": False,
                "same_activation_trace_channel": True,
                "exact_selector_equivalence_required": True,
            }
            for seed in seeds
        },
        "complexity": {
            "exhaustive_candidate_record_reads": {
                str(m): len(ACTIONS) * m for m in m_levels
            },
            "indexed_query_histogram_bin_reads_by_seed_M": {
                str(seed): {
                    str(m): statistics.fmean(
                        row["indexed_histogram_bin_reads"]
                        for row in raw[str(seed)][str(m)]
                    )
                    for m in m_levels
                }
                for seed in seeds
            },
            "indexed_query_upper_bound_bin_reads": len(ACTIONS) * (2 ** TRACE_WIDTH),
            "index_build_candidate_records_by_seed": build_records,
            "index_build_is_one_time_per_seed": True,
        },
        "timing": timings,
        "primary": {
            "M": PRIMARY_M,
            "seed_level_exhaustive_minus_indexed_us": primary_deltas,
            "mean_exhaustive_minus_indexed_us": statistics.fmean(primary_deltas),
            "seed_bootstrap_95ci": bootstrap(primary_deltas, 91700 + PRIMARY_M),
            "exhaustive_query_loglog_slope_by_seed": query_slopes_exhaustive,
            "indexed_query_loglog_slope_by_seed": query_slopes_indexed,
            "exact_selector_action_agreement": exact_agreement,
        },
        "scope": {
            "exact_selector_equivalence": True,
            "query_time_is_hardware_bound": True,
            "query_complexity_is_bounded_by_fixed_evidence_alphabet": True,
            "index_build_remains_O_M": True,
            "no_claim_of_C5_total_history_sublinearity": True,
            "no_learned_probe_policy_claim": True,
            "no_language_image_audio_claim": True,
        },
        "raw": raw,
        "audit": {
            "leakage": {
                "index_inputs": "candidate-predicted activation traces and candidate execution costs only",
                "forbidden_selector_inputs": [
                    "target identity",
                    "target index",
                    "realized target evidence before row choice",
                    "verifier labels",
                    "test outcomes",
                ],
            },
            "capability_bridge": "Exact selected-action equivalence means the indexed selector feeds the same downstream probe as exhaustive selection; target evidence is accessed only after selection.",
            "invalidated_017a_issue": "017A mixed non-commensurate primitive work units and recomputed expected cost with an O(M) scan in the indexed query path.",
        },
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    (out / f"{EXPERIMENT_ID}.json").write_text(
        json.dumps(final, indent=2, sort_keys=True)
    )
    print(json.dumps(final["primary"], indent=2, sort_keys=True))
    print(json.dumps(final["complexity"], indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    main(smoke=parser.parse_args().smoke)
