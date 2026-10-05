#!/usr/bin/env python3
"""G-CASM-015: structured evidence-channel ceiling.

One-step, target-identity-blind action selection compares a scalar external
output channel against an instrumented activation-trace channel. All
candidate-side predictions are cached before query evaluation; their cost is
reported separately from target-environment acquisition work.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
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
from tac_osm.structured_action_probe import ProbeAction, choose_best_action, score_action


EXPERIMENT_ID = "TACOSM-GRAPH-CASM-STRUCTURED-ACTION-PROBE-015"
SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (32, 64, 128, 256, 512)
TASKS_PER_M = 32
TRAIN_PROGRAMS = 256
LIBRARY_SIZE = 512
INPUT_COUNT = g010.INPUT_COUNT
INPUT_ROWS = tuple(itertools.product((0, 1), repeat=INPUT_COUNT))
MAX_NODES = g010.MAX_NODES
TRACE_READ_UNITS = MAX_NODES - INPUT_COUNT
REGISTERED_TASKS_PER_SEED = len(M_LEVELS) * TASKS_PER_M
CHANNELS = ("scalar_row", "activation_trace")
GENERATOR_COMMIT = "c31554413301e3c9d3e6b3f8c8c6be572a74a748"


@dataclass(frozen=True)
class QueryTask:
    task_id: str
    target_index: int


@dataclass(frozen=True)
class ProbeCache:
    scalar: dict[tuple[int, int], tuple[int, int]]
    trace: dict[tuple[int, int], tuple[tuple[int, ...], int]]


def load_registered_contract():
    from tac_osm.contract import load_contract
    return load_contract(EXPERIMENT_ID)


def build_query_task(seed: int, m: int, task_i: int) -> QueryTask:
    rng = random.Random(seed * 3000001 + m * 10007 + task_i * 1009 + 17)
    target_index = rng.randrange(m)
    digest = hashlib.sha256(
        repr((seed, m, task_i, target_index)).encode()
    ).hexdigest()[:16]
    return QueryTask(
        task_id=f"g15:{seed}:{m}:{task_i}:{digest}",
        target_index=target_index,
    )


def execute_trace(ep, bits: Sequence[int]) -> tuple[int, tuple[int, ...], int]:
    values = {i: int(v) for i, v in zip(ep.inputs, bits)}
    incoming = {}
    for edge in ep.true_edges:
        incoming.setdefault(edge.dst, {})[edge.port] = edge.src

    edge_ops = node_ops = 0
    internal: list[int] = []
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
        else:
            raise RuntimeError(f"unsupported operator: {node.op.value}")
        node_ops += 1
        internal.append(int(values[node.index]))

    if len(internal) != TRACE_READ_UNITS:
        raise RuntimeError(
            f"expected {TRACE_READ_UNITS} internal activations, got {len(internal)}"
        )
    return int(values[ep.output]), tuple(internal), edge_ops + node_ops


def build_cache(library: Sequence) -> tuple[ProbeCache, dict]:
    scalar: dict[tuple[int, int], tuple[int, int]] = {}
    trace: dict[tuple[int, int], tuple[tuple[int, ...], int]] = {}
    scalar_work = 0
    trace_work = 0

    for idx, ep in enumerate(library):
        if ep.active_count != MAX_NODES or len(ep.inputs) != INPUT_COUNT:
            raise RuntimeError("registered generator shape changed")
        for row_index, bits in enumerate(INPUT_ROWS):
            out, work = g010.execute_exact(ep, bits)
            scalar[(idx, row_index)] = (int(out), int(work.total))
            scalar_work += int(work.total)

            trace_out, activation, trace_exec_work = execute_trace(ep, bits)
            if trace_out != out or trace_exec_work != work.total:
                raise RuntimeError("trace executor diverges from exact executor")
            trace[(idx, row_index)] = (activation, int(trace_exec_work))
            trace_work += int(trace_exec_work + TRACE_READ_UNITS)

    return ProbeCache(scalar=scalar, trace=trace), {
        "candidate_cache_scalar_work_units": scalar_work,
        "candidate_cache_trace_work_units": trace_work,
        "candidate_cache_trace_read_units": len(library) * len(INPUT_ROWS) * TRACE_READ_UNITS,
    }


def channel_signature(cache: ProbeCache, channel: str, indices: Sequence[int], row: int):
    if channel == "scalar_row":
        return tuple(cache.scalar[(idx, row)][0] for idx in indices)
    if channel == "activation_trace":
        return tuple(cache.trace[(idx, row)][0] for idx in indices)
    raise ValueError(channel)


def action_scores(cache: ProbeCache, indices: Sequence[int], channel: str):
    scores = []
    for row in range(len(INPUT_ROWS)):
        action = ProbeAction(channel, (row,))
        if channel == "scalar_row":
            evidence = tuple(cache.scalar[(idx, row)][0] for idx in indices)
            cost = statistics.fmean(cache.scalar[(idx, row)][1] for idx in indices)
        elif channel == "activation_trace":
            evidence = tuple(cache.trace[(idx, row)][0] for idx in indices)
            cost = statistics.fmean(
                cache.trace[(idx, row)][1] + TRACE_READ_UNITS
                for idx in indices
            )
        else:
            raise ValueError(channel)
        scores.append(score_action(action, evidence, cost))
    return scores


def evaluate_task(cache: ProbeCache, task: QueryTask, m: int) -> dict:
    indices = tuple(range(m))
    out = {
        "target_index": task.target_index,
        "channels": {},
    }
    for channel in CHANNELS:
        best = choose_best_action(action_scores(cache, indices, channel))
        selected_row = int(best.action.parameters[0])
        if channel == "scalar_row":
            target_evidence = (cache.scalar[(task.target_index, selected_row)][0],)
        else:
            target_evidence = cache.trace[(task.target_index, selected_row)][0]
        out["channels"][channel] = {
            "selected_row": selected_row,
            "information_gain_bits": best.information_gain_bits,
            "expected_remaining_candidates": best.expected_remaining_candidates,
            "signature_count": best.signature_count,
            "expected_environment_work_units": best.expected_cost,
            "information_per_work": best.information_per_work,
            "target_evidence_signature": list(target_evidence),
            "information_alphabet_ceiling": (
                2 if channel == "scalar_row" else 2 ** TRACE_READ_UNITS
            ),
            "selector_prediction_units": int(
                m * len(INPUT_ROWS) * (
                    1 if channel == "scalar_row" else TRACE_READ_UNITS
                )
            ),
        }
    return out


def bootstrap_ci(values: Sequence[float], seed: int, rounds: int = 4000):
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


def summarize(raw, cache_meta, seeds: Sequence[int], m_levels: Sequence[int]):
    channels = ("scalar_row", "activation_trace")
    by_channel = {}
    for channel in channels:
        by_channel[channel] = {}
        for m in m_levels:
            trials = []
            seed_means = []
            for seed in seeds:
                rows = raw[str(seed)][str(m)]
                vals = [r["channels"][channel] for r in rows]
                trials.extend(vals)
                seed_means.append(
                    statistics.fmean(v["information_per_work"] for v in vals)
                )
            by_channel[channel][str(m)] = {
                "trials": len(trials),
                "information_gain_bits_mean": statistics.fmean(
                    v["information_gain_bits"] for v in trials
                ),
                "expected_remaining_candidates_mean": statistics.fmean(
                    v["expected_remaining_candidates"] for v in trials
                ),
                "signature_count_mean": statistics.fmean(
                    v["signature_count"] for v in trials
                ),
                "expected_environment_work_units_mean": statistics.fmean(
                    v["expected_environment_work_units"] for v in trials
                ),
                "information_per_work_mean": statistics.fmean(
                    v["information_per_work"] for v in trials
                ),
                "amortized_total_work_units_mean": statistics.fmean(
                    r["expected_environment_work_units"]
                    + r["selector_prediction_units"]
                    + cache_meta[str(seed)][
                        "candidate_cache_scalar_work_units"
                        if channel == "scalar_row"
                        else "candidate_cache_trace_work_units"
                    ] / REGISTERED_TASKS_PER_SEED
                    for seed in seeds
                    for row in raw[str(seed)][str(m)]
                    for r in (row["channels"][channel],)
                ),
                "amortized_total_information_per_work_mean": statistics.fmean(
                    r["information_gain_bits"] / max(
                        1.0,
                        r["expected_environment_work_units"]
                        + r["selector_prediction_units"]
                        + cache_meta[str(seed)][
                            "candidate_cache_scalar_work_units"
                            if channel == "scalar_row"
                            else "candidate_cache_trace_work_units"
                        ] / REGISTERED_TASKS_PER_SEED,
                    )
                    for seed in seeds
                    for row in raw[str(seed)][str(m)]
                    for r in (row["channels"][channel],)
                ),
                "seed_information_per_work_means": seed_means,
            }

    primary = {}
    for m in m_levels:
        deltas = []
        seed_scalar = []
        seed_trace = []
        for seed in seeds:
            rows = raw[str(seed)][str(m)]
            s = statistics.fmean(
                r["channels"]["scalar_row"]["information_per_work"] for r in rows
            )
            t = statistics.fmean(
                r["channels"]["activation_trace"]["information_per_work"] for r in rows
            )
            seed_scalar.append(s)
            seed_trace.append(t)
            deltas.append(t - s)
        primary[str(m)] = {
            "seed_scalar_information_per_work": seed_scalar,
            "seed_trace_information_per_work": seed_trace,
            "seed_level_paired_delta": deltas,
            "mean_paired_delta": statistics.fmean(deltas),
            "seed_bootstrap_95ci": bootstrap_ci(deltas, 91500 + m),
        }

    return {
        "by_channel": by_channel,
        "primary": primary,
        "cache_accounting": cache_meta,
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
        contract.require_arms(["scalar_row", "activation_trace"])

    raw = {str(seed): {str(m): [] for m in m_levels} for seed in seeds}
    checks = {}
    cache_meta = {}

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

        cache, meta = build_cache(library)
        cache_meta[str(seed)] = meta
        checks[str(seed)] = {
            "train_programs": TRAIN_PROGRAMS,
            "eval_library_size": LIBRARY_SIZE,
            "train_eval_structure_disjoint": True,
            "train_eval_truth_disjoint": True,
            "candidate_library_built_once": True,
            "selector_receives_target_identity": False,
            "selector_receives_target_evidence_before_choice": False,
            "generator_commit": GENERATOR_COMMIT,
            "activation_trace_bits": TRACE_READ_UNITS,
        }

        for m in m_levels:
            for task_i in range(tasks_per_m):
                task = build_query_task(seed, m, task_i)
                result = evaluate_task(cache, task, m)
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
            "channels": ["scalar_row", "activation_trace"],
            "input_count": INPUT_COUNT,
            "input_domain_rows": len(INPUT_ROWS),
            "library_size": LIBRARY_SIZE,
            "train_programs": TRAIN_PROGRAMS,
            "activation_trace_bits": TRACE_READ_UNITS,
            "trace_read_units_per_bit": 1,
            "registered_tasks_per_seed_for_cache_amortization": REGISTERED_TASKS_PER_SEED,
            "selector_prediction_units": "M * 16 * 1 for scalar_row; M * 16 * 6 for activation_trace",
            "primary_cost_scope": "environment acquisition only",
            "secondary_cost_scope": "candidate cache amortized over full registered seed grid plus selector prediction scan",
        },
        "checks": checks,
        "summary": summarize(raw, cache_meta, seeds, m_levels),
        "results": raw,
        "scope": {
            "exact_observation_channel_ceiling": True,
            "target_identity_blind": True,
            "learned_probe_policy_claim": False,
            "global_optimal_sequential_policy_claim": False,
            "asymptotic_claim": False,
            "language_image_audio_claim": False,
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
    args = parser.parse_args()
    main(smoke=args.smoke)
