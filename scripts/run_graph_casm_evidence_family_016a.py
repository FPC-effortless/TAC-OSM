#!/usr/bin/env python3
"""G-CASM-016A: fixed-cost external structured evidence comparison.

The action selector is an exact finite-domain information-per-environment-cost
oracle. Target identity is used only after action selection to instantiate the
post-selection verification/utility measurement.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_graph_casm_selective_010 as g010
from tac_osm.evidence_family import (
    budget_success,
    effective_information_bits,
    expected_remaining,
    information_bits,
)

EXPERIMENT_ID = "TACOSM-GRAPH-CASM-EVIDENCE-FAMILY-016A"
SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (32, 64, 128, 256, 512)
TASKS_PER_M = 32
TRAIN_PROGRAMS = 256
LIBRARY_SIZE = 512
BUDGET = 8
INPUT_COUNT = g010.INPUT_COUNT
INPUT_ROWS = tuple(itertools.product((0, 1), repeat=INPUT_COUNT))
ROW_TO_INDEX = {row: i for i, row in enumerate(INPUT_ROWS)}
DELTA = (1, 0, 0, 0)
TRACE_BITS = g010.MAX_NODES - g010.INPUT_COUNT
CHANNELS = (
    "scalar_output",
    "paired_output",
    "quad_output",
    "activation_trace_ceiling",
)
EXTERNAL_CHANNELS = ("paired_output", "quad_output")
ENVIRONMENT_WORK = {
    "scalar_output": 1.0,
    "paired_output": 2.0,
    "quad_output": 4.0,
    "activation_trace_ceiling": 1.0 + TRACE_BITS,
}
REGISTERED_TASKS_PER_SEED = len(M_LEVELS) * TASKS_PER_M
GENERATOR_COMMIT = "c31554413301e3c9d3e6b3f8c8c6be572a74a748"


def truth_signature(ep):
    return tuple(int(ep.truth_table[k]) for k in sorted(ep.truth_table))


def structure_key(ep):
    return (
        tuple(
            (n.index, n.op.value, n.depth, n.arity)
            for n in ep.nodes[:ep.active_count]
        ),
        tuple(sorted((e.src, e.dst, e.port) for e in ep.true_edges)),
        tuple(ep.inputs),
        ep.output,
    )


def task_descriptor(seed: int, m: int, task_i: int):
    rng = random.Random(seed * 3000001 + m * 10007 + task_i * 1009 + 17)
    target_index = rng.randrange(m)
    digest = hashlib.sha256(
        repr((seed, m, task_i, target_index)).encode()
    ).hexdigest()[:16]
    return f"g16a:{seed}:{m}:{task_i}:{digest}", target_index


def execute_trace(ep, bits):
    values = {i: int(v) for i, v in zip(ep.inputs, bits)}
    incoming = {}
    for e in ep.true_edges:
        incoming.setdefault(e.dst, {})[e.port] = e.src
    edge_ops = 0
    node_ops = 0
    internal = []
    for node in ep.nodes[:ep.active_count]:
        if node.op.value == "INPUT":
            continue
        args = [values[incoming[node.index][p]] for p in range(node.arity)]
        edge_ops += node.arity
        if node.op.value == "NOT":
            values[node.index] = 1 - args[0]
        elif node.op.value == "AND":
            values[node.index] = args[0] & args[1]
        elif node.op.value == "OR":
            values[node.index] = args[0] | args[1]
        elif node.op.value == "XOR":
            values[node.index] = args[0] ^ args[1]
        else:
            raise RuntimeError(f"unsupported operator {node.op.value}")
        node_ops += 1
        internal.append(int(values[node.index]))
    if len(internal) != TRACE_BITS:
        raise RuntimeError(f"expected {TRACE_BITS} trace bits, got {len(internal)}")
    return int(values[ep.output]), tuple(internal), edge_ops + node_ops


def build_cache(library):
    scalar = {}
    trace = {}
    cache_work = {"scalar": 0, "trace": 0}
    for idx, ep in enumerate(library):
        if ep.active_count != g010.MAX_NODES or len(ep.inputs) != g010.INPUT_COUNT:
            raise RuntimeError("registered generator shape changed")
        for row_index, bits in enumerate(INPUT_ROWS):
            out, work = g010.execute_exact(ep, bits)
            scalar[(idx, row_index)] = int(out)
            cache_work["scalar"] += int(work.total)
            tout, activation, trace_work = execute_trace(ep, bits)
            if tout != out or trace_work != work.total:
                raise RuntimeError("trace executor diverges from exact executor")
            trace[(idx, row_index)] = tuple(activation)
            cache_work["trace"] += int(trace_work + TRACE_BITS)
    return scalar, trace, cache_work


def action_rows(kind: str, base_row: int):
    x = INPUT_ROWS[base_row]
    if kind in ("scalar_output", "activation_trace_ceiling"):
        return (base_row,)
    if kind == "paired_output":
        paired = tuple(v ^ d for v, d in zip(x, DELTA))
        return (base_row, ROW_TO_INDEX[paired])
    if kind == "quad_output":
        rows = []
        for bit in range(INPUT_COUNT):
            rows.append(
                tuple(v ^ (1 if j == bit else 0) for j, v in enumerate(x))
            )
        return tuple(ROW_TO_INDEX[row] for row in rows)
    raise ValueError(kind)


def candidate_evidence(scalar, trace, idx: int, kind: str, base_row: int):
    rows = action_rows(kind, base_row)
    if kind == "activation_trace_ceiling":
        return trace[(idx, rows[0])]
    return tuple(scalar[(idx, row)] for row in rows)


def selector_prediction_units(kind: str, m: int) -> int:
    per_action = {
        "scalar_output": 1,
        "paired_output": 2,
        "quad_output": 4,
        "activation_trace_ceiling": TRACE_BITS,
    }[kind]
    return int(m * len(INPUT_ROWS) * per_action)


def select_action(scalar, trace, candidates, kind):
    scores = []
    for base_row in range(len(INPUT_ROWS)):
        evidence = tuple(
            candidate_evidence(scalar, trace, idx, kind, base_row)
            for idx in candidates
        )
        info = information_bits(evidence)
        rem = expected_remaining(evidence)
        utility = info / ENVIRONMENT_WORK[kind]
        scores.append(
            (
                utility,
                info,
                -rem,
                -base_row,
                base_row,
            )
        )
    return max(scores), len(scores)


def post_selection_verify(scalar, target, candidates, kind, selected_row, target_evidence):
    predicted = tuple(
        candidate_evidence(scalar, {}, idx, kind, selected_row)
        for idx in range(len(candidates))
    ) if kind != "activation_trace_ceiling" else None
    # The reusable cache is handled by verify_task below. This function exists
    # only to make the causal boundary explicit.
    del scalar, target, candidates, kind, selected_row, target_evidence, predicted
    return None


def exact_verify_shortlist(target, candidates, shortlist):
    target_truth = truth_signature(target)
    for idx in shortlist:
        candidate = candidates[idx]
        if truth_signature(candidate) == target_truth:
            return 1.0
        for bits in INPUT_ROWS:
            got, _ = g010.execute_exact(candidate, bits)
            if got != int(target.truth_table[bits]):
                break
        else:
            return 1.0
    return 0.0


def evaluate_channel(scalar, trace, candidates, target_index, kind):
    indices = tuple(range(len(candidates)))
    best, _ = select_action(scalar, trace, indices, kind)
    selected_row = int(best[-1])
    candidate_evidence_rows = tuple(
        candidate_evidence(scalar, trace, idx, kind, selected_row)
        for idx in indices
    )
    target_evidence = candidate_evidence(
        scalar, trace, target_index, kind, selected_row
    )
    bucket = [
        i for i, evidence in enumerate(candidate_evidence_rows)
        if evidence == target_evidence
    ]
    shortlist = bucket[:BUDGET]
    target = candidates[target_index]
    verified = exact_verify_shortlist(target, candidates, shortlist)
    oracle_budget_success = budget_success(
        candidate_evidence_rows, target_index, target_evidence, BUDGET
    )
    if abs(verified - oracle_budget_success) > 1e-9:
        raise RuntimeError("post-selection verifier disagrees with evidence shortlist")
    info = float(best[1])
    remaining = expected_remaining(candidate_evidence_rows)
    return {
        "selected_row": selected_row,
        "information_gain_bits": info,
        "information_per_work": info / ENVIRONMENT_WORK[kind],
        "expected_remaining_candidates": remaining,
        "effective_information_bits": effective_information_bits(candidate_evidence_rows),
        "signature_count": len(set(candidate_evidence_rows)),
        "target_bucket_size": len(bucket),
        "verified_candidate_budget_success": verified,
        "environment_work_units": ENVIRONMENT_WORK[kind],
        "selector_prediction_units": selector_prediction_units(kind, len(candidates)),
    }


def bootstrap_max_external(primary_rows, rounds=4000, seed=16016):
    rng = random.Random(seed)
    n = len(primary_rows)
    statistics_draws = []
    for _ in range(rounds):
        sample = [primary_rows[rng.randrange(n)] for _ in range(n)]
        arm_means = [
            statistics.fmean(
                row[arm] - row["scalar_output"] for row in sample
            )
            for arm in EXTERNAL_CHANNELS
        ]
        statistics_draws.append(max(arm_means))
    statistics_draws.sort()
    lo = statistics_draws[int(0.025 * (len(statistics_draws) - 1))]
    hi = statistics_draws[int(0.975 * (len(statistics_draws) - 1))]
    return [float(lo), float(hi)]


def main(smoke=False):
    from tac_osm.contract import load_contract

    contract = load_contract(EXPERIMENT_ID)
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
        contract.require_arms(CHANNELS)

    raw = {str(s): {str(m): [] for m in m_levels} for s in seeds}
    cache_accounting = {}
    checks = {}

    for seed in seeds:
        training = g010.generate(seed + 100, TRAIN_PROGRAMS)
        train_s = {structure_key(ep) for ep in training}
        train_t = {truth_signature(ep) for ep in training}
        library = g010.generate(
            seed + 5000,
            LIBRARY_SIZE,
            exclude_structures=train_s,
            exclude_truths=train_t,
        )
        if len(library) != LIBRARY_SIZE:
            raise RuntimeError("evaluation library construction failed")
        library_s = {structure_key(ep) for ep in library}
        library_t = {truth_signature(ep) for ep in library}
        if train_s & library_s or train_t & library_t:
            raise RuntimeError("train/eval leakage gate failed")
        scalar, trace, cache_work = build_cache(library)
        cache_accounting[str(seed)] = {
            "scalar_output_candidate_cache_work_units": cache_work["scalar"],
            "paired_output_candidate_cache_work_units": cache_work["scalar"],
            "quad_output_candidate_cache_work_units": cache_work["scalar"],
            "activation_trace_ceiling_candidate_cache_work_units": cache_work["trace"],
        }
        checks[str(seed)] = {
            "training_programs": TRAIN_PROGRAMS,
            "evaluation_library_size": LIBRARY_SIZE,
            "train_eval_structure_disjoint": True,
            "train_eval_truth_disjoint": True,
            "target_identity_available_to_selector": False,
            "target_index_available_to_selector": False,
            "realized_target_evidence_available_before_selection": False,
            "verifier_labels_available_to_selector": False,
            "generator_commit": GENERATOR_COMMIT,
        }

        for m in m_levels:
            candidates = library[:m]
            for task_i in range(tasks_per_m):
                task_id, target_index = task_descriptor(seed, m, task_i)
                task = {"task_id": task_id, "target_index": target_index, "channels": {}}
                for kind in CHANNELS:
                    task["channels"][kind] = evaluate_channel(
                        scalar, trace, candidates, target_index, kind
                    )
                raw[str(seed)][str(m)].append(task)

    summary = {"arms": {}, "primary": {}, "cache_accounting": cache_accounting}
    for kind in CHANNELS:
        summary["arms"][kind] = {}
        for m in m_levels:
            seed_values = []
            trials = []
            for seed in seeds:
                vals = [
                    task["channels"][kind]
                    for task in raw[str(seed)][str(m)]
                ]
                seed_values.append(
                    statistics.fmean(v["information_per_work"] for v in vals)
                )
                trials.extend(vals)
            summary["arms"][kind][str(m)] = {
                "trials": len(trials),
                "information_gain_bits_mean": statistics.fmean(
                    v["information_gain_bits"] for v in trials
                ),
                "information_per_work_mean": statistics.fmean(
                    v["information_per_work"] for v in trials
                ),
                "expected_remaining_candidates_mean": statistics.fmean(
                    v["expected_remaining_candidates"] for v in trials
                ),
                "effective_information_bits_mean": statistics.fmean(
                    v["effective_information_bits"] for v in trials
                ),
                "target_bucket_size_mean": statistics.fmean(
                    v["target_bucket_size"] for v in trials
                ),
                "verified_candidate_budget_success_mean": statistics.fmean(
                    v["verified_candidate_budget_success"] for v in trials
                ),
                "amortized_total_work_units_mean": statistics.fmean(
                    v["environment_work_units"]
                    + v["selector_prediction_units"]
                    + cache_accounting[str(seed)][
                        kind + "_candidate_cache_work_units"
                    ] / REGISTERED_TASKS_PER_SEED
                    for seed in seeds
                    for v in raw[str(seed)][str(m)]
                ),
                "amortized_total_information_per_work_mean": statistics.fmean(
                    v["information_gain_bits"]
                    / max(
                        1.0,
                        v["environment_work_units"]
                        + v["selector_prediction_units"]
                        + cache_accounting[str(seed)][
                            kind + "_candidate_cache_work_units"
                        ] / REGISTERED_TASKS_PER_SEED,
                    )
                    for seed in seeds
                    for v in raw[str(seed)][str(m)]
                ),
                "seed_mean_information_per_work": seed_values,
            }

    primary_seed_rows = []
    for seed in seeds:
        row = {}
        for kind in CHANNELS:
            row[kind] = statistics.fmean(
                t["channels"][kind]["information_per_work"]
                for t in raw[str(seed)]["512"]
            )
        primary_seed_rows.append(row)

    differences = {
        arm: [
            row[arm] - row["scalar_output"] for row in primary_seed_rows
        ]
        for arm in EXTERNAL_CHANNELS
    }
    arm_means = {
        arm: statistics.fmean(values)
        for arm, values in differences.items()
    }
    summary["primary"] = {
        "M": 512,
        "seed_level_arm_information_per_work": primary_seed_rows,
        "external_seed_level_differences": differences,
        "external_mean_differences": arm_means,
        "max_external_minus_scalar_mean": max(arm_means.values()),
        "winning_external_arm": max(arm_means, key=arm_means.get),
        "seed_bootstrap_95ci_max_statistic": bootstrap_max_external(primary_seed_rows),
        "activation_trace_ceiling_mean_difference": statistics.fmean(
            row["activation_trace_ceiling"] - row["scalar_output"]
            for row in primary_seed_rows
        ),
    }

    result = {
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
            "channels": list(CHANNELS),
            "input_rows": len(INPUT_ROWS),
            "library_size": LIBRARY_SIZE,
            "train_programs": TRAIN_PROGRAMS,
            "budget": BUDGET,
            "trace_bits": TRACE_BITS,
            "environment_work_units": ENVIRONMENT_WORK,
            "primary_scope": "M=512 seed-level max-statistic",
        },
        "checks": checks,
        "cache_accounting": cache_accounting,
        "results": raw,
        "summary": summary,
        "scope": {
            "finite_domain_observation_channel_comparison": True,
            "target_identity_blind_probe_selection": True,
            "activation_trace_is_non_deployable_ceiling": True,
            "learned_probe_policy_claim": False,
            "optimal_sequential_policy_claim": False,
            "asymptotic_claim": False,
            "language_image_audio_claim": False,
        },
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    path = Path("artifacts") / f"{EXPERIMENT_ID}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args().smoke)
