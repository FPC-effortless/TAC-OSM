#!/usr/bin/env python3
"""Run preregistered TACOSM-C5-END-TO-END-001."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm.c5_end_to_end import (
    EndToEndExecutor,
    build_candidate_index,
    build_population,
    build_state_index,
    build_task,
    full_state_scan,
    prepare_state,
)
from tac_osm.contract import load_contract

SEEDS = (0, 1, 2, 3, 4)
H_LEVELS = (64, 128, 256)
M = 64
K = 4
EVAL_STEPS = 100
TRAIN_STEPS = 100
WORK = 12


def run_cell(seed: int, h: int) -> dict:
    first = build_task(seed, h, 0)
    population = first.candidates
    state = prepare_state(first)

    state_index = build_state_index(state)
    candidate_index = build_candidate_index(population)
    executor = EndToEndExecutor()

    exhaustive_success = []
    selective_success = []
    full_state_ops = []
    selective_state_ops = []
    full_candidate_ops = []
    selective_candidate_ops = []
    full_executor_work = []
    selective_executor_work = []
    full_executor_calls = []
    selective_executor_calls = []
    retained = []
    state_target_retained = []
    indexed_candidate_target_retained = []

    for step in range(EVAL_STEPS):
        task = build_task(seed, h, step)
        query = task.query
        target_address = task.target_address

        full_address, full_value, state_ops = full_state_scan(task, state)
        full_state_ops.append(state_ops)

        all_output, all_work, all_calls = executor.execute_population(
            task.candidates, full_value
        )
        exhaustive_success.append(
            int(all_output == task.expected_output)
        )
        full_executor_work.append(all_work)
        full_executor_calls.append(all_calls)

        state_hit = state_index.lookup(query)
        state_target_retained.append(
            int(target_address in state_hit.addresses)
        )
        retrieved_state_addresses = state_hit.addresses
        if not retrieved_state_addresses:
            raise RuntimeError("state index returned no address")

        retrieved_value = tuple(
            state.read(
                __import__("tac_osm").Query(
                    text="\t" + retrieved_state_addresses[0],
                    step=state.current_step,
                )
            ).values[0]
        )

        candidate_hit = candidate_index.lookup(
            task.query,
            reference=retrieved_value,
            k=K,
            relation="equality",
        )
        indexed_candidate_target_retained.append(
            int(set(candidate_hit.candidate_indices) == set(task.relevant_indices))
        )
        retained.append(len(candidate_hit.candidate_indices))

        selected_candidates = tuple(
            task.candidates[i] for i in candidate_hit.candidate_indices
        )
        selective_output, selective_work, selective_calls = (
            executor.execute_population(selected_candidates, retrieved_value)
        )
        selective_success.append(
            int(selective_output == task.expected_output)
        )
        selective_executor_work.append(selective_work)
        selective_executor_calls.append(selective_calls)
        selective_state_ops.append(state_hit.query_operations)
        full_candidate_ops.append(len(task.candidates) * len(task.target_value))
        selective_candidate_ops.append(candidate_hit.inspected_positions)

        if full_address != target_address or full_value != task.target_value:
            raise AssertionError("full state scan failed to recover target")
    return {
        "seed": seed,
        "H": h,
        "M": M,
        "K": K,
        "R": len(first.relevant_indices),
        "queries": EVAL_STEPS,
        "exhaustive_success_rate": statistics.fmean(exhaustive_success),
        "selective_success_rate": statistics.fmean(selective_success),
        "full_state_scan_ops_mean": statistics.fmean(full_state_ops),
        "indexed_state_probe_ops_mean": statistics.fmean(selective_state_ops),
        "full_candidate_match_ops_mean": statistics.fmean(full_candidate_ops),
        "indexed_candidate_positions_mean": statistics.fmean(selective_candidate_ops),
        "full_executor_work_mean": statistics.fmean(full_executor_work),
        "selective_executor_work_mean": statistics.fmean(selective_executor_work),
        "full_executor_calls_mean": statistics.fmean(full_executor_calls),
        "selective_executor_calls_mean": statistics.fmean(selective_executor_calls),
        "retained_candidates_mean": statistics.fmean(retained),
        "retained_fraction": statistics.fmean(retained) / h,
        "state_target_retained_rate": statistics.fmean(state_target_retained),
        "candidate_bucket_exact_rate": statistics.fmean(indexed_candidate_target_retained),
        "index_build_state_items": state_index.build_diagnostics.state_items,
        "index_build_candidate_items": candidate_index.built_candidates,
        "amortized_state_build_items_per_query": M / EVAL_STEPS,
        "amortized_candidate_build_items_per_query": h / EVAL_STEPS,
    }


def main() -> None:
    contract = load_contract("TACOSM-C5-END-TO-END-001")
    contract.require_levels(H_LEVELS)
    contract.require_seeds(SEEDS)
    contract.require_steps(TRAIN_STEPS)
    contract.require_eval_steps(EVAL_STEPS)
    contract.require_k_levels([K])
    contract.require_arms(["exhaustive", "selective"])

    cells = [
        run_cell(seed, h)
        for h in H_LEVELS
        for seed in SEEDS
    ]
    result = {
        "protocol": {
            "name": "TACOSM-C5-END-TO-END-001",
            "seeds": list(SEEDS),
            "H_levels": list(H_LEVELS),
            "M": M,
            "K": K,
            "R": 4,
            "queries_per_cell": EVAL_STEPS,
            "work_units_per_candidate": WORK,
            "state_query": "one-bit-noisy 10-bit code",
            "state_index": "Hamming radius 2, shortlist 1",
            "candidate_index": "exact equality, shortlist 4",
            "executor": "fixed-work aggregate",
        },
        "cells": cells,
    }
    out = Path("artifacts/TACOSM-C5-END-TO-END-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
