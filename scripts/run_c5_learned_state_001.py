#!/usr/bin/env python3
"""Run preregistered TACOSM-C5-LEARNED-STATE-001."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import Query
from tac_osm.c5_end_to_end import (
    EndToEndExecutor,
    build_candidate_index,
    build_population,
    build_task,
    full_state_scan,
    prepare_state,
)
from tac_osm.learned_state_index import (
    LearnedSemanticStateIndex,
    LearnedStateIndexConfig,
)
from tac_osm.noisy_state_tasks import CODEBOOK
from tac_osm.contract import load_contract

SEEDS = (0, 1, 2, 3, 4)
H_LEVELS = (64, 128, 256)
M = 64
K = 1
EVAL_STEPS = 100
TRAIN_EPOCHS = 32
R = 4
WORK = 12
TRAIN_CODES = tuple(CODEBOOK[16:64])
EVAL_CODES = tuple(CODEBOOK[:16])


def _read_state_value(state, address: str) -> tuple[int, ...]:
    read = state.read(
        Query(
            text="\t" + address,
            step=state.current_step,
            provenance="c5_learned_state_internal_read",
        )
    )
    if not read.keys:
        raise RuntimeError(f"state address {address!r} was not readable")
    return tuple(int(x) for x in read.values[0])


def _build_h_invariant_task(seed: int, h: int, step: int):
    # Amendment A1: query/state generation is anchored at H=64 so H changes
    # only candidate-population size, not the noisy query distribution.
    from tac_osm.c5_end_to_end import EndToEndTask

    base = build_task(seed, 64, step)
    candidates = build_population(seed, h)
    relevant = frozenset(
        i
        for i, candidate in enumerate(candidates)
        if tuple(candidate.descriptor) == base.target_value
    )
    if len(relevant) != R:
        raise AssertionError("registered task must have exactly four relevant programs")
    expected, work, calls = EndToEndExecutor().execute_population(
        tuple(candidates[i] for i in sorted(relevant)),
        base.target_value,
    )
    assert work == R * WORK
    assert calls == R
    return EndToEndTask(
        query=base.query,
        state_updates=base.state_updates,
        candidates=candidates,
        target_address=base.target_address,
        target_value=base.target_value,
        expected_output=expected,
        relevant_indices=relevant,
        step=step,
    )


def run_cell(seed: int, h: int) -> dict:
    first = _build_h_invariant_task(seed, h, 0)
    population = first.candidates
    state = prepare_state(first)

    index = LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10,
            latent_dim=8,
            learning_rate=0.02,
            margin=0.25,
            epochs=TRAIN_EPOCHS,
            bucket_bits=8,
            probe_radius=1,
            shortlist_k=K,
            seed=seed,
        )
    )
    trained_pairs = index.train(TRAIN_CODES)
    build = index.build(state)
    candidate_index = build_candidate_index(population)
    executor = EndToEndExecutor()

    exhaustive_success = []
    learned_success = []
    target_retained = []
    shortlist_sizes = []
    query_probes = []
    bucket_candidates = []
    query_macs = []
    build_macs = []
    candidate_retained = []
    learned_executor_work = []
    learned_executor_calls = []
    full_executor_work = []
    full_executor_calls = []
    full_state_ops = []
    candidate_positions = []

    for step in range(EVAL_STEPS):
        task = _build_h_invariant_task(seed, h, step)

        full_address, full_value, state_ops = full_state_scan(task, state)
        full_state_ops.append(state_ops)

        all_output, all_work, all_calls = executor.execute_population(
            task.candidates,
            full_value,
        )
        full_executor_work.append(all_work)
        full_executor_calls.append(all_calls)
        exhaustive_success.append(int(all_output == task.expected_output))

        hit = index.lookup(task.query)
        shortlist_sizes.append(hit.shortlist_size)
        query_probes.append(hit.query_probes)
        bucket_candidates.append(hit.bucket_candidates)
        query_macs.append(index.query_embedding_macs)
        build_macs.append(build.state_items * index.state_embedding_macs)

        retained = int(task.target_address in hit.addresses)
        target_retained.append(retained)

        if not hit.addresses:
            learned_executor_work.append(0)
            learned_executor_calls.append(0)
            learned_success.append(0)
            candidate_retained.append(0)
            continue

        retrieved_address = hit.addresses[0]
        retrieved_value = _read_state_value(state, retrieved_address)
        candidate_hit = candidate_index.lookup(
            task.query,
            reference=retrieved_value,
            k=R,
            relation="equality",
        )
        kept_all_relevant = set(candidate_hit.candidate_indices) == set(
            task.relevant_indices
        )
        candidate_retained.append(int(kept_all_relevant))
        candidate_positions.append(candidate_hit.inspected_positions)

        selected = tuple(
            task.candidates[i] for i in candidate_hit.candidate_indices
        )
        selected_output, selected_work, selected_calls = (
            executor.execute_population(selected, retrieved_value)
        )
        learned_executor_work.append(selected_work)
        learned_executor_calls.append(selected_calls)
        learned_success.append(
            int(selected_output == task.expected_output)
        )

        if full_address != task.target_address or full_value != task.target_value:
            raise AssertionError("exhaustive state path did not recover target")

    exhaustive_rate = statistics.fmean(exhaustive_success)
    learned_rate = statistics.fmean(learned_success)

    return {
        "seed": seed,
        "H": h,
        "M": M,
        "R": R,
        "K": K,
        "training_codes": len(TRAIN_CODES),
        "evaluation_codes": len(EVAL_CODES),
        "trained_pairs": trained_pairs,
        "queries": EVAL_STEPS,
        "exhaustive_success_rate": exhaustive_rate,
        "learned_selective_success_rate": learned_rate,
        "capability_delta": learned_rate - exhaustive_rate,
        "capability_rule_pass": learned_rate >= exhaustive_rate - 0.05,
        "state_target_retention_rate": statistics.fmean(target_retained),
        "shortlist_size_mean": statistics.fmean(shortlist_sizes),
        "query_probes_mean": statistics.fmean(query_probes),
        "bucket_candidates_mean": statistics.fmean(bucket_candidates),
        "query_embedding_macs": index.query_embedding_macs,
        "build_embedding_macs_total": build.state_items * index.state_embedding_macs,
        "build_embedding_macs_per_query": statistics.fmean(build_macs) / EVAL_STEPS,
        "full_state_scan_ops_mean": statistics.fmean(full_state_ops),
        "candidate_index_positions_mean": statistics.fmean(candidate_positions) if candidate_positions else 0.0,
        "candidate_target_retention_rate": statistics.fmean(candidate_retained),
        "full_executor_work_mean": statistics.fmean(full_executor_work),
        "learned_executor_work_mean": statistics.fmean(learned_executor_work),
        "full_executor_calls_mean": statistics.fmean(full_executor_calls),
        "learned_executor_calls_mean": statistics.fmean(learned_executor_calls),
        "retained_fraction": statistics.fmean(shortlist_sizes) / M,
        "unique_buckets": build.unique_buckets,
        "index_build_state_items": build.state_items,
    }


def main() -> None:
    contract = load_contract("TACOSM-C5-LEARNED-STATE-001")
    contract.require_levels(H_LEVELS)
    contract.require_seeds(SEEDS)
    contract.require_steps(TRAIN_EPOCHS)
    contract.require_eval_steps(EVAL_STEPS)
    contract.require_k_levels([K])
    contract.require_arms(["exhaustive", "learned_selective"])

    if set(TRAIN_CODES) & set(EVAL_CODES):
        raise AssertionError("training/evaluation semantic codes overlap")
    if len(TRAIN_CODES) != 48 or len(EVAL_CODES) != 16:
        raise AssertionError("registered code split changed")

    cells = [run_cell(seed, h) for h in H_LEVELS for seed in SEEDS]
    result = {
        "protocol": {
            "name": "TACOSM-C5-LEARNED-STATE-001",
            "seeds": list(SEEDS),
            "H_levels": list(H_LEVELS),
            "M": M,
            "R": R,
            "K": K,
            "training_epochs": TRAIN_EPOCHS,
            "training_codes": len(TRAIN_CODES),
            "evaluation_codes": len(EVAL_CODES),
            "queries_per_cell": EVAL_STEPS,
            "encoder": "dual linear 10->8 semantic encoder",
            "binary_index": "8-bit code, Hamming radius 1, shortlist 1",
            "candidate_index": "exact equality control, shortlist 4",
            "executor": "fixed-work aggregate",
            "capability_rule": "learned_selective >= exhaustive - 0.05",
        },
        "cells": cells,
    }
    out = Path("artifacts/TACOSM-C5-LEARNED-STATE-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
