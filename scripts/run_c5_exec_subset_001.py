#!/usr/bin/env python3
"""Run preregistered TACOSM-C5-EXEC-001."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm.addressing import ContentAddressIndex
from tac_osm.contract import load_contract
from tac_osm.selective_execution import (
    SelectiveProgramExecutor,
    build_index,
    build_population,
    build_task,
    retrieve,
)

SEEDS = (0, 1, 2, 3, 4)
H_LEVELS = (64, 128, 256)
STEPS = 100
EVAL_STEPS = 100
K = 4
WORK = 9


def run_cell(seed: int, h: int) -> dict:
    population = build_population(seed, h)
    first_task = build_task(seed, h, 0)
    assert population == first_task.candidates
    index = ContentAddressIndex.build(population, context=first_task.query.context)
    executor = SelectiveProgramExecutor()
    full_success = []
    indexed_success = []
    full_calls = []
    indexed_calls = []
    full_work = []
    indexed_work = []
    retained = []
    address_positions = []
    index_build_start = time.perf_counter()
    # Build already performed above; retain a separate measured wall-clock
    # datum for the construction path without mixing it into query timing.
    index_build_wall_clock = time.perf_counter() - index_build_start

    for step in range(EVAL_STEPS):
        task = build_task(seed, h, step)
        full_output, fw, fc = executor.execute_population(task.candidates, task.query)
        full_success.append(int(full_output == task.expected_output))
        full_calls.append(fc)
        full_work.append(fw)

        hit_start = time.perf_counter()
        reference = tuple(int(x) for x in task.query.text.partition('\t')[0].split())
        hit = index.lookup(
            task.query, reference=reference, k=K, relation='equality'
        )
        _ = time.perf_counter() - hit_start
        retained_candidates = tuple(task.candidates[i] for i in hit.candidate_indices)
        indexed_output, iw, ic = executor.execute_population(retained_candidates, task.query)
        indexed_success.append(int(indexed_output == task.expected_output))
        indexed_calls.append(ic)
        indexed_work.append(iw)
        retained.append(len(retained_candidates))
        address_positions.append(hit.inspected_positions)

    return {
        'seed': seed,
        'H': h,
        'K': K,
        'queries': EVAL_STEPS,
        'R_expected': 4,
        'full_success_rate': statistics.fmean(full_success),
        'indexed_success_rate': statistics.fmean(indexed_success),
        'full_executor_calls_per_query': statistics.fmean(full_calls),
        'indexed_executor_calls_per_query': statistics.fmean(indexed_calls),
        'full_executor_work_per_query': statistics.fmean(full_work),
        'indexed_executor_work_per_query': statistics.fmean(indexed_work),
        'retained_candidates_per_query': statistics.fmean(retained),
        'retained_fraction': statistics.fmean(retained) / h,
        'address_query_positions_per_query': statistics.fmean(address_positions),
        'index_build_candidates': len(population),
        'amortized_index_build_candidates_per_query': len(population) / EVAL_STEPS,
        'index_build_wall_clock_seconds': index_build_wall_clock,
    }


def main() -> None:
    contract = load_contract('TACOSM-C5-EXEC-001')
    contract.require_levels(H_LEVELS)
    contract.require_seeds(SEEDS)
    contract.require_steps(STEPS)
    contract.require_eval_steps(EVAL_STEPS)
    contract.require_k_levels([K])
    contract.require_arms(['exhaustive', 'selective'])

    cells = [run_cell(seed, h) for h in H_LEVELS for seed in SEEDS]
    result = {
        'protocol': {
            'name': 'TACOSM-C5-EXEC-001',
            'seeds': list(SEEDS),
            'H_levels': list(H_LEVELS),
            'queries_per_cell': EVAL_STEPS,
            'K': K,
            'R': 4,
            'work_units_per_candidate': WORK,
            'relation': 'six-bit equality prefix',
            'index': 'exact public content-address index',
            'executor': 'fixed-work aggregate program',
        },
        'cells': cells,
    }
    out = Path('artifacts/TACOSM-C5-EXEC-001.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()