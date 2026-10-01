#!/usr/bin/env python3
"""Run preregistered TACOSM-STATE-REP-008 exact index ceiling."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm.contract import load_contract
from tac_osm.exact_state_index import ExactSemanticStateIndex
from tac_osm.semantic_state_addressor import SemanticStateAddressor
from tac_osm.semantic_state_tasks import build_semantic_state_task
from tac_osm.temporal import TemporalPersistentState

SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (2, 4, 8, 16, 32)
QUERY_COUNT = 256
DELAY = 1


def prepare(task):
    state = TemporalPersistentState()
    if task.write_step:
        state.advance_to(task.write_step)
    task.stage(state)
    return state


def run(m: int, seed: int):
    task = build_semantic_state_task(
        seed * 900001 + m * 101,
        write_step=0, delay=DELAY, n_states=m, n_candidates=8,
    )
    state = prepare(task)
    index = ExactSemanticStateIndex()
    build = index.build(state)

    exact_hits = 0
    query_ops = []
    for _ in range(QUERY_COUNT):
        lookup = index.lookup(task.query)
        exact_hits += int(lookup.address == task.target_address)
        query_ops.append(lookup.query_operations)

    scan = SemanticStateAddressor()
    scan_hits = 0
    scan_macs = []
    for _ in range(QUERY_COUNT):
        decision = scan.select(task.query, state, target_address=task.target_address)
        scan_hits += int(decision.selected_address == task.target_address)
        scan_macs.append(decision.total_macs)

    return {
        'm': m,
        'seed': seed,
        'queries': QUERY_COUNT,
        'index_build_reads': build.build_reads,
        'index_entries': build.build_entries,
        'indexed_top1_recall': exact_hits / QUERY_COUNT,
        'indexed_query_operations_mean': statistics.fmean(query_ops),
        'scan_top1_recall': scan_hits / QUERY_COUNT,
        'scan_macs_mean': statistics.fmean(scan_macs),
        'index_total_query_operations': sum(query_ops),
        'amortized_build_reads_per_query': build.build_reads / QUERY_COUNT,
    }


def main():
    contract = load_contract('TACOSM-STATE-REP-008')
    contract.require_levels(M_LEVELS)
    contract.require_seeds(SEEDS)
    contract.require_steps(QUERY_COUNT)
    contract.require_eval_steps(QUERY_COUNT)
    contract.require_arms(['exact_index', 'full_scan'])
    result = {
        'protocol': {
            'name': 'TACOSM-STATE-REP-008',
            'seeds': list(SEEDS),
            'm_levels': list(M_LEVELS),
            'queries_per_seed_level': QUERY_COUNT,
            'state_delay': DELAY,
            'lookup': 'exact semantic inverted index',
            'control': 'full bilinear scan',
        },
        'cells': [],
    }
    for m in M_LEVELS:
        for seed in SEEDS:
            result['cells'].append(run(m, seed))
    out = Path('artifacts/TACOSM-STATE-REP-008.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()