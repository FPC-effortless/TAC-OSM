#!/usr/bin/env python3
"""Run TACOSM-C5-MULTIFACTOR-BEAM-001."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tac_osm.contract import load_contract
from tac_osm.multifactor_product_key_index import (
    MultiFactorProductKeyConfig,
    MultiFactorProductKeyStateIndex,
)
from scripts.run_c5_frontier_robustness_002 import (
    H_FIXED, K, LATENT_DIM, M_LEVELS, NEGATIVE_COUNT, SEEDS, TEACHER_EPOCHS,
    TRAIN_CODES, build_scaled_task, make_teacher, normalized_embeddings, normalized_query,
)
from scripts.run_c5_multifactor_001 import StateDistinctEndToEndExecutor

EVAL_STEPS = 100
FACTOR_COUNT = 3
FACTOR_SIZE = 8
FACTOR_BEAMS = (3, 4, 5, 6)
FACTOR_KMEANS_ITERATIONS = 8
CAPABILITY_FLOOR = 0.85
ARMS = tuple((f'three_factor_8_beam{beam}', beam) for beam in FACTOR_BEAMS)


def evaluate_arm(seed, m, arm_name, beam, teacher, embeddings, tasks):
    first = tasks[0]
    train_set = set(TRAIN_CODES)
    train_items = [
        item for item, update in zip(embeddings, first.state_updates)
        if tuple(update.value) in train_set
    ]
    if len(train_items) != len(TRAIN_CODES):
        raise AssertionError("training-state population mismatch")

    index = MultiFactorProductKeyStateIndex(MultiFactorProductKeyConfig(
        factor_count=FACTOR_COUNT,
        factor_size=FACTOR_SIZE,
        factor_beam=beam,
        iterations=FACTOR_KMEANS_ITERATIONS,
        max_shortlist=K,
    ))
    build = index.build(embeddings, codebook_items=train_items)
    executor = StateDistinctEndToEndExecutor()

    ex_target = []
    sel_target = []
    proposal = []
    ex_success = []
    sel_success = []
    scored = []
    shortlist = []
    pair_ops = []
    macs = []

    for task in tasks:
        query = normalized_query(teacher, task)
        ranked = [
            (sum(a * b for a, b in zip(query, emb)), address)
            for address, emb in embeddings
        ]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        ex_address = ranked[0][1]
        ex_value = next(
            tuple(update.value) for update in task.state_updates
            if update.key == ex_address
        )
        ex_target.append(int(ex_address == task.target_address))
        ex_relevant = tuple(
            i for i, candidate in enumerate(task.candidates)
            if tuple(candidate.descriptor) == ex_value
        )
        ex_output, _, _ = executor.execute_population(
            tuple(task.candidates[i] for i in ex_relevant), ex_value
        )
        ex_success.append(int(ex_output == task.expected_output))

        hit = index.lookup(query, beam=beam, max_shortlist=K)
        selected = hit.selected_address
        sel_target.append(int(selected == task.target_address))
        proposal.append(int(task.target_address in hit.candidate_addresses))
        selected_value = next(
            (tuple(update.value) for update in task.state_updates if update.key == selected),
            None,
        )
        if selected_value is None:
            sel_success.append(0)
        else:
            relevant = tuple(
                i for i, candidate in enumerate(task.candidates)
                if tuple(candidate.descriptor) == selected_value
            )
            output, _, _ = executor.execute_population(
                tuple(task.candidates[i] for i in relevant), selected_value
            )
            sel_success.append(int(output == task.expected_output))

        scored.append(hit.state_candidates_scored)
        shortlist.append(len(hit.candidate_addresses))
        pair_ops.append(hit.pair_generation_ops)
        macs.append(
            LATENT_DIM * 10 + hit.factor_score_macs + hit.state_rerank_macs
        )

    ex_target_rate = statistics.fmean(ex_target)
    sel_target_rate = statistics.fmean(sel_target)
    return {
        'seed': seed, 'M': m, 'arm': arm_name, 'factor_beam': beam,
        'exhaustive_target_recall': ex_target_rate,
        'selective_target_recall': sel_target_rate,
        'proposal_target_retention': statistics.fmean(proposal),
        'exhaustive_end_to_end_success': statistics.fmean(ex_success),
        'selective_end_to_end_success': statistics.fmean(sel_success),
        'state_candidates_scored_mean': statistics.fmean(scored),
        'states_scored_over_M': statistics.fmean(scored) / m,
        'shortlist_size_mean': statistics.fmean(shortlist),
        'pair_generation_ops_mean': statistics.fmean(pair_ops),
        'total_macs_mean': statistics.fmean(macs),
        'nonempty_cells': build.nonempty_cells,
        'max_cell_size': build.max_cell_size,
        'build_total_macs': build.total_build_macs,
        'eval_count': len(tasks),
    }


def pool_cells(cells, m, arm_name):
    selected = [cell for cell in cells if cell['M'] == m and cell['arm'] == arm_name]
    if not selected:
        return None
    n = sum(cell['eval_count'] for cell in selected)
    ex_t = sum(cell['exhaustive_target_recall'] * cell['eval_count'] for cell in selected)
    sel_t = sum(cell['selective_target_recall'] * cell['eval_count'] for cell in selected)
    ex_s = sum(cell['exhaustive_end_to_end_success'] * cell['eval_count'] for cell in selected)
    sel_s = sum(cell['selective_end_to_end_success'] * cell['eval_count'] for cell in selected)
    return {
        'M': m, 'arm': arm_name, 'factor_count': FACTOR_COUNT, 'factor_size': FACTOR_SIZE,
        'factor_beam': selected[0]['factor_beam'], 'eval_count': n,
        'exhaustive_target_successes': ex_t,
        'selective_target_successes': sel_t,
        'target_recall_retention_ratio': sel_t / ex_t if ex_t else float('nan'),
        'exhaustive_end_to_end_successes': ex_s,
        'selective_end_to_end_successes': sel_s,
        'proposal_target_retention': statistics.fmean(c['proposal_target_retention'] for c in selected),
        'state_candidates_scored_mean': statistics.fmean(c['state_candidates_scored_mean'] for c in selected),
        'states_scored_over_M': statistics.fmean(c['states_scored_over_M'] for c in selected),
        'shortlist_size_mean': statistics.fmean(c['shortlist_size_mean'] for c in selected),
        'pair_generation_ops_mean': statistics.fmean(c['pair_generation_ops_mean'] for c in selected),
        'total_macs_mean': statistics.fmean(c['total_macs_mean'] for c in selected),
        'nonempty_cells': statistics.fmean(c['nonempty_cells'] for c in selected),
        'max_cell_size': statistics.fmean(c['max_cell_size'] for c in selected),
        'build_total_macs': statistics.fmean(c['build_total_macs'] for c in selected),
    }


def run(smoke: bool):
    seeds = (10,) if smoke else SEEDS
    ms = (128,) if smoke else M_LEVELS
    steps = 5 if smoke else EVAL_STEPS
    cells = []
    for seed in seeds:
        teacher = make_teacher(seed)
        for m in ms:
            first = build_scaled_task(seed, m, 0)
            embeddings = normalized_embeddings(teacher, first)
            tasks = tuple(build_scaled_task(seed, m, step) for step in range(steps))
            for arm_name, beam in ARMS:
                cells.append(evaluate_arm(seed, m, arm_name, beam, teacher, embeddings, tasks))

    pooled = {}
    for m in ms:
        for arm_name, _ in ARMS:
            value = pool_cells(cells, m, arm_name)
            if value is not None:
                pooled[f'{m}:{arm_name}'] = value

    fixed_feasible = []
    if not smoke:
        for arm_name, beam in ARMS:
            values = [pooled[f'{m}:{arm_name}'] for m in M_LEVELS]
            if all(v['target_recall_retention_ratio'] >= CAPABILITY_FLOOR for v in values):
                fixed_feasible.append({
                    'arm': arm_name, 'factor_beam': beam,
                    'min_target_recall_retention_ratio': min(v['target_recall_retention_ratio'] for v in values),
                    'mean_states_scored_over_M': statistics.fmean(v['states_scored_over_M'] for v in values),
                })

    return {
        'protocol': {
            'name': 'TACOSM-C5-MULTIFACTOR-BEAM-001', 'smoke': smoke,
            'seeds': list(seeds), 'M_levels': list(ms), 'H_fixed': H_FIXED, 'K': K,
            'factor_count': FACTOR_COUNT, 'factor_size': FACTOR_SIZE,
            'factor_beams': list(FACTOR_BEAMS), 'factor_kmeans_iterations': FACTOR_KMEANS_ITERATIONS,
            'teacher_epochs': TEACHER_EPOCHS, 'negative_count': NEGATIVE_COUNT,
            'eval_steps': steps, 'capability_floor': CAPABILITY_FLOOR,
        },
        'pooled': pooled, 'fixed_capability_qualified_arms': fixed_feasible, 'cells': cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--no-contract', action='store_true')
    args = parser.parse_args()
    if not args.no_contract:
        contract = load_contract('TACOSM-C5-MULTIFACTOR-BEAM-001')
        contract.require_levels((H_FIXED,))
        contract.require_m_levels(M_LEVELS)
        contract.require_seeds(SEEDS)
        contract.require_k_levels([K])
        contract.require_arms([arm[0] for arm in ARMS])
        if not args.smoke:
            contract.require_eval_steps(EVAL_STEPS)
    result = run(args.smoke)
    output = Path('artifacts/TACOSM-C5-MULTIFACTOR-BEAM-001.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
