#!/usr/bin/env python3
"""Run preregistered TACOSM-C5-PROTOTYPE-SELECTIVE-001."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import Query
from tac_osm.c5_end_to_end import EndToEndExecutor, EndToEndTask, build_population, build_task, prepare_state
from tac_osm.contract import load_contract
from tac_osm.learned_prototype_state import LearnedPrototypeStateIndex, PrototypeStateIndexConfig
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from tac_osm.noisy_state_tasks import CODEBOOK

SEEDS = (0, 1, 2, 3, 4)
H_LEVELS = (64, 128, 256)
H_ANCHOR = 64
M = 64
R = 4
K = 4
EPOCHS = 512
EVAL_STEPS = 100
NEGATIVE_COUNT = 8
LATENT_DIM = 16
TRAIN_CODES = tuple(CODEBOOK[16:64])
EVAL_CODES = tuple(CODEBOOK[:16])
QUERY_PROJECTION_MACS = 10 * LATENT_DIM
EXHAUSTIVE_STATE_SCORE_MACS = M * LATENT_DIM
PROTOTYPE_SCORE_MACS = 16 * LATENT_DIM
MAX_RERANK_MACS = K * LATENT_DIM
BUILD_TRAIN_EMBED_MACS = len(TRAIN_CODES) * 10 * LATENT_DIM
KMEANS_MACS = 8 * len(TRAIN_CODES) * 16 * LATENT_DIM
BUILD_STATE_EMBED_MACS = M * 10 * LATENT_DIM
BUILD_ASSIGNMENT_MACS = M * 16 * LATENT_DIM
BUILD_TOTAL_MACS = BUILD_TRAIN_EMBED_MACS + KMEANS_MACS + BUILD_STATE_EMBED_MACS + BUILD_ASSIGNMENT_MACS

def _model(seed: int) -> LearnedSemanticStateIndex:
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10, latent_dim=LATENT_DIM, learning_rate=0.02,
            margin=0.25, epochs=EPOCHS, bucket_bits=8, probe_radius=1,
            shortlist_k=1, seed=seed,
        )
    )

def _h_invariant_task(seed: int, h: int, step: int) -> EndToEndTask:
    base = build_task(seed, H_ANCHOR, step)
    candidates = build_population(seed, h)
    relevant = frozenset(i for i, candidate in enumerate(candidates)
                         if tuple(candidate.descriptor) == base.target_value)
    if len(relevant) != R:
        raise AssertionError("registered task must have four relevant programs")
    expected, work, calls = EndToEndExecutor().execute_population(
        tuple(candidates[i] for i in sorted(relevant)), base.target_value
    )
    assert work == R * 12 and calls == R
    return EndToEndTask(
        query=base.query, state_updates=base.state_updates, candidates=candidates,
        target_address=base.target_address, target_value=base.target_value,
        expected_output=expected, relevant_indices=relevant, step=step,
    )

def _pool(state):
    pool = []
    for address in state.addresses():
        read = state.read(Query(
            text="\t" + address, step=state.current_step,
            provenance="c5_prototype_selective_pool",
        ))
        if read.keys:
            pool.append((address, tuple(int(x) for x in read.values[0])))
    if len(pool) != M:
        raise AssertionError(f"expected {M} readable states, got {len(pool)}")
    return tuple(pool)

def _cosine(values_a, values_b):
    return sum(a * b for a, b in zip(values_a, values_b))

def _normalize(values):
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-8:
        raise AssertionError("zero embedding encountered")
    return tuple(v / norm for v in values)

def run_cell(seed: int, h: int) -> dict:
    first = _h_invariant_task(seed, h, 0)
    state = prepare_state(first)
    pool = _pool(state)
    model = _model(seed)
    updates = model.train_with_negative_coverage(
        TRAIN_CODES, negative_count=NEGATIVE_COUNT, aggregation="mean", positive_views=1
    )
    if updates != EPOCHS * len(TRAIN_CODES):
        raise AssertionError("optimizer update count changed")

    exhaustive_embeddings = tuple(_normalize(model.encode_state(value)) for _, value in pool)
    selective = LearnedPrototypeStateIndex(
        model, PrototypeStateIndexConfig(prototype_count=16, bucket_capacity=4, kmeans_iterations=8)
    )
    build = selective.build(state, TRAIN_CODES)
    executor = EndToEndExecutor()

    actual_target = []
    proposal_target = []
    exhaustive_target = []
    selective_success = []
    exhaustive_success = []
    shortlist_sizes = []
    prototype_macs = []
    rerank_macs = []
    norm_ops = []
    selected_exec_work = []
    exhaustive_exec_work = []

    for step in range(EVAL_STEPS):
        task = _h_invariant_task(seed, h, step)
        q = _normalize(model.encode_query(task.query))
        scores = [_cosine(q, emb) for emb in exhaustive_embeddings]
        exhaustive_idx = max(range(M), key=lambda i: (scores[i], -i))
        exhaustive_address = pool[exhaustive_idx][0]
        exhaustive_value = pool[exhaustive_idx][1]
        exhaustive_target.append(int(exhaustive_address == task.target_address))

        exhaustive_relevant = tuple(
            i for i, candidate in enumerate(task.candidates)
            if tuple(candidate.descriptor) == tuple(exhaustive_value)
        )
        exhaustive_output, ew, _ = executor.execute_population(
            tuple(task.candidates[i] for i in exhaustive_relevant), exhaustive_value
        )
        exhaustive_success.append(int(exhaustive_output == task.expected_output))
        exhaustive_exec_work.append(ew)

        hit = selective.lookup(task.query)
        proposal_target.append(int(task.target_address in hit.candidate_addresses))
        shortlist_sizes.append(len(hit.candidate_addresses))
        prototype_macs.append(hit.prototype_score_macs)
        rerank_macs.append(hit.state_rerank_macs)
        norm_ops.append(hit.normalization_ops)

        if hit.address is None:
            actual_target.append(0)
            selective_success.append(0)
            selected_exec_work.append(0)
            continue

        selected_value = next(value for address, value in pool if address == hit.address)
        actual_target.append(int(hit.address == task.target_address))
        selective_relevant = tuple(
            i for i, candidate in enumerate(task.candidates)
            if tuple(candidate.descriptor) == tuple(selected_value)
        )
        selective_output, sw, _ = executor.execute_population(
            tuple(task.candidates[i] for i in selective_relevant), selected_value
        )
        selective_success.append(int(selective_output == task.expected_output))
        selected_exec_work.append(sw)

    return {
        'seed': seed, 'H': h, 'M': M, 'R': R, 'K': K, 'queries': EVAL_STEPS,
        'exhaustive_target_recall': statistics.fmean(exhaustive_target),
        'selective_actual_target_recall': statistics.fmean(actual_target),
        'proposal_actual_target_retention': statistics.fmean(proposal_target),
        'exhaustive_end_to_end_success': statistics.fmean(exhaustive_success),
        'selective_actual_end_to_end_success': statistics.fmean(selective_success),
        'shortlist_size_mean': statistics.fmean(shortlist_sizes),
        'prototype_score_macs_mean': statistics.fmean(prototype_macs),
        'state_rerank_macs_mean': statistics.fmean(rerank_macs),
        'selective_query_macs_mean': QUERY_PROJECTION_MACS + statistics.fmean(prototype_macs) + statistics.fmean(rerank_macs),
        'exhaustive_query_macs': QUERY_PROJECTION_MACS + EXHAUSTIVE_STATE_SCORE_MACS,
        'binary_probe_count': 0,
        'normalization_ops_mean': statistics.fmean(norm_ops),
        'build_macs_total': build.total_build_macs,
        'build_train_embedding_macs': build.train_embedding_macs,
        'build_kmeans_macs': build.kmeans_macs,
        'build_state_embedding_macs': build.state_embedding_macs,
        'build_assignment_macs': build.state_assignment_macs,
        'candidate_executor_work_mean': statistics.fmean(selected_exec_work),
        'exhaustive_executor_work_mean': statistics.fmean(exhaustive_exec_work),
        'state_score_arithmetic_reduction': 1.0 - statistics.fmean(rerank_macs) / EXHAUSTIVE_STATE_SCORE_MACS,
        'state_target_gap_vs_exhaustive': statistics.fmean(exhaustive_target) - statistics.fmean(actual_target),
        'end_to_end_gap_vs_exhaustive': statistics.fmean(exhaustive_success) - statistics.fmean(selective_success),
    }

def main() -> None:
    contract = load_contract('TACOSM-C5-PROTOTYPE-SELECTIVE-001')
    contract.require_levels(H_LEVELS)
    contract.require_seeds(SEEDS)
    contract.require_steps(EPOCHS)
    contract.require_eval_steps(EVAL_STEPS)
    contract.require_k_levels([K])
    contract.require_arms(['exhaustive_cosine', 'prototype_selective'])
    if set(TRAIN_CODES) & set(EVAL_CODES):
        raise AssertionError('training/evaluation semantic codes overlap')

    cells = [run_cell(seed, h) for h in H_LEVELS for seed in SEEDS]
    by_h = {}
    for h in H_LEVELS:
        group = [c for c in cells if c['H'] == h]
        by_h[str(h)] = {
            'exhaustive_target_recall_mean': statistics.fmean(c['exhaustive_target_recall'] for c in group),
            'selective_actual_target_recall_mean': statistics.fmean(c['selective_actual_target_recall'] for c in group),
            'proposal_actual_target_retention_mean': statistics.fmean(c['proposal_actual_target_retention'] for c in group),
            'exhaustive_end_to_end_success_mean': statistics.fmean(c['exhaustive_end_to_end_success'] for c in group),
            'selective_actual_end_to_end_success_mean': statistics.fmean(c['selective_actual_end_to_end_success'] for c in group),
            'shortlist_size_mean': statistics.fmean(c['shortlist_size_mean'] for c in group),
            'selective_query_macs_mean': statistics.fmean(c['selective_query_macs_mean'] for c in group),
            'state_score_arithmetic_reduction_mean': statistics.fmean(c['state_score_arithmetic_reduction'] for c in group),
        }

    exhaustive_target = statistics.fmean(c['exhaustive_target_recall'] for c in cells)
    selective_target = statistics.fmean(c['selective_actual_target_recall'] for c in cells)
    exhaustive_success = statistics.fmean(c['exhaustive_end_to_end_success'] for c in cells)
    selective_success = statistics.fmean(c['selective_actual_end_to_end_success'] for c in cells)
    max_shortlist = max(c['shortlist_size_mean'] for c in cells)
    rule_pass = (
        selective_target >= exhaustive_target - 0.05
        and selective_success >= exhaustive_success - 0.05
        and max_shortlist <= K
    )

    result = {
        'protocol': {
            'name':'TACOSM-C5-PROTOTYPE-SELECTIVE-001', 'seeds':list(SEEDS),
            'H_levels':list(H_LEVELS), 'M':M, 'R':R, 'K':K, 'latent_dim':LATENT_DIM,
            'epochs':EPOCHS, 'negative_count':NEGATIVE_COUNT, 'positive_views':1,
            'queries_per_seed_per_H':EVAL_STEPS, 'prototype_count':16,
            'bucket_capacity':4, 'reranker':'cosine', 'candidate_index':'exact',
        },
        'overall': {
            'exhaustive_target_recall_mean': exhaustive_target,
            'selective_actual_target_recall_mean': selective_target,
            'exhaustive_end_to_end_success_mean': exhaustive_success,
            'selective_actual_end_to_end_success_mean': selective_success,
            'state_target_gap': exhaustive_target - selective_target,
            'end_to_end_gap': exhaustive_success - selective_success,
            'max_shortlist_size_mean': max_shortlist,
            'capability_rule_pass': rule_pass,
            'build_total_macs': cells[0]['build_macs_total'],
            'exhaustive_query_macs': cells[0]['exhaustive_query_macs'],
            'selective_query_macs_mean': statistics.fmean(c['selective_query_macs_mean'] for c in cells),
        },
        'by_H': by_h, 'cells': cells,
    }
    out = Path('artifacts/TACOSM-C5-PROTOTYPE-SELECTIVE-001.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()