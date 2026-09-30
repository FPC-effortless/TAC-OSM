#!/usr/bin/env python3
"""Run preregistered TACOSM-C5-COSINE-SELECTIVE-001."""

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
from tac_osm.cosine_selective_state import CosineRerankedStateIndex
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
POSITIVE_VIEWS = 1
LATENT_DIM = 16
TRAIN_CODES = tuple(CODEBOOK[16:64])
EVAL_CODES = tuple(CODEBOOK[:16])
QUERY_PROJECTION_MACS = 10 * LATENT_DIM
EXHAUSTIVE_SCORE_MACS = M * LATENT_DIM
BUILD_MACS = M * QUERY_PROJECTION_MACS
BINARY_PROBES_RADIUS2 = 1 + 8 + 28
EXEC_WORK_PER_CANDIDATE = 12

def _model(seed: int) -> LearnedSemanticStateIndex:
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10, latent_dim=LATENT_DIM, learning_rate=0.02,
            margin=0.25, epochs=EPOCHS, bucket_bits=8, probe_radius=2,
            shortlist_k=K, seed=seed,
        )
    )

def _h_invariant_task(seed: int, h: int, step: int) -> EndToEndTask:
    base = build_task(seed, H_ANCHOR, step)
    candidates = build_population(seed, h)
    relevant = frozenset(
        i for i, candidate in enumerate(candidates)
        if tuple(candidate.descriptor) == base.target_value
    )
    if len(relevant) != R:
        raise AssertionError("registered task must have four relevant programs")
    expected, work, calls = EndToEndExecutor().execute_population(
        tuple(candidates[i] for i in sorted(relevant)), base.target_value
    )
    assert work == R * EXEC_WORK_PER_CANDIDATE and calls == R
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
            provenance="c5_cosine_selective_pool",
        ))
        if read.keys:
            pool.append((address, tuple(int(x) for x in read.values[0])))
    if len(pool) != M:
        raise AssertionError(f"expected {M} readable states, got {len(pool)}")
    return tuple(pool)

def _cosine(vec_a, vec_b):
    return sum(a * b for a, b in zip(vec_a, vec_b))

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
        TRAIN_CODES, negative_count=NEGATIVE_COUNT, aggregation="mean",
        positive_views=POSITIVE_VIEWS,
    )
    expected_updates = EPOCHS * len(TRAIN_CODES)
    if updates != expected_updates:
        raise AssertionError("optimizer update count changed")

    exhaustive_state_embeddings = tuple(
        _normalize(model.encode_state(value)) for _, value in pool
    )
    selective = CosineRerankedStateIndex(model)
    build = selective.build(state)
    executor = EndToEndExecutor()

    actual_target = []
    exhaustive_target = []
    proposal_target = []
    actual_end_to_end = []
    exhaustive_end_to_end = []
    shortlist_sizes = []
    rerank_macs = []
    normalization_ops = []
    executor_work_selective = []
    executor_work_exhaustive = []

    for step in range(EVAL_STEPS):
        task = _h_invariant_task(seed, h, step)
        q = _normalize(model.encode_query(task.query))
        scores = [_cosine(q, emb) for emb in exhaustive_state_embeddings]
        exhaustive_index = max(range(M), key=lambda i: (scores[i], -i))
        exhaustive_address = pool[exhaustive_index][0]
        exhaustive_value = pool[exhaustive_index][1]

        exhaustive_relevant = tuple(
            i for i, candidate in enumerate(task.candidates)
            if tuple(candidate.descriptor) == tuple(exhaustive_value)
        )
        exhaustive_output, ew, _ = executor.execute_population(
            tuple(task.candidates[i] for i in exhaustive_relevant), exhaustive_value
        )
        exhaustive_end_to_end.append(int(exhaustive_output == task.expected_output))
        executor_work_exhaustive.append(ew)
        exhaustive_target.append(int(exhaustive_address == task.target_address))

        hit = selective.lookup(task.query)
        proposal_target.append(int(task.target_address in hit.proposal.addresses))
        shortlist_sizes.append(hit.shortlist_size)
        rerank_macs.append(hit.cosine_score_macs)
        normalization_ops.append(hit.normalization_ops)

        if hit.address is None:
            actual_target.append(0)
            actual_end_to_end.append(0)
            executor_work_selective.append(0)
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
        actual_end_to_end.append(int(selective_output == task.expected_output))
        executor_work_selective.append(sw)

    exhaustive_target_recall = statistics.fmean(exhaustive_target)
    exhaustive_success = statistics.fmean(exhaustive_end_to_end)
    selective_target_recall = statistics.fmean(actual_target)
    selective_success = statistics.fmean(actual_end_to_end)

    return {
        "seed": seed, "H": h, "M": M, "R": R, "K": K, "queries": EVAL_STEPS,
        "exhaustive_target_recall": exhaustive_target_recall,
        "selective_actual_target_recall": selective_target_recall,
        "proposal_actual_target_retention": statistics.fmean(proposal_target),
        "exhaustive_end_to_end_success": exhaustive_success,
        "selective_actual_end_to_end_success": selective_success,
        "shortlist_size_mean": statistics.fmean(shortlist_sizes),
        "cosine_rerank_macs_mean": statistics.fmean(rerank_macs),
        "normalization_ops_mean": statistics.fmean(normalization_ops),
        "query_projection_macs": QUERY_PROJECTION_MACS,
        "exhaustive_state_score_macs": EXHAUSTIVE_SCORE_MACS,
        "selective_state_score_macs_mean": QUERY_PROJECTION_MACS + statistics.fmean(rerank_macs),
        "binary_probe_count": BINARY_PROBES_RADIUS2,
        "build_macs_total": BUILD_MACS,
        "exhaustive_executor_work_mean": statistics.fmean(executor_work_exhaustive),
        "selective_executor_work_mean": statistics.fmean(executor_work_selective),
        "state_score_arithmetic_reduction": 1.0 - statistics.fmean(rerank_macs) / EXHAUSTIVE_SCORE_MACS,
        "state_retention_gap_vs_exhaustive": exhaustive_target_recall - selective_target_recall,
        "end_to_end_gap_vs_exhaustive": exhaustive_success - selective_success,
    }

def main() -> None:
    contract = load_contract("TACOSM-C5-COSINE-SELECTIVE-001")
    contract.require_levels(H_LEVELS)
    contract.require_seeds(SEEDS)
    contract.require_steps(EPOCHS)
    contract.require_eval_steps(EVAL_STEPS)
    contract.require_k_levels([K])
    contract.require_arms(["exhaustive_cosine", "selective_cosine"])
    if set(TRAIN_CODES) & set(EVAL_CODES):
        raise AssertionError("training/evaluation semantic codes overlap")

    cells = [run_cell(seed, h) for h in H_LEVELS for seed in SEEDS]
    by_h = {}
    for h in H_LEVELS:
        group = [c for c in cells if c["H"] == h]
        by_h[str(h)] = {
            "exhaustive_target_recall_mean": statistics.fmean(c["exhaustive_target_recall"] for c in group),
            "selective_actual_target_recall_mean": statistics.fmean(c["selective_actual_target_recall"] for c in group),
            "proposal_actual_target_retention_mean": statistics.fmean(c["proposal_actual_target_retention"] for c in group),
            "exhaustive_end_to_end_success_mean": statistics.fmean(c["exhaustive_end_to_end_success"] for c in group),
            "selective_actual_end_to_end_success_mean": statistics.fmean(c["selective_actual_end_to_end_success"] for c in group),
            "shortlist_size_mean": statistics.fmean(c["shortlist_size_mean"] for c in group),
            "cosine_rerank_macs_mean": statistics.fmean(c["cosine_rerank_macs_mean"] for c in group),
            "state_score_arithmetic_reduction_mean": statistics.fmean(c["state_score_arithmetic_reduction"] for c in group),
        }

    selective_target = statistics.fmean(c["selective_actual_target_recall"] for c in cells)
    exhaustive_target = statistics.fmean(c["exhaustive_target_recall"] for c in cells)
    selective_success = statistics.fmean(c["selective_actual_end_to_end_success"] for c in cells)
    exhaustive_success = statistics.fmean(c["exhaustive_end_to_end_success"] for c in cells)
    max_shortlist = max(c["shortlist_size_mean"] for c in cells)
    rule_pass = (
        selective_target >= exhaustive_target - 0.05
        and selective_success >= exhaustive_success - 0.05
        and max_shortlist <= K
    )

    result = {
        "protocol": {
            "name":"TACOSM-C5-COSINE-SELECTIVE-001", "seeds":list(SEEDS),
            "H_levels":list(H_LEVELS), "M":M, "R":R, "K":K,
            "latent_dim":LATENT_DIM, "epochs":EPOCHS,
            "negative_count":NEGATIVE_COUNT, "positive_views":POSITIVE_VIEWS,
            "queries_per_seed_per_H":EVAL_STEPS,
            "state_index":"8-bit learned sign code, Hamming radius 2",
            "reranker":"cosine over retained K<=4 states",
        },
        "overall": {
            "exhaustive_target_recall_mean": exhaustive_target,
            "selective_actual_target_recall_mean": selective_target,
            "exhaustive_end_to_end_success_mean": exhaustive_success,
            "selective_actual_end_to_end_success_mean": selective_success,
            "state_target_gap": exhaustive_target - selective_target,
            "end_to_end_gap": exhaustive_success - selective_success,
            "max_shortlist_size_mean": max_shortlist,
            "capability_rule_pass": rule_pass,
        },
        "by_H": by_h,
        "cells": cells,
    }
    out = Path("artifacts/TACOSM-C5-COSINE-SELECTIVE-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()