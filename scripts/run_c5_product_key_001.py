#!/usr/bin/env python3
"""Run TACOSM-C5-PRODUCT-KEY-001."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm.c5_end_to_end import EndToEndExecutor, build_task, prepare_state
from tac_osm.contract import load_contract
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from tac_osm.noisy_state_tasks import CODEBOOK
from tac_osm.product_key_index import ProductKeyConfig, ProductKeyStateIndex

SEEDS = (0, 1, 2, 3, 4)
H_LEVELS = (64, 128, 256)
K_LEVELS = (4, 8, 16)
BEAM_FOR_K = {4: 1, 8: 2, 16: 4}
TRAIN_CODES = tuple(CODEBOOK[16:64])
M = 64
R = 4
LATENT_DIM = 16
TEACHER_EPOCHS = 512
EVAL_STEPS = 100
NEGATIVE_COUNT = 8
FACTOR_SIZE = 8
KMEANS_ITERATIONS = 8
QUERY_PROJECTION_MACS = 10 * LATENT_DIM
EXHAUSTIVE_QUERY_MACS = QUERY_PROJECTION_MACS + M * LATENT_DIM


def make_teacher(seed: int) -> LearnedSemanticStateIndex:
    model = LearnedSemanticStateIndex(LearnedStateIndexConfig(
        input_dim=10, latent_dim=LATENT_DIM, learning_rate=0.02, margin=0.25,
        epochs=TEACHER_EPOCHS, bucket_bits=8, probe_radius=1, shortlist_k=1, seed=seed,
    ))
    updates = model.train_with_negative_coverage(
        TRAIN_CODES, negative_count=NEGATIVE_COUNT, aggregation="mean", positive_views=1
    )
    expected = TEACHER_EPOCHS * len(TRAIN_CODES)
    if updates != expected:
        raise AssertionError(f"unexpected teacher update count: {updates} != {expected}")
    return model


def exhaustive_cosine(teacher, task):
    q_raw = teacher.encode_query(task.query)
    q_norm = math.sqrt(sum(x * x for x in q_raw))
    if q_norm <= 1e-8:
        raise AssertionError("teacher query embedding is zero")
    q = tuple(x / q_norm for x in q_raw)
    scored = []
    for update in task.state_updates:
        z_raw = teacher.encode_state(update.value)
        z_norm = math.sqrt(sum(x * x for x in z_raw))
        if z_norm <= 1e-8:
            raise AssertionError("teacher state embedding is zero")
        z = tuple(x / z_norm for x in z_raw)
        scored.append((sum(a * b for a, b in zip(q, z)), update.key, tuple(update.value)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    score, address, value = scored[0]
    return address, value, score


def runtime_items(teacher, task):
    all_items = [(u.key, tuple(teacher.encode_state(u.value))) for u in task.state_updates]
    train_set = set(TRAIN_CODES)
    train_items = [
        (u.key, tuple(teacher.encode_state(u.value)))
        for u in task.state_updates
        if tuple(u.value) in train_set
    ]
    if len(train_items) != len(TRAIN_CODES):
        raise AssertionError("runtime state must contain every training code exactly once")
    return all_items, train_items


def run_seed(seed: int, steps: int, smoke: bool) -> list[dict]:
    first = build_task(seed, 64, 0)
    teacher = make_teacher(seed)
    all_items, train_items = runtime_items(teacher, first)
    index = ProductKeyStateIndex(ProductKeyConfig(
        factor_size=FACTOR_SIZE, factor_beam=2, iterations=KMEANS_ITERATIONS, max_shortlist=16
    ))
    build = index.build(all_items, codebook_items=train_items)
    executor = EndToEndExecutor()
    results = []
    for h in (64,) if smoke else H_LEVELS:
        accum = {k: {"ex_target": [], "proposal": [], "sel_target": [], "ex_success": [],
                      "sel_success": [], "shortlist": [], "macs": [], "scored": []} for k in K_LEVELS}
        for step in range(steps):
            task = build_task(seed, h, step)
            ex_address, ex_value, _ = exhaustive_cosine(teacher, task)
            ex_target = int(ex_address == task.target_address)
            ex_relevant = tuple(i for i, c in enumerate(task.candidates) if tuple(c.descriptor) == ex_value)
            ex_output, _, _ = executor.execute_population(tuple(task.candidates[i] for i in ex_relevant), ex_value)
            ex_success = int(ex_output == task.expected_output)
            q = teacher.encode_query(task.query)
            for k in K_LEVELS:
                hit = index.lookup(q, beam=BEAM_FOR_K[k], max_shortlist=k)
                selected = hit.selected_address
                selected_value = next((tuple(u.value) for u in task.state_updates if u.key == selected), None)
                retention = int(task.target_address in hit.candidate_addresses)
                selected_target = int(selected == task.target_address)
                if selected_value is None:
                    success = 0
                else:
                    relevant = tuple(i for i, c in enumerate(task.candidates) if tuple(c.descriptor) == selected_value)
                    output, _, _ = executor.execute_population(tuple(task.candidates[i] for i in relevant), selected_value)
                    success = int(output == task.expected_output)
                acc = accum[k]
                acc["ex_target"].append(ex_target)
                acc["proposal"].append(retention)
                acc["sel_target"].append(selected_target)
                acc["ex_success"].append(ex_success)
                acc["sel_success"].append(success)
                acc["shortlist"].append(len(hit.candidate_addresses))
                acc["macs"].append(QUERY_PROJECTION_MACS + hit.factor_score_macs + hit.state_rerank_macs)
                acc["scored"].append(hit.state_candidates_scored)
        cells = {}
        for k in K_LEVELS:
            v = accum[k]
            cells[str(k)] = {
                "k": k, "beam_width": BEAM_FOR_K[k],
                "exhaustive_target_recall": statistics.fmean(v["ex_target"]),
                "proposal_target_retention": statistics.fmean(v["proposal"]),
                "selective_target_recall": statistics.fmean(v["sel_target"]),
                "exhaustive_end_to_end_success": statistics.fmean(v["ex_success"]),
                "selective_end_to_end_success": statistics.fmean(v["sel_success"]),
                "shortlist_size_mean": statistics.fmean(v["shortlist"]),
                "state_candidates_scored_mean": statistics.fmean(v["scored"]),
                "total_macs_mean": statistics.fmean(v["macs"]),
                "arithmetic_reduction": 1.0 - statistics.fmean(v["macs"]) / EXHAUSTIVE_QUERY_MACS,
            }
        results.append({
            "seed": seed, "H": h, "eval_steps": steps, "cells": cells,
            "prototype_build": {
                "state_items": build.state_items,
                "codebook_training_items": build.codebook_training_items,
                "codebook_fit_macs": build.build_similarity_macs,
                "state_assignment_macs": build.state_assignment_macs,
                "total_build_macs": build.total_build_macs,
                "nonempty_cells": build.nonempty_cells,
                "max_cell_size": build.max_cell_size,
            },
            "exhaustive_query_macs": EXHAUSTIVE_QUERY_MACS,
        })
    return results


def run(smoke: bool) -> dict:
    seeds = (0,) if smoke else SEEDS
    steps = 5 if smoke else EVAL_STEPS
    results = []
    for seed in seeds:
        results.extend(run_seed(seed, steps, smoke))
    pooled = {}
    for k in K_LEVELS:
        values = [r["cells"][str(k)] for r in results]
        pooled[str(k)] = {
            "exhaustive_target_recall_mean": statistics.fmean(v["exhaustive_target_recall"] for v in values),
            "proposal_target_retention_mean": statistics.fmean(v["proposal_target_retention"] for v in values),
            "selective_target_recall_mean": statistics.fmean(v["selective_target_recall"] for v in values),
            "exhaustive_end_to_end_success_mean": statistics.fmean(v["exhaustive_end_to_end_success"] for v in values),
            "selective_end_to_end_success_mean": statistics.fmean(v["selective_end_to_end_success"] for v in values),
            "shortlist_size_mean": statistics.fmean(v["shortlist_size_mean"] for v in values),
            "state_candidates_scored_mean": statistics.fmean(v["state_candidates_scored_mean"] for v in values),
            "total_macs_mean": statistics.fmean(v["total_macs_mean"] for v in values),
            "arithmetic_reduction_mean": statistics.fmean(v["arithmetic_reduction"] for v in values),
        }
    return {
        "protocol": {"name": "TACOSM-C5-PRODUCT-KEY-001", "smoke": smoke,
                     "seeds": list(seeds), "H_levels": [64] if smoke else list(H_LEVELS),
                     "K_levels": list(K_LEVELS), "beam_for_K": {str(k): BEAM_FOR_K[k] for k in K_LEVELS},
                     "M": M, "R": R, "latent_dim": LATENT_DIM, "factor_size": FACTOR_SIZE,
                     "kmeans_iterations": KMEANS_ITERATIONS, "teacher_epochs": TEACHER_EPOCHS,
                     "negative_count": NEGATIVE_COUNT, "eval_steps": steps},
        "pooled": pooled, "cells": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-contract", action="store_true")
    args = parser.parse_args()
    if not args.no_contract:
        contract = load_contract("TACOSM-C5-PRODUCT-KEY-001")
        contract.require_levels(H_LEVELS)
        contract.require_seeds(SEEDS)
        contract.require_k_levels(K_LEVELS)
    result = run(args.smoke)
    out = Path("artifacts/TACOSM-C5-PRODUCT-KEY-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
