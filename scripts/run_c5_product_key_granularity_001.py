#!/usr/bin/env python3
"""Run TACOSM-C5-PRODUCT-KEY-GRANULARITY-001."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import Query, StateUpdate
from tac_osm.c5_end_to_end import EndToEndExecutor, EndToEndTask, build_population
from tac_osm.contract import load_contract
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from tac_osm.noisy_state_tasks import CODEBOOK
from tac_osm.product_key_index import ProductKeyConfig, ProductKeyStateIndex

SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (128, 256, 512)
H_FIXED = 256
K = 32
FACTOR_BEAM = 6
FACTOR_SIZES = (8, 16, 32)
ARM_NAMES = ("factor_size_8", "factor_size_16", "factor_size_32")
FACTOR_KMEANS_ITERATIONS = 8
LATENT_DIM = 16
TEACHER_EPOCHS = 512
NEGATIVE_COUNT = 8
EVAL_STEPS = 100
TARGET_CODES = tuple(CODEBOOK[:16])
TRAIN_CODES = tuple(CODEBOOK[16:64])
STATE_BITS = 10
QUERY_PROJECTION_MACS = STATE_BITS * LATENT_DIM


def _all_binary_codes() -> tuple[tuple[int, ...], ...]:
    excluded = set(TARGET_CODES) | set(TRAIN_CODES)
    out = []
    for value in range(1 << STATE_BITS):
        code = tuple(
            (value >> (STATE_BITS - 1 - i)) & 1
            for i in range(STATE_BITS)
        )
        if code not in excluded:
            out.append(code)
    return tuple(out)


DECOY_CODES = _all_binary_codes()


def make_teacher(seed: int) -> LearnedSemanticStateIndex:
    model = LearnedSemanticStateIndex(LearnedStateIndexConfig(
        input_dim=STATE_BITS,
        latent_dim=LATENT_DIM,
        learning_rate=0.02,
        margin=0.25,
        epochs=TEACHER_EPOCHS,
        bucket_bits=8,
        probe_radius=1,
        shortlist_k=1,
        seed=seed,
    ))
    updates = model.train_with_negative_coverage(
        TRAIN_CODES,
        negative_count=NEGATIVE_COUNT,
        aggregation="mean",
        positive_views=1,
    )
    expected = TEACHER_EPOCHS * len(TRAIN_CODES)
    if updates != expected:
        raise AssertionError(
            f"unexpected teacher update count: {updates} != {expected}"
        )
    return model


def build_scaled_task(seed: int, m: int, step: int) -> EndToEndTask:
    if m not in M_LEVELS:
        raise ValueError(f"M must be one of {M_LEVELS}")
    target_rng = random.Random(seed * 1009 + m * 7919 + step * 104729 + 31)
    state_rng = random.Random(seed * 3001 + m * 15485863 + 31)
    target_value = tuple(target_rng.choice(TARGET_CODES))
    flip = (seed + step * 3 + m) % STATE_BITS
    bits = list(target_value)
    bits[flip] ^= 1
    query = Query(
        text=" ".join(str(int(x)) for x in bits),
        context=(1,) * STATE_BITS,
        step=step + 1,
        provenance="c5_product_key_granularity_query",
    )

    values = list(TARGET_CODES) + list(TRAIN_CODES)
    if len(values) != 64:
        raise AssertionError("base state pool must contain exactly 64 items")
    decoys = list(DECOY_CODES)
    state_rng.shuffle(decoys)
    if m > 64:
        values.extend(decoys[:m - 64])
    state_rng.shuffle(values)

    addresses = [f"granularity-state-{seed}-{m}-{i:04d}" for i in range(m)]
    updates = tuple(
        StateUpdate(
            key=addresses[i],
            value=tuple(values[i]),
            step=0,
            success_score=1.0,
        )
        for i in range(m)
    )
    target_address = next(
        update.key for update in updates
        if tuple(update.value) == target_value
    )

    candidates = build_population(seed, H_FIXED)
    relevant = frozenset(
        i for i, candidate in enumerate(candidates)
        if tuple(candidate.descriptor) == target_value
    )
    if len(relevant) != 4:
        raise AssertionError("task must contain four relevant programs")

    executor = EndToEndExecutor()
    expected, work, calls = executor.execute_population(
        tuple(candidates[i] for i in sorted(relevant)),
        target_value,
    )
    if work != 4 * 12 or calls != 4:
        raise AssertionError("fixed executor changed")

    return EndToEndTask(
        query=query,
        state_updates=updates,
        candidates=candidates,
        target_address=target_address,
        target_value=target_value,
        expected_output=expected,
        relevant_indices=relevant,
        step=step,
    )


def normalized_embeddings(
    teacher: LearnedSemanticStateIndex,
    task: EndToEndTask,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    items = []
    for update in task.state_updates:
        raw = teacher.encode_state(update.value)
        norm = math.sqrt(sum(x * x for x in raw))
        if norm <= 1e-8:
            raise AssertionError("zero state embedding")
        items.append((update.key, tuple(x / norm for x in raw)))
    return tuple(items)


def normalized_query(teacher: LearnedSemanticStateIndex, task: EndToEndTask) -> tuple[float, ...]:
    raw = teacher.encode_query(task.query)
    norm = math.sqrt(sum(x * x for x in raw))
    if norm <= 1e-8:
        raise AssertionError("zero query embedding")
    return tuple(x / norm for x in raw)


def exhaustive(
    task: EndToEndTask,
    query: tuple[float, ...],
    embeddings: tuple[tuple[str, tuple[float, ...]], ...],
) -> tuple[str, int]:
    scored = [
        (sum(a * b for a, b in zip(query, emb)), address)
        for address, emb in embeddings
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1], QUERY_PROJECTION_MACS + len(embeddings) * LATENT_DIM


def run_cell(
    seed: int,
    m: int,
    factor_size: int,
    teacher: LearnedSemanticStateIndex,
    embeddings: tuple[tuple[str, tuple[float, ...]], ...],
    tasks: tuple[EndToEndTask, ...],
) -> dict:
    if factor_size not in FACTOR_SIZES:
        raise ValueError(f"factor_size must be one of {FACTOR_SIZES}")

    first = tasks[0]
    train_set = set(TRAIN_CODES)
    train_items = [
        item
        for item, update in zip(embeddings, first.state_updates)
        if tuple(update.value) in train_set
    ]
    if len(train_items) != len(TRAIN_CODES):
        raise AssertionError("training-state population mismatch")

    index = ProductKeyStateIndex(ProductKeyConfig(
        factor_size=factor_size,
        factor_beam=FACTOR_BEAM,
        iterations=FACTOR_KMEANS_ITERATIONS,
        max_shortlist=K,
    ))
    build = index.build(embeddings, codebook_items=train_items)
    executor = EndToEndExecutor()

    ex_target = []
    proposal = []
    target_sel = []
    success_ex = []
    success_sel = []
    shortlist = []
    scored = []
    macs = []

    for task in tasks:
        query = normalized_query(teacher, task)
        ex_address, _ = exhaustive(task, query, embeddings)
        ex_value = next(
            tuple(u.value)
            for u in task.state_updates
            if u.key == ex_address
        )
        ex_target.append(int(ex_address == task.target_address))
        ex_rel = tuple(
            i for i, c in enumerate(task.candidates)
            if tuple(c.descriptor) == ex_value
        )
        ex_out, _, _ = executor.execute_population(
            tuple(task.candidates[i] for i in ex_rel),
            ex_value,
        )
        success_ex.append(int(ex_out == task.expected_output))

        hit = index.lookup(
            query,
            beam=FACTOR_BEAM,
            max_shortlist=K,
        )
        selected = hit.selected_address
        selected_value = next(
            (tuple(u.value) for u in task.state_updates if u.key == selected),
            None,
        )
        proposal.append(int(task.target_address in hit.candidate_addresses))
        target_sel.append(int(selected == task.target_address))

        if selected_value is None:
            success_sel.append(0)
        else:
            rel = tuple(
                i for i, c in enumerate(task.candidates)
                if tuple(c.descriptor) == selected_value
            )
            output, _, _ = executor.execute_population(
                tuple(task.candidates[i] for i in rel),
                selected_value,
            )
            success_sel.append(int(output == task.expected_output))

        shortlist.append(len(hit.candidate_addresses))
        scored.append(hit.state_candidates_scored)
        macs.append(
            QUERY_PROJECTION_MACS
            + hit.factor_score_macs
            + hit.state_rerank_macs
        )

    return {
        "M": m,
        "factor_size": factor_size,
        "factor_beam": FACTOR_BEAM,
        "exhaustive_target_recall": statistics.fmean(ex_target),
        "proposal_target_retention": statistics.fmean(proposal),
        "selective_target_recall": statistics.fmean(target_sel),
        "exhaustive_end_to_end_success": statistics.fmean(success_ex),
        "selective_end_to_end_success": statistics.fmean(success_sel),
        "shortlist_size_mean": statistics.fmean(shortlist),
        "state_candidates_scored_mean": statistics.fmean(scored),
        "states_scored_over_M": statistics.fmean(scored) / m,
        "exhaustive_query_macs": QUERY_PROJECTION_MACS + m * LATENT_DIM,
        "total_macs_mean": statistics.fmean(macs),
        "arithmetic_reduction": 1.0 - statistics.fmean(macs) / (QUERY_PROJECTION_MACS + m * LATENT_DIM),
        "nonempty_cells": build.nonempty_cells,
        "max_cell_size": build.max_cell_size,
        "build_total_macs": build.total_build_macs,
    }


def run(smoke: bool) -> dict:
    seeds = (0,) if smoke else SEEDS
    ms = (128,) if smoke else M_LEVELS
    steps = 5 if smoke else EVAL_STEPS

    cells = []
    for seed in seeds:
        teacher = make_teacher(seed)
        for m in ms:
            first = build_scaled_task(seed, m, 0)
            embeddings = normalized_embeddings(teacher, first)
            tasks = tuple(build_scaled_task(seed, m, step) for step in range(steps))
            for factor_size in FACTOR_SIZES:
                cells.append(
                    run_cell(seed, m, factor_size, teacher, embeddings, tasks)
                )

    pooled = {}
    for m in ms:
        for factor_size in FACTOR_SIZES:
            values = [
                cell for cell in cells
                if cell["M"] == m and cell["factor_size"] == factor_size
            ]
            pooled[f"{m}:{factor_size}"] = {
                key: statistics.fmean(v[key] for v in values)
                for key in (
                    "exhaustive_target_recall",
                    "proposal_target_retention",
                    "selective_target_recall",
                    "exhaustive_end_to_end_success",
                    "selective_end_to_end_success",
                    "shortlist_size_mean",
                    "state_candidates_scored_mean",
                    "states_scored_over_M",
                    "total_macs_mean",
                    "arithmetic_reduction",
                    "nonempty_cells",
                    "max_cell_size",
                    "build_total_macs",
                )
            }
            pooled[f"{m}:{factor_size}"]["M"] = m
            pooled[f"{m}:{factor_size}"]["factor_size"] = factor_size

    return {
        "protocol": {
            "name": "TACOSM-C5-PRODUCT-KEY-GRANULARITY-001",
            "smoke": smoke,
            "seeds": list(seeds),
            "M_levels": list(ms),
            "H_fixed": H_FIXED,
            "K": K,
            "factor_beam": FACTOR_BEAM,
            "factor_sizes": list(FACTOR_SIZES),
            "factor_kmeans_iterations": FACTOR_KMEANS_ITERATIONS,
            "latent_dim": LATENT_DIM,
            "teacher_epochs": TEACHER_EPOCHS,
            "negative_count": NEGATIVE_COUNT,
            "eval_steps": steps,
            "arms": list(ARM_NAMES),
        },
        "pooled": pooled,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-contract", action="store_true")
    args = parser.parse_args()

    if not args.no_contract:
        contract = load_contract("TACOSM-C5-PRODUCT-KEY-GRANULARITY-001")
        contract.require_levels((H_FIXED,))
        contract.require_m_levels(M_LEVELS)
        contract.require_seeds(SEEDS)
        contract.require_k_levels([K])
        contract.require_eval_steps(EVAL_STEPS if not args.smoke else 5)
        contract.require_arms(ARM_NAMES)

    result = run(args.smoke)
    out = Path("artifacts/TACOSM-C5-PRODUCT-KEY-GRANULARITY-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
