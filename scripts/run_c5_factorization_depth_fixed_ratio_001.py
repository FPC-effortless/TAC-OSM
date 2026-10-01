#!/usr/bin/env python3
"""Run TACOSM-C5-FACTORIZATION-DEPTH-FIXED-RATIO-001."""

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
from tac_osm.c5_end_to_end import build_population
from tac_osm.learned_state_index import LearnedSemanticStateIndex
from tac_osm.multifactor_product_key_index import (
    MultiFactorProductKeyConfig,
    MultiFactorProductKeyStateIndex,
)
from tac_osm.noisy_state_tasks import CODEBOOK
from scripts.run_c5_frontier_robustness_002 import (
    H_FIXED,
    K,
    M_LEVELS,
    SEEDS,
    TEACHER_EPOCHS,
    TRAIN_CODES,
    build_scaled_task,
    make_teacher,
    normalized_embeddings,
    normalized_query,
)

EVAL_STEPS = 100
FACTOR_SIZE = 8
FACTOR_BEAM = 5
FACTOR_COUNTS = (2, 3, 4, 5)
FACTOR_KMEANS_ITERATIONS = 8
ARMS = tuple((d, FACTOR_SIZE, FACTOR_BEAM) for d in FACTOR_COUNTS)
CAPABILITY_FLOOR = 0.90


class StateDistinctEndToEndExecutor:
    """Fixed downstream task whose successful output identifies the state value."""

    @staticmethod
    def _state_code(reference: tuple[int, ...]) -> int:
        value = 0
        for bit in reference:
            value = (value << 1) | int(bit)
        return value

    def execute_candidate(self, candidate, reference) -> tuple[int, int]:
        if not candidate.executable_edges:
            raise ValueError("candidate program must expose executable_edges")
        if tuple(candidate.descriptor) != tuple(reference):
            return 0, 12
        return 1_000_000 + self._state_code(tuple(reference)), 12

    def execute_population(self, candidates, reference) -> tuple[int, int, int]:
        output = 0
        work = 0
        invocations = 0
        for candidate in candidates:
            contribution, units = self.execute_candidate(candidate, reference)
            output += contribution
            work += units
            invocations += 1
        return output, work, invocations


def assert_executor_distinctness() -> None:
    executor = StateDistinctEndToEndExecutor()
    population = build_population(0, H_FIXED)
    outputs = []
    for reference in CODEBOOK[:2]:
        relevant = tuple(
            candidate for candidate in population
            if tuple(candidate.descriptor) == tuple(reference)
        )
        if len(relevant) != 4:
            raise AssertionError("state-distinct audit expected four relevant programs")
        output, work, calls = executor.execute_population(relevant, reference)
        if work != 4 * 12 or calls != 4:
            raise AssertionError("state-distinct executor accounting changed")
        outputs.append(output)
    if outputs[0] == outputs[1]:
        raise AssertionError("state-distinct executor collapsed two target codes")


def evaluate_arm(
    seed: int,
    m: int,
    factor_count: int,
    teacher: LearnedSemanticStateIndex,
    embeddings,
    tasks,
) -> dict:
    train_set = set(TRAIN_CODES)
    first = tasks[0]
    train_items = [
        item
        for item, update in zip(embeddings, first.state_updates)
        if tuple(update.value) in train_set
    ]
    if len(train_items) != len(TRAIN_CODES):
        raise AssertionError("training-state population mismatch")

    index = MultiFactorProductKeyStateIndex(
        MultiFactorProductKeyConfig(
            factor_count=factor_count,
            factor_size=FACTOR_SIZE,
            factor_beam=FACTOR_BEAM,
            iterations=FACTOR_KMEANS_ITERATIONS,
            max_shortlist=K,
        )
    )
    build = index.build(embeddings, codebook_items=train_items)
    executor = StateDistinctEndToEndExecutor()

    ex_target = 0
    sel_target = 0
    proposal_target = 0
    ex_success = 0
    sel_success = 0
    scored = 0
    shortlist = 0
    pair_ops = 0
    macs = 0

    for task in tasks:
        query = normalized_query(teacher, task)
        exhaustive_scores = [
            (sum(a * b for a, b in zip(query, embedding)), address)
            for address, embedding in embeddings
        ]
        exhaustive_scores.sort(key=lambda item: (-item[0], item[1]))
        ex_address = exhaustive_scores[0][1]
        ex_value = next(
            tuple(update.value)
            for update in task.state_updates
            if update.key == ex_address
        )
        ex_target += int(ex_address == task.target_address)

        ex_relevant = tuple(
            i
            for i, candidate in enumerate(task.candidates)
            if tuple(candidate.descriptor) == ex_value
        )
        ex_output, _, _ = executor.execute_population(
            tuple(task.candidates[i] for i in ex_relevant),
            ex_value,
        )
        ex_success += int(ex_output == task.expected_output)

        hit = index.lookup(query, beam=FACTOR_BEAM, max_shortlist=K)
        sel_address = hit.selected_address
        sel_target += int(sel_address == task.target_address)
        proposal_target += int(task.target_address in hit.candidate_addresses)

        sel_value = next(
            (
                tuple(update.value)
                for update in task.state_updates
                if update.key == sel_address
            ),
            None,
        )
        if sel_value is not None:
            relevant = tuple(
                i
                for i, candidate in enumerate(task.candidates)
                if tuple(candidate.descriptor) == sel_value
            )
            output, _, _ = executor.execute_population(
                tuple(task.candidates[i] for i in relevant),
                sel_value,
            )
            sel_success += int(output == task.expected_output)

        scored += hit.state_candidates_scored
        shortlist += len(hit.candidate_addresses)
        pair_ops += hit.pair_generation_ops
        macs += (
            16 * 10
            + hit.factor_score_macs
            + hit.state_rerank_macs
        )

    tasks_count = len(tasks)
    ex_recall = ex_target / tasks_count
    sel_recall = sel_target / tasks_count
    return {
        "seed": seed,
        "M": m,
        "factor_count": factor_count,
        "factor_size": FACTOR_SIZE,
        "factor_beam": FACTOR_BEAM,
        "B_over_F": FACTOR_BEAM / FACTOR_SIZE,
        "tasks": tasks_count,
        "exhaustive_target_successes": ex_target,
        "selective_target_successes": sel_target,
        "exhaustive_target_recall": ex_recall,
        "selective_target_recall": sel_recall,
        "target_recall_retention_ratio": (
            sel_recall / ex_recall if ex_recall > 0 else float("nan")
        ),
        "proposal_target_successes": proposal_target,
        "proposal_target_retention": proposal_target / tasks_count,
        "exhaustive_end_to_end_success": ex_success / tasks_count,
        "selective_end_to_end_success": sel_success / tasks_count,
        "state_candidates_scored_mean": scored / tasks_count,
        "states_scored_over_M": scored / tasks_count / m,
        "shortlist_size_mean": shortlist / tasks_count,
        "pair_generation_ops_mean": pair_ops / tasks_count,
        "total_macs_mean": macs / tasks_count,
        "nonempty_cells": build.nonempty_cells,
        "max_cell_size": build.max_cell_size,
        "build_total_macs": build.total_build_macs,
    }


def run(smoke: bool) -> dict:
    assert_executor_distinctness()
    seeds = (10,) if smoke else SEEDS
    ms = (128,) if smoke else M_LEVELS
    steps = 5 if smoke else EVAL_STEPS
    arms = ((3, FACTOR_SIZE, FACTOR_BEAM),) if smoke else ARMS

    cells = []
    for seed in seeds:
        teacher = make_teacher(seed)
        for m in ms:
            first = build_scaled_task(seed, m, 0)
            embeddings = normalized_embeddings(teacher, first)
            tasks = tuple(build_scaled_task(seed, m, step) for step in range(steps))
            for factor_count, _, _ in arms:
                cells.append(
                    evaluate_arm(
                        seed,
                        m,
                        factor_count,
                        teacher,
                        embeddings,
                        tasks,
                    )
                )

    pooled = {}
    for m in ms:
        for factor_count, _, _ in arms:
            values = [
                row for row in cells
                if row["M"] == m and row["factor_count"] == factor_count
            ]
            if not values:
                continue
            task_total = sum(row["tasks"] for row in values)
            ex = sum(row["exhaustive_target_successes"] for row in values)
            sel = sum(row["selective_target_successes"] for row in values)
            proposal = sum(row["proposal_target_successes"] for row in values)
            ex_end = sum(
                round(row["exhaustive_end_to_end_success"] * row["tasks"])
                for row in values
            )
            sel_end = sum(
                round(row["selective_end_to_end_success"] * row["tasks"])
                for row in values
            )
            raw_retention = sel / ex if ex else float("nan")
            reference = (FACTOR_BEAM / FACTOR_SIZE) ** factor_count
            pooled[f"{m}:d{factor_count}"] = {
                "M": m,
                "factor_count": factor_count,
                "factor_size": FACTOR_SIZE,
                "factor_beam": FACTOR_BEAM,
                "B_over_F": FACTOR_BEAM / FACTOR_SIZE,
                "exhaustive_target_successes": ex,
                "selective_target_successes": sel,
                "exhaustive_target_recall": ex / task_total,
                "selective_target_recall": sel / task_total,
                "target_recall_retention_ratio": raw_retention,
                "proposal_target_retention": proposal / task_total,
                "exhaustive_end_to_end_success": ex_end / task_total,
                "selective_end_to_end_success": sel_end / task_total,
                "state_candidates_scored_mean": statistics.fmean(
                    row["state_candidates_scored_mean"] for row in values
                ),
                "states_scored_over_M": statistics.fmean(
                    row["states_scored_over_M"] for row in values
                ),
                "geometric_scored_fraction_reference": reference,
                "geometric_reference_minus_observed": (
                    reference
                    - statistics.fmean(row["states_scored_over_M"] for row in values)
                ),
                "shortlist_size_mean": statistics.fmean(
                    row["shortlist_size_mean"] for row in values
                ),
                "pair_generation_ops_mean": statistics.fmean(
                    row["pair_generation_ops_mean"] for row in values
                ),
                "total_macs_mean": statistics.fmean(
                    row["total_macs_mean"] for row in values
                ),
                "nonempty_cells": statistics.fmean(
                    row["nonempty_cells"] for row in values
                ),
                "max_cell_size": statistics.fmean(
                    row["max_cell_size"] for row in values
                ),
                "build_total_macs": statistics.fmean(
                    row["build_total_macs"] for row in values
                ),
                "registered_capability_floor": CAPABILITY_FLOOR,
                "meets_capability_floor": raw_retention >= CAPABILITY_FLOOR,
            }

    return {
        "protocol": {
            "name":"TACOSM-C5-FACTORIZATION-DEPTH-FIXED-RATIO-001",
            "smoke": smoke,
            "seeds": list(seeds),
            "M_levels": list(ms),
            "H_fixed": H_FIXED,
            "K": K,
            "eval_steps": steps,
            "factor_size": FACTOR_SIZE,
            "factor_beam": FACTOR_BEAM,
            "B_over_F": FACTOR_BEAM / FACTOR_SIZE,
            "factor_counts": [arm[0] for arm in arms],
            "teacher_epochs": TEACHER_EPOCHS,
            "capability_floor": CAPABILITY_FLOOR,
        },
        "pooled": pooled,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-contract", action="store_true")
    args = parser.parse_args()

    if not args.no_contract and not args.smoke:
        contract = load_contract(
            "TACOSM-C5-FACTORIZATION-DEPTH-FIXED-RATIO-001"
        )
        contract.require_levels((H_FIXED,))
        contract.require_m_levels(M_LEVELS)
        contract.require_seeds(SEEDS)
        contract.require_k_levels([K])
        contract.require_arms(
            ["d2_f8_b5", "d3_f8_b5", "d4_f8_b5", "d5_f8_b5"]
        )
        contract.require_eval_steps(EVAL_STEPS)

    result = run(args.smoke)
    output = Path(
        "artifacts/TACOSM-C5-FACTORIZATION-DEPTH-FIXED-RATIO-001.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
