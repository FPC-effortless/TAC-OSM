#!/usr/bin/env python3
"""Run TACOSM-C5-MULTIFACTOR-GRANULARITY-BEAM-FRONTIER-001."""

from __future__ import annotations

import argparse
import json
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
    NEGATIVE_COUNT,
    SEEDS,
    TEACHER_EPOCHS,
    TRAIN_CODES,
    build_scaled_task,
    make_teacher,
    normalized_embeddings,
    normalized_query,
)

EVAL_STEPS = 100
LATENT_DIM = 16
FACTOR_KMEANS_ITERATIONS = 8
FACTOR_COUNT = 3
FACTOR_SIZE = 16
FACTOR_BEAMS = (4, 5, 6)
CAPABILITY_FLOOR = 0.90
ARMS = tuple((FACTOR_COUNT, FACTOR_SIZE, beam) for beam in FACTOR_BEAMS)


class StateDistinctEndToEndExecutor:
    """Fixed executor whose successful output identifies the state value."""

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
            candidate
            for candidate in population
            if tuple(candidate.descriptor) == tuple(reference)
        )
        if len(relevant) != 4:
            raise AssertionError(
                "state-distinct audit expected four relevant programs"
            )
        output, work, calls = executor.execute_population(relevant, reference)
        if work != 4 * 12 or calls != 4:
            raise AssertionError("state-distinct audit executor accounting changed")
        outputs.append(output)
    if outputs[0] == outputs[1]:
        raise AssertionError("state-distinct executor collapsed two target codes")


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def evaluate_arm(
    seed: int,
    m: int,
    factor_beam: int,
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
            factor_count=FACTOR_COUNT,
            factor_size=FACTOR_SIZE,
            factor_beam=factor_beam,
            iterations=FACTOR_KMEANS_ITERATIONS,
            max_shortlist=K,
        )
    )
    build = index.build(embeddings, codebook_items=train_items)
    executor = StateDistinctEndToEndExecutor()

    exhaustive_target_successes = 0
    selective_target_successes = 0
    proposal_target_successes = 0
    exhaustive_end_to_end_successes = 0
    selective_end_to_end_successes = 0
    task_count = 0
    state_scored_total = 0
    shortlist_total = 0
    mac_total = 0
    pair_ops_total = 0

    for task in tasks:
        query = normalized_query(teacher, task)
        exhaustive_scores = [
            (sum(a * b for a, b in zip(query, embedding)), address)
            for address, embedding in embeddings
        ]
        exhaustive_scores.sort(key=lambda item: (-item[0], item[1]))
        exhaustive_address = exhaustive_scores[0][1]
        exhaustive_value = next(
            tuple(update.value)
            for update in task.state_updates
            if update.key == exhaustive_address
        )
        exhaustive_target_successes += int(
            exhaustive_address == task.target_address
        )

        exhaustive_relevant = tuple(
            i
            for i, candidate in enumerate(task.candidates)
            if tuple(candidate.descriptor) == exhaustive_value
        )
        exhaustive_output, _, _ = executor.execute_population(
            tuple(task.candidates[i] for i in exhaustive_relevant),
            exhaustive_value,
        )
        exhaustive_end_to_end_successes += int(
            exhaustive_output == task.expected_output
        )

        hit = index.lookup(
            query,
            beam=factor_beam,
            max_shortlist=K,
        )
        selective = hit.selected_address
        proposal_target_successes += int(
            task.target_address in hit.candidate_addresses
        )
        selective_target_successes += int(
            selective == task.target_address
        )

        selected_value = next(
            (
                tuple(update.value)
                for update in task.state_updates
                if update.key == selective
            ),
            None,
        )
        if selected_value is not None:
            relevant = tuple(
                i
                for i, candidate in enumerate(task.candidates)
                if tuple(candidate.descriptor) == selected_value
            )
            output, _, _ = executor.execute_population(
                tuple(task.candidates[i] for i in relevant),
                selected_value,
            )
            selective_end_to_end_successes += int(
                output == task.expected_output
            )

        state_scored_total += hit.state_candidates_scored
        shortlist_total += len(hit.candidate_addresses)
        pair_ops_total += hit.pair_generation_ops
        mac_total += (
            LATENT_DIM * 10
            + hit.factor_score_macs
            + hit.state_rerank_macs
        )
        task_count += 1

    exhaustive_recall = safe_ratio(exhaustive_target_successes, task_count)
    selective_recall = safe_ratio(selective_target_successes, task_count)
    exhaustive_success = safe_ratio(
        exhaustive_end_to_end_successes, task_count
    )
    selective_success = safe_ratio(
        selective_end_to_end_successes, task_count
    )

    return {
        "seed": seed,
        "M": m,
        "factor_count": FACTOR_COUNT,
        "factor_size": FACTOR_SIZE,
        "factor_beam": factor_beam,
        "tasks": task_count,
        "exhaustive_target_successes": exhaustive_target_successes,
        "selective_target_successes": selective_target_successes,
        "exhaustive_target_recall": exhaustive_recall,
        "selective_target_recall": selective_recall,
        "target_recall_retention_ratio": safe_ratio(
            selective_target_successes,
            exhaustive_target_successes,
        ),
        "proposal_target_successes": proposal_target_successes,
        "proposal_target_retention": safe_ratio(
            proposal_target_successes,
            task_count,
        ),
        "exhaustive_end_to_end_successes": exhaustive_end_to_end_successes,
        "selective_end_to_end_successes": selective_end_to_end_successes,
        "exhaustive_end_to_end_success": exhaustive_success,
        "selective_end_to_end_success": selective_success,
        "state_candidates_scored_mean": state_scored_total / task_count,
        "states_scored_over_M": state_scored_total / task_count / m,
        "shortlist_size_mean": shortlist_total / task_count,
        "pair_generation_ops_mean": pair_ops_total / task_count,
        "total_macs_mean": mac_total / task_count,
        "nonempty_cells": build.nonempty_cells,
        "max_cell_size": build.max_cell_size,
        "build_total_macs": build.total_build_macs,
    }


def run(smoke: bool) -> dict:
    assert_executor_distinctness()
    seeds = (10,) if smoke else SEEDS
    ms = (128,) if smoke else M_LEVELS
    steps = 5 if smoke else EVAL_STEPS

    cells = []
    for seed in seeds:
        teacher = make_teacher(seed)
        for m in ms:
            first = build_scaled_task(seed, m, 0)
            embeddings = normalized_embeddings(teacher, first)
            tasks = tuple(
                build_scaled_task(seed, m, step)
                for step in range(steps)
            )
            for _, _, beam in ARMS:
                cells.append(
                    evaluate_arm(
                        seed,
                        m,
                        beam,
                        teacher,
                        embeddings,
                        tasks,
                    )
                )

    pooled = {}
    for m in ms:
        for _, _, beam in ARMS:
            values = [
                cell
                for cell in cells
                if cell["M"] == m and cell["factor_beam"] == beam
            ]
            if not values:
                continue

            task_total = sum(value["tasks"] for value in values)
            ex_targets = sum(
                value["exhaustive_target_successes"] for value in values
            )
            sel_targets = sum(
                value["selective_target_successes"] for value in values
            )
            proposal_targets = sum(
                value["proposal_target_successes"] for value in values
            )
            ex_successes = sum(
                value["exhaustive_end_to_end_successes"] for value in values
            )
            sel_successes = sum(
                value["selective_end_to_end_successes"] for value in values
            )

            key = f"{m}:3:8:{beam}"
            pooled[key] = {
                "M": m,
                "factor_count": FACTOR_COUNT,
                "factor_size": FACTOR_SIZE,
                "factor_beam": beam,
                "tasks": task_total,
                "exhaustive_target_successes": ex_targets,
                "selective_target_successes": sel_targets,
                "exhaustive_target_recall": safe_ratio(
                    ex_targets, task_total
                ),
                "selective_target_recall": safe_ratio(
                    sel_targets, task_total
                ),
                "target_recall_retention_ratio": safe_ratio(
                    sel_targets, ex_targets
                ),
                "proposal_target_retention": safe_ratio(
                    proposal_targets, task_total
                ),
                "exhaustive_end_to_end_success": safe_ratio(
                    ex_successes, task_total
                ),
                "selective_end_to_end_success": safe_ratio(
                    sel_successes, task_total
                ),
                "state_candidates_scored_mean": statistics.fmean(
                    value["state_candidates_scored_mean"] for value in values
                ),
                "states_scored_over_M": statistics.fmean(
                    value["states_scored_over_M"] for value in values
                ),
                "shortlist_size_mean": statistics.fmean(
                    value["shortlist_size_mean"] for value in values
                ),
                "pair_generation_ops_mean": statistics.fmean(
                    value["pair_generation_ops_mean"] for value in values
                ),
                "total_macs_mean": statistics.fmean(
                    value["total_macs_mean"] for value in values
                ),
                "nonempty_cells": statistics.fmean(
                    value["nonempty_cells"] for value in values
                ),
                "max_cell_size": statistics.fmean(
                    value["max_cell_size"] for value in values
                ),
                "build_total_macs": statistics.fmean(
                    value["build_total_macs"] for value in values
                ),
            }

    eligible_arms = []
    for _, _, beam in ARMS:
        ratios = [
            pooled[f"{m}:3:8:{beam}"]["target_recall_retention_ratio"]
            for m in ms
            if f"{m}:3:8:{beam}" in pooled
        ]
        if ratios and all(ratio >= CAPABILITY_FLOOR for ratio in ratios):
            eligible_arms.append(
                {
                    "factor_count": FACTOR_COUNT,
                    "factor_size": FACTOR_SIZE,
                    "factor_beam": beam,
                    "min_target_recall_retention_ratio": min(ratios),
                    "max_states_scored_over_M": max(
                        pooled[f"{m}:3:8:{beam}"]["states_scored_over_M"]
                        for m in ms
                        if f"{m}:3:8:{beam}" in pooled
                    ),
                }
            )

    global_eligible = []
    for _, _, beam in ARMS:
        rows = [
            row for row in pooled.values()
            if row["factor_beam"] == beam
        ]
        if len(rows) == len(ms) and all(
            row["target_recall_retention_ratio"] >= CAPABILITY_FLOOR
            for row in rows
        ):
            global_eligible.append({
                "factor_count": FACTOR_COUNT,
                "factor_size": FACTOR_SIZE,
                "factor_beam": beam,
                "min_target_recall_retention_ratio": min(
                    row["target_recall_retention_ratio"] for row in rows
                ),
                "mean_states_scored_over_M": statistics.fmean(
                    row["states_scored_over_M"] for row in rows
                ),
                "max_states_scored_over_M": max(
                    row["states_scored_over_M"] for row in rows
                ),
            })

    frontier_improving = [
        row for row in global_eligible
        if row["mean_states_scored_over_M"] < 0.1542864583
    ]

    return {
        "protocol": {
            "name": "TACOSM-C5-MULTIFACTOR-GRANULARITY-BEAM-FRONTIER-001",
            "smoke": smoke,
            "seeds": list(seeds),
            "M_levels": list(ms),
            "H_fixed": H_FIXED,
            "K": K,
            "factor_count": FACTOR_COUNT,
            "factor_size": FACTOR_SIZE,
            "factor_beams": list(FACTOR_BEAMS),
            "eval_steps": steps,
            "capability_floor": CAPABILITY_FLOOR,
            "arms": [
                "three_factor_16_beam4",
                "three_factor_16_beam5",
                "three_factor_16_beam6",
            ],
        },
        "pooled": pooled,
        "eligible_arms_at_capability_floor": eligible_arms,
        "global_eligible_arms": global_eligible,
        "frontier_improving_arms": frontier_improving,
        "frontier_improvement_threshold_mean_states_scored_over_M": 0.1542864583,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-contract", action="store_true")
    args = parser.parse_args()

    if not args.no_contract:
        contract = load_contract(
            "TACOSM-C5-MULTIFACTOR-GRANULARITY-BEAM-FRONTIER-001"
        )
        contract.require_levels((H_FIXED,))
        contract.require_m_levels(M_LEVELS)
        contract.require_seeds(SEEDS)
        contract.require_k_levels([K])
        contract.require_arms(
            [
                "three_factor_16_beam4",
                "three_factor_16_beam5",
                "three_factor_16_beam6",
            ]
        )
        if not args.smoke:
            contract.require_eval_steps(EVAL_STEPS)

    result = run(args.smoke)
    output = Path(
        "artifacts/TACOSM-C5-MULTIFACTOR-GRANULARITY-BEAM-FRONTIER-001.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
