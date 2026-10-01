#!/usr/bin/env python3
"""Run TACOSM-C5-FACTOR16-BEAM-7-8-FRONTIER-001."""

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
FACTOR_BEAMS = (7, 8)
CAPABILITY_FLOOR = 0.90
FRONTIER_REFERENCE = 0.1542864583
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
            raise AssertionError("state-distinct audit expected four relevant programs")
        output, work, calls = executor.execute_population(relevant, reference)
        if work != 4 * 12 or calls != 4:
            raise AssertionError("state-distinct executor accounting changed")
        outputs.append(output)
    if outputs[0] == outputs[1]:
        raise AssertionError("state-distinct executor collapsed two target codes")


def evaluate_arm(seed, m, factor_beam, teacher, embeddings, tasks) -> dict:
    first = tasks[0]
    train_set = set(TRAIN_CODES)
    train_items = [
        item for item, update in zip(embeddings, first.state_updates)
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
            i for i, candidate in enumerate(task.candidates)
            if tuple(candidate.descriptor) == exhaustive_value
        )
        exhaustive_output, _, _ = executor.execute_population(
            tuple(task.candidates[i] for i in exhaustive_relevant),
            exhaustive_value,
        )
        exhaustive_end_to_end_successes += int(
            exhaustive_output == task.expected_output
        )

        hit = index.lookup(query, beam=factor_beam, max_shortlist=K)
        selected = hit.selected_address
        admitted = int(task.target_address in hit.candidate_addresses)
        proposal_target_successes += admitted
        selective_target_successes += int(selected == task.target_address)

        selected_value = next(
            (
                tuple(update.value)
                for update in task.state_updates
                if update.key == selected
            ),
            None,
        )
        if selected_value is not None:
            relevant = tuple(
                i for i, candidate in enumerate(task.candidates)
                if tuple(candidate.descriptor) == selected_value
            )
            output, _, _ = executor.execute_population(
                tuple(task.candidates[i] for i in relevant),
                selected_value,
            )
            selective_end_to_end_successes += int(
                output == task.expected_output
            )

        scored += hit.state_candidates_scored
        shortlist += len(hit.candidate_addresses)
        pair_ops += hit.pair_generation_ops
        macs += LATENT_DIM * 10 + hit.factor_score_macs + hit.state_rerank_macs

    task_count = len(tasks)
    ex = exhaustive_target_successes / task_count
    sel = selective_target_successes / task_count
    prop = proposal_target_successes / task_count
    return {
        "seed": seed,
        "M": m,
        "factor_count": FACTOR_COUNT,
        "factor_size": FACTOR_SIZE,
        "factor_beam": factor_beam,
        "tasks": task_count,
        "exhaustive_target_successes": exhaustive_target_successes,
        "selective_target_successes": selective_target_successes,
        "proposal_target_successes": proposal_target_successes,
        "exhaustive_target_recall": ex,
        "selective_target_recall": sel,
        "proposal_target_retention": prop,
        "target_recall_retention_ratio": sel / ex if ex else float("nan"),
        "conditional_selection_given_admission": (
            sel / prop if prop else float("nan")
        ),
        "exhaustive_end_to_end_successes": exhaustive_end_to_end_successes,
        "selective_end_to_end_successes": selective_end_to_end_successes,
        "exhaustive_end_to_end_success": exhaustive_end_to_end_successes / task_count,
        "selective_end_to_end_success": selective_end_to_end_successes / task_count,
        "state_candidates_scored_mean": scored / task_count,
        "states_scored_over_M": scored / task_count / m,
        "shortlist_size_mean": shortlist / task_count,
        "pair_generation_ops_mean": pair_ops / task_count,
        "total_macs_mean": macs / task_count,
        "nonempty_cells": build.nonempty_cells,
        "max_cell_size": build.max_cell_size,
        "build_total_macs": build.total_build_macs,
    }


def run(smoke: bool) -> dict:
    assert_executor_distinctness()
    seeds = (10,) if smoke else SEEDS
    ms = (128,) if smoke else M_LEVELS
    steps = 5 if smoke else EVAL_STEPS
    arms = ((7, FACTOR_SIZE, 7),) if smoke else ARMS

    cells = []
    for seed in seeds:
        teacher = make_teacher(seed)
        for m in ms:
            first = build_scaled_task(seed, m, 0)
            embeddings = normalized_embeddings(teacher, first)
            tasks = tuple(build_scaled_task(seed, m, step) for step in range(steps))
            for _, _, beam in arms:
                cells.append(
                    evaluate_arm(seed, m, beam, teacher, embeddings, tasks)
                )

    pooled = {}
    for m in ms:
        for _, _, beam in arms:
            values = [
                row for row in cells
                if row["M"] == m and row["factor_beam"] == beam
            ]
            if not values:
                continue
            task_total = sum(row["tasks"] for row in values)
            ex = sum(row["exhaustive_target_successes"] for row in values)
            sel = sum(row["selective_target_successes"] for row in values)
            proposal = sum(row["proposal_target_successes"] for row in values)
            ex_end = sum(row["exhaustive_end_to_end_successes"] for row in values)
            sel_end = sum(row["selective_end_to_end_successes"] for row in values)
            retention = sel / ex if ex else float("nan")
            proposal_retention = proposal / task_total
            pooled[f"{m}:3:16:{beam}"] = {
                "M": m,
                "factor_count": FACTOR_COUNT,
                "factor_size": FACTOR_SIZE,
                "factor_beam": beam,
                "tasks": task_total,
                "exhaustive_target_successes": ex,
                "selective_target_successes": sel,
                "proposal_target_successes": proposal,
                "exhaustive_target_recall": ex / task_total,
                "selective_target_recall": sel / task_total,
                "proposal_target_retention": proposal_retention,
                "target_recall_retention_ratio": retention,
                "conditional_selection_given_admission": (
                    (sel / task_total) / proposal_retention
                    if proposal_retention else float("nan")
                ),
                "exhaustive_end_to_end_success": ex_end / task_total,
                "selective_end_to_end_success": sel_end / task_total,
                "state_candidates_scored_mean": statistics.fmean(
                    row["state_candidates_scored_mean"] for row in values
                ),
                "states_scored_over_M": statistics.fmean(
                    row["states_scored_over_M"] for row in values
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
                "meets_capability_floor": retention >= CAPABILITY_FLOOR,
            }

    global_eligible = []
    if not smoke:
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
        if row["mean_states_scored_over_M"] < FRONTIER_REFERENCE
    ]

    return {
        "protocol": {
            "name": "TACOSM-C5-FACTOR16-BEAM-7-8-FRONTIER-001",
            "smoke": smoke,
            "seeds": list(seeds),
            "M_levels": list(ms),
            "H_fixed": H_FIXED,
            "K": K,
            "eval_steps": steps,
            "factor_count": FACTOR_COUNT,
            "factor_size": FACTOR_SIZE,
            "factor_beams": [7, 8],
            "capability_floor": CAPABILITY_FLOOR,
            "frontier_reference": FRONTIER_REFERENCE,
            "arms": [
                "three_factor_16_beam7",
                "three_factor_16_beam8",
            ],
        },
        "pooled": pooled,
        "global_eligible_arms": global_eligible,
        "frontier_improving_arms": frontier_improving,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-contract", action="store_true")
    args = parser.parse_args()

    if not args.no_contract and not args.smoke:
        contract = load_contract(
            "TACOSM-C5-FACTOR16-BEAM-7-8-FRONTIER-001"
        )
        contract.require_levels((H_FIXED,))
        contract.require_m_levels(M_LEVELS)
        contract.require_seeds(SEEDS)
        contract.require_k_levels([K])
        contract.require_arms([
            "three_factor_16_beam7",
            "three_factor_16_beam8",
        ])
        contract.require_eval_steps(EVAL_STEPS)

    result = run(args.smoke)
    output = Path(
        "artifacts/TACOSM-C5-FACTOR16-BEAM-7-8-FRONTIER-001.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
