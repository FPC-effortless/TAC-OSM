#!/usr/bin/env python3
"""Run TACOSM-C5-MULTIFACTOR-001."""

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
from tac_osm.learned_state_index import LearnedSemanticStateIndex
from tac_osm.multifactor_product_key_index import (
    MultiFactorProductKeyConfig,
    MultiFactorProductKeyStateIndex,
)
from tac_osm.product_key_index import ProductKeyConfig, ProductKeyStateIndex
from tac_osm.noisy_state_tasks import CODEBOOK
from scripts.run_c5_frontier_robustness_002 import (
    H_FIXED,
    K,
    LATENT_DIM,
    M_LEVELS,
    NEGATIVE_COUNT,
    SEEDS,
    TEACHER_EPOCHS,
    build_scaled_task,
    make_teacher,
    normalized_embeddings,
    normalized_query,
    TRAIN_CODES,
)

EVAL_STEPS = 100
FACTOR_KMEANS_ITERATIONS = 8

ARMS = (
    ("two_factor_16_beam6", 2, 16, 6),
    ("three_factor_8_beam3", 3, 8, 3),
    ("three_factor_16_beam3", 3, 16, 3),
)



class StateDistinctEndToEndExecutor:
    """Fixed downstream task whose successful output uniquely identifies the state value."""

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
            raise AssertionError("state-distinct audit executor accounting changed")
        outputs.append(output)
    if outputs[0] == outputs[1]:
        raise AssertionError("state-distinct executor collapsed two target codes")


def evaluate_arm(
    seed: int,
    m: int,
    arm_name: str,
    factor_count: int,
    factor_size: int,
    factor_beam: int,
    teacher: LearnedSemanticStateIndex,
    embeddings,
    tasks,
) -> dict:
    first = tasks[0]
    train_set = set(TRAIN_CODES)
    train_items = [
        item
        for item, update in zip(embeddings, first.state_updates)
        if tuple(update.value) in train_set
    ]
    if len(train_items) != len(TRAIN_CODES):
        raise AssertionError("training-state population mismatch")

    if factor_count == 2:
        index = ProductKeyStateIndex(
            ProductKeyConfig(
                factor_size=factor_size,
                factor_beam=factor_beam,
                iterations=FACTOR_KMEANS_ITERATIONS,
                max_shortlist=K,
            )
        )
    elif factor_count == 3:
        index = MultiFactorProductKeyStateIndex(
            MultiFactorProductKeyConfig(
                factor_count=factor_count,
                factor_size=factor_size,
                factor_beam=factor_beam,
                iterations=FACTOR_KMEANS_ITERATIONS,
                max_shortlist=K,
            )
        )
    else:
        raise AssertionError(f"unsupported factor count: {factor_count}")

    build = index.build(embeddings, codebook_items=train_items)
    executor = StateDistinctEndToEndExecutor()

    exhaustive_target = []
    selective_target = []
    proposal = []
    exhaustive_success = []
    selective_success = []
    scored = []
    shortlist_sizes = []
    macs = []

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
        exhaustive_target.append(int(ex_address == task.target_address))

        ex_relevant = tuple(
            i
            for i, candidate in enumerate(task.candidates)
            if tuple(candidate.descriptor) == ex_value
        )
        ex_output, _, _ = executor.execute_population(
            tuple(task.candidates[i] for i in ex_relevant),
            ex_value,
        )
        exhaustive_success.append(int(ex_output == task.expected_output))

        hit = index.lookup(query, beam=factor_beam, max_shortlist=K)
        selected = hit.selected_address
        selective_target.append(int(selected == task.target_address))
        proposal.append(int(task.target_address in hit.candidate_addresses))

        selected_value = next(
            (
                tuple(update.value)
                for update in task.state_updates
                if update.key == selected
            ),
            None,
        )
        if selected_value is None:
            selective_success.append(0)
        else:
            relevant = tuple(
                i
                for i, candidate in enumerate(task.candidates)
                if tuple(candidate.descriptor) == selected_value
            )
            output, _, _ = executor.execute_population(
                tuple(task.candidates[i] for i in relevant),
                selected_value,
            )
            selective_success.append(int(output == task.expected_output))

        scored.append(hit.state_candidates_scored)
        shortlist_sizes.append(len(hit.candidate_addresses))
        macs.append(
            K * 0
            + LATENT_DIM * 10
            + hit.factor_score_macs
            + hit.state_rerank_macs
        )

    return {
        "seed": seed,
        "M": m,
        "arm": arm_name,
        "factor_count": factor_count,
        "factor_size": factor_size,
        "factor_beam": factor_beam,
        "exhaustive_target_recall": statistics.fmean(exhaustive_target),
        "selective_target_recall": statistics.fmean(selective_target),
        "proposal_target_retention": statistics.fmean(proposal),
        "exhaustive_end_to_end_success": statistics.fmean(exhaustive_success),
        "selective_end_to_end_success": statistics.fmean(selective_success),
        "target_recall_retention_ratio": (
            statistics.fmean(selective_target) / statistics.fmean(exhaustive_target)
            if statistics.fmean(exhaustive_target) > 0.0 else float("nan")
        ),
        "state_candidates_scored_mean": statistics.fmean(scored),
        "states_scored_over_M": statistics.fmean(scored) / m,
        "shortlist_size_mean": statistics.fmean(shortlist_sizes),
        "total_macs_mean": statistics.fmean(macs),
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
            tasks = tuple(build_scaled_task(seed, m, step) for step in range(steps))
            for arm in ARMS:
                cells.append(
                    evaluate_arm(
                        seed,
                        m,
                        *arm,
                        teacher,
                        embeddings,
                        tasks,
                    )
                )

    pooled = {}
    for m in ms:
        for arm_name, factor_count, factor_size, factor_beam in ARMS:
            values = [
                cell for cell in cells
                if cell["M"] == m
                and cell["arm"] == arm_name
            ]
            if not values:
                continue
            pooled[f"{m}:{arm_name}"] = {
                key: statistics.fmean(value[key] for value in values)
                for key in (
                    "exhaustive_target_recall",
                    "selective_target_recall",
                    "proposal_target_retention",
                    "exhaustive_end_to_end_success",
                    "selective_end_to_end_success",
                    "target_recall_retention_ratio",
                    "state_candidates_scored_mean",
                    "states_scored_over_M",
                    "shortlist_size_mean",
                    "total_macs_mean",
                    "nonempty_cells",
                    "max_cell_size",
                    "build_total_macs",
                )
            }
            pooled[f"{m}:{arm_name}"].update(
                {
                    "M": m,
                    "arm": arm_name,
                    "factor_count": factor_count,
                    "factor_size": factor_size,
                    "factor_beam": factor_beam,
                }
            )

    return {
        "protocol": {
            "name": "TACOSM-C5-MULTIFACTOR-001",
            "smoke": smoke,
            "seeds": list(seeds),
            "M_levels": list(ms),
            "H_fixed": H_FIXED,
            "K": K,
            "eval_steps": steps,
            "factor_kmeans_iterations": FACTOR_KMEANS_ITERATIONS,
            "teacher_epochs": TEACHER_EPOCHS,
            "negative_count": NEGATIVE_COUNT,
            "arms": [list(arm) for arm in ARMS],
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
        contract = load_contract("TACOSM-C5-MULTIFACTOR-001")
        contract.require_levels((H_FIXED,))
        contract.require_m_levels(M_LEVELS)
        contract.require_seeds(SEEDS)
        contract.require_k_levels([K])
        contract.require_arms([arm[0] for arm in ARMS])
        if not args.smoke:
            contract.require_eval_steps(EVAL_STEPS)

    result = run(args.smoke)
    output = Path("artifacts/TACOSM-C5-MULTIFACTOR-001.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
