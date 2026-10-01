#!/usr/bin/env python3
"""Run preregistered TACOSM-C5-LEARNED-STATE-BUDGET-001."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import Query
from tac_osm.c5_end_to_end import build_task, prepare_state
from tac_osm.learned_state_index import (
    LearnedSemanticStateIndex,
    LearnedStateIndexConfig,
)
from tac_osm.noisy_state_tasks import CODEBOOK
from tac_osm.contract import load_contract

SEEDS = (0, 1, 2, 3, 4)
H_ANCHOR = 64
M = 64
BUDGETS = (32, 128, 512)
EVAL_STEPS = 100
TRAIN_CODES = tuple(CODEBOOK[16:64])
EVAL_CODES = tuple(CODEBOOK[:16])
INPUT_DIM = 10
LATENT_DIM = 8
QUERY_MACS = INPUT_DIM * LATENT_DIM
STATE_SCORE_MACS = M * LATENT_DIM
TOTAL_QUERY_MACS = QUERY_MACS + STATE_SCORE_MACS


def _model(seed: int, epochs: int) -> LearnedSemanticStateIndex:
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=INPUT_DIM,
            latent_dim=LATENT_DIM,
            learning_rate=0.02,
            margin=0.25,
            epochs=epochs,
            bucket_bits=8,
            probe_radius=1,
            shortlist_k=1,
            seed=seed,
        )
    )


def _pool(state) -> tuple[tuple[str, tuple[int, ...]], ...]:
    out: list[tuple[str, tuple[int, ...]]] = []
    for address in state.addresses():
        read = state.read(
            Query(
                text="\t" + address,
                step=state.current_step,
                provenance="c5_budget_internal_read",
            )
        )
        if not read.keys:
            continue
        out.append((address, tuple(int(x) for x in read.values[0])))
    if len(out) != M:
        raise AssertionError(f"expected {M} readable state items, got {len(out)}")
    return tuple(out)


def _continuous_select(
    model: LearnedSemanticStateIndex,
    query: Query,
    pool: tuple[tuple[str, tuple[int, ...]], ...],
    cached_embeddings: tuple[tuple[float, ...], ...],
    target_address: str,
) -> tuple[str, int]:
    query_embedding = model.encode_query(query)
    scores = [
        sum(a * b for a, b in zip(query_embedding, state_embedding))
        for state_embedding in cached_embeddings
    ]
    selected = max(range(len(scores)), key=lambda i: (scores[i], -i))
    target_index = next(
        i for i, (address, _) in enumerate(pool) if address == target_address
    )
    target_score = scores[target_index]
    rank = 1 + sum(score > target_score for score in scores)
    return pool[selected][0], rank


def run_seed(seed: int) -> dict:
    first = build_task(seed, H_ANCHOR, 0)
    state = prepare_state(first)
    pool = _pool(state)

    control = _model(seed + 10000, epochs=32)
    control_embeddings = tuple(
        tuple(control.encode_state(value)) for _, value in pool
    )

    budget_results: dict[str, dict] = {}
    for budget in BUDGETS:
        model = _model(seed, epochs=budget)
        training_pairs = model.train(TRAIN_CODES)
        cached_embeddings = tuple(
            tuple(model.encode_state(value)) for _, value in pool
        )
        recall: list[int] = []
        ranks: list[int] = []

        for step in range(EVAL_STEPS):
            task = build_task(seed, H_ANCHOR, step)
            selected, rank = _continuous_select(
                model,
                task.query,
                pool,
                cached_embeddings,
                task.target_address,
            )
            recall.append(int(selected == task.target_address))
            ranks.append(rank)

        budget_results[str(budget)] = {
            "epochs": budget,
            "training_pairs": training_pairs,
            "target_top1_recall": statistics.fmean(recall),
            "target_rank_mean": statistics.fmean(ranks),
            "query_macs": TOTAL_QUERY_MACS,
            "state_build_macs": M * model.state_embedding_macs,
        }

    control_recall: list[int] = []
    control_ranks: list[int] = []
    for step in range(EVAL_STEPS):
        task = build_task(seed, H_ANCHOR, step)
        selected, rank = _continuous_select(
            control,
            task.query,
            pool,
            control_embeddings,
            task.target_address,
        )
        control_recall.append(int(selected == task.target_address))
        control_ranks.append(rank)

    return {
        "seed": seed,
        "M": M,
        "training_codes": len(TRAIN_CODES),
        "evaluation_codes": len(EVAL_CODES),
        "queries": EVAL_STEPS,
        "no_learning_continuous_recall": statistics.fmean(control_recall),
        "no_learning_target_rank_mean": statistics.fmean(control_ranks),
        "budgets": budget_results,
    }


def main() -> None:
    contract = load_contract("TACOSM-C5-LEARNED-STATE-BUDGET-001")
    contract.require_levels([H_ANCHOR])
    contract.require_seeds(SEEDS)
    contract.require_steps(512)
    contract.require_eval_steps(EVAL_STEPS)
    contract.require_k_levels([])
    contract.require_arms(
        ["no_learning_continuous", "learned_32", "learned_128", "learned_512"]
    )

    if set(TRAIN_CODES) & set(EVAL_CODES):
        raise AssertionError("training/evaluation semantic codes overlap")

    cells = [run_seed(seed) for seed in SEEDS]

    aggregate: dict[str, float] = {}
    for budget in BUDGETS:
        aggregate[f"learned_{budget}_recall_mean"] = statistics.fmean(
            cell["budgets"][str(budget)]["target_top1_recall"]
            for cell in cells
        )
        aggregate[f"learned_{budget}_rank_mean"] = statistics.fmean(
            cell["budgets"][str(budget)]["target_rank_mean"]
            for cell in cells
        )
    aggregate["no_learning_recall_mean"] = statistics.fmean(
        cell["no_learning_continuous_recall"] for cell in cells
    )
    aggregate["learned_512_minus_32"] = (
        aggregate["learned_512_recall_mean"]
        - aggregate["learned_32_recall_mean"]
    )
    aggregate["learned_512_gain_vs_no_learning"] = (
        aggregate["learned_512_recall_mean"]
        - aggregate["no_learning_recall_mean"]
    )

    result = {
        "protocol": {
            "name": "TACOSM-C5-LEARNED-STATE-BUDGET-001",
            "seeds": list(SEEDS),
            "H_anchor": H_ANCHOR,
            "M": M,
            "budgets": list(BUDGETS),
            "queries_per_seed_per_budget": EVAL_STEPS,
            "training_codes": len(TRAIN_CODES),
            "evaluation_codes": len(EVAL_CODES),
            "encoder": "dual linear 10->8 semantic encoder",
            "continuous_scoring": "64 cached state embeddings, 8-dimensional dot product",
            "query_macs": TOTAL_QUERY_MACS,
            "state_build_macs": M * QUERY_MACS,
            "code_split": "train CODEBOOK[16:64], eval CODEBOOK[:16]",
        },
        "aggregate": aggregate,
        "seeds": cells,
    }
    out = Path("artifacts/TACOSM-C5-LEARNED-STATE-BUDGET-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
