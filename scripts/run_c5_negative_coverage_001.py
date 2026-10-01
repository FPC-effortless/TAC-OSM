#!/usr/bin/env python3
"""Run preregistered TACOSM-C5-NEGATIVE-COVERAGE-001."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import Query
from tac_osm.c5_end_to_end import build_task, prepare_state
from tac_osm.contract import load_contract
from tac_osm.learned_state_index import LearnedSemanticStateIndex, LearnedStateIndexConfig
from tac_osm.noisy_state_tasks import CODEBOOK

SEEDS = (0, 1, 2, 3, 4)
H_ANCHOR = 64
M = 64
BUDGET = 512
EVAL_STEPS = 100
TRAIN_CODES = tuple(CODEBOOK[16:64])
EVAL_CODES = tuple(CODEBOOK[:16])
QUERY_MACS = 592
STATE_BUILD_MACS = 5120
ARMS = (
    ("single_negative", 1, "mean"),
    ("mean_8_negatives", 8, "mean"),
    ("hardest_8_negatives", 8, "hardest"),
)


def _model(seed: int) -> LearnedSemanticStateIndex:
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10,
            latent_dim=8,
            learning_rate=0.02,
            margin=0.25,
            epochs=BUDGET,
            bucket_bits=8,
            probe_radius=1,
            shortlist_k=1,
            seed=seed,
        )
    )


def _pool(state):
    out = []
    for address in state.addresses():
        read = state.read(
            Query(
                text="\t" + address,
                step=state.current_step,
                provenance="c5_negative_coverage_internal_read",
            )
        )
        if not read.keys:
            continue
        out.append((address, tuple(int(x) for x in read.values[0])))
    if len(out) != M:
        raise AssertionError(f"expected {M} readable states, got {len(out)}")
    return tuple(out)


def _select(model, query, pool, cached_embeddings, target_address):
    query_embedding = model.encode_query(query)
    scores = [
        sum(a * b for a, b in zip(query_embedding, emb))
        for emb in cached_embeddings
    ]
    selected = max(range(len(scores)), key=lambda i: (scores[i], -i))
    target_index = next(i for i, (address, _) in enumerate(pool) if address == target_address)
    target_score = scores[target_index]
    rank = 1 + sum(score > target_score for score in scores)
    return pool[selected][0], rank


def run_arm(seed: int, arm_name: str, negative_count: int, aggregation: str, state, pool):
    model = _model(seed)
    updates = model.train_with_negative_coverage(
        TRAIN_CODES,
        negative_count=negative_count,
        aggregation=aggregation,
    )
    if updates != BUDGET * len(TRAIN_CODES):
        raise AssertionError("optimizer update count changed")

    cached_embeddings = tuple(
        tuple(model.encode_state(value)) for _, value in pool
    )
    recall = []
    ranks = []

    for step in range(EVAL_STEPS):
        task = build_task(seed, H_ANCHOR, step)
        selected, rank = _select(
            model, task.query, pool, cached_embeddings, task.target_address
        )
        recall.append(int(selected == task.target_address))
        ranks.append(rank)

    return {
        "arm": arm_name,
        "epochs": BUDGET,
        "negative_count": negative_count,
        "aggregation": aggregation,
        "training_updates": updates,
        "negative_evaluations_total": updates * negative_count,
        "negative_evaluations_per_update": negative_count,
        "target_top1_recall": statistics.fmean(recall),
        "target_rank_mean": statistics.fmean(ranks),
        "query_macs": QUERY_MACS,
        "state_build_macs": STATE_BUILD_MACS,
    }


def run_seed(seed: int) -> dict:
    first = build_task(seed, H_ANCHOR, 0)
    state = prepare_state(first)
    pool = _pool(state)

    arms = {}
    for arm_name, count, aggregation in ARMS:
        arms[arm_name] = run_arm(
            seed, arm_name, count, aggregation, state, pool
        )

    return {
        "seed": seed,
        "M": M,
        "queries": EVAL_STEPS,
        "training_codes": len(TRAIN_CODES),
        "evaluation_codes": len(EVAL_CODES),
        "arms": arms,
    }


def main() -> None:
    contract = load_contract("TACOSM-C5-NEGATIVE-COVERAGE-001")
    contract.require_levels([H_ANCHOR])
    contract.require_seeds(SEEDS)
    contract.require_steps(BUDGET)
    contract.require_eval_steps(EVAL_STEPS)
    contract.require_k_levels([])
    contract.require_arms(
        ["single_negative", "mean_8_negatives", "hardest_8_negatives"]
    )
    if set(TRAIN_CODES) & set(EVAL_CODES):
        raise AssertionError("training/evaluation semantic codes overlap")

    seeds = [run_seed(seed) for seed in SEEDS]
    aggregate = {}
    for arm_name, _, _ in ARMS:
        aggregate[f"{arm_name}_recall_mean"] = statistics.fmean(
            row["arms"][arm_name]["target_top1_recall"] for row in seeds
        )
        aggregate[f"{arm_name}_rank_mean"] = statistics.fmean(
            row["arms"][arm_name]["target_rank_mean"] for row in seeds
        )

    baseline = aggregate["single_negative_recall_mean"]
    for arm_name, _, _ in ARMS[1:]:
        delta = aggregate[f"{arm_name}_recall_mean"] - baseline
        aggregate[f"{arm_name}_delta_vs_single"] = delta

    aggregate["strong_threshold_met"] = any(
        aggregate[f"{arm_name}_delta_vs_single"] >= 0.20
        and aggregate[f"{arm_name}_recall_mean"] >= 0.50
        for arm_name, _, _ in ARMS[1:]
    )
    aggregate["partial_threshold_met"] = any(
        aggregate[f"{arm_name}_delta_vs_single"] >= 0.10
        for arm_name, _, _ in ARMS[1:]
    )

    result = {
        "protocol": {
            "name": "TACOSM-C5-NEGATIVE-COVERAGE-001",
            "seeds": list(SEEDS),
            "H_anchor": H_ANCHOR,
            "M": M,
            "epochs": BUDGET,
            "queries_per_seed": EVAL_STEPS,
            "training_codes": len(TRAIN_CODES),
            "evaluation_codes": len(EVAL_CODES),
            "query_macs": QUERY_MACS,
            "state_build_macs": STATE_BUILD_MACS,
        },
        "aggregate": aggregate,
        "seeds": seeds,
    }
    out = Path("artifacts/TACOSM-C5-NEGATIVE-COVERAGE-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
