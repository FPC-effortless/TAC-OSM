#!/usr/bin/env python3
"""Run preregistered TACOSM-C5-COSINE-OBJECTIVE-001."""

from __future__ import annotations

import json
import math
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
LATENT_DIM = 16
EPOCHS = 512
EVAL_STEPS = 100
NEGATIVE_COUNT = 8
POSITIVE_VIEWS = 1
TRAIN_CODES = tuple(CODEBOOK[16:64])
EVAL_CODES = tuple(CODEBOOK[:16])
QUERY_PROJECTION_MACS = 10 * LATENT_DIM
STATE_SCORE_MACS = M * LATENT_DIM
QUERY_MACS = QUERY_PROJECTION_MACS + STATE_SCORE_MACS
STATE_BUILD_MACS = M * QUERY_PROJECTION_MACS


def _model(seed: int) -> LearnedSemanticStateIndex:
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10,
            latent_dim=LATENT_DIM,
            learning_rate=0.02,
            margin=0.25,
            epochs=EPOCHS,
            bucket_bits=8,
            probe_radius=1,
            shortlist_k=1,
            seed=seed,
        )
    )


def _normalize(values):
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-8:
        raise AssertionError("zero embedding encountered")
    return [v / norm for v in values], norm


def _pool(state):
    out = []
    for address in state.addresses():
        read = state.read(
            Query(
                text="	" + address,
                step=state.current_step,
                provenance="c5_cosine_objective_internal_read",
            )
        )
        if not read.keys:
            continue
        out.append((address, tuple(int(x) for x in read.values[0])))
    if len(out) != M:
        raise AssertionError(f"expected {M} readable states, got {len(out)}")
    return tuple(out)


def _cosine_select(model, query, pool, cached_embeddings, target_address):
    query_embedding, _ = _normalize(model.encode_query(query))
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


def run_arm(seed: int, arm: str, state, pool):
    model = _model(seed)
    if arm == "raw_objective":
        updates = model.train_with_negative_coverage(
            TRAIN_CODES,
            negative_count=NEGATIVE_COUNT,
            aggregation="mean",
            positive_views=POSITIVE_VIEWS,
        )
    elif arm == "cosine_objective":
        updates = model.train_with_cosine_negative_coverage(
            TRAIN_CODES,
            negative_count=NEGATIVE_COUNT,
            positive_views=POSITIVE_VIEWS,
        )
    else:
        raise ValueError(arm)

    expected_updates = EPOCHS * len(TRAIN_CODES)
    if updates != expected_updates:
        raise AssertionError("optimizer update count changed")

    cached_embeddings = tuple(
        tuple(_normalize(model.encode_state(value))[0]) for _, value in pool
    )
    recall = []
    ranks = []

    for step in range(EVAL_STEPS):
        task = build_task(seed, H_ANCHOR, step)
        selected, rank = _cosine_select(
            model,
            task.query,
            pool,
            cached_embeddings,
            task.target_address,
        )
        recall.append(int(selected == task.target_address))
        ranks.append(rank)

    return {
        "arm": arm,
        "latent_dim": LATENT_DIM,
        "negative_count": NEGATIVE_COUNT,
        "positive_views": POSITIVE_VIEWS,
        "epochs": EPOCHS,
        "training_updates": updates,
        "negative_evaluations_total": updates * NEGATIVE_COUNT * POSITIVE_VIEWS,
        "target_top1_recall": statistics.fmean(recall),
        "target_rank_mean": statistics.fmean(ranks),
        "query_projection_macs": QUERY_PROJECTION_MACS,
        "state_score_macs": STATE_SCORE_MACS,
        "query_macs": QUERY_MACS,
        "state_build_macs": STATE_BUILD_MACS,
        "normalization_ops_inference": 2 * (M + 1),
    }


def run_seed(seed: int) -> dict:
    first = build_task(seed, H_ANCHOR, 0)
    state = prepare_state(first)
    pool = _pool(state)
    return {
        "seed": seed,
        "M": M,
        "queries": EVAL_STEPS,
        "training_codes": len(TRAIN_CODES),
        "evaluation_codes": len(EVAL_CODES),
        "arms": {
            "raw": run_arm(seed, "raw_objective", state, pool),
            "cosine": run_arm(seed, "cosine_objective", state, pool),
        },
    }


def main() -> None:
    contract = load_contract("TACOSM-C5-COSINE-OBJECTIVE-001")
    contract.require_levels([H_ANCHOR])
    contract.require_seeds(SEEDS)
    contract.require_steps(EPOCHS)
    contract.require_eval_steps(EVAL_STEPS)
    contract.require_k_levels([])
    contract.require_arms(["raw_objective", "cosine_objective"])

    if set(TRAIN_CODES) & set(EVAL_CODES):
        raise AssertionError("training/evaluation semantic codes overlap")

    seeds = [run_seed(seed) for seed in SEEDS]
    aggregate = {}
    for key, arm_name in (("raw", "raw_objective"), ("cosine", "cosine_objective")):
        aggregate[f"{key}_recall_mean"] = statistics.fmean(
            row["arms"][key]["target_top1_recall"] for row in seeds
        )
        aggregate[f"{key}_rank_mean"] = statistics.fmean(
            row["arms"][key]["target_rank_mean"] for row in seeds
        )
        aggregate[f"{key}_negative_evaluations_total"] = seeds[0]["arms"][key][
            "negative_evaluations_total"
        ]

    delta = aggregate["cosine_recall_mean"] - aggregate["raw_recall_mean"]
    if delta >= 0.10 and aggregate["cosine_recall_mean"] >= 0.50:
        decision = "material_cosine_effect"
    elif delta <= -0.05:
        decision = "cosine_harmful"
    elif abs(delta) <= 0.05:
        decision = "no_material_change"
    else:
        decision = "partial_effect"

    aggregate["delta_cosine_minus_raw"] = delta
    aggregate["decision"] = decision

    result = {
        "protocol": {
            "name": "TACOSM-C5-COSINE-OBJECTIVE-001",
            "seeds": list(SEEDS),
            "H_anchor": H_ANCHOR,
            "M": M,
            "latent_dim": LATENT_DIM,
            "epochs": EPOCHS,
            "negative_count": NEGATIVE_COUNT,
            "positive_views": POSITIVE_VIEWS,
            "queries_per_seed_per_arm": EVAL_STEPS,
            "training_codes": len(TRAIN_CODES),
            "evaluation_codes": len(EVAL_CODES),
            "evaluation_metric": "cosine similarity for both arms",
        },
        "aggregate": aggregate,
        "seeds": seeds,
    }
    out = Path("artifacts/TACOSM-C5-COSINE-OBJECTIVE-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
