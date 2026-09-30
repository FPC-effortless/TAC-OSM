#!/usr/bin/env python3
"""Run preregistered TACOSM-C5-SIMILARITY-EVAL-001."""

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
TRAIN_CODES = tuple(CODEBOOK[16:64])
EVAL_CODES = tuple(CODEBOOK[:16])
QUERY_PROJECTION_MACS = 10 * LATENT_DIM
STATE_SCORE_MACS = M * LATENT_DIM


def _model(seed: int):
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


def _pool(state):
    pool = []
    for address in state.addresses():
        read = state.read(
            Query(
                text="	" + address,
                step=state.current_step,
                provenance="c5_similarity_eval_internal_read",
            )
        )
        if not read.keys:
            continue
        pool.append((address, tuple(int(x) for x in read.values[0])))
    if len(pool) != M:
        raise AssertionError(f"expected {M} readable states, got {len(pool)}")
    return tuple(pool)


def _normalize(values):
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-8:
        raise AssertionError("zero embedding encountered")
    return [value / norm for value in values]


def _select_raw(model, query, pool, cached):
    q = model.encode_query(query)
    scores = [sum(a * b for a, b in zip(q, emb)) for emb in cached]
    selected = max(range(M), key=lambda i: (scores[i], -i))
    return selected, scores


def _select_cosine(model, query, pool, cached):
    q = _normalize(model.encode_query(query))
    scores = [sum(a * b for a, b in zip(q, emb)) for emb in cached]
    selected = max(range(M), key=lambda i: (scores[i], -i))
    return selected, scores


def _run_arm(model, pool, queries, mode):
    cached_raw = tuple(
        tuple(model.encode_state(value)) for _, value in pool
    )
    cached_cos = tuple(
        tuple(_normalize(model.encode_state(value))) for _, value in pool
    )
    recall = []
    ranks = []
    for task in queries:
        if mode == "raw":
            selected, scores = _select_raw(model, task.query, pool, cached_raw)
        else:
            selected, scores = _select_cosine(model, task.query, pool, cached_cos)
        target_index = next(
            i for i, (address, _) in enumerate(pool)
            if address == task.target_address
        )
        target_score = scores[target_index]
        rank = 1 + sum(score > target_score for score in scores)
        recall.append(int(pool[selected][0] == task.target_address))
        ranks.append(rank)
    return {
        "target_top1_recall": statistics.fmean(recall),
        "target_rank_mean": statistics.fmean(ranks),
    }


def run_seed(seed: int) -> dict:
    first = build_task(seed, H_ANCHOR, 0)
    state = prepare_state(first)
    pool = _pool(state)

    model = _model(seed)
    updates = model.train_with_negative_coverage(
        TRAIN_CODES,
        negative_count=NEGATIVE_COUNT,
        aggregation="mean",
        positive_views=1,
    )
    expected = EPOCHS * len(TRAIN_CODES)
    if updates != expected:
        raise AssertionError("optimizer update count changed")

    queries = [build_task(seed, H_ANCHOR, step) for step in range(EVAL_STEPS)]
    raw = _run_arm(model, pool, queries, "raw")
    cosine = _run_arm(model, pool, queries, "cosine")

    return {
        "seed": seed,
        "M": M,
        "queries": EVAL_STEPS,
        "training_updates": updates,
        "raw": {
            **raw,
            "query_projection_macs": QUERY_PROJECTION_MACS,
            "state_score_macs": STATE_SCORE_MACS,
            "query_macs": QUERY_PROJECTION_MACS + STATE_SCORE_MACS,
            "normalization_ops_inference": 0,
        },
        "cosine": {
            **cosine,
            "query_projection_macs": QUERY_PROJECTION_MACS,
            "state_score_macs": STATE_SCORE_MACS,
            "query_macs": QUERY_PROJECTION_MACS + STATE_SCORE_MACS,
            "normalization_ops_inference": 2 * (M + 1),
        },
    }


def main() -> None:
    contract = load_contract("TACOSM-C5-SIMILARITY-EVAL-001")
    contract.require_levels([H_ANCHOR])
    contract.require_seeds(SEEDS)
    contract.require_steps(EPOCHS)
    contract.require_eval_steps(EVAL_STEPS)
    contract.require_k_levels([])
    contract.require_arms(["raw_dot_eval", "cosine_eval"])

    if set(TRAIN_CODES) & set(EVAL_CODES):
        raise AssertionError("training/evaluation semantic codes overlap")

    seeds = [run_seed(seed) for seed in SEEDS]
    raw_mean = statistics.fmean(row["raw"]["target_top1_recall"] for row in seeds)
    cosine_mean = statistics.fmean(
        row["cosine"]["target_top1_recall"] for row in seeds
    )
    delta = cosine_mean - raw_mean

    if delta >= 0.10 and cosine_mean >= 0.50:
        decision = "material_cosine_effect"
    elif delta <= -0.05:
        decision = "cosine_harmful"
    elif abs(delta) <= 0.05:
        decision = "no_material_change"
    else:
        decision = "partial_effect"

    result = {
        "protocol": {
            "name": "TACOSM-C5-SIMILARITY-EVAL-001",
            "seeds": list(SEEDS),
            "H_anchor": H_ANCHOR,
            "M": M,
            "latent_dim": LATENT_DIM,
            "epochs": EPOCHS,
            "negative_count": NEGATIVE_COUNT,
            "positive_views": 1,
            "queries_per_seed": EVAL_STEPS,
            "training_codes": len(TRAIN_CODES),
            "evaluation_codes": len(EVAL_CODES),
            "training_objective": "raw dot-product margin",
            "evaluation_metrics": ["raw dot product", "cosine similarity"],
        },
        "aggregate": {
            "raw_recall_mean": raw_mean,
            "cosine_recall_mean": cosine_mean,
            "delta_cosine_minus_raw": delta,
            "decision": decision,
        },
        "seeds": seeds,
    }
    out = Path("artifacts/TACOSM-C5-SIMILARITY-EVAL-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
