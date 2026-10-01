#!/usr/bin/env python3
"""Run preregistered TACOSM-C5-LATENT-WIDTH-001."""

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
EPOCHS = 512
EVAL_STEPS = 100
NEGATIVE_COUNT = 8
POSITIVE_VIEWS = 1
LATENT_WIDTHS = (8, 16, 32)
TRAIN_CODES = tuple(CODEBOOK[16:64])
EVAL_CODES = tuple(CODEBOOK[:16])
INPUT_DIM = 10


def _model(seed: int, latent_dim: int) -> LearnedSemanticStateIndex:
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=INPUT_DIM,
            latent_dim=latent_dim,
            learning_rate=0.02,
            margin=0.25,
            epochs=EPOCHS,
            bucket_bits=min(8, latent_dim),
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
                text="	" + address,
                step=state.current_step,
                provenance="c5_latent_width_internal_read",
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
    target_index = next(
        i for i, (address, _) in enumerate(pool) if address == target_address
    )
    target_score = scores[target_index]
    rank = 1 + sum(score > target_score for score in scores)
    return pool[selected][0], rank


def run_arm(seed: int, latent_dim: int, state, pool):
    model = _model(seed, latent_dim)
    updates = model.train_with_negative_coverage(
        TRAIN_CODES,
        negative_count=NEGATIVE_COUNT,
        aggregation="mean",
        positive_views=POSITIVE_VIEWS,
    )
    expected_updates = EPOCHS * len(TRAIN_CODES)
    if updates != expected_updates:
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

    query_macs = INPUT_DIM * latent_dim + M * latent_dim
    state_build_macs = M * INPUT_DIM * latent_dim

    return {
        "latent_dim": latent_dim,
        "epochs": EPOCHS,
        "negative_count": NEGATIVE_COUNT,
        "positive_views": POSITIVE_VIEWS,
        "training_updates": updates,
        "negative_evaluations_total": updates * NEGATIVE_COUNT * POSITIVE_VIEWS,
        "target_top1_recall": statistics.fmean(recall),
        "target_rank_mean": statistics.fmean(ranks),
        "query_macs": query_macs,
        "state_build_macs": state_build_macs,
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
            str(width): run_arm(seed, width, state, pool)
            for width in LATENT_WIDTHS
        },
    }


def main() -> None:
    contract = load_contract("TACOSM-C5-LATENT-WIDTH-001")
    contract.require_levels([H_ANCHOR])
    contract.require_seeds(SEEDS)
    contract.require_steps(EPOCHS)
    contract.require_eval_steps(EVAL_STEPS)
    contract.require_k_levels([])
    contract.require_arms(["latent_8", "latent_16", "latent_32"])

    if set(TRAIN_CODES) & set(EVAL_CODES):
        raise AssertionError("training/evaluation semantic codes overlap")

    seeds = [run_seed(seed) for seed in SEEDS]
    aggregate = {}
    for width in LATENT_WIDTHS:
        key = str(width)
        aggregate[f"latent_{width}_recall_mean"] = statistics.fmean(
            row["arms"][key]["target_top1_recall"] for row in seeds
        )
        aggregate[f"latent_{width}_rank_mean"] = statistics.fmean(
            row["arms"][key]["target_rank_mean"] for row in seeds
        )

    delta = (
        aggregate["latent_32_recall_mean"]
        - aggregate["latent_8_recall_mean"]
    )
    if delta >= 0.15 and aggregate["latent_32_recall_mean"] >= 0.50:
        decision = "material_capacity_effect"
    elif abs(delta) <= 0.05:
        decision = "saturation"
    elif delta > 0.05:
        decision = "partial_capacity_effect"
    elif delta < -0.05:
        decision = "wider_reversal"
    else:
        decision = "near_zero"

    aggregate["delta_8_to_32"] = delta
    aggregate["decision"] = decision

    result = {
        "protocol": {
            "name": "TACOSM-C5-LATENT-WIDTH-001",
            "seeds": list(SEEDS),
            "H_anchor": H_ANCHOR,
            "M": M,
            "epochs": EPOCHS,
            "latent_widths": list(LATENT_WIDTHS),
            "negative_count": NEGATIVE_COUNT,
            "positive_views": POSITIVE_VIEWS,
            "queries_per_seed_per_arm": EVAL_STEPS,
            "training_codes": len(TRAIN_CODES),
            "evaluation_codes": len(EVAL_CODES),
            "aggregation": "mean",
        },
        "aggregate": aggregate,
        "seeds": seeds,
    }
    out = Path("artifacts/TACOSM-C5-LATENT-WIDTH-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
