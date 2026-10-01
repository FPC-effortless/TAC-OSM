#!/usr/bin/env python3
"""Run preregistered TACOSM-C5-NEGATIVE-SWEEP-001."""

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
NEGATIVE_COUNTS = (1, 4, 8, 16)
TRAIN_CODES = tuple(CODEBOOK[16:64])
EVAL_CODES = tuple(CODEBOOK[:16])
QUERY_MACS = 592
STATE_BUILD_MACS = 5120


def _model(seed: int) -> LearnedSemanticStateIndex:
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10,
            latent_dim=8,
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
    out = []
    for address in state.addresses():
        read = state.read(
            Query(
                text="\t" + address,
                step=state.current_step,
                provenance="c5_negative_sweep_internal_read",
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


def run_arm(seed: int, negative_count: int, state, pool):
    model = _model(seed)
    updates = model.train_with_negative_coverage(
        TRAIN_CODES,
        negative_count=negative_count,
        aggregation="mean",
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
            model,
            task.query,
            pool,
            cached_embeddings,
            task.target_address,
        )
        recall.append(int(selected == task.target_address))
        ranks.append(rank)

    return {
        "negative_count": negative_count,
        "epochs": EPOCHS,
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

    arms = {
        str(count): run_arm(seed, count, state, pool)
        for count in NEGATIVE_COUNTS
    }
    return {
        "seed": seed,
        "M": M,
        "queries": EVAL_STEPS,
        "training_codes": len(TRAIN_CODES),
        "evaluation_codes": len(EVAL_CODES),
        "arms": arms,
    }


def main() -> None:
    contract = load_contract("TACOSM-C5-NEGATIVE-SWEEP-001")
    contract.require_levels([H_ANCHOR])
    contract.require_seeds(SEEDS)
    contract.require_steps(EPOCHS)
    contract.require_eval_steps(EVAL_STEPS)
    contract.require_k_levels([])
    contract.require_arms(["mean_1", "mean_4", "mean_8", "mean_16"])

    if set(TRAIN_CODES) & set(EVAL_CODES):
        raise AssertionError("training/evaluation semantic codes overlap")
    if len(TRAIN_CODES) != 48 or len(EVAL_CODES) != 16:
        raise AssertionError("registered semantic code split changed")
    if not all(count < len(TRAIN_CODES) for count in NEGATIVE_COUNTS):
        raise AssertionError("negative pool must remain smaller than training pool")

    seeds = [run_seed(seed) for seed in SEEDS]
    aggregate = {}
    for count in NEGATIVE_COUNTS:
        key = str(count)
        aggregate[f"mean_{count}_recall_mean"] = statistics.fmean(
            row["arms"][key]["target_top1_recall"] for row in seeds
        )
        aggregate[f"mean_{count}_rank_mean"] = statistics.fmean(
            row["arms"][key]["target_rank_mean"] for row in seeds
        )
        aggregate[f"mean_{count}_negative_evaluations"] = seeds[0]["arms"][key][
            "negative_evaluations_total"
        ]

    delta_8_to_16 = (
        aggregate["mean_16_recall_mean"] - aggregate["mean_8_recall_mean"]
    )
    if delta_8_to_16 >= 0.10:
        decision = "continued_gain_beyond_eight"
    elif delta_8_to_16 >= -0.05:
        decision = "plateau_after_eight"
    else:
        decision = "reversal_after_eight"

    aggregate["delta_8_to_16"] = delta_8_to_16
    aggregate["decision"] = decision

    result = {
        "protocol": {
            "name": "TACOSM-C5-NEGATIVE-SWEEP-001",
            "seeds": list(SEEDS),
            "H_anchor": H_ANCHOR,
            "M": M,
            "epochs": EPOCHS,
            "negative_counts": list(NEGATIVE_COUNTS),
            "queries_per_seed_per_arm": EVAL_STEPS,
            "training_codes": len(TRAIN_CODES),
            "evaluation_codes": len(EVAL_CODES),
            "aggregation": "mean",
            "query_macs": QUERY_MACS,
            "state_build_macs": STATE_BUILD_MACS,
        },
        "aggregate": aggregate,
        "seeds": seeds,
    }

    out = Path("artifacts/TACOSM-C5-NEGATIVE-SWEEP-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
