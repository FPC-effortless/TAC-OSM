#!/usr/bin/env python3
"""Run preregistered TACOSM-C5-LEARNED-STATE-DIAG-001."""

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
EVAL_STEPS = 100
TRAIN_EPOCHS = 32
TRAIN_CODES = tuple(CODEBOOK[16:64])
EVAL_CODES = tuple(CODEBOOK[:16])


def _model(seed: int) -> LearnedSemanticStateIndex:
    return LearnedSemanticStateIndex(
        LearnedStateIndexConfig(
            input_dim=10,
            latent_dim=8,
            learning_rate=0.02,
            margin=0.25,
            epochs=TRAIN_EPOCHS,
            bucket_bits=8,
            probe_radius=1,
            shortlist_k=1,
            seed=seed,
        )
    )


def _state_pool(state) -> tuple[tuple[str, tuple[int, ...]], ...]:
    out = []
    for address in state.addresses():
        read = state.read(
            Query(
                text="\t" + address,
                step=state.current_step,
                provenance="c5_state_diag_internal_read",
            )
        )
        if not read.keys:
            continue
        out.append((address, tuple(int(x) for x in read.values[0])))
    if len(out) != M:
        raise AssertionError("registered diagnostic requires exactly 64 readable states")
    return tuple(out)


def _continuous_lookup(
    model: LearnedSemanticStateIndex,
    query: Query,
    pool: tuple[tuple[str, tuple[int, ...]], ...],
    cached_embeddings: tuple[tuple[float, ...], ...],
    target_address: str,
) -> tuple[str, int, int]:
    query_embedding = model.encode_query(query)
    scores = [
        sum(a * b for a, b in zip(query_embedding, state_embedding))
        for state_embedding in cached_embeddings
    ]
    selected = max(range(len(scores)), key=lambda i: (scores[i], -i))
    target_index = next(i for i, (address, _) in enumerate(pool) if address == target_address)
    target_score = scores[target_index]
    rank = 1 + sum(score > target_score for score in scores)
    macs = model.query_embedding_macs + len(pool) * model.config.latent_dim
    return pool[selected][0], rank, macs


def run_seed(seed: int) -> dict:
    first = build_task(seed, H_ANCHOR, 0)
    state = prepare_state(first)
    pool = _state_pool(state)

    no_learning = _model(seed + 10000)
    learned = _model(seed)
    trained_pairs = learned.train(TRAIN_CODES)

    no_learning_embeddings = tuple(
        tuple(no_learning.encode_state(value)) for _, value in pool
    )
    learned_embeddings = tuple(
        tuple(learned.encode_state(value)) for _, value in pool
    )

    learned_binary = _model(seed)
    learned_binary.wq = [row[:] for row in learned.wq]
    learned_binary.ws = [row[:] for row in learned.ws]
    learned_binary.bq = learned.bq[:]
    learned_binary.bs = learned.bs[:]
    learned_binary._trained_pairs = learned.trained_pairs
    binary_build = learned_binary.build(state)

    no_learning_recall = []
    learned_continuous_recall = []
    learned_binary_recall = []
    no_learning_ranks = []
    learned_continuous_ranks = []
    learned_binary_probes = []
    learned_binary_bucket_candidates = []

    for step in range(EVAL_STEPS):
        task = build_task(seed, H_ANCHOR, step)

        selected_no, rank_no, macs_no = _continuous_lookup(
            no_learning,
            task.query,
            pool,
            no_learning_embeddings,
            task.target_address,
        )
        selected_cont, rank_cont, macs_cont = _continuous_lookup(
            learned,
            task.query,
            pool,
            learned_embeddings,
            task.target_address,
        )
        binary_hit = learned_binary.lookup(task.query)

        no_learning_recall.append(int(selected_no == task.target_address))
        learned_continuous_recall.append(int(selected_cont == task.target_address))
        learned_binary_recall.append(int(task.target_address in binary_hit.addresses))
        no_learning_ranks.append(rank_no)
        learned_continuous_ranks.append(rank_cont)
        learned_binary_probes.append(binary_hit.query_probes)
        learned_binary_bucket_candidates.append(binary_hit.bucket_candidates)

        if macs_no != 592 or macs_cont != 592:
            raise AssertionError("continuous cost ledger changed")

    return {
        "seed": seed,
        "M": M,
        "training_codes": len(TRAIN_CODES),
        "evaluation_codes": len(EVAL_CODES),
        "trained_pairs": trained_pairs,
        "queries": EVAL_STEPS,
        "no_learning_continuous_recall": statistics.fmean(no_learning_recall),
        "learned_continuous_recall": statistics.fmean(learned_continuous_recall),
        "learned_binary_recall": statistics.fmean(learned_binary_recall),
        "continuous_recall_gain": statistics.fmean(learned_continuous_recall)
        - statistics.fmean(no_learning_recall),
        "binary_gap_vs_continuous": statistics.fmean(learned_continuous_recall)
        - statistics.fmean(learned_binary_recall),
        "continuous_target_rank_mean_no_learning": statistics.fmean(no_learning_ranks),
        "continuous_target_rank_mean_learned": statistics.fmean(learned_continuous_ranks),
        "continuous_query_macs": learned.query_embedding_macs,
        "continuous_total_macs_per_query": 592,
        "binary_query_macs": learned_binary.query_embedding_macs,
        "binary_probe_count": statistics.fmean(learned_binary_probes),
        "binary_bucket_candidates_mean": statistics.fmean(learned_binary_bucket_candidates),
        "binary_build_macs_total": binary_build.state_items * learned_binary.state_embedding_macs,
        "binary_build_macs_per_query": (
            binary_build.state_items * learned_binary.state_embedding_macs / EVAL_STEPS
        ),
        "binary_unique_buckets": binary_build.unique_buckets,
    }


def main() -> None:
    contract = load_contract("TACOSM-C5-LEARNED-STATE-DIAG-001")
    contract.require_levels([H_ANCHOR])
    contract.require_seeds(SEEDS)
    contract.require_steps(TRAIN_EPOCHS)
    contract.require_eval_steps(EVAL_STEPS)
    contract.require_k_levels([1])
    contract.require_arms(
        ["no_learning_continuous", "learned_continuous", "learned_binary"]
    )

    if set(TRAIN_CODES) & set(EVAL_CODES):
        raise AssertionError("training/evaluation semantic codes overlap")

    cells = [run_seed(seed) for seed in SEEDS]
    result = {
        "protocol": {
            "name": "TACOSM-C5-LEARNED-STATE-DIAG-001",
            "seeds": list(SEEDS),
            "H_anchor": H_ANCHOR,
            "M": M,
            "training_epochs": TRAIN_EPOCHS,
            "training_codes": len(TRAIN_CODES),
            "evaluation_codes": len(EVAL_CODES),
            "queries_per_seed": EVAL_STEPS,
            "continuous": "8-d latent dot-product over all 64 cached state embeddings",
            "binary": "8-bit sign code, Hamming radius 1, K=1",
            "diagnostic_gap_threshold": 0.20,
        },
        "cells": cells,
    }
    out = Path("artifacts/TACOSM-C5-LEARNED-STATE-DIAG-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
