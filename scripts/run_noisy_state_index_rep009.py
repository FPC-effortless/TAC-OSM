#!/usr/bin/env python3
"""Run preregistered TACOSM-STATE-REP-009 noisy selective state index."""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import Query
from tac_osm.contract import load_contract
from tac_osm.semantic_state_addressor import SemanticAddressingConfig, SemanticStateAddressor
from tac_osm.hamming_state_index import HammingStateIndex, pool_for_addresses
from tac_osm.noisy_state_tasks import (
    STATE_BITS,
    build_pool,
    build_query_for_target,
    prepare_state,
)

SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (8, 16, 32, 64)
TRAIN_EPISODES = 512
HELDOUT_EPISODES = 256
INDEX_RADIUS = 2
SHORTLIST_K = 4


def _train_task_seed(seed: int, m: int) -> int:
    return seed * 1000003 + m * 9176 + 503


def _eval_task_seed(seed: int, m: int) -> int:
    return seed * 2000003 + m * 12011 + 907


def _query(pool, target_index: int, query_step: int, nonce: int) -> Query:
    return build_query_for_target(
        pool,
        target_index,
        step=query_step,
        flip=(target_index * 3 + nonce * 5 + 7) % STATE_BITS,
    ).query


def _train_addressor(seed: int, m: int) -> SemanticStateAddressor:
    addressor = SemanticStateAddressor(
        SemanticAddressingConfig(
            input_dim=STATE_BITS,
            latent_dim=8,
            learning_rate=0.01,
            margin=0.1,
            seed=seed,
        )
    )
    pool = build_pool(_train_task_seed(seed, m), n_states=m, write_step=0, delay=1)
    state = prepare_state(pool)
    rng = random.Random(seed * 701 + m * 31 + 19)
    for step in range(TRAIN_EPISODES):
        target_index = rng.randrange(m)
        query = _query(pool, target_index, state.current_step, step)
        decision = addressor.select(
            query,
            state,
            target_address=pool.updates[target_index].key,
        )
        addressor.learn_from_success(
            query,
            addressor.state_pool(query, state),
            decision.selected_index,
            success=decision.selected_address == pool.updates[target_index].key,
            scores=decision.scores,
        )
    return addressor


def _evaluate(seed: int, m: int) -> dict:
    addressor = _train_addressor(seed, m)
    eval_pool = build_pool(_eval_task_seed(seed, m), n_states=m, write_step=0, delay=1)
    state = prepare_state(eval_pool)
    index = HammingStateIndex(bits=STATE_BITS, radius=INDEX_RADIUS, shortlist_k=SHORTLIST_K)
    build = index.build(state)
    rng = random.Random(seed * 1301 + m * 97 + 43)

    full_hits = 0
    indexed_hits = 0
    no_learning_hits = 0
    no_learning = SemanticStateAddressor(
        SemanticAddressingConfig(
            input_dim=STATE_BITS,
            latent_dim=8,
            learning_rate=0.01,
            margin=0.1,
            seed=seed + 10000,
        )
    )
    index_hits = 0
    shortlist_sizes = []
    full_macs = []
    indexed_macs = []
    index_ops = []

    for step in range(HELDOUT_EPISODES):
        target_index = rng.randrange(m)
        query = _query(eval_pool, target_index, state.current_step, step)

        full = addressor.select(
            query,
            state,
            target_address=eval_pool.updates[target_index].key,
        )
        full_hits += int(full.selected_address == eval_pool.updates[target_index].key)
        full_macs.append(full.total_macs)

        lookup = index.lookup(query)
        target_address = eval_pool.updates[target_index].key
        index_hits += int(target_address in lookup.addresses)
        shortlist = pool_for_addresses(state, lookup.addresses)
        indexed = addressor.select_from_pool(
            query,
            shortlist,
            target_address=target_address,
        )
        indexed_hits += int(indexed.selected_address == target_address)
        indexed_macs.append(indexed.total_macs)
        shortlist_sizes.append(lookup.shortlist_size)
        index_ops.append(lookup.query_operations)

    return {
        "m": m,
        "seed": seed,
        "queries": HELDOUT_EPISODES,
        "learned_full_top1": full_hits / HELDOUT_EPISODES,
        "learned_indexed_top1": indexed_hits / HELDOUT_EPISODES,
        "no_learning_full_top1": no_learning_hits / HELDOUT_EPISODES,
        "index_target_recall": index_hits / HELDOUT_EPISODES,
        "shortlist_size_mean": statistics.fmean(shortlist_sizes),
        "shortlist_size_max": max(shortlist_sizes),
        "full_route_macs_mean": statistics.fmean(full_macs),
        "indexed_route_macs_mean": statistics.fmean(indexed_macs),
        "index_query_operations_mean": statistics.fmean(index_ops),
        "index_build_reads": build.build_reads,
        "index_build_entries": build.unique_codes,
        "amortized_build_reads_per_query": build.build_reads / HELDOUT_EPISODES,
        "updates": addressor.updates,
    }


def _analytic(seed: int, m: int) -> dict:
    addressor = SemanticStateAddressor(
        SemanticAddressingConfig(
            input_dim=STATE_BITS,
            latent_dim=STATE_BITS,
            seed=seed,
        )
    )
    addressor.set_identity()
    pool = build_pool(_eval_task_seed(seed, m), n_states=m, write_step=0, delay=1)
    state = prepare_state(pool)
    index = HammingStateIndex(bits=STATE_BITS, radius=INDEX_RADIUS, shortlist_k=SHORTLIST_K)
    index.build(state)
    rng = random.Random(seed * 1703 + m * 113 + 59)
    full_hits = 0
    indexed_hits = 0
    for step in range(HELDOUT_EPISODES):
        target_index = rng.randrange(m)
        query = _query(pool, target_index, state.current_step)
        target = pool.updates[target_index].key
        full_hits += int(
            addressor.select(query, state, target_address=target).selected_address == target
        )
        lookup = index.lookup(query)
        shortlist = pool_for_addresses(state, lookup.addresses)
        indexed_hits += int(
            addressor.select_from_pool(
                query, shortlist, target_address=target
            ).selected_address == target
        )
    return {
        "m": m,
        "seed": seed,
        "top1_full": full_hits / HELDOUT_EPISODES,
        "top1_indexed": indexed_hits / HELDOUT_EPISODES,
    }


def main() -> None:
    contract = load_contract("TACOSM-STATE-REP-009")
    contract.require_levels(M_LEVELS)
    contract.require_seeds(SEEDS)
    contract.require_steps(TRAIN_EPISODES)
    contract.require_eval_steps(HELDOUT_EPISODES)
    contract.require_arms(["analytic", "learned_full", "learned_indexed", "no_learning"])

    result = {
        "protocol": {
            "name": "TACOSM-STATE-REP-009",
            "seeds": list(SEEDS),
            "m_levels": list(M_LEVELS),
            "train_episodes_per_seed_level": TRAIN_EPISODES,
            "heldout_queries_per_seed_level": HELDOUT_EPISODES,
            "state_bits": STATE_BITS,
            "query_noise": "exactly one bit flip",
            "index": "Hamming radius 2",
            "shortlist_k": SHORTLIST_K,
            "full_addressor": "10->8 bilinear semantic matcher",
            "full_route_cost": "80 + 88M MACs",
            "indexed_route_cost": "80 + 88K MACs, K <= 4",
            "index_query_operations": "56 bucket probes",
            "index_build": "M state reads",
        },
        "analytic": [],
        "cells": [],
    }

    for m in M_LEVELS:
        for seed in SEEDS:
            result["analytic"].append(_analytic(seed, m))
            cell = _evaluate(seed, m)
            result["cells"].append(cell)


    out = Path("artifacts/TACOSM-STATE-REP-009.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
