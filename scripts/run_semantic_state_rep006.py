#!/usr/bin/env python3
"""Run preregistered TACOSM-STATE-REP-006."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import Query
from tac_osm.contract import load_contract
from tac_osm.graph_program_router import semantic_match_rank
from tac_osm.semantic_state_addressor import (
    SemanticAddressingConfig,
    SemanticStateAddressor,
)
from tac_osm.semantic_state_tasks import (
    STATE_ADDRESS_COUNT,
    build_semantic_state_task,
)
from tac_osm.temporal import TemporalPersistentState


SEEDS = (0, 1, 2, 3, 4)
TRAIN_EPISODES = 512
HELDOUT_EPISODES = 256
N_STATES = STATE_ADDRESS_COUNT
N_CANDIDATES = 8
DELAY = 1


def task_seed(seed: int, step: int, heldout: bool) -> int:
    return (
        seed * 2000003 + 400000 + step * 7919 + 71
        if heldout
        else seed * 1000003 + step * 7919 + 61
    )


def empty_metrics() -> dict:
    return {
        "episodes": 0,
        "state_top1": [],
        "state_rank": [],
        "end_to_end_top1": [],
        "end_to_end_rank": [],
        "address_macs": [],
        "inspected_items": [],
        "pool_size": [],
        "training_successes": 0,
    }


def summarise(metrics: dict, updates: int) -> dict:
    mean = lambda xs: statistics.fmean(xs) if xs else None
    return {
        "episodes": metrics["episodes"],
        "state_top1_recall": mean(metrics["state_top1"]),
        "mean_target_state_rank": mean(metrics["state_rank"]),
        "end_to_end_top1_recall": mean(metrics["end_to_end_top1"]),
        "mean_end_to_end_rank": mean(metrics["end_to_end_rank"]),
        "address_macs_mean": mean(metrics["address_macs"]),
        "inspected_state_items_mean": mean(metrics["inspected_items"]),
        "state_pool_size_mean": mean(metrics["pool_size"]),
        "training_successes": metrics["training_successes"],
        "updates": updates,
    }


def prepare_state(task) -> TemporalPersistentState:
    state = TemporalPersistentState()
    task.stage(state)
    return state


def learned_arm(seed: int, learned: bool) -> dict:
    addressor = SemanticStateAddressor(
        SemanticAddressingConfig(
            input_dim=5,
            latent_dim=8,
            learning_rate=0.01,
            margin=0.1,
            seed=seed,
        )
    )
    metrics = empty_metrics()

    for step in range(TRAIN_EPISODES):
        task = build_semantic_state_task(
            task_seed(seed, step, False),
            write_step=step,
            delay=DELAY,
            n_states=N_STATES,
            n_candidates=N_CANDIDATES,
        )
        # Advance a fresh state to the task's write/read boundary. State
        # addressing is evaluated independently of previous episode contents.
        state = prepare_state(task)
        pool = addressor.state_pool(task.query, state)
        decision = addressor.select_from_pool(
            task.query, pool, target_address=task.target_address
        )
        success = pool[decision.selected_index].value == task.target_signature
        metrics["training_successes"] += int(success)
        if learned:
            addressor.learn_from_success(
                task.query,
                pool,
                decision.selected_index,
                success=success,
                scores=decision.scores,
            )

    for step in range(HELDOUT_EPISODES):
        task = build_semantic_state_task(
            task_seed(seed, step, True),
            write_step=step,
            delay=DELAY,
            n_states=N_STATES,
            n_candidates=N_CANDIDATES,
        )
        state = prepare_state(task)
        pool = addressor.state_pool(task.query, state)
        decision = addressor.select_from_pool(
            task.query, pool, target_address=task.target_address
        )
        selected_value = pool[decision.selected_index].value
        state_query = Query(
            text=" ".join(str(int(x)) for x in selected_value),
            step=task.read_step,
            provenance="retrieved_semantic_state",
        )
        _, program_index_rank = semantic_match_rank(
            state_query, task.candidates
        )
        # semantic_match_rank returns the selected rank, which is 1 exactly
        # when the retrieved state value resolves the target semantic class.
        end_to_end_success = (
            selected_value == task.target_signature and program_index_rank == 1
        )
        metrics["episodes"] += 1
        metrics["state_top1"].append(int(decision.selected_address == task.target_address))
        metrics["state_rank"].append(decision.target_rank)
        metrics["end_to_end_top1"].append(int(end_to_end_success))
        metrics["end_to_end_rank"].append(
            1 if end_to_end_success else program_index_rank
        )
        metrics["address_macs"].append(decision.total_macs)
        metrics["inspected_items"].append(decision.inspected_items)
        metrics["pool_size"].append(decision.pool_size)

    return summarise(metrics, addressor.updates)


def analytic_arm(seed: int) -> dict:
    addressor = SemanticStateAddressor(
        SemanticAddressingConfig(seed=seed)
    )
    addressor.set_identity()
    metrics = empty_metrics()
    for step in range(HELDOUT_EPISODES):
        task = build_semantic_state_task(
            task_seed(seed, step, True),
            write_step=step,
            delay=DELAY,
            n_states=N_STATES,
            n_candidates=N_CANDIDATES,
        )
        state = prepare_state(task)
        pool = addressor.state_pool(task.query, state)
        decision = addressor.select_from_pool(
            task.query, pool, target_address=task.target_address
        )
        metrics["episodes"] += 1
        metrics["state_top1"].append(
            int(decision.selected_address == task.target_address)
        )
        metrics["state_rank"].append(decision.target_rank)
        metrics["address_macs"].append(decision.total_macs)
        metrics["inspected_items"].append(decision.inspected_items)
        metrics["pool_size"].append(decision.pool_size)
        selected = pool[decision.selected_index].value
        metrics["end_to_end_top1"].append(
            int(selected == task.target_signature)
        )
        metrics["end_to_end_rank"].append(
            1 if selected == task.target_signature else 8
        )
    return summarise(metrics, 0)


def shuffle_probe(seed: int) -> dict:
    task = build_semantic_state_task(
        seed, write_step=0, delay=DELAY, n_states=N_STATES
    )
    state = prepare_state(task)
    before = state.addresses()
    state.swap_values(task.target_address, before[0])
    after = state.addresses()
    return {
        "address_set_preserved": before == after,
        "target_address_preserved": task.target_address in after,
        "value_swap_performed": True,
    }


def reset_probe(seed: int) -> dict:
    task = build_semantic_state_task(
        seed, write_step=0, delay=DELAY, n_states=N_STATES
    )
    state = prepare_state(task)
    state.clear()
    addressor = SemanticStateAddressor(
        SemanticAddressingConfig(seed=seed)
    )
    try:
        addressor.select(task.query, state)
    except ValueError:
        return {"failed_closed": True}
    return {"failed_closed": False}


def main() -> None:
    contract = load_contract("TACOSM-STATE-REP-006")
    contract.require_levels([N_STATES])
    contract.require_seeds(SEEDS)
    contract.require_steps(TRAIN_EPISODES)
    contract.require_eval_steps(HELDOUT_EPISODES)
    contract.require_arms(
        ["analytic", "learned", "no_learning", "shuffle", "reset"]
    )
    result = {
        "protocol": {
            "name": "TACOSM-STATE-REP-006",
            "seeds": list(SEEDS),
            "train_episodes_per_seed": TRAIN_EPISODES,
            "heldout_episodes_per_seed": HELDOUT_EPISODES,
            "state_items": N_STATES,
            "candidates": N_CANDIDATES,
            "delay": DELAY,
            "query": "five-bit semantic signature, no address",
            "state_addresses": "opaque randomized keys",
            "candidate_router": "exact semantic structural matcher",
            "state_addressor": "5->8 bilinear learned matcher",
            "learning_rate": 0.01,
            "margin": 0.1,
        },
        "condition_A_analytic": {},
        "condition_B_learned": {},
        "condition_C_no_learning": {},
        "condition_D_shuffle": {},
        "condition_E_reset": {},
    }

    for seed in SEEDS:
        result["condition_A_analytic"][str(seed)] = analytic_arm(seed)
        result["condition_B_learned"][str(seed)] = learned_arm(seed, True)
        result["condition_C_no_learning"][str(seed)] = learned_arm(seed, False)
        result["condition_D_shuffle"][str(seed)] = shuffle_probe(seed)
        result["condition_E_reset"][str(seed)] = reset_probe(seed)

    out = Path("artifacts/TACOSM-STATE-REP-006.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
