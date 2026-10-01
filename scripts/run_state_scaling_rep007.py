#!/usr/bin/env python3
"""Run preregistered TACOSM-STATE-REP-007 state-population scaling."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm.contract import load_contract
from tac_osm.semantic_state_addressor import SemanticAddressingConfig, SemanticStateAddressor
from tac_osm.semantic_state_tasks import build_semantic_state_task
from tac_osm.temporal import TemporalPersistentState

SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (2, 4, 8, 16, 32)
TRAIN_EPISODES = 512
HELDOUT_EPISODES = 256
DELAY = 1
N_CANDIDATES = 8


def task_seed(seed: int, step: int, heldout: bool, m: int) -> int:
    phase = 700000 if heldout else 0
    return seed * 3000017 + m * 100003 + phase + step * 7919 + 101


def prepare(task) -> TemporalPersistentState:
    state = TemporalPersistentState()
    if task.write_step > 0:
        state.advance_to(task.write_step)
    task.stage(state)
    return state


def empty():
    return {'episodes': 0, 'top1': [], 'rank': [], 'macs': [], 'inspected': [], 'pool': [], 'train_successes': 0}


def summary(m, seed, arm, x, updates):
    mean = lambda xs: statistics.fmean(xs) if xs else None
    return {
        'm': m,
        'seed': seed,
        'arm': arm,
        'episodes': x['episodes'],
        'top1_recall': mean(x['top1']),
        'mean_target_rank': mean(x['rank']),
        'address_macs_mean': mean(x['macs']),
        'inspected_items_mean': mean(x['inspected']),
        'state_pool_size_mean': mean(x['pool']),
        'training_successes': x['train_successes'],
        'updates': updates,
    }


def run(m: int, seed: int, learned: bool):
    addressor = SemanticStateAddressor(
        SemanticAddressingConfig(seed=seed, learning_rate=0.01, margin=0.1)
    )
    train_successes = 0
    for step in range(TRAIN_EPISODES):
        task = build_semantic_state_task(
            task_seed(seed, step, False, m),
            write_step=step, delay=DELAY, n_states=m, n_candidates=N_CANDIDATES,
        )
        state = prepare(task)
        pool = addressor.state_pool(task.query, state)
        decision = addressor.select_from_pool(task.query, pool, target_address=task.target_address)
        success = decision.selected_address == task.target_address
        train_successes += int(success)
        if learned:
            addressor.learn_from_success(
                task.query, pool, decision.selected_index, success=success, scores=decision.scores
            )

    x = empty()
    x['train_successes'] = train_successes
    for step in range(HELDOUT_EPISODES):
        task = build_semantic_state_task(
            task_seed(seed, step, True, m),
            write_step=step, delay=DELAY, n_states=m, n_candidates=N_CANDIDATES,
        )
        state = prepare(task)
        pool = addressor.state_pool(task.query, state)
        decision = addressor.select_from_pool(task.query, pool, target_address=task.target_address)
        x['episodes'] += 1
        x['top1'].append(int(decision.selected_address == task.target_address))
        x['rank'].append(decision.target_rank)
        x['macs'].append(decision.total_macs)
        x['inspected'].append(decision.inspected_items)
        x['pool'].append(decision.pool_size)
    return summary(m, seed, 'learned' if learned else 'no_learning', x, addressor.updates)


def analytic(m: int, seed: int):
    addressor = SemanticStateAddressor(SemanticAddressingConfig(seed=seed))
    addressor.set_identity()
    x = empty()
    for step in range(HELDOUT_EPISODES):
        task = build_semantic_state_task(
            task_seed(seed, step, True, m),
            write_step=step, delay=DELAY, n_states=m, n_candidates=N_CANDIDATES,
        )
        state = prepare(task)
        pool = addressor.state_pool(task.query, state)
        decision = addressor.select_from_pool(task.query, pool, target_address=task.target_address)
        x['episodes'] += 1
        x['top1'].append(int(decision.selected_address == task.target_address))
        x['rank'].append(decision.target_rank)
        x['macs'].append(decision.total_macs)
        x['inspected'].append(decision.inspected_items)
        x['pool'].append(decision.pool_size)
    return summary(m, seed, 'analytic', x, 0)


def main():
    contract = load_contract("TACOSM-STATE-REP-007")
    contract.require_levels(M_LEVELS)
    contract.require_seeds(SEEDS)
    contract.require_steps(TRAIN_EPISODES)
    contract.require_eval_steps(HELDOUT_EPISODES)
    contract.require_arms(["analytic", "learned", "no_learning"])
    result = {
        'protocol': {
            'name': 'TACOSM-STATE-REP-007',
            'seeds': list(SEEDS),
            'm_levels': list(M_LEVELS),
            'train_episodes_per_seed': TRAIN_EPISODES,
            'heldout_episodes_per_seed': HELDOUT_EPISODES,
            'delay': DELAY,
            'candidates': N_CANDIDATES,
            'addressor': '5->8 bilinear semantic matcher',
            'learning_rate': 0.01,
            'margin': 0.1,
            'address_cost_formula': '40 + 48M MACs',
            'lookup_strategy': 'exhaustive semantic scan',
        },
        'analytic': [],
        'learned': [],
        'no_learning': [],
    }
    for m in M_LEVELS:
        for seed in SEEDS:
            result['analytic'].append(analytic(m, seed))
            result['learned'].append(run(m, seed, True))
            result['no_learning'].append(run(m, seed, False))
    out = Path('artifacts/TACOSM-STATE-REP-007.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()