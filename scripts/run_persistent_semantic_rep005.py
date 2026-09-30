#!/usr/bin/env python3
"""Run preregistered TACOSM-PERSISTENT-SEMANTIC-REP-005."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import Query, Structure
from tac_osm.explicit_executor import ExplicitGraphExecutor
from tac_osm.graph_program_router import GraphProgramRouterConfig, semantic_match_rank
from tac_osm.persistent_semantic import build_persistent_semantic_task
from tac_osm.persistent_semantic_router import PersistentSemanticGraphRouter
from tac_osm.state_addressing import StateAddressor
from tac_osm.temporal import TemporalPersistentState
from tac_osm.topology_tasks import TOPOLOGY_EDGE_UNIVERSE

SEEDS = (0, 1, 2, 3, 4)
TRAIN_EPISODES = 512
HELDOUT_EPISODES = 256
N_CANDIDATES = 8
DELAY = 1
ADDRESS = 'goal:semantic'


def task_seed(seed: int, step: int, heldout: bool) -> int:
    if heldout:
        return seed * 2000003 + 300000 + step * 7919 + 67
    return seed * 1000003 + step * 7919 + 59


def empty_metrics() -> dict:
    return {
        'episodes': 0,
        'top1': [],
        'rank': [],
        'margin': [],
        'route_macs': [],
        'state_inspected_slots': [],
        'state_pool_size': [],
        'exact_execution': 0,
        'active_edges': [],
        'candidate_edges': [],
    }


def summarise(metrics: dict, updates: int, training_successes: int) -> dict:
    mean = lambda xs: statistics.fmean(xs) if xs else None
    return {
        'episodes': metrics['episodes'],
        'top1_recall': mean(metrics['top1']),
        'mean_target_rank': mean(metrics['rank']),
        'hard_negative_margin_mean': mean(metrics['margin']),
        'route_macs_mean': mean(metrics['route_macs']),
        'state_inspected_slots_mean': mean(metrics['state_inspected_slots']),
        'state_pool_size_mean': mean(metrics['state_pool_size']),
        'exact_edge_execution_rate': metrics['exact_execution'] / metrics['episodes'],
        'active_edges_mean': mean(metrics['active_edges']),
        'candidate_edges_mean': mean(metrics['candidate_edges']),
        'updates': updates,
        'training_successes': training_successes,
    }


def stage_and_advance(state: TemporalPersistentState, task):
    if state.current_step != task.write_step:
        raise RuntimeError(
            f'state clock {state.current_step} != task write step {task.write_step}'
        )
    state.stage_world_write(task.write, delay=task.delay)
    state.advance_to(task.read_step)


def evaluate_one(task, state, router, executor, metrics) -> None:
    addressor = StateAddressor()
    memory = addressor.address(task.query, state)
    decision = router.route(task.query, state, task.candidates)
    diag = router.diagnostics_with_target(
        task.query, state, task.candidates, task.target_action
    )
    selected = task.candidates[decision.selected]
    execution = executor.execute_with_work(
        Structure(
            key=selected.key,
            spec=__import__('tac_osm.persistent_semantic', fromlist=['program_for_candidate'])
            .program_for_candidate(selected, input_values=task.input_values),
        ),
        [],
    )
    expected = tuple(
        edge in selected.executable_edges for edge in TOPOLOGY_EDGE_UNIVERSE
    )
    metrics['episodes'] += 1
    metrics['top1'].append(int(diag.selected_rank == 1))
    metrics['rank'].append(diag.selected_rank)
    if diag.hard_negative_margin is not None:
        metrics['margin'].append(diag.hard_negative_margin)
    metrics['route_macs'].append(diag.total_macs)
    metrics['state_inspected_slots'].append(memory.inspected_slots)
    metrics['state_pool_size'].append(memory.pool_size)
    metrics['exact_execution'] += int(tuple(bool(g) for g in execution.result.gates) == expected)
    metrics['active_edges'].append(execution.work.active_edges)
    metrics['candidate_edges'].append(execution.work.candidate_edges_metadata)


def run_arm(seed: int, *, learned: bool) -> dict:
    router = PersistentSemanticGraphRouter(
        GraphProgramRouterConfig(
            hidden_dim=16, latent_dim=8, learning_rate=0.01, margin=0.1, seed=seed
        )
    )
    state = TemporalPersistentState()
    executor = ExplicitGraphExecutor()
    successes = 0

    for step in range(TRAIN_EPISODES):
        task = build_persistent_semantic_task(
            task_seed(seed, step, False),
            write_step=step, delay=DELAY, n_candidates=N_CANDIDATES, address=ADDRESS,
        )
        stage_and_advance(state, task)
        decision = router.route(task.query, state, task.candidates)
        success = decision.selected == task.target_action
        successes += int(success)
        if learned:
            router.learn_from_outcome(
                task.query, state, task.candidates, decision.selected,
                success=success, scores=decision.scores,
            )

    metrics = empty_metrics()
    for step in range(HELDOUT_EPISODES):
        write_step = TRAIN_EPISODES + step
        task = build_persistent_semantic_task(
            task_seed(seed, step, True),
            write_step=write_step, delay=DELAY, n_candidates=N_CANDIDATES, address=ADDRESS,
        )
        stage_and_advance(state, task)
        evaluate_one(task, state, router, executor, metrics)

    return summarise(metrics, router.updates, successes)


def run_analytic(seed: int) -> dict:
    state = TemporalPersistentState()
    top1 = []
    ranks = []
    margins = []
    for step in range(HELDOUT_EPISODES):
        task = build_persistent_semantic_task(
            task_seed(seed, step, True),
            write_step=TRAIN_EPISODES + step,
            delay=DELAY, n_candidates=N_CANDIDATES, address=ADDRESS,
        )
        stage_and_advance(state, task)
        memory = StateAddressor().address(task.query, state)
        state_query = Query(
            text=' '.join(str(int(x)) for x in memory.value) + '\t',
            context=(), step=task.read_step, provenance='persistent_state_analytic',
        )
        rank, margin = semantic_match_rank(state_query, task.candidates)
        top1.append(int(rank == 1))
        ranks.append(rank)
        margins.append(margin)
    return {
        'episodes': HELDOUT_EPISODES,
        'top1_recall': statistics.fmean(top1),
        'mean_target_rank': statistics.fmean(ranks),
        'hard_negative_margin_mean': statistics.fmean(margins),
    }


def reset_probe(seed: int) -> dict:
    state = TemporalPersistentState()
    router = PersistentSemanticGraphRouter(GraphProgramRouterConfig(seed=seed))
    failures_closed = 0
    for step in range(32):
        task = build_persistent_semantic_task(
            task_seed(seed, 1000 + step, True),
            write_step=step, delay=DELAY, n_candidates=N_CANDIDATES, address=ADDRESS,
        )
        stage_and_advance(state, task)
        state.clear()
        try:
            router.route(task.query, state, task.candidates)
        except ValueError:
            failures_closed += 1
    return {'episodes': 32, 'failed_closed': failures_closed}


def oracle(seed: int) -> dict:
    return {'success_rate': 1.0}


def main() -> None:
    result = {
        'protocol': {
            'name': 'TACOSM-PERSISTENT-SEMANTIC-REP-005',
            'seeds': list(SEEDS),
            'train_episodes_per_seed': TRAIN_EPISODES,
            'heldout_episodes_per_seed': HELDOUT_EPISODES,
            'candidates': N_CANDIDATES,
            'state_address': ADDRESS,
            'state_delay': DELAY,
            'query_payload': 'address_only',
            'candidate_representation': 'explicit_executable_topology',
            'encoder': '7->16-tanh->8 candidate; 5->8 query',
            'learning_rate': 0.01,
            'margin': 0.1,
            'executor': 'explicit_graph_exact',
        },
        'condition_A_analytic': {},
        'condition_B_learned': {},
        'condition_C_no_learning': {},
        'condition_D_reset': {},
        'condition_E_oracle': {},
    }
    for seed in SEEDS:
        result['condition_A_analytic'][str(seed)] = run_analytic(seed)
        result['condition_B_learned'][str(seed)] = run_arm(seed, learned=True)
        result['condition_C_no_learning'][str(seed)] = run_arm(seed, learned=False)
        result['condition_D_reset'][str(seed)] = reset_probe(seed)
        result['condition_E_oracle'][str(seed)] = oracle(seed)

    out = Path('artifacts/TACOSM-PERSISTENT-SEMANTIC-REP-005.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()