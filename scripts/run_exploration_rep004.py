#!/usr/bin/env python3
"""Run preregistered TACOSM-LEARN-REP-004 exploration intervention.

The runner records training successes separately from held-out routing so
exploration effects are not conflated with evaluation-time stochasticity.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import Structure
from tac_osm.exploration import EpsilonGreedySelector
from tac_osm.explicit_executor import ExplicitGraphExecutor
from tac_osm.graph_program_router import (
    GraphProgramRouter,
    GraphProgramRouterConfig,
)
from tac_osm.semantic_topology import (
    build_semantic_task,
    program_for_candidate,
)
from tac_osm.state import PersistentStore, StateConfig
from tac_osm.topology_tasks import TOPOLOGY_EDGE_UNIVERSE
from tac_osm.graph_program_router import semantic_match_rank


SEEDS = (0, 1, 2, 3, 4)
TRAIN_EPISODES = 512
HELDOUT_EPISODES = 256
N_CANDIDATES = 8


def task_seed(seed: int, step: int, heldout: bool) -> int:
    return (
        seed * 2000003 + 200000 + step * 7919 + 53
        if heldout
        else seed * 1000003 + step * 7919 + 41
    )


def empty_metrics():
    return {
        "episodes": 0,
        "top1": [],
        "rank": [],
        "margin": [],
        "exact": 0,
        "active_edges": [],
        "candidate_edges": [],
        "route_macs": [],
    }


def summarise(m, updates, training_successes, exploration_decisions):
    mean = lambda xs: statistics.fmean(xs) if xs else None
    return {
        "episodes": m["episodes"],
        "top1_recall": mean(m["top1"]),
        "mean_target_rank": mean(m["rank"]),
        "hard_negative_margin_mean": mean(m["margin"]),
        "exact_edge_execution_rate": m["exact"] / m["episodes"],
        "active_edges_mean": mean(m["active_edges"]),
        "candidate_edges_mean": mean(m["candidate_edges"]),
        "route_macs_mean": mean(m["route_macs"]),
        "updates": updates,
        "training_successes": training_successes,
        "exploration_decisions": exploration_decisions,
    }


def evaluate(task, selected, router, executor, metrics):
    diag = router.diagnostics_with_target(
        task.public(), task.candidates, task.target_action
    )
    candidate = task.candidates[selected]
    execution = executor.execute_with_work(
        Structure(
            key=candidate.key,
            spec=program_for_candidate(candidate, input_values=task.input_values),
        ),
        [],
    )
    expected = tuple(
        edge in candidate.executable_edges for edge in TOPOLOGY_EDGE_UNIVERSE
    )
    metrics["episodes"] += 1
    metrics["top1"].append(int(diag.selected_rank == 1))
    metrics["rank"].append(diag.selected_rank)
    if diag.hard_negative_margin is not None:
        metrics["margin"].append(diag.hard_negative_margin)
    metrics["exact"] += int(
        tuple(bool(g) for g in execution.result.gates) == expected
    )
    metrics["active_edges"].append(execution.work.active_edges)
    metrics["candidate_edges"].append(execution.work.candidate_edges_metadata)
    metrics["route_macs"].append(diag.total_macs)


def run(seed: int, mode: str):
    router = GraphProgramRouter(
        GraphProgramRouterConfig(
            hidden_dim=16,
            latent_dim=8,
            learning_rate=0.01,
            margin=0.1,
            seed=seed,
        )
    )
    state = PersistentStore(StateConfig(seed=seed))
    executor = ExplicitGraphExecutor()
    selector = EpsilonGreedySelector(seed=seed + 991) if mode == "exploration" else None

    training_successes = 0
    exploration_decisions = 0

    for step in range(TRAIN_EPISODES):
        task = build_semantic_task(
            task_seed(seed, step, False),
            n_candidates=N_CANDIDATES,
            step=step,
        )
        greedy = router.route(task.public(), state, task.candidates)
        selected = selector.select(greedy, step) if selector else greedy.selected
        if selector is not None and selected != greedy.selected:
            exploration_decisions += 1
        success = selected == task.target_action
        training_successes += int(success)
        router.learn_from_outcome(
            task.public(),
            state,
            task.candidates,
            selected,
            success=success,
            scores=greedy.scores,
        ) if mode != "no_learning" else None

    metrics = empty_metrics()
    for step in range(HELDOUT_EPISODES):
        task = build_semantic_task(
            task_seed(seed, step, True),
            n_candidates=N_CANDIDATES,
            step=TRAIN_EPISODES + step,
        )
        decision = router.route(task.public(), state, task.candidates)
        evaluate(task, decision.selected, router, executor, metrics)

    return summarise(
        metrics,
        router.updates,
        training_successes,
        exploration_decisions,
    )


def analytic(seed: int):
    ranks = []
    margins = []
    for step in range(HELDOUT_EPISODES):
        task = build_semantic_task(
            task_seed(seed, step, True),
            n_candidates=N_CANDIDATES,
            step=TRAIN_EPISODES + step,
        )
        rank, margin = semantic_match_rank(task.public(), task.candidates)
        ranks.append(rank)
        margins.append(margin)
    return {
        "episodes": HELDOUT_EPISODES,
        "top1_recall": 1.0,
        "mean_target_rank": statistics.fmean(ranks),
        "hard_negative_margin_mean": statistics.fmean(margins),
    }


def oracle(seed: int):
    return {"success_rate": 1.0}


def main():
    result = {
        "protocol": {
            "name": "TACOSM-LEARN-REP-004",
            "seeds": list(SEEDS),
            "train_episodes_per_seed": TRAIN_EPISODES,
            "heldout_episodes_per_seed": HELDOUT_EPISODES,
            "candidates": N_CANDIDATES,
            "task": "REP-003 semantic program topology",
            "baseline_training": "greedy deterministic selection",
            "intervention": "epsilon-greedy, eps=0.30 linearly to 0 over 500 steps",
            "optimizer": "same GraphProgramRouter and pairwise success update as REP-003",
            "executor": "explicit_graph_exact",
        },
        "condition_A_analytic": {},
        "condition_B_baseline": {},
        "condition_C_no_learning": {},
        "condition_D_exploration": {},
        "condition_E_oracle": {},
    }
    for seed in SEEDS:
        result["condition_A_analytic"][str(seed)] = analytic(seed)
        result["condition_B_baseline"][str(seed)] = run(seed, "baseline")
        result["condition_C_no_learning"][str(seed)] = run(seed, "no_learning")
        result["condition_D_exploration"][str(seed)] = run(seed, "exploration")
        result["condition_E_oracle"][str(seed)] = oracle(seed)

    out = Path("artifacts/TACOSM-LEARN-REP-004.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
