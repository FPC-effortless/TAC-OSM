#!/usr/bin/env python3
"""Run preregistered TACOSM-SEMANTIC-PROGRAM-REP-003."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import StateConfig, Structure
from tac_osm.explicit_executor import ExplicitGraphExecutor
from tac_osm.graph_program_router import (
    GraphProgramRouter,
    GraphProgramRouterConfig,
    semantic_match_rank,
)
from tac_osm.semantic_topology import build_semantic_task
from tac_osm.state import PersistentStore


SEEDS = (0, 1, 2, 3, 4)
TRAIN_EPISODES = 512
HELDOUT_EPISODES = 256
N_CANDIDATES = 8


def _task_seed(seed: int, step: int, heldout: bool) -> int:
    if heldout:
        return seed * 2000003 + 200000 + step * 7919 + 53
    return seed * 1000003 + step * 7919 + 41


def _summarise(metrics: dict, updates: int) -> dict:
    mean = lambda xs: statistics.fmean(xs) if xs else None
    return {
        "episodes": metrics["episodes"],
        "top1_recall": mean(metrics["top1"]),
        "mean_target_rank": mean(metrics["rank"]),
        "hard_negative_margin_mean": mean(metrics["margin"]),
        "exact_edge_execution_rate": (
            metrics["exact_execution"] / metrics["episodes"]
            if metrics["episodes"]
            else 0.0
        ),
        "active_edges_mean": mean(metrics["active_edges"]),
        "candidate_edges_mean": mean(metrics["candidate_edges"]),
        "route_macs_mean": mean(metrics["route_macs"]),
        "updates": updates,
    }


def _empty_metrics() -> dict:
    return {
        "episodes": 0,
        "top1": [],
        "rank": [],
        "margin": [],
        "exact_execution": 0,
        "active_edges": [],
        "candidate_edges": [],
        "route_macs": [],
    }


def _evaluate(task, decision, router, executor, state, metrics):
    diag = router.diagnostics_with_target(
        task.public(), task.candidates, task.target_action
    )
    selected = task.candidates[decision.selected]
    execution = executor.execute_with_work(
        Structure(
            key=selected.key,
            spec=__import__("tac_osm.semantic_topology", fromlist=["program_for_candidate"])
            .program_for_candidate(selected, input_values=task.input_values),
        ),
        [],
    )
    expected = tuple(
        edge in selected.executable_edges
        for edge in __import__("tac_osm.topology_tasks", fromlist=["TOPOLOGY_EDGE_UNIVERSE"])
        .TOPOLOGY_EDGE_UNIVERSE
    )
    exact = tuple(bool(g) for g in execution.result.gates) == expected
    metrics["episodes"] += 1
    metrics["top1"].append(int(diag.selected_rank == 1))
    metrics["rank"].append(diag.selected_rank)
    if diag.hard_negative_margin is not None:
        metrics["margin"].append(diag.hard_negative_margin)
    metrics["exact_execution"] += int(exact)
    metrics["active_edges"].append(execution.work.active_edges)
    metrics["candidate_edges"].append(execution.work.candidate_edges_metadata)
    metrics["route_macs"].append(diag.total_macs)


def run_learned(seed: int, learned: bool) -> dict:
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

    for step in range(TRAIN_EPISODES):
        task = build_semantic_task(
            _task_seed(seed, step, heldout=False),
            n_candidates=N_CANDIDATES,
            step=step,
        )
        decision = router.route(task.public(), state, task.candidates)
        if learned:
            router.learn_from_outcome(
                task.public(),
                state,
                task.candidates,
                decision.selected,
                success=decision.selected == task.target_action,
                scores=decision.scores,
            )

    metrics = _empty_metrics()
    for step in range(HELDOUT_EPISODES):
        task = build_semantic_task(
            _task_seed(seed, step, heldout=True),
            n_candidates=N_CANDIDATES,
            step=TRAIN_EPISODES + step,
        )
        decision = router.route(task.public(), state, task.candidates)
        _evaluate(task, decision, router, executor, state, metrics)

    return _summarise(metrics, router.updates)


def run_analytic(seed: int) -> dict:
    metrics = _empty_metrics()
    router = GraphProgramRouter()
    state = PersistentStore(StateConfig(seed=seed))
    executor = ExplicitGraphExecutor()
    for step in range(HELDOUT_EPISODES):
        task = build_semantic_task(
            _task_seed(seed, step, heldout=True),
            n_candidates=N_CANDIDATES,
            step=TRAIN_EPISODES + step,
        )
        rank, margin = semantic_match_rank(task.public(), task.candidates)
        decision = type("Decision", (), {"selected": task.target_action})()
        _evaluate(task, decision, router, executor, state, metrics)
        metrics["top1"][-1] = int(rank == 1)
        metrics["rank"][-1] = rank
        metrics["margin"][-1] = margin
        # The analytic witness is not represented by GraphProgramRouter's
        # random score path; its cost is structural matching, so leave route
        # MACs unset rather than pretending it used the learned model.
        metrics["route_macs"][-1] = None
    valid = [x for x in metrics["route_macs"] if x is not None]
    result = _summarise(metrics, 0)
    result["route_macs_mean"] = None
    return result


def collision_probe(seed: int) -> dict:
    task = build_semantic_task(seed, n_candidates=N_CANDIDATES)
    return {
        "candidate_count": len(task.candidates),
        "target_semantic_signature": [
            int(x)
            for x in __import__("tac_osm.semantic_topology", fromlist=["semantic_signature"])
            .semantic_signature(task.target_edges)
        ],
        "target_signature_count_in_pool": sum(
            1
            for c in task.candidates
            if __import__("tac_osm.semantic_topology", fromlist=["semantic_signature"])
            .semantic_signature(c.executable_edges)
            == __import__("tac_osm.semantic_topology", fromlist=["semantic_signature"])
            .semantic_signature(task.target_edges)
        ),
        "unique_raw_topologies": len(
            {
                tuple(c.executable_edges)
                for c in task.candidates
            }
        ),
    }


def oracle(seed: int) -> dict:
    successes = []
    for step in range(HELDOUT_EPISODES):
        task = build_semantic_task(
            _task_seed(seed, step, heldout=True),
            n_candidates=N_CANDIDATES,
            step=TRAIN_EPISODES + step,
        )
        successes.append(task.target_action < len(task.candidates))
    return {"success_rate": statistics.fmean(successes)}


def main() -> None:
    result = {
        "protocol": {
            "name": "TACOSM-SEMANTIC-PROGRAM-REP-003",
            "seeds": list(SEEDS),
            "train_episodes_per_seed": TRAIN_EPISODES,
            "heldout_episodes_per_seed": HELDOUT_EPISODES,
            "candidates": N_CANDIDATES,
            "query_width": 5,
            "edge_width": 7,
            "hidden_dim": 16,
            "latent_dim": 8,
            "learning_rate": 0.01,
            "margin": 0.1,
            "executor": "explicit_graph_exact",
        },
        "semantic_probe": collision_probe(17),
        "condition_A_analytic": {},
        "condition_B_learned": {},
        "condition_C_no_learning": {},
        "condition_D_oracle": {},
    }
    for seed in SEEDS:
        result["condition_A_analytic"][str(seed)] = run_analytic(seed)
        result["condition_B_learned"][str(seed)] = run_learned(seed, True)
        result["condition_C_no_learning"][str(seed)] = run_learned(seed, False)
        result["condition_D_oracle"][str(seed)] = oracle(seed)

    out = Path("artifacts/TACOSM-SEMANTIC-PROGRAM-REP-003.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
