#!/usr/bin/env python3
"""Run preregistered TACOSM-IDENTIFIABILITY-REP-002."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import Structure
from tac_osm.energy_router import EnergyRouterConfig, RepresentationEnergyRouter
from tac_osm.explicit_executor import ExplicitGraphExecutor
from tac_osm.state import PersistentStore, StateConfig
from tac_osm.topology_router import ExplicitProgramEnergyRouter
from tac_osm.topology_tasks import build_topology_task, program_for_candidate


SEEDS = (0, 1, 2, 3, 4)
TRAIN_EPISODES = 128
HELDOUT_EPISODES = 128
N_CANDIDATES = 8
INPUT_DIM = 7
LATENT_DIM = 7


def _metric():
    return {
        "episodes": 0,
        "top1": [],
        "rank": [],
        "margin": [],
        "exact_edge_execution": 0,
        "active_edges": [],
        "candidate_edges": [],
        "updates": 0,
    }


def _summarise(m):
    mean = lambda xs: statistics.fmean(xs) if xs else None
    return {
        "episodes": m["episodes"],
        "top1_recall": mean(m["top1"]),
        "mean_target_rank": mean(m["rank"]),
        "hard_negative_margin_mean": mean(m["margin"]),
        "exact_edge_execution_rate": (
            m["exact_edge_execution"] / m["episodes"] if m["episodes"] else 0.0
        ),
        "active_edges_mean": mean(m["active_edges"]),
        "candidate_edges_mean": mean(m["candidate_edges"]),
        "updates": m["updates"],
    }


def _run_arm(seed: int, learned: bool):
    router = ExplicitProgramEnergyRouter(
        {
            "input_dim": INPUT_DIM
        }
    ) if False else ExplicitProgramEnergyRouter(
        __import__("tac_osm").topology_router.ExplicitProgramRouterConfig(
            input_dim=INPUT_DIM,
            latent_dim=LATENT_DIM,
            learning_rate=0.01,
            margin=0.1,
            seed=seed,
        )
    )
    executor = ExplicitGraphExecutor()
    metrics = _metric()

    for step in range(TRAIN_EPISODES):
        task = build_topology_task(
            seed * 1000003 + step * 7919 + 17,
            n_candidates=N_CANDIDATES,
            step=step,
        )
        decision = router.route(task.public(), PersistentStore(StateConfig(seed=seed)), task.candidates)
        if learned:
            router.learn_from_outcome(
                task.public(),
                PersistentStore(StateConfig(seed=seed)),
                task.candidates,
                decision.selected,
                success=decision.selected == task.target_action,
                scores=decision.scores,
            )
        metrics["episodes"] += 1

    heldout = _metric()
    eval_state = PersistentStore(StateConfig(seed=seed))
    for step in range(HELDOUT_EPISODES):
        task = build_topology_task(
            seed * 2000003 + 100000 + step * 7919 + 23,
            n_candidates=N_CANDIDATES,
            step=TRAIN_EPISODES + step,
        )
        decision = router.route(task.public(), eval_state, task.candidates)
        diag = router.diagnostics_with_target(
            task.public(), task.candidates, task.target_action
        )
        selected = task.candidates[decision.selected]
        program = program_for_candidate(selected, input_values=task.input_values)
        execution = executor.execute_with_work(
            Structure(key=selected.key, spec=program),
            [],
        )
        expected_edges = tuple(
            edge in selected.executable_edges
            for edge in __import__("tac_osm").topology_tasks.TOPOLOGY_EDGE_UNIVERSE
        )
        exact_edges = tuple(bool(g) for g in execution.result.gates) == expected_edges
        heldout["episodes"] += 1
        heldout["top1"].append(int(diag.selected_rank == 1))
        heldout["rank"].append(diag.selected_rank)
        if diag.hard_negative_margin is not None:
            heldout["margin"].append(diag.hard_negative_margin)
        heldout["exact_edge_execution"] += int(exact_edges)
        heldout["active_edges"].append(execution.work.active_edges)
        heldout["candidate_edges"].append(execution.work.candidate_edges_metadata)

    heldout["updates"] = router.updates
    return _summarise(heldout)


def _collision_probe(seed: int):
    task = build_topology_task(seed, n_candidates=N_CANDIDATES)
    # Legacy router sees only the shared descriptor; its candidate scores
    # therefore collide exactly.
    legacy_state = PersistentStore(StateConfig(seed=seed))
    legacy = RepresentationEnergyRouter(
        EnergyRouterConfig(seed=seed, dim=INPUT_DIM, max_state_slots=0)
    )
    legacy_scores = legacy.score(task.public(), legacy_state, task.candidates)

    explicit = ExplicitProgramEnergyRouter()
    observations = [
        explicit.candidate_observation(candidate) for candidate in task.candidates
    ]
    return {
        "legacy_unique_candidate_scores": len(set(legacy_scores)),
        "legacy_all_scores_equal": len(set(legacy_scores)) == 1,
        "explicit_unique_topology_observations": len(set(observations)),
        "candidate_count": len(task.candidates),
    }


def _analytic(seed: int):
    router = ExplicitProgramEnergyRouter()
    router.set_analytic_relation()
    state = PersistentStore(StateConfig(seed=seed))
    top1 = []
    margins = []
    for step in range(HELDOUT_EPISODES):
        task = build_topology_task(
            seed * 2000003 + 100000 + step * 7919 + 23,
            n_candidates=N_CANDIDATES,
            step=TRAIN_EPISODES + step,
        )
        decision = router.route(task.public(), state, task.candidates)
        diag = router.last_diagnostics
        top1.append(decision.selected == task.target_action)
        margins.append(diag.hard_negative_margin)
    return {
        "top1_recall": statistics.fmean(top1),
        "hard_negative_margin_mean": statistics.fmean(margins),
    }


def _oracle(seed: int):
    top1 = []
    for step in range(HELDOUT_EPISODES):
        task = build_topology_task(
            seed * 3000017 + 200000 + step * 7919 + 31,
            n_candidates=N_CANDIDATES,
            step=step,
        )
        top1.append(task.target_action in range(len(task.candidates)))
    return {"success_rate": statistics.fmean(top1)}


def main():
    result = {
        "protocol": {
            "name": "TACOSM-IDENTIFIABILITY-REP-002",
            "seeds": list(SEEDS),
            "train_episodes_per_seed": TRAIN_EPISODES,
            "heldout_episodes_per_seed": HELDOUT_EPISODES,
            "candidates": N_CANDIDATES,
            "input_dim": INPUT_DIM,
            "latent_dim": LATENT_DIM,
            "learning_rate": 0.01,
            "margin": 0.1,
            "executor": "explicit_graph_exact",
        },
        "identifiability_probe": _collision_probe(17),
        "condition_A_analytic": {},
        "condition_B_learned": {},
        "condition_C_no_learning": {},
        "condition_D_oracle": {},
    }

    for seed in SEEDS:
        result["condition_A_analytic"][str(seed)] = _analytic(seed)
        result["condition_B_learned"][str(seed)] = _run_arm(seed, learned=True)
        result["condition_C_no_learning"][str(seed)] = _run_arm(seed, learned=False)
        result["condition_D_oracle"][str(seed)] = _oracle(seed)

    out = Path("artifacts/TACOSM-IDENTIFIABILITY-REP-002.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
