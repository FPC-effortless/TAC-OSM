#!/usr/bin/env python3
"""Run preregistered TACOSM-SUCCESSOR-REP-001.

This runner intentionally keeps the analytic witness (A) separate from the
trained capability arms (B-E). It writes machine-readable JSON so the result
can be audited without reconstructing metrics from prose.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm.energy_router import EnergyRouterConfig, RepresentationEnergyRouter
from tac_osm.environment import (
    build_lookup_task,
    build_replay_task,
    build_relational_task,
)
from tac_osm.executor import relevance_program
from tac_osm.explicit_executor import ExplicitExecutorConfig, ExplicitGraphExecutor
from tac_osm.state import PersistentStore, StateConfig


SEEDS = (0, 1, 2, 3, 4)
TRAIN_EPISODES = 128
HELDOUT_EPISODES = 128
DIM = 8
LATENT = 8
N_CANDIDATES = 8
FAMILIES = ("relational", "state_lookup", "replay")


def make_task(family: str, seed: int, state: PersistentStore, step: int):
    if family == "relational":
        return build_relational_task(
            seed, dim=DIM, n_candidates=N_CANDIDATES, step=step
        )
    if family == "state_lookup":
        return build_lookup_task(
            seed, state, dim=DIM, n_candidates=N_CANDIDATES, step=step
        )
    if family == "replay":
        return build_replay_task(
            seed, state, dim=DIM, n_candidates=N_CANDIDATES, step=step
        )
    raise ValueError(family)


def reference_for(task):
    detail = task.detail
    if task.family == "relational":
        return tuple(detail.query_bits)
    return tuple(detail.written_bits)


def marks_for(task):
    return tuple(i for i, bit in enumerate(task.query.context) if bit == 1)


def execute(task, selected: int, executor: ExplicitGraphExecutor):
    reference = reference_for(task)
    candidate = task.candidates[selected]
    program = relevance_program(
        reference,
        candidate.descriptor,
        marks_for(task),
        max_nodes=executor.config.max_nodes,
    )
    result = executor.execute(
        __import__("tac_osm").Structure(
            key=candidate.key,
            spec=program,
            provenance="REP-001",
        ),
        [],
    )
    expected = 1.0 if selected == task.target_action else 0.0
    return result, expected, program


def record_route(router, task, state, target):
    decision = router.route(task.public(), state, task.candidates)
    diag = router.diagnostics_with_target(
        task.public(), state, task.candidates, target
    )
    return decision, diag


def empty_metric():
    return {
        "episodes": 0,
        "successes": 0,
        "exact_execution_successes": 0,
        "target_rank": [],
        "target_recall_at_1": [],
        "selected_energy": [],
        "hard_negative_margin": [],
        "candidate_coverage": [],
        "router_macs": [],
        "state_slots_inspected": [],
        "state_pool_size": [],
        "active_nodes": [],
        "active_edges": [],
        "candidate_edges": [],
        "updates": 0,
    }


def add_metric(m, task, diag, decision, execution, expected, program, learned_update):
    m["episodes"] += 1
    success = decision.selected == task.target_action
    exact_success = execution.output == expected
    m["successes"] += int(success)
    m["exact_execution_successes"] += int(exact_success)
    m["target_rank"].append(diag.selected_rank)
    m["target_recall_at_1"].append(int(diag.selected_rank == 1))
    m["selected_energy"].append(diag.selected_energy)
    if diag.hard_negative_margin is not None:
        m["hard_negative_margin"].append(diag.hard_negative_margin)
    m["candidate_coverage"].append(diag.candidate_coverage)
    m["router_macs"].append(diag.total_macs)
    m["state_slots_inspected"].append(diag.state_inspected_slots)
    m["state_pool_size"].append(diag.state_pool_size)
    m["active_nodes"].append(program.active_count)
    m["active_edges"].append(len(program.true_edges))
    m["candidate_edges"].append(len(program.candidate_edges))
    m["updates"] += int(learned_update)


def summarise(m):
    def mean(xs):
        return statistics.fmean(xs) if xs else None

    ranks = m["target_rank"]
    return {
        "episodes": m["episodes"],
        "success_rate": m["successes"] / m["episodes"] if m["episodes"] else 0.0,
        "exact_execution_success_rate": (
            m["exact_execution_successes"] / m["episodes"] if m["episodes"] else 0.0
        ),
        "top1_recall": mean(m["target_recall_at_1"]),
        "mean_target_rank": mean(ranks),
        "selected_energy_mean": mean(m["selected_energy"]),
        "hard_negative_margin_mean": mean(m["hard_negative_margin"]),
        "candidate_coverage_mean": mean(m["candidate_coverage"]),
        "router_macs_mean": mean(m["router_macs"]),
        "state_slots_inspected_mean": mean(m["state_slots_inspected"]),
        "state_pool_size_mean": mean(m["state_pool_size"]),
        "active_nodes_mean": mean(m["active_nodes"]),
        "active_edges_mean": mean(m["active_edges"]),
        "candidate_edges_mean": mean(m["candidate_edges"]),
        "updates": m["updates"],
    }


def run_arm(seed: int, *, learned: bool, train: bool):
    state = PersistentStore(StateConfig(seed=seed, n_slots=2048))
    router = RepresentationEnergyRouter(
        EnergyRouterConfig(
            input_dim=DIM,
            latent_dim=LATENT,
            learning_rate=0.01,
            margin=0.1,
            top_k=1,
            seed=seed,
        )
    )
    executor = ExplicitGraphExecutor(ExplicitExecutorConfig(mode="exact"))
    metrics = {family: empty_metric() for family in FAMILIES}

    if train:
        for step in range(TRAIN_EPISODES):
            family = FAMILIES[step % len(FAMILIES)]
            task_seed = seed * 1000003 + step * 7919 + 17
            task = make_task(family, task_seed, state, step)
            decision, diag = record_route(router, task, state, task.target_action)
            execution, expected, program = execute(task, decision.selected, executor)
            success = decision.selected == task.target_action
            if learned:
                router.learn_from_outcome(
                    task.public(),
                    state,
                    task.candidates,
                    decision.selected,
                    success=success,
                    scores=decision.scores,
                )
            add_metric(
                metrics[family], task, diag, decision, execution, expected, program, learned and success
            )

    heldout = {family: empty_metric() for family in FAMILIES}
    for step in range(HELDOUT_EPISODES):
        family = FAMILIES[step % len(FAMILIES)]
        task_seed = seed * 2000003 + 100000 + step * 7919 + 23
        task = make_task(family, task_seed, state, TRAIN_EPISODES + step)
        decision, diag = record_route(router, task, state, task.target_action)
        execution, expected, program = execute(task, decision.selected, executor)
        add_metric(
            heldout[family], task, diag, decision, execution, expected, program, False
        )

    return {
        "train": {family: summarise(metrics[family]) for family in FAMILIES},
        "heldout": {family: summarise(heldout[family]) for family in FAMILIES},
        "router_updates": router.updates,
    }


def run_no_learning(seed: int):
    # Same task/state construction protocol as B, but no router updates.
    return run_arm(seed, learned=False, train=True)


def run_learned(seed: int):
    return run_arm(seed, learned=True, train=True)


def run_oracle(seed: int):
    state = PersistentStore(StateConfig(seed=seed, n_slots=2048))
    executor = ExplicitGraphExecutor(ExplicitExecutorConfig(mode="exact"))
    metrics = {family: empty_metric() for family in FAMILIES}
    for step in range(HELDOUT_EPISODES):
        family = FAMILIES[step % len(FAMILIES)]
        task_seed = seed * 3000017 + 200000 + step * 7919 + 31
        task = make_task(family, task_seed, state, step)
        selected = task.target_action
        # Oracle arm is deliberately outside the router information boundary.
        router_diag = RepresentationEnergyRouter(
            EnergyRouterConfig(input_dim=DIM, latent_dim=LATENT, seed=seed)
        ).diagnostics_with_target(
            task.public(), state, task.candidates, task.target_action
        )
        decision = type("OracleDecision", (), {"selected": selected})()
        execution, expected, program = execute(task, selected, executor)
        add_metric(
            metrics[family],
            task,
            router_diag,
            decision,
            execution,
            expected,
            program,
            False,
        )
    return {"heldout": {family: summarise(metrics[family]) for family in FAMILIES}}


def run_analytic(seed: int):
    state = PersistentStore(StateConfig(seed=seed, n_slots=64))
    router = RepresentationEnergyRouter(
        EnergyRouterConfig(input_dim=DIM, latent_dim=LATENT, seed=seed)
    )
    router.set_analytic_relation()
    results = {family: [] for family in FAMILIES}
    for family in FAMILIES:
        for i in range(16):
            task = make_task(family, seed * 10007 + i + 1, state, i)
            decision, diag = record_route(router, task, state, task.target_action)
            results[family].append(
                {
                    "selected_target": decision.selected == task.target_action,
                    "target_rank": diag.selected_rank,
                    "hard_negative_margin": diag.hard_negative_margin,
                }
            )
    return results


def main():
    result = {
        "protocol": {
            "name": "TACOSM-SUCCESSOR-REP-001",
            "seeds": list(SEEDS),
            "train_episodes_per_seed": TRAIN_EPISODES,
            "heldout_episodes_per_seed": HELDOUT_EPISODES,
            "candidates": N_CANDIDATES,
            "dim": DIM,
            "latent_dim": LATENT,
            "top_k": 1,
            "executor": "exact",
            "families": list(FAMILIES),
        },
        "condition_A_analytic_witness": {},
        "condition_B_learned": {},
        "condition_C_no_learning": {},
        "condition_D_oracle": {},
    }

    for seed in SEEDS:
        result["condition_A_analytic_witness"][str(seed)] = run_analytic(seed)
        learned = run_learned(seed)
        result["condition_B_learned"][str(seed)] = learned
        # E is the execution-facing view of the same learned held-out arm.
        result.setdefault("condition_E_learned_execution", {})[str(seed)] = learned["heldout"]
        result["condition_C_no_learning"][str(seed)] = run_no_learning(seed)
        result["condition_D_oracle"][str(seed)] = run_oracle(seed)

    out = Path("artifacts/TACOSM-SUCCESSOR-REP-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
