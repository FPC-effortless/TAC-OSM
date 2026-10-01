#!/usr/bin/env python3
"""TACOSM-FUSED-FRONTIER-004: larger-M absolute-budget scaling.

Frontier 003 selected 4 factors x 16 cells, beam 4 as the lowest-cost tested
geometry with execution capability close to the 3-factor control. Frontier 004
tests whether that geometry remains sparse at larger populations.

Arms:
  control_3f16_b7
  candidate_4f16_b4

M = 512, 1024, 2048
B = 16, 32, 64
seeds = 10, 11, 12
StructMeans family gate + PST remain fixed.
"""
from __future__ import annotations

import json
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_fused_frontier_002 import (
    DIM,
    EVAL_SINGLE,
    SEEDS,
    build_structural,
    make_library,
    q_for,
)
from tac_osm.fused_architecture import candidate_operator
from tac_osm.operator_learning import apply_operator
from tac_osm.state import PersistentStore, StateConfig
from tac_osm.structural_product_key import StructuralProductKeyRouter

M_LEVELS = (512, 1024, 2048)
BUDGETS = (16, 32, 64)
CONFIGS = (
    ("3f16_b7", 3, 16, 7),
    ("4f16_b4", 4, 16, 4),
)
CLUSTER_BUDGET = 2


def unique_single_tasks(candidates, seed):
    """Construct exactly the same identifiable task family without an O(M) scan.

    For a held-out toggle operator, state XOR mask is a bijection over states:
    two distinct toggle masks cannot yield the same goal for the same state.
    Requiring at least one 0->1 and one 1->0 changed bit additionally excludes
    set1/set0 as alternative explanations. Therefore every sampled task in
    this family is uniquely identifiable by construction.
    """
    rng = random.Random(seed + 901)
    held_toggle = [
        i for i in range(48, len(candidates))
        if candidate_operator(candidates[i]).kind == "toggle"
    ]
    if not held_toggle:
        raise RuntimeError("no held-out toggle operators")
    tasks = []
    while len(tasks) < EVAL_SINGLE:
        target = held_toggle[rng.randrange(len(held_toggle))]
        op = candidate_operator(candidates[target])
        state = tuple(rng.randrange(2) for _ in range(DIM))
        changed = [i for i, bit in enumerate(op.mask) if bit]
        if any(state[i] == 0 for i in changed) and any(state[i] == 1 for i in changed):
            goal = apply_operator(state, op)
            tasks.append((state, goal, target))
    return tasks


def evaluate(
    candidates,
    rows,
    sm,
    pst,
    state,
    seed,
    budget,
    name,
    factor_count,
    factor_size,
    factor_beam,
):
    router = StructuralProductKeyRouter(
        factor_count=factor_count,
        factor_size=factor_size,
        factor_beam=factor_beam,
        max_shortlist=64,
    )
    router.build(candidates, training_records=rows, structmeans=sm)
    tasks = unique_single_tasks(candidates, seed)

    execution = []
    exact = []
    admission = []
    final_admission = []
    factor_macs = []
    pair_ops = []
    exact_addresses = []
    exact_rerank_ops = []
    pst_ops = []
    gate_ops = []

    for step, (before, goal, target) in enumerate(tasks):
        route = router.route(
            q_for(before, goal, step),
            state,
            candidates,
            training_records=rows,
            budget=budget,
            cluster_budget=CLUSTER_BUDGET,
            pst=pst,
        )
        selected = route.decision.selected
        candidate = candidates[selected] if selected >= 0 else None
        admission.append(int(target in set(route.admitted_indices)))
        final_admission.append(int(target in set(route.final_indices)))
        execution.append(
            int(
                candidate is not None
                and apply_operator(before, candidate_operator(candidate)) == goal
            )
        )
        exact.append(int(selected == target))
        d = route.diagnostics
        factor_macs.append(d.factor_score_macs)
        pair_ops.append(d.pair_generation_ops)
        exact_addresses.append(d.internal_exact_addresses_scored)
        exact_rerank_ops.append(d.internal_exact_rerank_ops)
        pst_ops.append(d.pst_prediction_ops)
        gate_ops.append(d.structural_gate_ops)

    total = [
        f + p + x + pp + g
        for f, p, x, pp, g in zip(
            factor_macs, pair_ops, exact_rerank_ops, pst_ops, gate_ops
        )
    ]
    return {
        "M": len(candidates),
        "budget": budget,
        "seed": seed,
        "config": name,
        "factor_count": factor_count,
        "factor_size": factor_size,
        "factor_beam": factor_beam,
        "execution_success": statistics.fmean(execution),
        "exact_target_success": statistics.fmean(exact),
        "pkm_admission_recall": statistics.fmean(admission),
        "final_admission_recall": statistics.fmean(final_admission),
        "internal_exact_addresses_scored": statistics.fmean(exact_addresses),
        "internal_exact_addresses_scored_over_M": statistics.fmean(exact_addresses) / len(candidates),
        "internal_exact_rerank_ops": statistics.fmean(exact_rerank_ops),
        "factor_score_macs": statistics.fmean(factor_macs),
        "pair_generation_ops": statistics.fmean(pair_ops),
        "pst_prediction_ops": statistics.fmean(pst_ops),
        "structural_gate_ops": statistics.fmean(gate_ops),
        "route_ops_excluding_build": statistics.fmean(total),
        "build_macs": router.index.build_diagnostics.total_build_macs,
        "build_macs_amortized_over_eval": router.index.build_diagnostics.total_build_macs / EVAL_SINGLE,
        "max_cell_size": router.index.build_diagnostics.max_cell_size,
        "nonempty_cells": router.index.build_diagnostics.nonempty_cells,
    }


def run_cell(args):
    seed, m, name, fc, fs, fb, budget = args
    candidates = make_library(seed, m)
    rows, store, sm, pst, _ = build_structural(candidates, seed)
    state = PersistentStore(StateConfig(seed=seed + 92000, n_slots=128))
    return evaluate(
        candidates, rows, sm, pst, state, seed,
        budget, name, fc, fs, fb
    )


def main():
    jobs = [
        (seed, m, name, fc, fs, fb, budget)
        for seed in SEEDS
        for m in M_LEVELS
        for name, fc, fs, fb in CONFIGS
        for budget in BUDGETS
    ]

    # Independent experimental cells are embarrassingly parallel. Results are
    # sorted by the same canonical key before aggregation to preserve stable
    # artifact ordering despite parallel completion order.
    with ProcessPoolExecutor(max_workers=2) as executor:
        rows_out = list(executor.map(run_cell, jobs))

    rows_out.sort(key=lambda r: (int(r["M"]), str(r["config"]), int(r["budget"]), int(r["seed"])))

    metrics = (
        "execution_success",
        "exact_target_success",
        "pkm_admission_recall",
        "final_admission_recall",
        "internal_exact_addresses_scored",
        "internal_exact_addresses_scored_over_M",
        "internal_exact_rerank_ops",
        "factor_score_macs",
        "pair_generation_ops",
        "pst_prediction_ops",
        "structural_gate_ops",
        "route_ops_excluding_build",
        "build_macs",
        "build_macs_amortized_over_eval",
        "max_cell_size",
        "nonempty_cells",
    )
    grouped = {}
    for row in rows_out:
        grouped.setdefault(
            (int(row["M"]), str(row["config"]), int(row["budget"])), []
        ).append(row)

    pooled = []
    for key, members in sorted(grouped.items()):
        m, config, budget = key
        out = {"M": m, "config": config, "budget": budget, "seeds": len(members)}
        for metric in metrics:
            out[metric] = statistics.fmean(float(x[metric]) for x in members)
        pooled.append(out)

    result = {
        "protocol": {
            "name": "TACOSM-FUSED-FRONTIER-004",
            "question": "Does 4f16/b4 remain sparse and capable at M=1024/2048 under fixed absolute decision budgets?",
            "M_levels": list(M_LEVELS),
            "absolute_budgets": list(BUDGETS),
            "seeds": list(SEEDS),
            "configs": [
                {"name": n, "factor_count": fc, "factor_size": fs, "factor_beam": fb}
                for n, fc, fs, fb in CONFIGS
            ],
            "cluster_budget": CLUSTER_BUDGET,
            "task_protocol": "unique-target held-out toggle tasks from frontier 002/003",
            "route_cost": "factor score + cell expansion + exact address score + PST + family gate",
            "C5_rule": "No bounded/sublinear claim unless total routing work and internal address work stop increasing with M."
        },
        "pooled": pooled,
        "runs": rows_out,
    }

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/TACOSM-FUSED-FRONTIER-004.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
