#!/usr/bin/env python3
"""TACOSM-FUSED-FRONTIER-003: factor-geometry scaling of structural routing.

Frontier 002 established that a deterministic operator descriptor restores
held-out admission but leaves a dense exact-address region. Frontier 003 keeps
that representation, StructMeans family gate, and PST fixed while varying
product-key factor geometry/beam.

Primary question:
  Can factorization reduce total internal exact-address work without losing
  the execution capability recovered in frontier 002?

Configurations:
  3f x 16, beam 7  (control)
  4f x 16, beam 4
  4f x 16, beam 5
  4f x 16, beam 7
  5f x 8,  beam 4

All use StructMeans family admission (cluster budget 2) and PST reranking.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_fused_frontier_002 import (
    BUDGETS,
    DIM,
    EVAL_SINGLE,
    M_LEVELS,
    SEEDS,
    build_structural,
    make_library,
    q_for,
    unique_single_tasks,
)
from tac_osm.fused_architecture import candidate_operator
from tac_osm.structural_product_key import StructuralProductKeyRouter

CONFIGS = (
    ("3f16_b7", 3, 16, 7),
    ("4f16_b4", 4, 16, 4),
    ("4f16_b5", 4, 16, 5),
    ("4f16_b7", 4, 16, 7),
    ("5f8_b4", 5, 8, 4),
)
CLUSTER_BUDGET = 2


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
        max_shortlist=32,
    )
    router.build(candidates, training_records=rows, structmeans=sm)
    build_macs = router.index.build_diagnostics.total_build_macs
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
        out = router.route(
            q_for(before, goal, step),
            state,
            candidates,
            training_records=rows,
            budget=budget,
            cluster_budget=CLUSTER_BUDGET,
            pst=pst,
        )
        selected = out.decision.selected
        selected_candidate = candidates[selected] if selected >= 0 else None
        admission.append(int(target in set(out.admitted_indices)))
        final_admission.append(int(target in set(out.final_indices)))
        execution.append(
            int(
                selected_candidate is not None
                and tuple(
                    __import__("tac_osm.operator_learning", fromlist=["apply_operator"]).apply_operator(
                        before, candidate_operator(selected_candidate)
                    )
                ) == tuple(goal)
            )
        )
        exact.append(int(selected == target))
        factor_macs.append(out.diagnostics.factor_score_macs)
        pair_ops.append(out.diagnostics.pair_generation_ops)
        exact_addresses.append(out.diagnostics.internal_exact_addresses_scored)
        exact_rerank_ops.append(out.diagnostics.internal_exact_rerank_ops)
        pst_ops.append(out.diagnostics.pst_prediction_ops)
        gate_ops.append(out.diagnostics.structural_gate_ops)

    route_ops = [
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
        "route_ops_excluding_build": statistics.fmean(route_ops),
        "build_macs": build_macs,
        "build_macs_amortized_over_eval": build_macs / EVAL_SINGLE,
    }


def main():
    all_rows = []
    for seed in SEEDS:
        for m in M_LEVELS:
            candidates = make_library(seed, m)
            rows, store, sm, pst, structural_base = build_structural(candidates, seed)
            state = __import__(
                "tac_osm.state", fromlist=["PersistentStore", "StateConfig"]
            ).PersistentStore(
                __import__("tac_osm.state", fromlist=["StateConfig"]).StateConfig(
                    seed=seed + 91000, n_slots=128
                )
            )
            for name, factor_count, factor_size, factor_beam in CONFIGS:
                for budget in BUDGETS:
                    all_rows.append(
                        evaluate(
                            candidates, rows, sm, pst, state, seed, budget,
                            name, factor_count, factor_size, factor_beam
                        )
                    )

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
    )
    grouped = {}
    for row in all_rows:
        grouped.setdefault(
            (int(row["M"]), str(row["config"]), int(row["budget"])), []
        ).append(row)

    pooled = []
    for (m, config, budget), members in sorted(grouped.items()):
        out = {"M": m, "config": config, "budget": budget, "seeds": len(members)}
        for metric in metrics:
            out[metric] = statistics.fmean(float(x[metric]) for x in members)
        pooled.append(out)

    result = {
        "protocol": {
            "name": "TACOSM-FUSED-FRONTIER-003",
            "question": "Can factor geometry reduce exact-address routing work while retaining the unique-target execution capability recovered in frontier 002?",
            "M_levels": list(M_LEVELS),
            "absolute_budgets": list(BUDGETS),
            "seeds": list(SEEDS),
            "configs": [
                {"name": n, "factor_count": fc, "factor_size": fs, "factor_beam": fb}
                for n, fc, fs, fb in CONFIGS
            ],
            "cluster_budget": CLUSTER_BUDGET,
            "held_out_protocol": "unique-target held-out toggle tasks",
            "primary_cost": "route_ops_excluding_build plus internal_exact_addresses_scored_over_M",
            "C5_rule": "No bounded/sublinear claim unless internal address work and total route arithmetic both stop growing with M.",
        },
        "pooled": pooled,
        "runs": all_rows,
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/TACOSM-FUSED-FRONTIER-003.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
