#!/usr/bin/env python3
"""Efficient execution of TACOSM-C5-COMPOSITION-BUDGET-AUDIT-001.

The scientific protocol is unchanged. This runner avoids recomputing the same
LSH index once per execution budget and restricts the budget curve to the
registered M=1024 endpoint. Dense rank and empirical L90 remain evaluated for
all registered M levels.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm.c5_composition_budget_audit import (
    BUDGETS,
    BOOTSTRAP_SEED,
    CALIBRATION_TRIALS,
    DISTILL_STEPS,
    M_LEVELS,
    REP_LEVELS,
    SEEDS,
    BASELINE_STEPS,
    TRAIN_STEPS,
    AnalyticRelationalRouter,
    BitwiseLateInteractionRouter,
    ProductKeyRelationRouter,
    analytic_relational_initialize,
    build_router,
    dense_rank,
    empirical_l90,
    exact_relation_hash_lookup,
    fit_power,
    rank_summary,
    rank_op,
    stable_arm_seed,
    trials_for_split,
    verify_split_disjointness,
    operand_hamming_rank,
    sparse_adaptive_run,
)
from tac_osm.c5_noisy_full_phase import ORLSHIndex
from tac_osm.c5_persistent_relational_loop import (
    DIM,
    LATENT_DIM,
    OPS,
    PersistentRelationCASMVerifier,
)
 
DISTILL_TABLES = 8
DISTILL_K = 16
LSH_MAX_TABLES = 128


def fast_budget_curve(router, trial, state, *, tables: int, k: int, seed: int):
    index = ORLSHIndex(
        latent_dim=LATENT_DIM,
        bits=max(1, math.ceil(math.log2(len(trial.candidates)))),
        tables=tables,
        cap=tables,
        seed=seed,
    )
    index.build(router.candidate_embeddings(trial.candidates))
    qz = router.encode_query(trial.query, state)
    lookup = index.lookup(
        qz,
        k=min(k, len(trial.candidates)),
        score=lambda key: router.score_embeddings(qz, index.embeddings[key]),
    )
    positions = {c.key: i for i, c in enumerate(trial.candidates)}
    admitted = [positions[key] for key in lookup.addresses]
    target_rank = (
        admitted.index(trial.target_index) + 1
        if trial.target_index in admitted
        else None
    )
    verifier = PersistentRelationCASMVerifier()
    max_budget = min(max(BUDGETS), len(admitted))
    verified_at = None
    for j, idx in enumerate(admitted[:max_budget], start=1):
        if verifier.execute_and_verify(trial, state, idx).verified:
            verified_at = j
            break
    return [
        {
            "budget": b,
            "success": verified_at is not None and verified_at <= b,
            "executed": min(b, len(admitted)),
            "admission": trial.target_index in admitted,
            "admitted_rank": target_rank,
            "rerank_count": lookup.rerank_count,
            "routing_ops": lookup.hash_ops + lookup.rerank_count * LATENT_DIM,
        }
        for b in BUDGETS
    ]


def evaluate_control(seed: int, router):
    dense_rows = []
    per_op = {}
    l90_curve = []
    mean_exec = []
    k90_curve = []
    split_checks = {}
    for m in M_LEVELS:
        calibration = trials_for_split(seed, m, "calibration", CALIBRATION_TRIALS)
        heldout = trials_for_split(seed, m, "heldout", CALIBRATION_TRIALS)
        split_checks[str(m)] = verify_split_disjointness(seed, m)
        cal_ranks = [dense_rank(router, *x)[0] for x in calibration]
        k90 = max(1, min(m, rank_summary(cal_ranks, BOOTSTRAP_SEED + seed + m)["P90_rank"]))
        l90, curve = empirical_l90(router, seed, m, calibration)
        ranks = []
        op_rows = {op: [] for op in range(4)}
        hamming = []
        for trial, state in heldout:
            rank, _ = dense_rank(router, trial, state)
            ranks.append(rank)
            hamming.append(operand_hamming_rank(trial, state))
            op_rows[rank_op(trial, state)].append(rank)
        row = rank_summary(ranks, BOOTSTRAP_SEED + seed * 100 + m)
        row.update({
            "M": m,
            "K90_calibration": k90,
            "L90_calibration": l90,
            "mean_expected_dense_executions": statistics.fmean(ranks),
            "operand_hamming_top1": statistics.fmean(r == 1 for r in hamming),
        })
        dense_rows.append(row)
        mean_exec.append(statistics.fmean(ranks))
        k90_curve.append(k90)
        l90_curve.append({"M": m, "L90": l90, "curve": curve})
        per_op[str(m)] = {
            OPS[op]: {
                "n": len(vals),
                "top1": statistics.fmean(v == 1 for v in vals) if vals else 0.0,
                "mean_rank": statistics.fmean(vals) if vals else None,
            }
            for op, vals in op_rows.items()
        }

        if m == 1024:
            budget_rows = []
            for trial, state in heldout:
                budget_rows.extend(
                    fast_budget_curve(
                        router,
                        trial,
                        state,
                        tables=l90,
                        k=k90,
                        seed=seed * 700_003 + m * 31,
                    )
                )
            budgets = [
                {
                    "budget": b,
                    "success": statistics.fmean(r["success"] for r in budget_rows if r["budget"] == b),
                    "mean_executed": statistics.fmean(r["executed"] for r in budget_rows if r["budget"] == b),
                    "admission": statistics.fmean(r["admission"] for r in budget_rows if r["budget"] == b),
                    "mean_admitted_rank": (
                        statistics.fmean(
                            r["admitted_rank"]
                            for r in budget_rows
                            if r["budget"] == b and r["admitted_rank"] is not None
                        )
                        if any(r["budget"] == b and r["admitted_rank"] is not None for r in budget_rows)
                        else None
                    ),
                    "mean_routing_ops": statistics.fmean(r["routing_ops"] for r in budget_rows if r["budget"] == b),
                }
                for b in BUDGETS
            ]
            adaptive_rows = [
                sparse_adaptive_run(
                    router,
                    trial,
                    state,
                    tables=l90,
                    k=k90,
                    seed=seed * 900_001 + m,
                )
                for trial, state in heldout
            ]
            sparse = {
                "M": 1024,
                "K90": k90,
                "L90": l90,
                "budgets": budgets,
                "adaptive": {
                    "success": statistics.fmean(r["success"] for r in adaptive_rows),
                    "mean_executed": statistics.fmean(r["executed"] for r in adaptive_rows),
                    "mean_budget": statistics.fmean(r["budget"] for r in adaptive_rows),
                    "admission": statistics.fmean(r["target_admitted"] for r in adaptive_rows),
                },
            }
        else:
            sparse = None
    return {
        "dense": dense_rows,
        "per_op": per_op,
        "split_checks": split_checks,
        "sparse_M1024": sparse,
        "L90": l90_curve,
        "K90": k90_curve,
        "mean_exec": mean_exec,
        "hamming_ref": {
            str(m): dense_rows[i]["operand_hamming_top1"]
            for i, m in enumerate(M_LEVELS)
        },
        "mean_exec_gamma": fit_power(M_LEVELS, mean_exec),
    }


def bootstrap_per_seed_control(control_results):
    out = {}
    for seed in SEEDS:
        row = control_results[str(seed)]["dense"][4]
        out[str(seed)] = {
            "mean_rank": row["mean_rank"],
            "mean_rank_ci95": row["mean_rank_ci95"],
            "P90_rank": row["P90_rank"],
            "P90_rank_ci95": row["P90_rank_ci95"],
            "top1": row["top1"],
        }
    return out


def main():
    arms = (
        "control",
        "analytic_init",
        "product_key",
        "late_interaction",
        "outcome_distill",
        "product_key_distill",
        "late_interaction_distill",
    )
    result = {
        "protocol": {
            "id": "TACOSM-C5-COMPOSITION-BUDGET-AUDIT-001",
            "runner": "run_c5_composition_budget_audit_fast_001.py",
            "status": "measured",
            "M_levels": list(M_LEVELS),
            "representation_M": list(REP_LEVELS),
            "seeds": list(SEEDS),
            "training_updates": TRAIN_STEPS,
            "calibration_trials": CALIBRATION_TRIALS,
            "heldout_trials": CALIBRATION_TRIALS,
            "budgets": list(BUDGETS),
        },
        "diagnostics": {
            "baseline_query_tower_hidden_nonlinearity": False,
            "baseline_explicit_pairwise_feature": True,
            "baseline_relation_basis": ["a", "b", "a*b"],
            "xor_xnor_have_explicit_pairwise_basis": True,
            "calibration_heldout_are_distinct_by_manifest": True,
            "budget_curve_scope": "M=1024 primary endpoint; one LSH build per held-out trial",
        },
        "exact_reference": {
            "capability_top1": 1.0,
            "query_relation_computations": 4,
            "query_hash_lookups": 4,
            "runtime_in_M": "O(1) after descriptor-to-index hash tables are built",
        },
        "arms": {},
        "persistent_reset": {},
    }

    control_results = {}
    for seed in SEEDS:
        control_results[str(seed)] = evaluate_control(seed, build_router("control", seed))
    result["arms"]["control"] = control_results

    for arm in arms:
        if arm == "control":
            continue
        result["arms"][arm] = {}
        for seed in SEEDS:
            router = build_router(arm, seed)
            calibration = trials_for_split(seed, 1024, "calibration", CALIBRATION_TRIALS)
            heldout = trials_for_split(seed, 1024, "heldout", CALIBRATION_TRIALS)
            cal_ranks = [dense_rank(router, *x)[0] for x in calibration]
            k90 = max(1, min(1024, rank_summary(cal_ranks, BOOTSTRAP_SEED + seed + stable_arm_seed(arm) % 1000)["P90_rank"]))
            l90, _ = empirical_l90(router, seed, 1024, calibration)
            ranks = [dense_rank(router, trial, state)[0] for trial, state in heldout]
            row = rank_summary(
                ranks,
                BOOTSTRAP_SEED + seed * 100 + stable_arm_seed(arm) % 1000,
            )
            budget_rows = []
            for trial, state in heldout:
                budget_rows.extend(
                    fast_budget_curve(
                        router,
                        trial,
                        state,
                        tables=l90,
                        k=k90,
                        seed=seed * 700_003 + 1024 * 31,
                    )
                )
            budgets = [
                {
                    "budget": b,
                    "success": statistics.fmean(r["success"] for r in budget_rows if r["budget"] == b),
                    "mean_executed": statistics.fmean(r["executed"] for r in budget_rows if r["budget"] == b),
                    "admission": statistics.fmean(r["admission"] for r in budget_rows if r["budget"] == b),
                }
                for b in BUDGETS
            ]
            adaptive_rows = [
                sparse_adaptive_run(
                    router,
                    trial,
                    state,
                    tables=l90,
                    k=k90,
                    seed=seed * 900_001 + 1024,
                )
                for trial, state in heldout
            ]
            row.update({"M": 1024, "K90": k90, "L90": l90, "budgets": budgets})
            row["adaptive"] = {
                "success": statistics.fmean(r["success"] for r in adaptive_rows),
                "mean_executed": statistics.fmean(r["executed"] for r in adaptive_rows),
                "mean_budget": statistics.fmean(r["budget"] for r in adaptive_rows),
            }
            result["arms"][arm][str(seed)] = {"1024": row}

    for arm in arms:
        result["persistent_reset"][arm] = {}
        for seed in SEEDS:
            router = build_router(arm, seed)
            heldout = trials_for_split(seed, 1024, "heldout", CALIBRATION_TRIALS)
            persistent = []
            reset = []
            per_op = {name: [] for name in OPS}
            for trial, state in heldout:
                p_rank, _ = dense_rank(router, trial, state)
                persistent.append(p_rank)
                read = state.read(trial.query)
                persisted_op = rank_op(trial, state)
                state.clear()
                qz = router.encode_query(trial.query, state)
                scores = [router.score_embeddings(qz, router.encode_candidate(c)) for c in trial.candidates]
                order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                reset.append(order.index(trial.target_index) + 1)
                if read.values:
                    per_op[OPS[persisted_op]].append(p_rank)
            result["persistent_reset"][arm][str(seed)] = {
                "persistent_top1": statistics.fmean(x == 1 for x in persistent),
                "reset_top1": statistics.fmean(x == 1 for x in reset),
                "persistent_P90": sorted(persistent)[min(len(persistent)-1, math.ceil(.90*len(persistent))-1)],
                "reset_P90": sorted(reset)[min(len(reset)-1, math.ceil(.90*len(reset))-1)],
                "per_op_top1": {
                    name: statistics.fmean(v == 1 for v in vals) if vals else None
                    for name, vals in per_op.items()
                },
            }

    pooled_L90 = [
        {"M": m, "L90": statistics.fmean(control_results[str(s)]["L90"][i]["L90"] for s in SEEDS)}
        for i, m in enumerate(M_LEVELS)
    ]
    pooled_exec = [
        {"M": m, "mean_executions": statistics.fmean(control_results[str(s)]["mean_exec"][i] for s in SEEDS)}
        for i, m in enumerate(M_LEVELS)
    ]
    pooled_P90 = [
        {"M": m, "P90": statistics.fmean(control_results[str(s)]["dense"][i]["P90_rank"] for s in SEEDS)}
        for i, m in enumerate(M_LEVELS)
    ]
    pooled_K90 = [
        {"M": m, "K90": statistics.fmean(control_results[str(s)]["K90"][i] for s in SEEDS)}
        for i, m in enumerate(M_LEVELS)
    ]
    result["control_scaling"] = {
        "pooled_L90": pooled_L90,
        "L90_fit_gamma": fit_power(M_LEVELS, [x["L90"] for x in pooled_L90]),
        "pooled_mean_expected_dense_executions": pooled_exec,
        "mean_execution_fit_gamma": fit_power(M_LEVELS, [x["mean_executions"] for x in pooled_exec]),
        "pooled_P90": pooled_P90,
        "pooled_K90": pooled_K90,
        "L90_definition": "smallest integer L in [1,128] with >=0.90 calibration target collision",
    }
    result["bootstrap_M1024"] = bootstrap_per_seed_control(control_results)
    for arm in arms[1:]:
        result["bootstrap_M1024"][arm] = {
            str(seed): {
                "mean_rank": result["arms"][arm][str(seed)]["1024"]["mean_rank"],
                "mean_rank_ci95": result["arms"][arm][str(seed)]["mean_rank_ci95"],
                "P90_rank": result["arms"][arm][str(seed)]["P90_rank"],
                "P90_rank_ci95": result["arms"][arm][str(seed)]["P90_rank_ci95"],
                "top1": result["arms"][arm][str(seed)]["top1"],
            }
            for seed in SEEDS
        }

    split_checks = [
        control_results[str(s)]["split_checks"][str(m)]
        for s in SEEDS for m in M_LEVELS
    ]
    result["scope"] = {
        "semantic_language_claim": False,
        "complexity_theorem": False,
        "causal_learning_claim": False,
        "exact_reference_is_task_semantic_upper_bound": True,
        "heldout_selection": False,
        "calibration_heldout_disjoint_verified": all(x["intersection_count"] == 0 for x in split_checks),
    }

    out = Path("artifacts/TACOSM-C5-COMPOSITION-BUDGET-AUDIT-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "artifact": str(out),
        "disjoint": result["scope"]["calibration_heldout_disjoint_verified"],
        "pooled_L90": result["control_scaling"]["pooled_L90"],
        "mean_expected_executions": result["control_scaling"]["pooled_mean_expected_dense_executions"],
        "M1024_control": {
            s: {
                "top1": result["arms"]["control"][s]["dense"][4]["top1"],
                "P90": result["arms"]["control"][s]["dense"][4]["P90_rank"],
                "mean_rank": result["arms"]["control"][s]["dense"][4]["mean_rank"],
            } for s in map(str, SEEDS)
        },
    }, indent=2))

if __name__ == "__main__":
    main()
