#!/usr/bin/env python3
"""Measure TACOSM-C5-PERSISTENT-RELATIONAL-LOOP-001."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm.c5_persistent_relational_loop import (
    DIM,
    CDLPersistentRelationRouter,
    PersistentRelationConfig,
    PersistentRelationTrial,
    build_population,
    derive_tables,
    empirical_quantile,
    estimate_lsh_geometry,
    fit_power,
    make_episode,
    persistent_oracle_rank,
    HammingOperandBaseline,
    run_sparse,
    apply_relation,
    decode_state_value,
)
from tac_osm.temporal import TemporalPersistentState


CONFIG = PersistentRelationConfig()


def train_router(seed: int) -> CDLPersistentRelationRouter:
    router = CDLPersistentRelationRouter(
        seed=seed,
        learning_rate=CONFIG.learning_rate,
        soft_target_epsilon=CONFIG.soft_target_epsilon,
    )
    # Exhaustive outcome supervision: the environment target is checked only
    # after the candidate action; the router receives the resulting positive index.
    for step in range(CONFIG.train_steps):
        m = CONFIG.train_m_levels[step % len(CONFIG.train_m_levels)]
        trial, state = make_episode(seed=seed + step * 11, step=step, m=m)
        router.train_exhaustive(
            trial.query, state, trial.candidates, trial.target_index
        )
    return router


def fresh_trials(seed: int, m: int, count: int, offset: int):
    return [
        make_episode(seed=seed + offset, step=10000 + offset * 1000 + i, m=m)
        for i in range(count)
    ]


def dense_rank(router, trial, state) -> int:
    order = router.rank(trial.query, state, trial.candidates)
    return order.index(trial.target_index) + 1


def hamming_rank(trial, state) -> int:
    return HammingOperandBaseline().rank(trial, state)


def evaluate_seed(seed: int, router):
    dense_rows = []
    sparse_rows = []
    for m in CONFIG.eval_levels:
        calibration = fresh_trials(seed, m, CONFIG.calibration_trials, offset=7)
        heldout = fresh_trials(seed, m, CONFIG.eval_trials, offset=9)

        cdl_cal = [dense_rank(router, *episode) for episode in calibration]
        k90 = max(1, min(m, empirical_quantile(cdl_cal, 0.90)))
        k95 = max(1, min(m, empirical_quantile(cdl_cal, 0.95)))

        cdl_eval = [dense_rank(router, *episode) for episode in heldout]
        h_eval = [hamming_rank(*episode) for episode in heldout]
        reset_eval = []
        for trial, state in heldout:
            state.clear()
            reset_eval.append(
                router.rank(trial.query, state, trial.candidates).index(trial.target_index) + 1
            )
        dense_rows.append({
            "M": m,
            "mean_rank": statistics.fmean(cdl_eval),
            "P90_rank": empirical_quantile(cdl_eval, 0.90),
            "top1": sum(r == 1 for r in cdl_eval) / len(cdl_eval),
            "hamming_mean_rank": statistics.fmean(h_eval),
            "hamming_top1": sum(r == 1 for r in h_eval) / len(h_eval),
            "reset_mean_rank": statistics.fmean(reset_eval),
            "reset_P90_rank": empirical_quantile(reset_eval, 0.90),
            "reset_top1": sum(r == 1 for r in reset_eval) / len(reset_eval),
            "oracle_rank": persistent_oracle_rank(heldout[0][0]),
            "K90": k90,
            "K95": k95,
        })

        # Build one shared LSH family from the calibration population and
        # derive its table count from measured collision geometry.
        from tac_osm.c5_noisy_full_phase import ORLSHIndex
        calibration_trial, calibration_state = calibration[0]
        bits = max(1, math.ceil(math.log2(m)))
        probe = ORLSHIndex(
            latent_dim=16,
            bits=bits,
            tables=1,
            cap=CONFIG.lsh_cap,
            seed=seed * 1009 + m,
        )
        probe.build(router.candidate_embeddings(calibration_trial.candidates))
        planes = probe.planes
        trials_only = [x[0] for x in calibration]
        states_only = [x[1] for x in calibration]
        p1, p2 = estimate_lsh_geometry(router, planes, trials_only, states_only)
        rho, tables = derive_tables(p1, p2, m)

        sparse = [
            run_sparse(
                router, trial, state, k=k90, tables=tables,
                seed=seed * 9176 + m,
            )
            for trial, state in heldout
        ]
        sparse_rows.append({
            "M": m,
            "K90": k90,
            "K95": k95,
            "p1": p1,
            "p2": p2,
            "rho": rho,
            "tables": tables,
            "admission_recall": statistics.fmean(r["target_admitted"] for r in sparse),
            "final_success": statistics.fmean(r["success"] for r in sparse),
            "first_attempt_success": statistics.fmean(r["first_attempt_success"] for r in sparse),
            "mean_executed": statistics.fmean(r["executed_count"] for r in sparse),
            "mean_repairs": statistics.fmean(r["repair_attempts"] for r in sparse),
            "rerank_fraction": statistics.fmean(r["rerank_count"] for r in sparse) / m,
            "routing_ops": statistics.fmean(r["routing_ops"] for r in sparse),
        })

    # Direct persistence boundary controls at fixed M=128.
    persistence_success = 0
    reset_success = 0
    boundary_pass = 0
    for i in range(CONFIG.eval_trials):
        trial, state = make_episode(seed=seed + 20000, step=i, m=128)
        read = state.read(trial.query)
        boundary_pass += int(bool(read.values))
        if read.values:
            left, right, op = decode_state_value(read.values[0])
            selected = apply_relation(left, right, op)
            persistence_success += int(selected == trial.target_descriptor)
        state.clear()
        read_reset = state.read(trial.query)
        if read_reset.values:
            left, right, op = decode_state_value(read_reset.values[0])
            selected = apply_relation(left, right, op)
            reset_success += int(selected == trial.target_descriptor)

    return {
        "seed": seed,
        "router_updates": router.updates,
        "dense": dense_rows,
        "sparse": sparse_rows,
        "persistence_control_M128": persistence_success / CONFIG.eval_trials,
        "reset_control_M128": reset_success / CONFIG.eval_trials,
        "boundary_read_success": boundary_pass / CONFIG.eval_trials,
    }


def online_loop(seed: int, router: CDLPersistentRelationRouter):
    m = 128
    rows = []
    for step in range(CONFIG.online_trials):
        trial, state = make_episode(seed=seed + 30000, step=step, m=m)
        from tac_osm.c5_noisy_full_phase import ORLSHIndex
        index = ORLSHIndex(
            latent_dim=16, bits=7, tables=8, cap=CONFIG.lsh_cap,
            seed=seed * 131 + step,
        )
        index.build(router.candidate_embeddings(trial.candidates))
        qz = router.encode_query(trial.query, state)
        lookup = index.lookup(
            qz,
            k=4,
            score=lambda key: router.score_embeddings(qz, index.embeddings[key]),
        )
        positions = {c.key: i for i, c in enumerate(trial.candidates)}
        admitted = [positions[k] for k in lookup.addresses]
        first = admitted[0] if admitted else None
        failed = None
        final = None
        attempts = 0
        verifier = __import__(
            "tac_osm.c5_persistent_relational_loop",
            fromlist=["PersistentRelationCASMVerifier"],
        ).PersistentRelationCASMVerifier()
        for idx in admitted:
            attempts += 1
            record = verifier.execute_and_verify(trial, state, idx)
            if record.verified:
                final = idx
                break
            if failed is None:
                failed = idx
        if final is not None:
            router.learn_verified(
                trial.query, state, trial.candidates, final, failed
            )
        rows.append({
            "admission": trial.target_index in admitted,
            "first": first == trial.target_index if first is not None else False,
            "success": final is not None,
            "executed": attempts,
            "repairs": max(0, attempts - 1),
        })
    n = max(1, len(rows))
    return {
        "trials": len(rows),
        "admission_recall": sum(r["admission"] for r in rows) / n,
        "first_attempt_success": sum(r["first"] for r in rows) / n,
        "final_success": sum(r["success"] for r in rows) / n,
        "mean_executed": statistics.fmean(r["executed"] for r in rows),
        "mean_repairs": statistics.fmean(r["repairs"] for r in rows),
        "updates": router.updates,
    }


def main() -> None:
    per_seed = []
    for seed in CONFIG.seeds:
        router = train_router(seed)
        scaling = evaluate_seed(seed, router)
        scaling["online"] = online_loop(seed, train_router(seed))
        per_seed.append(scaling)

    dense_pooled = []
    sparse_pooled = []
    for i, m in enumerate(CONFIG.eval_levels):
        dr = [s["dense"][i] for s in per_seed]
        sr = [s["sparse"][i] for s in per_seed]
        dense_pooled.append({
            "M": m,
            "mean_rank": statistics.fmean(r["mean_rank"] for r in dr),
            "P90_rank": statistics.fmean(r["P90_rank"] for r in dr),
            "top1": statistics.fmean(r["top1"] for r in dr),
            "hamming_mean_rank": statistics.fmean(r["hamming_mean_rank"] for r in dr),
            "hamming_top1": statistics.fmean(r["hamming_top1"] for r in dr),
            "reset_mean_rank": statistics.fmean(r["reset_mean_rank"] for r in dr),
            "reset_P90_rank": statistics.fmean(r["reset_P90_rank"] for r in dr),
            "reset_top1": statistics.fmean(r["reset_top1"] for r in dr),
            "K90": statistics.fmean(r["K90"] for r in dr),
            "K95": statistics.fmean(r["K95"] for r in dr),
        })
        sparse_pooled.append({
            "M": m,
            "K90": statistics.fmean(r["K90"] for r in sr),
            "p1": statistics.fmean(r["p1"] for r in sr),
            "p2": statistics.fmean(r["p2"] for r in sr),
            "rho": statistics.fmean(r["rho"] for r in sr),
            "tables": statistics.fmean(r["tables"] for r in sr),
            "admission_recall": statistics.fmean(r["admission_recall"] for r in sr),
            "final_success": statistics.fmean(r["final_success"] for r in sr),
            "first_attempt_success": statistics.fmean(r["first_attempt_success"] for r in sr),
            "mean_executed": statistics.fmean(r["mean_executed"] for r in sr),
            "mean_repairs": statistics.fmean(r["mean_repairs"] for r in sr),
            "rerank_fraction": statistics.fmean(r["rerank_fraction"] for r in sr),
            "routing_ops": statistics.fmean(r["routing_ops"] for r in sr),
        })

    gamma = fit_power(
        [r["M"] for r in dense_pooled],
        [r["P90_rank"] for r in dense_pooled],
    )
    local = []
    for a, b in zip(dense_pooled, dense_pooled[1:]):
        local.append({
            "M_from": a["M"],
            "M_to": b["M"],
            "local_P90_exponent": math.log(b["P90_rank"] / a["P90_rank"]) / math.log(2.0),
        })

    online_first = statistics.fmean(s["online"]["first_attempt_success"] for s in per_seed)
    online_final = statistics.fmean(s["online"]["final_success"] for s in per_seed)
    online_admission = statistics.fmean(s["online"]["admission_recall"] for s in per_seed)

    result = {
        "protocol": {
            "name": "TACOSM-C5-PERSISTENT-RELATIONAL-LOOP-001",
            "status": "measured",
            "task": {
                "relations": list(CONFIG.ops),
                "state_width": 2 * DIM + 1,
                "query_has_target_descriptor": False,
                "query_has_noisy_hint": True,
                "target_stored_in_persistent_state": False,
                "persistent_delay": 3,
                "intervening_decoy_writes": 3,
            },
            "M": list(CONFIG.eval_levels),
            "seeds": list(CONFIG.seeds),
            "train_steps": CONFIG.train_steps,
            "calibration_trials": CONFIG.calibration_trials,
            "eval_trials": CONFIG.eval_trials,
            "online_trials": CONFIG.online_trials,
        },
        "dense_scaling": {
            "pooled": dense_pooled,
            "gamma_full_range": gamma,
            "local_P90_exponents": local,
        },
        "sparse_scaling": {"pooled": sparse_pooled},
        "controls": {
            "persistent_state_relation_reconstruction_M128": statistics.fmean(
                s["persistence_control_M128"] for s in per_seed
            ),
            "reset_relation_reconstruction_M128": statistics.fmean(
                s["reset_control_M128"] for s in per_seed
            ),
            "boundary_read_success_M128": statistics.fmean(
                s["boundary_read_success"] for s in per_seed
            ),
        },
        "online": {
            "admission_recall_M128": online_admission,
            "first_attempt_success_M128": online_first,
            "final_success_M128": online_final,
        },
        "scope": {
            "semantic_language_claim": False,
            "hamming_shortcut_removed": True,
            "persistence_is_in_main_funnel": True,
            "causal_learning_claim": False,
            "complexity_theorem": False,
        },
    }
    out = Path("artifacts/TACOSM-C5-PERSISTENT-RELATIONAL-LOOP-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
