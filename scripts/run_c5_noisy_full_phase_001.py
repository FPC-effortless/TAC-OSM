#!/usr/bin/env python3
"""Measure TACOSM-C5-NOISY-FULL-PHASE-001."""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm.c5_noisy_full_phase import (
    CDLDenseNoisyRouter,
    HammingBaseline,
    NoisyCASMExecutor,
    NoisyEnvironment,
    NoisyOutcomeVerifier,
    ORLSHIndex,
    PhaseConfig,
    estimate_p1_p2,
    derive_tables,
    build_population,
    make_trial,
    execute_one,
    persistent_clean_control,
)


CONFIG = PhaseConfig()
DIM = CONFIG.dim


def trials_for(seed: int, m: int, count: int, offset: int = 0):
    candidates = build_population(seed, m)
    return [
        make_trial(seed=seed + offset, step=i + offset * 1000, candidates=candidates)
        for i in range(count)
    ]


def fit_power(xs, ys):
    pairs = [
        (math.log(float(x)), math.log(max(float(y), 1e-9)))
        for x, y in zip(xs, ys)
    ]
    if len(pairs) < 2:
        return 0.0
    mx = statistics.fmean(x for x, _ in pairs)
    my = statistics.fmean(y for _, y in pairs)
    den = sum((x - mx) ** 2 for x, _ in pairs)
    return sum((x - mx) * (y - my) for x, y in pairs) / den if den else 0.0


def train_router(seed: int):
    router = CDLDenseNoisyRouter(
        seed=seed,
        learning_rate=CONFIG.learning_rate,
        soft_target_epsilon=CONFIG.soft_target_epsilon,
    )
    executor = NoisyCASMExecutor(dim=DIM)
    verifier = NoisyOutcomeVerifier()
    for step in range(CONFIG.train_steps):
        # Matched training distribution in the extrapolation regime:
        # M is sampled across the registered lower and middle levels.
        m = CONFIG.eval_levels[
            step % 5
        ]
        if m > 256:
            m = 256
        candidates = build_population(seed + step * 3, m)
        trial = make_trial(
            seed=seed + step * 11,
            step=step,
            candidates=candidates,
        )
        env = NoisyEnvironment(trial)
        successes = []
        for i in range(len(candidates)):
            computation, value = executor.execute(trial, candidates[i])
            outcome = env.act(i, casm_output=value)
            if outcome.success:
                successes.append(i)
        expected = len(trial.valid_indices)
        if len(successes) != expected:
            raise AssertionError(
                f"environment success count {len(successes)} != valid count {expected}"
            )
        router.train_exhaustive(trial, successes)
    return router


def aggregate_sparse(rows):
    n = max(1, len(rows))
    return {
        "trials": len(rows),
        "admission_recall": sum(r["target_admitted"] for r in rows) / n,
        "first_attempt_success": sum(r["first_attempt_success"] for r in rows) / n,
        "final_success": sum(r["success"] for r in rows) / n,
        "mean_admitted_count": statistics.fmean(r["admitted_count"] for r in rows),
        "mean_admitted_valid_count": statistics.fmean(r["admitted_valid_count"] for r in rows),
        "mean_executed_count": statistics.fmean(r["executed_count"] for r in rows),
        "mean_verified_count": statistics.fmean(r["verified_count"] for r in rows),
        "mean_repair_attempts": statistics.fmean(r["repair_attempts"] for r in rows),
        "mean_rerank_count": statistics.fmean(r["rerank_count"] for r in rows),
        "rerank_fraction": statistics.fmean(r["rerank_count"] for r in rows) / rows[0]["M"],
        "mean_routing_ops": statistics.fmean(r["routing_ops"] for r in rows),
        "mean_routing_fraction": statistics.fmean(r["routing_ops"] for r in rows) / max(1, rows[0]["M"]),
        "target_admission_and_final_success_gap": (
            sum(int(r["target_admitted"]) - int(r["success"]) for r in rows) / n
        ),
    }


def evaluate_seed(seed: int, router: CDLDenseNoisyRouter):
    executor = NoisyCASMExecutor(dim=DIM)
    verifier = NoisyOutcomeVerifier()
    hamming = HammingBaseline()
    by_m = []
    for m in CONFIG.eval_levels:
        calibration = trials_for(seed, m, CONFIG.calibration_trials, offset=7)
        heldout = trials_for(seed + 100, m, CONFIG.eval_trials, offset=9)

        cdl_ranks = [router.inner.best_valid_rank(trial) for trial in calibration]
        hamming_ranks = [hamming.best_valid_rank(trial) for trial in calibration]
        k90 = max(1, min(m, sorted(cdl_ranks)[max(0, math.ceil(0.90 * len(cdl_ranks)) - 1)]))
        k95 = max(1, min(m, sorted(cdl_ranks)[max(0, math.ceil(0.95 * len(cdl_ranks)) - 1)]))

        bits = max(1, math.ceil(math.log2(m)))
        # One-table collision diagnostic; table count is then derived, not tuned.
        probe = ORLSHIndex(
            latent_dim=16,
            bits=bits,
            tables=1,
            cap=CONFIG.lsh_tables_cap,
            seed=seed * 1009 + m,
        )
        population = build_population(seed, m)
        probe.build(router.inner.candidate_embeddings(population))
        p1s, p2s = [], []
        for trial in calibration[: min(32, len(calibration))]:
            p1, p2 = estimate_p1_p2(router, probe, trial, samples=4)
            p1s.append(p1)
            p2s.append(p2)
        p1, p2 = statistics.fmean(p1s), statistics.fmean(p2s)
        rho, requested_tables = derive_tables(p1, p2, m)
        lsh = ORLSHIndex(
            latent_dim=16,
            bits=bits,
            tables=requested_tables,
            cap=CONFIG.lsh_tables_cap,
            seed=seed * 9176 + m,
        )
        lsh.build(router.inner.candidate_embeddings(population))

        cov90 = []
        cov95 = []
        sparse90 = []
        sparse95 = []
        for trial in heldout:
            rank = router.inner.best_valid_rank(trial)
            cov90.append(int(rank <= k90))
            cov95.append(int(rank <= k95))
            for k, rows in ((k90, sparse90), (k95, sparse95)):
                qz = router.inner.encode_query(trial.noisy_query, __import__("tac_osm.temporal", fromlist=["TemporalPersistentState"]).TemporalPersistentState())
                lookup = lsh.lookup(
                    qz,
                    k=min(k, m),
                    score=lambda key, qz=qz: router.inner.score_embeddings(
                        qz, lsh.embeddings[key]
                    ),
                )
                pos = {c.key: i for i, c in enumerate(trial.candidates)}
                admitted = [pos[key] for key in lookup.addresses]
                executed = 0
                verified = 0
                first = False
                final = False
                repairs = 0
                for j, index in enumerate(admitted):
                    rec = execute_one(
                        trial, index, executor, NoisyEnvironment(trial), verifier
                    )
                    executed += 1
                    verified += int(rec.verified)
                    if j == 0:
                        first = rec.verified
                    if rec.verified:
                        final = True
                        repairs = j
                        break
                rows.append({
                    "M": m,
                    "target_admitted": any(i in trial.valid_indices for i in admitted),
                    "first_attempt_success": first,
                    "success": final,
                    "admitted_count": len(admitted),
                    "admitted_valid_count": sum(i in trial.valid_indices for i in admitted),
                    "executed_count": executed,
                    "verified_count": verified,
                    "repair_attempts": repairs,
                    "rerank_count": lookup.rerank_count,
                    "routing_ops": lookup.hash_ops + lookup.rerank_count * 16,
                })
        by_m.append({
            "M": m,
            "cdl_mean_best_valid_rank": statistics.fmean(cdl_ranks),
            "cdl_p90_best_valid_rank": sorted(cdl_ranks)[max(0, math.ceil(0.90 * len(cdl_ranks)) - 1)],
            "cdl_top1_valid": sum(r == 1 for r in cdl_ranks) / len(cdl_ranks),
            "hamming_mean_best_valid_rank": statistics.fmean(hamming_ranks),
            "hamming_top1_valid": sum(r == 1 for r in hamming_ranks) / len(hamming_ranks),
            "K90": k90,
            "K95": k95,
            "heldout_coverage90": statistics.fmean(cov90),
            "heldout_coverage95": statistics.fmean(cov95),
            "p1": p1,
            "p2": p2,
            "rho": rho,
            "requested_tables": requested_tables,
            "actual_tables": lsh.actual_tables,
            "table_cap_bound": lsh.requested_tables > lsh.actual_tables,
            "sparse90": aggregate_sparse(sparse90),
            "sparse95": aggregate_sparse(sparse95),
        })
    persistent, reset = persistent_clean_control(seed=seed, m=128, trials=CONFIG.eval_trials)
    return {
        "seed": seed,
        "router_updates": router.updates,
        "persistent_clean_control_M128": persistent,
        "reset_control_M128": reset,
        "by_m": by_m,
    }


def online_loop(seed: int, router: CDLDenseNoisyRouter):
    m = 128
    candidates = build_population(seed + 9000, m)
    executor = NoisyCASMExecutor(dim=DIM)
    verifier = NoisyOutcomeVerifier()
    rows = []
    for step in range(CONFIG.online_trials):
        trial = make_trial(
            seed=seed + 10000,
            step=step,
            candidates=candidates,
        )
        # Rebuild the candidate index after each verified learning event.
        index = ORLSHIndex(
            latent_dim=16,
            bits=7,
            tables=2,
            cap=CONFIG.lsh_tables_cap,
            seed=seed * 131 + step,
        )
        index.build(router.inner.candidate_embeddings(candidates))
        empty = __import__("tac_osm.temporal", fromlist=["TemporalPersistentState"]).TemporalPersistentState()
        qz = router.inner.encode_query(trial.noisy_query, empty)
        lookup = index.lookup(
            qz,
            k=4,
            score=lambda key, qz=qz: router.inner.score_embeddings(
                qz, index.embeddings[key]
            ),
        )
        pos = {c.key: i for i, c in enumerate(candidates)}
        admitted = [pos[k] for k in lookup.addresses]
        first_index = admitted[0] if admitted else None
        final_index = None
        failed_index = None
        attempts = 0
        for index_i in admitted:
            attempts += 1
            rec = execute_one(trial, index_i, executor, NoisyEnvironment(trial), verifier)
            if rec.verified:
                final_index = index_i
                break
            if failed_index is None:
                failed_index = index_i
        final_ok = final_index is not None
        first_ok = first_index is not None and first_index in trial.valid_indices
        if final_ok:
            # Update only after environment success and successful verification.
            router.learn_verified(
                trial,
                final_index,
                failed_index,
            )
        rows.append({
            "success": final_ok,
            "first_attempt_success": first_ok,
            "admission": any(i in trial.valid_indices for i in admitted),
            "executed": attempts,
            "repairs": max(0, attempts - 1),
        })
    n = max(1, len(rows))
    return {
        "trials": len(rows),
        "first_attempt_success": sum(r["first_attempt_success"] for r in rows) / n,
        "final_success": sum(r["success"] for r in rows) / n,
        "admission_recall": sum(r["admission"] for r in rows) / n,
        "mean_executed": statistics.fmean(r["executed"] for r in rows),
        "mean_repairs": statistics.fmean(r["repairs"] for r in rows),
        "updates_added": router.updates,
    }


def main():
    per_seed = []
    for seed in CONFIG.seeds:
        router = train_router(seed)
        per_seed.append(evaluate_seed(seed, router))
        # Separate fresh router for online run so online change is not conflated
        # with the final scaling evaluation.
        online_router = train_router(seed)
        per_seed[-1]["online"] = online_loop(seed, online_router)

    pooled = []
    for idx, m in enumerate(CONFIG.eval_levels):
        rows = [s["by_m"][idx] for s in per_seed]
        pooled.append({
            "M": m,
            "cdl_mean_best_valid_rank": statistics.fmean(r["cdl_mean_best_valid_rank"] for r in rows),
            "cdl_p90_best_valid_rank": statistics.fmean(r["cdl_p90_best_valid_rank"] for r in rows),
            "cdl_top1_valid": statistics.fmean(r["cdl_top1_valid"] for r in rows),
            "hamming_mean_best_valid_rank": statistics.fmean(r["hamming_mean_best_valid_rank"] for r in rows),
            "hamming_top1_valid": statistics.fmean(r["hamming_top1_valid"] for r in rows),
            "K90": statistics.fmean(r["K90"] for r in rows),
            "K95": statistics.fmean(r["K95"] for r in rows),
            "heldout_coverage90": statistics.fmean(r["heldout_coverage90"] for r in rows),
            "heldout_coverage95": statistics.fmean(r["heldout_coverage95"] for r in rows),
            "p1": statistics.fmean(r["p1"] for r in rows),
            "p2": statistics.fmean(r["p2"] for r in rows),
            "rho": statistics.fmean(r["rho"] for r in rows),
            "requested_tables": statistics.fmean(r["requested_tables"] for r in rows),
            "table_cap_bound": any(r["table_cap_bound"] for r in rows),
            "sparse90_admission_recall": statistics.fmean(r["sparse90"]["admission_recall"] for r in rows),
            "sparse90_final_success": statistics.fmean(r["sparse90"]["final_success"] for r in rows),
            "sparse90_first_attempt_success": statistics.fmean(r["sparse90"]["first_attempt_success"] for r in rows),
            "sparse90_executed": statistics.fmean(r["sparse90"]["mean_executed_count"] for r in rows),
            "sparse90_repairs": statistics.fmean(r["sparse90"]["mean_repair_attempts"] for r in rows),
            "sparse90_rerank_fraction": statistics.fmean(r["sparse90"]["rerank_fraction"] for r in rows),
            "sparse90_routing_ops": statistics.fmean(r["sparse90"]["mean_routing_ops"] for r in rows),
            "sparse95_final_success": statistics.fmean(r["sparse95"]["final_success"] for r in rows),
        })

    beta = fit_power([r["M"] for r in pooled], [r["sparse90_routing_ops"] for r in pooled])
    cdl_gamma = fit_power([r["M"] for r in pooled], [r["cdl_p90_best_valid_rank"] for r in pooled])
    online_first = statistics.fmean(s["online"]["first_attempt_success"] for s in per_seed)
    online_final = statistics.fmean(s["online"]["final_success"] for s in per_seed)
    online_admit = statistics.fmean(s["online"]["admission_recall"] for s in per_seed)

    result = {
        "protocol": {
            "name": "TACOSM-C5-NOISY-FULL-PHASE-001",
            "status": "measured",
            "config": {
                "dim": CONFIG.dim,
                "classes": CONFIG.classes,
                "class_copies_at_M1024": CLASS_COPIES,
                "noise_bits": 1,
                "train_steps": CONFIG.train_steps,
                "calibration_trials": CONFIG.calibration_trials,
                "eval_trials": CONFIG.eval_trials,
                "online_trials": CONFIG.online_trials,
                "M": list(CONFIG.eval_levels),
                "seeds": list(CONFIG.seeds),
                "repair_budget": CONFIG.repair_budget,
            },
        },
        "gates": {
            "clean_target_hidden_from_main_router": True,
            "query_contains_only_noisy_bits": True,
            "casm_receives_hidden_target": False,
            "verifier_receives_hidden_target": False,
            "persistent_state_used_only_in_control_arm": True,
            "experience_updates_only_after_verified_success": True,
            "exactly_64_semantic_classes": True,
            "minimum_codebook_hamming_distance": 3,
        },
        "headline_metrics": {
            "cdl_target_rank_scaling_exponent_gamma": cdl_gamma,
            "routing_work_scaling_exponent_beta": beta,
            "online_first_attempt_success_M128": online_first,
            "online_final_success_M128": online_final,
            "online_admission_recall_M128": online_admit,
            "persistent_clean_control_M128": statistics.fmean(
                s["persistent_clean_control_M128"] for s in per_seed
            ),
            "reset_control_M128": statistics.fmean(
                s["reset_control_M128"] for s in per_seed
            ),
            "all_lsh_table_caps_unbound": not any(
                r["table_cap_bound"] for r in pooled
            ),
        },
        "pooled_scaling": pooled,
        "per_seed": per_seed,
        "interpretation": {
            "scope": "synthetic 64-class binary codebook with one-bit public-query corruption and repeated valid actions",
            "primary_falsification": "If noisy CDL rank degrades substantially while Hamming remains stable, the learned representation is the bottleneck; if admission drops while dense rank remains acceptable, LSH is the bottleneck; if admission is adequate but final success is lower than admission, CASM/verifier/repair is the bottleneck.",
            "non_claims": [
                "No language-level semantic generalization.",
                "No hardware speedup from arithmetic counts.",
                "No causal claim from the descriptive online update.",
                "No sublinear theorem from empirical beta."
            ],
        },
    }
    out = Path("artifacts/TACOSM-C5-NOISY-FULL-PHASE-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
