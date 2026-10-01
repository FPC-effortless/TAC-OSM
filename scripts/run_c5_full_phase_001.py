#!/usr/bin/env python3
"""Execute TACOSM-C5-FULL-PHASE-001.

Outputs one JSON artifact containing:
  * exhaustive target-rank scaling;
  * split-conformal K calibration and held-out coverage;
  * p1/p2/rho measurement for OR-over-tables LSH;
  * sparse admission/rerank/execute/verify metrics;
  * oracle persistent-vs-reset control;
  * analytic-init diagnostic;
  * a closed execution->verification->experience->learning loop.

The workload is synthetic exact equality over a 16-bit state value. Conclusions
are therefore explicitly bounded to this workload.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm.c5_full_phase import (
    CASMVerifier,
    CDLDenseOutcomeRouter,
    ORLSHIndex,
    PhaseConfig,
    Trial,
    conformal_k,
    empirical_quantile,
    make_persistent_trial,
    sample_log_uniform_int,
    estimate_p1_p2,
)
from tac_osm import Query, StateUpdate
from tac_osm.temporal import TemporalPersistentState


CONFIG = PhaseConfig()
MARKS = tuple(1 for _ in range(CONFIG.dim))


def target_trial_from_population(
    *,
    rng: random.Random,
    seed_tag: str,
    step: int,
    candidates,
    state: TemporalPersistentState,
) -> Trial:
    target_index = rng.randrange(len(candidates))
    target = tuple(candidates[target_index].descriptor)
    address = f"world:eval:{seed_tag}:{step:06d}"
    state.stage_world_write(
        StateUpdate(key=address, value=target, step=state.current_step),
        delay=1,
    )
    state.advance_to(state.current_step + 1)
    return Trial(
        candidates=tuple(candidates),
        target_index=target_index,
        query=Query(
            text="\t" + address,
            context=MARKS,
            step=state.current_step,
            provenance="c5_full_phase_eval",
        ),
        reference=target,
        address=address,
    )


def build_fixed_population(seed: int, m: int, dim: int):
    rng = random.Random(seed * 100003 + m * 7919)
    values = set()
    while len(values) < m:
        values.add(tuple(rng.randrange(2) for _ in range(dim)))
    values = list(values)
    rng.shuffle(values)
    from tac_osm import Candidate
    return tuple(
        Candidate(
            key=f"eval-{seed}-{m}-{i:05d}",
            descriptor=tuple(value),
            action=i,
            provenance="c5_full_phase_fixed_population",
        )
        for i, value in enumerate(values)
    )


def make_trial_stream(seed: int, m: int, n: int, candidates):
    rng = random.Random(seed * 1_000_003 + m * 97 + 17)
    trials = []
    for step in range(n):
        state = TemporalPersistentState()
        trials.append(
            target_trial_from_population(
                rng=rng,
                seed_tag=f"s{seed}-m{m}",
                step=step,
                candidates=candidates,
                state=state,
            )
        )
    return trials


def state_for_trial(trial: Trial) -> TemporalPersistentState:
    state = TemporalPersistentState()
    state.stage_world_write(
        StateUpdate(
            key=trial.address,
            value=trial.reference,
            step=0,
        ),
        delay=1,
    )
    state.advance_to(1)
    return state


def fit_power_exponent(xs, ys) -> float:
    pairs = [
        (math.log(float(x)), math.log(float(y)))
        for x, y in zip(xs, ys)
        if float(y) > 0.0
    ]
    if len(pairs) < 2:
        return 0.0
    mx = statistics.fmean(x for x, _ in pairs)
    my = statistics.fmean(y for _, y in pairs)
    den = sum((x - mx) ** 2 for x, _ in pairs)
    return sum((x - mx) * (y - my) for x, y in pairs) / den if den else 0.0


def build_and_measure_index(router, candidates, seed, m, rho=None):
    bits = max(1, math.ceil(math.log2(m)))
    probe_tables = 1
    probe = ORLSHIndex(
        dim=CONFIG.latent_dim,
        bits=bits,
        tables=probe_tables,
        seed=seed * 7919 + m,
        tables_cap=CONFIG.lsh_tables_cap,
    )
    embeddings = router.candidate_embeddings(candidates)
    probe.build(embeddings)
    probe_trials = make_trial_stream(seed + 7000, m, min(CONFIG.calibration_trials, 32), candidates)
    p1, p2 = estimate_p1_p2(
        index=probe,
        router=router,
        trials=probe_trials,
        state_factory=state_for_trial,
        samples_per_trial=3,
    )
    if p1 <= p2 or p2 <= 0.0 or p2 >= 1.0:
        rho_value = 1.0
    else:
        rho_value = math.log(p1) / math.log(p2)
        rho_value = max(0.0, min(1.0, rho_value))
    requested_tables = max(1, math.ceil(m ** rho_value))
    index = ORLSHIndex(
        dim=CONFIG.latent_dim,
        bits=bits,
        tables=requested_tables,
        seed=seed * 104729 + m,
        tables_cap=CONFIG.lsh_tables_cap,
    )
    index.build(embeddings)
    return index, p1, p2, rho_value, requested_tables


def train_router(seed: int, *, analytic_init: bool = False, steps=None):
    steps = CONFIG.train_steps if steps is None else steps
    router = CDLDenseOutcomeRouter(
        dim=CONFIG.dim,
        latent_dim=CONFIG.latent_dim,
        seed=seed,
        learning_rate=CONFIG.learning_rate,
        soft_target_epsilon=CONFIG.soft_target_epsilon,
        analytic_init=analytic_init,
    )
    rng = random.Random(seed * 9_973 + 101)
    casm = CASMVerifier(dim=CONFIG.dim, max_nodes=CONFIG.max_nodes)
    for step in range(steps):
        m = sample_log_uniform_int(
            rng, CONFIG.train_min_m, CONFIG.train_max_m
        )
        state = TemporalPersistentState()
        trial = make_persistent_trial(
            rng,
            seed_tag=f"train-s{seed}",
            step=step,
            m=m,
            dim=CONFIG.dim,
            state=state,
        )
        outcomes = [
            casm.execute_and_verify(
                candidate=c, reference=trial.reference
            )
            for c in trial.candidates
        ]
        successes = [
            i for i, outcome in enumerate(outcomes)
            if outcome.verified and outcome.output >= 0.5
        ]
        if len(successes) != 1:
            raise AssertionError("training task must have exactly one CASM success")
        router.learn_from_dense_outcomes(
            trial.query, state, trial.candidates, successes
        )
    return router


def evaluate_scaling(seed: int, router):
    casm = CASMVerifier(dim=CONFIG.dim, max_nodes=CONFIG.max_nodes)
    all_rows = []
    scaling_by_m = {}
    for m in CONFIG.eval_levels:
        candidates = build_fixed_population(seed, m, CONFIG.dim)
        calibration = make_trial_stream(
            seed, m, CONFIG.calibration_trials, candidates
        )
        heldout = make_trial_stream(
            seed + 100, m, CONFIG.eval_trials, candidates
        )

        ranks = []
        for trial in calibration:
            state = state_for_trial(trial)
            ranks.append(
                router.target_rank(
                    trial.query, state, trial.candidates, trial.target_index
                )
            )
        k90 = conformal_k(ranks, CONFIG.alpha, m)
        k95 = conformal_k(ranks, CONFIG.alpha_95, m)

        index, p1, p2, rho, requested_tables = build_and_measure_index(
            router, candidates, seed, m
        )

        coverage90 = []
        coverage95 = []
        sparse_rows90 = []
        sparse_rows95 = []
        for trial in heldout:
            state = state_for_trial(trial)
            rank = router.target_rank(
                trial.query, state, trial.candidates, trial.target_index
            )
            coverage90.append(int(rank <= k90))
            coverage95.append(int(rank <= k95))

            for level, k, out_rows in (
                ("90", k90, sparse_rows90),
                ("95", k95, sparse_rows95),
            ):
                zq = router.encode_query(trial.query, state)
                lookup = index.lookup(
                    zq,
                    candidate_limit=k,
                    score_fn=lambda address, zq=zq: router.score_embeddings(
                        zq, index.embeddings[address]
                    ),
                )
                admitted_indices = [
                    next(i for i, c in enumerate(trial.candidates) if c.key == address)
                    for address in lookup.addresses
                ]
                executions = [
                    casm.execute_and_verify(
                        candidate=trial.candidates[i],
                        reference=trial.reference,
                    )
                    for i in admitted_indices
                ]
                first_success = bool(
                    executions
                    and executions[0].verified
                    and executions[0].output >= 0.5
                )
                final_success = any(
                    x.verified and x.output >= 0.5 for x in executions
                )
                out_rows.append(
                    {
                        "target_rank": rank,
                        "target_admitted": int(trial.target_index in admitted_indices),
                        "first_attempt_success": int(first_success),
                        "final_success": int(final_success),
                        "rerank_count": lookup.candidate_rerank_count,
                        "query_hash_ops": lookup.query_hash_ops,
                        "query_routing_ops": lookup.query_hash_ops
                        + lookup.candidate_rerank_count * CONFIG.latent_dim,
                        "executed_candidates": len(admitted_indices),
                        "executed_nodes": sum(x.node_count for x in executions),
                        "executed_edges": sum(x.edge_count for x in executions),
                        "tables": lookup.tables,
                        "bits": lookup.bits,
                        "tables_capped": lookup.capped_tables,
                        "build_macs_amortized": lookup.build_macs_amortized,
                    }
                )

        def aggregate(rows):
            denom = max(1, len(rows))
            admitted = sum(r["target_admitted"] for r in rows)
            finals = sum(r["final_success"] for r in rows)
            return {
                "trials": len(rows),
                "admission_recall": admitted / denom,
                "first_attempt_success": sum(r["first_attempt_success"] for r in rows) / denom,
                "final_success": finals / denom,
                "conditional_selection_given_admission": (
                    finals / admitted if admitted else 0.0
                ),
                "mean_rerank_count": statistics.fmean(r["rerank_count"] for r in rows),
                "rerank_fraction": statistics.fmean(r["rerank_count"] for r in rows) / m,
                "mean_query_hash_ops": statistics.fmean(r["query_hash_ops"] for r in rows),
                "mean_query_routing_ops": statistics.fmean(r["query_routing_ops"] for r in rows),
                "mean_executed_candidates": statistics.fmean(r["executed_candidates"] for r in rows),
                "mean_executed_nodes": statistics.fmean(r["executed_nodes"] for r in rows),
                "mean_executed_edges": statistics.fmean(r["executed_edges"] for r in rows),
                "mean_build_macs_amortized": statistics.fmean(r["build_macs_amortized"] for r in rows),
                "tables_capped": any(r["tables_capped"] for r in rows),
            }

        agg90 = aggregate(sparse_rows90)
        agg95 = aggregate(sparse_rows95)
        row = {
            "M": m,
            "mean_target_rank": statistics.fmean(ranks),
            "p90_target_rank": empirical_quantile(ranks, 0.90),
            "p95_target_rank": empirical_quantile(ranks, 0.95),
            "conformal_K_90": k90,
            "conformal_K_95": k95,
            "heldout_coverage_90": statistics.fmean(coverage90),
            "heldout_coverage_95": statistics.fmean(coverage95),
            "p1": p1,
            "p2": p2,
            "rho": rho,
            "requested_lsh_tables": requested_tables,
            "rank_scaling_reference_trials": len(ranks),
            "sparse_90": agg90,
            "sparse_95": agg95,
        }
        all_rows.append(row)
        scaling_by_m[m] = row
    return all_rows, scaling_by_m


def oracle_controls(seed: int, m: int):
    candidates = build_fixed_population(seed, m, CONFIG.dim)
    trials = make_trial_stream(seed + 500, m, CONFIG.eval_trials, candidates)
    persistent = 0
    reset = 0
    for trial in trials:
        state = state_for_trial(trial)
        read = state.read(trial.query)
        if read.values:
            reference = tuple(read.values[0])
            chosen = next(
                (
                    i for i, candidate in enumerate(trial.candidates)
                    if candidate.descriptor == reference
                ),
                0,
            )
        else:
            chosen = 0
        persistent += int(chosen == trial.target_index)
        state.clear()
        read = state.read(trial.query)
        if read.values:
            reference = tuple(read.values[0])
            chosen = next(
                (
                    i for i, candidate in enumerate(trial.candidates)
                    if candidate.descriptor == reference
                ),
                0,
            )
        else:
            chosen = 0
        reset += int(chosen == trial.target_index)
    n = max(1, len(trials))
    return persistent / n, reset / n


def initialization_diagnostic():
    rows = []
    for seed in CONFIG.seeds[:3]:
        for analytic in (False, True):
            router = train_router(seed + (100 if analytic else 0), analytic_init=analytic, steps=160)
            candidates = build_fixed_population(seed + 900, 32, CONFIG.dim)
            trials = make_trial_stream(seed + 901, 32, 32, candidates)
            ranks = []
            for trial in trials:
                state = state_for_trial(trial)
                ranks.append(router.target_rank(
                    trial.query, state, trial.candidates, trial.target_index
                ))
            rows.append({
                "seed": seed,
                "analytic_init": analytic,
                "mean_rank": statistics.fmean(ranks),
                "p90_rank": empirical_quantile(ranks, 0.90),
                "top1": sum(r == 1 for r in ranks) / len(ranks),
            })
    return rows


def closed_loop(seed: int, router, m: int = 128):
    rng = random.Random(seed * 91_771 + 3)
    casm = CASMVerifier(dim=CONFIG.dim, max_nodes=CONFIG.max_nodes)
    first = 0
    final = 0
    writes = 0
    admitted = 0
    experience_keys = set()
    for step in range(CONFIG.online_trials):
        state = TemporalPersistentState()
        trial = make_persistent_trial(
            rng,
            seed_tag=f"online-s{seed}",
            step=step,
            m=m,
            dim=CONFIG.dim,
            state=state,
        )
        embeddings = router.candidate_embeddings(trial.candidates)
        bits = math.ceil(math.log2(m))
        # Rebuild after each verified learning event so the index and
        # representation stay synchronized. Build cost is reported separately.
        probe = ORLSHIndex(
            dim=CONFIG.latent_dim,
            bits=bits,
            tables=1,
            seed=seed * 31 + step,
            tables_cap=CONFIG.lsh_tables_cap,
        )
        probe.build(embeddings)
        p1, p2 = estimate_p1_p2(
            index=probe,
            router=router,
            trials=[trial],
            state_factory=state_for_trial,
            samples_per_trial=2,
        )
        rho = 1.0 if p1 <= p2 or p2 in (0.0, 1.0) else max(
            0.0, min(1.0, math.log(p1) / math.log(p2))
        )
        tables = max(1, math.ceil(m ** rho))
        index = ORLSHIndex(
            dim=CONFIG.latent_dim,
            bits=bits,
            tables=tables,
            seed=seed * 31337 + step,
            tables_cap=CONFIG.lsh_tables_cap,
        )
        index.build(embeddings)
        k = min(m, 4)
        zq = router.encode_query(trial.query, state)
        lookup = index.lookup(
            zq,
            candidate_limit=k,
            score_fn=lambda address, zq=zq: router.score_embeddings(
                zq, index.embeddings[address]
            ),
        )
        admitted_indices = [
            next(i for i, c in enumerate(trial.candidates) if c.key == address)
            for address in lookup.addresses
        ]
        executions = [
            casm.execute_and_verify(
                candidate=trial.candidates[i], reference=trial.reference
            )
            for i in admitted_indices
        ]
        first_ok = bool(executions and executions[0].verified and executions[0].output >= 0.5)
        final_idx = next(
            (x.candidate_index for x in executions if x.verified and x.output >= 0.5),
            None,
        )
        final_ok = final_idx == trial.target_index
        first += int(first_ok)
        final += int(final_ok)
        admitted_now = int(trial.target_index in admitted_indices)
        admitted += admitted_now
        if final_ok:
            writes += 1
            key = f"experience:{trial.address}:{step}"
            state.write(
                StateUpdate(
                    key=key,
                    value=trial.reference,
                    task_key=trial.address,
                    success_score=1.0,
                    step=state.current_step,
                )
            )
            experience_keys.add(key)
            negative = next(
                (i for i in admitted_indices if i != trial.target_index),
                None,
            )
            router.learn_from_verified(
                trial.query,
                state,
                trial.candidates,
                trial.target_index,
                negative,
            )
    return {
        "M": m,
        "trials": CONFIG.online_trials,
        "first_attempt_success": first / max(1, CONFIG.online_trials),
        "final_success": final / max(1, CONFIG.online_trials),
        "admission_recall": admitted / max(1, CONFIG.online_trials),
        "experience_commit_rate": writes / max(1, CONFIG.online_trials),
        "experience_keys_written": len(experience_keys),
        "updates_after_online_loop": router.updates,
    }


def main():
    seed_rows = []
    all_scaling = []
    init_rows = initialization_diagnostic()
    for seed in CONFIG.seeds:
        router = train_router(seed)
        persistent8, reset8 = oracle_controls(seed, 8)
        scaling_rows, _ = evaluate_scaling(seed, router)
        online = closed_loop(seed, router, m=128)
        for row in scaling_rows:
            row = dict(row)
            row["seed"] = seed
            all_scaling.append(row)
        seed_rows.append({
            "seed": seed,
            "router_updates_after_training": router.updates,
            "oracle_persistent_success_M8": persistent8,
            "oracle_reset_success_M8": reset8,
            "online": online,
        })

    pooled = []
    for m in CONFIG.eval_levels:
        rows = [r for r in all_scaling if r["M"] == m]
        def mean_key(path):
            vals = []
            for r in rows:
                x = r
                for part in path.split("."):
                    x = x[part]
                vals.append(float(x))
            return statistics.fmean(vals)
        pooled.append({
            "M": m,
            "mean_target_rank": mean_key("mean_target_rank"),
            "p90_target_rank": mean_key("p90_target_rank"),
            "p95_target_rank": mean_key("p95_target_rank"),
            "conformal_K_90": statistics.fmean(r["conformal_K_90"] for r in rows),
            "conformal_K_95": statistics.fmean(r["conformal_K_95"] for r in rows),
            "heldout_coverage_90": statistics.fmean(r["heldout_coverage_90"] for r in rows),
            "heldout_coverage_95": statistics.fmean(r["heldout_coverage_95"] for r in rows),
            "p1": mean_key("p1"),
            "p2": mean_key("p2"),
            "rho": mean_key("rho"),
            "requested_lsh_tables": statistics.fmean(r["requested_lsh_tables"] for r in rows),
            "sparse90_admission_recall": statistics.fmean(r["sparse_90"]["admission_recall"] for r in rows),
            "sparse90_conditional_selection": statistics.fmean(r["sparse_90"]["conditional_selection_given_admission"] for r in rows),
            "sparse90_success": statistics.fmean(r["sparse_90"]["final_success"] for r in rows),
            "sparse90_rerank_fraction": statistics.fmean(r["sparse_90"]["rerank_fraction"] for r in rows),
            "sparse90_mean_rerank_count": statistics.fmean(r["sparse_90"]["mean_rerank_count"] for r in rows),
            "sparse90_query_routing_ops": statistics.fmean(r["sparse_90"]["mean_query_routing_ops"] for r in rows),
            "sparse90_executed_candidates": statistics.fmean(r["sparse_90"]["mean_executed_candidates"] for r in rows),
            "sparse90_build_macs_amortized": statistics.fmean(r["sparse_90"]["mean_build_macs_amortized"] for r in rows),
            "sparse95_admission_recall": statistics.fmean(r["sparse_95"]["admission_recall"] for r in rows),
            "sparse95_success": statistics.fmean(r["sparse_95"]["final_success"] for r in rows),
        })

    gamma_rank = fit_power_exponent(
        [r["M"] for r in pooled],
        [r["p90_target_rank"] for r in pooled],
    )
    beta_routing = fit_power_exponent(
        [r["M"] for r in pooled],
        [r["sparse90_query_routing_ops"] for r in pooled],
    )
    fixed_k_90 = max(int(round(max(r["conformal_K_90"] for r in pooled))), 1)
    oracle_persistent = statistics.fmean(
        r["oracle_persistent_success_M8"] for r in seed_rows
    )
    oracle_reset = statistics.fmean(
        r["oracle_reset_success_M8"] for r in seed_rows
    )
    online_first = statistics.fmean(
        r["online"]["first_attempt_success"] for r in seed_rows
    )
    online_final = statistics.fmean(
        r["online"]["final_success"] for r in seed_rows
    )
    online_writes = statistics.fmean(
        r["online"]["experience_commit_rate"] for r in seed_rows
    )

    result = {
        "protocol": {
            "name": "TACOSM-C5-FULL-PHASE-001",
            "status": "measured",
            "config": {
                "dim": CONFIG.dim,
                "max_nodes": CONFIG.max_nodes,
                "latent_dim": CONFIG.latent_dim,
                "train_steps": CONFIG.train_steps,
                "calibration_trials": CONFIG.calibration_trials,
                "eval_trials": CONFIG.eval_trials,
                "online_trials": CONFIG.online_trials,
                "train_M": [CONFIG.train_min_m, CONFIG.train_max_m],
                "eval_M": list(CONFIG.eval_levels),
                "seeds": list(CONFIG.seeds),
                "alpha": CONFIG.alpha,
                "alpha_95": CONFIG.alpha_95,
                "lsh_tables_cap": CONFIG.lsh_tables_cap,
            },
        },
        "gates": {
            "exactly_one_casm_success_per_trial": True,
            "router_has_no_target_index_input": True,
            "fresh_training_state_codes": True,
            "temporal_world_write_delay": 1,
            "experience_namespace_only": True,
        },
        "headline_metrics": {
            "rank_scaling_exponent_gamma": gamma_rank,
            "routing_work_scaling_exponent_beta": beta_routing,
            "oracle_persistent_success_M8": oracle_persistent,
            "oracle_reset_success_M8": oracle_reset,
            "online_first_attempt_success_M128": online_first,
            "online_final_success_M128": online_final,
            "online_experience_commit_rate_M128": online_writes,
            "max_conformal_K90": fixed_k_90,
            "sublinear_query_routing_observed": beta_routing < 1.0,
        },
        "pooled_scaling": pooled,
        "per_seed": seed_rows,
        "initialization_diagnostic": init_rows,
        "interpretation": {
            "scope": "synthetic exact-equality state-addressing workload",
            "sublinear_rule": "Only interpret sublinear query routing when beta<1 and no LSH table cap binds.",
            "coverage_rule": "Empirical held-out coverage is reported separately; conformal coverage is conditional on exchangeability.",
            "execution_rule": "CASM candidate execution count is bounded by the calibrated shortlist and verifier selects from executed candidates.",
            "non_claims": [
                "No language-level semantic generalization is established.",
                "No hardware wall-clock speedup is inferred from arithmetic counts.",
                "Index build cost is separate from query routing cost.",
                "Online learning results are descriptive and not a causal ablation."
            ],
        },
    }
    return result


if __name__ == "__main__":
    result = main()
    out = Path("artifacts/TACOSM-C5-FULL-PHASE-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
