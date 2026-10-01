#!/usr/bin/env python3
"""Registered fused TAC-OSM frontier experiment.

Fixed-budget routing:
  A product-key -> exact shortlist
  B product-key -> StructMeans -> exact shortlist
  C product-key -> StructMeans -> PST transition rerank

Closed-loop consolidation:
  verified execution -> REGM -> AXON consolidation
  failed base retrieval -> SECA composition -> independent verification
  accepted novel macro -> persistent operator memory

The primary compute test uses absolute budgets B={4,8,16} at
M={128,256,512}. Product-key internal routing work is reported separately;
no asymptotic claim is made from the final shortlist alone.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import Candidate, Outcome, Query, VerificationResult
from tac_osm.execution_product_key import ExecutionFeedbackProductKeyRouter
from tac_osm.operator_learning import (
    ExperienceStore,
    FixedAXONConsolidator,
    PSTLearner,
    StructMeans,
    TransitionRecord,
    apply_operator,
)
from tac_osm.state import PersistentStore, StateConfig
from tac_osm.fused_architecture import (
    FusedOperatorRouter,
    OperatorDescriptor,
    VerifiedOperatorMemory,
    candidate_operator,
    candidate_record,
    make_composite,
    macro_from_candidate,
    verify_macro,
)

SEEDS = (10, 11, 12)
M_LEVELS = (128, 256, 512)
BUDGETS = (4, 8, 16)
TRAIN_KEYS = 48
TRAIN_STEPS = 600
EVAL_SINGLE = 80
EVAL_COMPOSITE = 60
DIM = 8
ROUTER_INPUT_DIM = 11
FACTOR_COUNT = 3
FACTOR_SIZE = 16
FACTOR_BEAM = 7
COARSE_SHORTLIST = 32
REFRESH = 32
STRUCT_CLUSTERS = 3
STRUCT_CLUSTER_BUDGET = 2


def random_mask(rng: random.Random, width: int | None = None) -> tuple[int, ...]:
    width = rng.randint(2, 5) if width is None else int(width)
    active = set(rng.sample(range(DIM), width))
    return tuple(int(i in active) for i in range(DIM))


def make_library(seed: int, m: int) -> tuple[Candidate, ...]:
    rng = random.Random(seed * 100003 + m * 7919)
    seen: set[tuple[str, tuple[int, ...]]] = set()
    candidates: list[Candidate] = []
    per_kind = {kind: 0 for kind in ("toggle", "set1", "set0")}

    def add(kind: str, mask: tuple[int, ...]) -> bool:
        key = (kind, mask)
        if key in seen:
            return False
        seen.add(key)
        idx = len(candidates)
        candidates.append(
            Candidate(
                key=f"op-{seed}-{m}-{idx:04d}",
                descriptor=OperatorDescriptor(kind, mask).encode(),
                action=idx,
                provenance="fused_frontier_operator",
            )
        )
        per_kind[kind] += 1
        return True

    # Exactly 16 held-constant codebook items per kind.
    while len(candidates) < TRAIN_KEYS:
        kind = ("toggle", "set1", "set0")[len(candidates) // 16]
        add(kind, random_mask(rng))
    kinds = ("toggle", "set1", "set0")
    while len(candidates) < m:
        add(rng.choice(kinds), random_mask(rng))
    return tuple(candidates)


def training_keys(candidates: tuple[Candidate, ...]) -> tuple[str, ...]:
    return tuple(c.key for c in candidates[:TRAIN_KEYS])


def bootstrap_records(
    candidates: tuple[Candidate, ...],
    seed: int,
    repeats: int = 4,
) -> list[TransitionRecord]:
    rng = random.Random(seed + 7001)
    rows: list[TransitionRecord] = []
    for candidate in candidates[:TRAIN_KEYS]:
        op = candidate_operator(candidate)
        for rep in range(repeats):
            state = tuple(rng.randrange(2) for _ in range(DIM))
            rows.append(
                TransitionRecord(
                    before=state,
                    operator=op,
                    after=apply_operator(state, op),
                    verified=True,
                    episode=rep,
                    step=len(rows),
                )
            )
    return rows


def build_structural_memory(
    candidates: tuple[Candidate, ...],
    seed: int,
) -> tuple[StructMeans, PSTLearner, tuple[int, ...], ExperienceStore, tuple]:
    rows = bootstrap_records(candidates, seed)
    store = ExperienceStore()
    for row in rows:
        store.append(row)
    sm = StructMeans(STRUCT_CLUSTERS, ("toggle", "set1", "set0"), seed=seed)
    sm.fit(rows)
    pst = PSTLearner(("toggle", "set1", "set0"))
    pst.fit(rows)
    cluster_by_index = tuple(sm.assign(candidate_record(c)) for c in candidates)
    axon = FixedAXONConsolidator(min_support=2).consolidate(rows)
    return sm, pst, cluster_by_index, store, axon


def query_for(state: tuple[int, ...], goal: tuple[int, ...], step: int) -> Query:
    return Query(
        text=" ".join(str(int(x)) for x in state),
        context=goal,
        step=step,
        provenance="fused_frontier_query",
    )


def train_representation(
    candidates: tuple[Candidate, ...],
    *,
    seed: int,
) -> tuple[ExecutionFeedbackProductKeyRouter, PersistentStore]:
    router = ExecutionFeedbackProductKeyRouter(
        seed=seed,
        input_dim=ROUTER_INPUT_DIM,
        latent_dim=16,
        factor_count=FACTOR_COUNT,
        factor_size=FACTOR_SIZE,
        factor_beam=FACTOR_BEAM,
        max_shortlist=COARSE_SHORTLIST,
        refresh_interval_updates=REFRESH,
        learning_rate=0.01,
        margin=0.1,
    )
    state = PersistentStore(StateConfig(seed=seed + 90000, n_slots=128))
    keys = training_keys(candidates)
    for step in range(TRAIN_STEPS):
        rng = random.Random(seed * 1009 + step * 104729 + len(candidates) * 37)
        target_index = rng.randrange(TRAIN_KEYS)
        target = candidate_operator(candidates[target_index])
        current = tuple(rng.randrange(2) for _ in range(DIM))
        goal = apply_operator(current, target)
        query = query_for(current, goal, step)
        decision, _diag = router.route(query, state, candidates, training_keys=keys)
        selected = decision.selected
        ok = (
            selected >= 0
            and apply_operator(current, candidate_operator(candidates[selected])) == goal
        )
        detail = SimpleNamespace(gold_index=target_index, candidates=candidates)
        outcome = Outcome(ok, float(ok), "success" if ok else "wrong_operator", detail)
        verification = VerificationResult(ok, "accept" if ok else "reject")
        router.learn_from_verifier(
            query=query,
            state=state,
            candidates=candidates,
            selected=selected,
            outcome=outcome,
            verification=verification,
            scores=decision.scores,
        )
    return router, state


def single_tasks(
    candidates: tuple[Candidate, ...],
    seed: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...], int]]:
    rng = random.Random(seed + 111)
    held = list(range(TRAIN_KEYS, len(candidates)))
    rows = []
    for step in range(EVAL_SINGLE):
        target_index = held[rng.randrange(len(held))]
        op = candidate_operator(candidates[target_index])
        state = tuple(rng.randrange(2) for _ in range(DIM))
        goal = apply_operator(state, op)
        rows.append((state, goal, target_index))
    return rows


def evaluate_fixed(
    *,
    candidates: tuple[Candidate, ...],
    state: PersistentStore,
    pkm: ExecutionFeedbackProductKeyRouter,
    sm: StructMeans,
    pst: PSTLearner,
    cluster_by_index: tuple[int, ...],
    budget: int,
    mode: str,
    seed: int,
) -> dict[str, float | int | str]:
    router = FusedOperatorRouter(
        pkm,
        structmeans=sm,
        cluster_by_index=cluster_by_index,
        pst=pst,
        budget=budget,
        cluster_budget=STRUCT_CLUSTER_BUDGET,
        mode=mode,
    )
    keys = training_keys(candidates)
    rows = single_tasks(candidates, seed)
    pkm_admit: list[int] = []
    final_admit: list[int] = []
    selected: list[int] = []
    coarse_work: list[int] = []
    factor_work: list[int] = []
    struct_work: list[int] = []
    pst_work: list[int] = []
    final_counts: list[int] = []

    for step, (current, goal, target_index) in enumerate(rows):
        route = router.route(
            query_for(current, goal, step),
            state,
            candidates,
            training_keys=keys,
        )
        pkm_ids = set(route.pkm_indices)
        final_ids = {candidates.index(c) for c in route.final_candidates}
        pkm_admit.append(int(target_index in pkm_ids))
        final_admit.append(int(target_index in final_ids))
        chosen_ok = (
            route.final_selected_candidate is not None
            and candidates.index(route.final_selected_candidate) == target_index
        )
        selected.append(int(chosen_ok))
        coarse_work.append(route.pkm_diag.state_candidates_scored)
        factor_work.append(route.pkm_diag.factor_score_macs)
        struct_work.append(route.structural.structural_ops if route.structural else 0)
        pst_work.append(route.pst_prediction_ops)
        final_counts.append(len(route.final_candidates))

    return {
        "mode": mode,
        "budget": budget,
        "M": len(candidates),
        "seed": seed,
        "steps": len(rows),
        "pkm_admission_recall": statistics.fmean(pkm_admit),
        "final_admission_recall": statistics.fmean(final_admit),
        "top1_success": statistics.fmean(selected),
        "pkm_internal_scored_mean": statistics.fmean(coarse_work),
        "pkm_internal_scored_over_M": statistics.fmean(coarse_work) / len(candidates),
        "factor_score_macs_mean": statistics.fmean(factor_work),
        "structural_ops_mean": statistics.fmean(struct_work),
        "pst_prediction_ops_mean": statistics.fmean(pst_work),
        "final_candidates_mean": statistics.fmean(final_counts),
        "fixed_final_budget": budget,
    }


def composite_tasks(
    candidates: tuple[Candidate, ...],
    seed: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...], int, int]]:
    rng = random.Random(seed + 771)
    held = list(range(TRAIN_KEYS, len(candidates)))
    tasks: list[tuple[tuple[int, ...], tuple[int, ...], int, int]] = []
    attempts = 0
    while len(tasks) < EVAL_COMPOSITE and attempts < EVAL_COMPOSITE * 50:
        attempts += 1
        i, j = rng.sample(held, 2)
        first = candidate_operator(candidates[i])
        second = candidate_operator(candidates[j])
        state = tuple(rng.randrange(2) for _ in range(DIM))
        goal = apply_operator(apply_operator(state, first), second)
        if any(
            apply_operator(state, candidate_operator(candidate)) == goal
            for candidate in candidates
        ):
            continue
        tasks.append((state, goal, i, j))
    if len(tasks) < EVAL_COMPOSITE:
        raise RuntimeError("could not construct enough non-single-step composite tasks")
    return tasks


def evaluate_seca(
    *,
    candidates: tuple[Candidate, ...],
    state: PersistentStore,
    pkm: ExecutionFeedbackProductKeyRouter,
    sm: StructMeans,
    pst: PSTLearner,
    cluster_by_index: tuple[int, ...],
    seed: int,
    budget: int = 8,
) -> dict[str, float | int]:
    router = FusedOperatorRouter(
        pkm,
        structmeans=sm,
        cluster_by_index=cluster_by_index,
        pst=pst,
        budget=budget,
        cluster_budget=STRUCT_CLUSTER_BUDGET,
        mode="pst",
    )
    keys = training_keys(candidates)
    tasks = composite_tasks(candidates, seed)
    pkm_pair: list[int] = []
    final_pair: list[int] = []
    pre_success: list[int] = []
    post_success: list[int] = []
    operator_memory = VerifiedOperatorMemory()
    attempts = 0

    for step, (current, goal, first_index, second_index) in enumerate(tasks):
        route = router.route(
            query_for(current, goal, step),
            state,
            candidates,
            training_keys=keys,
        )
        pkm_ids = set(route.pkm_indices)
        final_candidates = list(route.final_candidates)
        final_ids = {candidates.index(candidate) for candidate in final_candidates}
        pkm_pair.append(int(first_index in pkm_ids and second_index in pkm_ids))
        final_pair.append(int(first_index in final_ids and second_index in final_ids))
        pre_ok = any(
            apply_operator(current, candidate_operator(candidate)) == goal
            for candidate in final_candidates
        )
        pre_success.append(int(pre_ok))

        macros = [macro_from_candidate(candidate) for candidate in final_candidates]
        accepted = None
        proposals = 0
        for first in macros:
            for second in macros:
                if first.name == second.name:
                    continue
                proposals += 1
                candidate_macro = make_composite(first, second)
                verdict = verify_macro(candidate_macro, state=current, goal=goal)
                if verdict.passed:
                    accepted = candidate_macro
                    break
                if proposals >= 32:
                    break
            if accepted is not None or proposals >= 32:
                break

        post_success.append(int(accepted is not None))
        attempts += proposals
        if accepted is not None:
            operator_memory.append(accepted, current, goal, "independent_verifier")

    return {
        "composite_steps": len(tasks),
        "budget": budget,
        "pkm_pair_admission_recall": statistics.fmean(pkm_pair),
        "final_pair_admission_recall": statistics.fmean(final_pair),
        "pre_seca_success": statistics.fmean(pre_success),
        "post_seca_success": statistics.fmean(post_success),
        "verified_novel_composites": operator_memory.verified_count,
        "operator_memory_unique": len(operator_memory.operators),
        "seca_verification_attempts": attempts,
    }


def consolidation_results(
    *,
    store: ExperienceStore,
    axon_macros: tuple,
    sm: StructMeans,
    candidates: tuple[Candidate, ...],
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed + 991)
    reuse: list[int] = []
    for macro in axon_macros[: min(24, len(axon_macros))]:
        if not macro.steps:
            continue
        primitive = macro.steps[0]
        for _ in range(4):
            state = tuple(rng.randrange(2) for _ in range(DIM))
            reuse.append(int(macro.execute(state) == apply_operator(state, primitive)))
    heldout_rows = bootstrap_records(candidates, seed + 333, repeats=2)
    reconstructed = store.reconstruct_pst(("toggle", "set1", "set0"))
    return {
        "regm_verified_records": store.stored_records,
        "pst_reconstruction_accuracy": reconstructed.transition_accuracy(heldout_rows),
        "structmeans_kind_purity": sm.purity(heldout_rows),
        "structmeans_signature_purity": sm.signature_purity(heldout_rows),
        "structmeans_compression_ratio": sm.compression_ratio(heldout_rows),
        "axon_macros": len(axon_macros),
        "axon_bound_reuse_accuracy": statistics.fmean(reuse) if reuse else 0.0,
    }


def run_seed_m(seed: int, m: int) -> dict[str, object]:
    candidates = make_library(seed, m)
    sm, pst, cluster_by_index, store, axon_macros = build_structural_memory(candidates, seed)
    pkm, state = train_representation(candidates, seed=seed + m)
    fixed = []
    for mode in ("pkm", "struct", "pst"):
        for budget in BUDGETS:
            fixed.append(
                evaluate_fixed(
                    candidates=candidates,
                    state=state,
                    pkm=pkm,
                    sm=sm,
                    pst=pst,
                    cluster_by_index=cluster_by_index,
                    budget=budget,
                    mode=mode,
                    seed=seed,
                )
            )
    seca = evaluate_seca(
        candidates=candidates,
        state=state,
        pkm=pkm,
        sm=sm,
        pst=pst,
        cluster_by_index=cluster_by_index,
        seed=seed,
        budget=8,
    )
    consolidation = consolidation_results(
        store=store,
        axon_macros=axon_macros,
        sm=sm,
        candidates=candidates,
        seed=seed,
    )
    return {
        "seed": seed,
        "M": m,
        "router_updates": pkm.updates,
        "pkm_refreshes": pkm.refresh_count,
        "fixed_budget": fixed,
        "seca": seca,
        "consolidation": consolidation,
    }


def pool_fixed(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["M"]), str(row["mode"]), int(row["budget"]))
        grouped.setdefault(key, []).append(row)
    pooled = []
    metric_names = (
        "pkm_admission_recall",
        "final_admission_recall",
        "top1_success",
        "pkm_internal_scored_mean",
        "pkm_internal_scored_over_M",
        "factor_score_macs_mean",
        "structural_ops_mean",
        "pst_prediction_ops_mean",
        "final_candidates_mean",
    )
    for (m, mode, budget), members in sorted(grouped.items()):
        row: dict[str, object] = {"M": m, "mode": mode, "budget": budget, "seeds": len(members)}
        for metric in metric_names:
            row[metric] = statistics.fmean(float(x[metric]) for x in members)
        pooled.append(row)
    return pooled


def pool_seca(runs: list[dict[str, object]]) -> dict[str, float]:
    return {
        "pkm_pair_admission_recall": statistics.fmean(float(r["seca"]["pkm_pair_admission_recall"]) for r in runs),
        "final_pair_admission_recall": statistics.fmean(float(r["seca"]["final_pair_admission_recall"]) for r in runs),
        "pre_seca_success": statistics.fmean(float(r["seca"]["pre_seca_success"]) for r in runs),
        "post_seca_success": statistics.fmean(float(r["seca"]["post_seca_success"]) for r in runs),
        "verified_novel_composites": statistics.fmean(float(r["seca"]["verified_novel_composites"]) for r in runs),
        "operator_memory_unique": statistics.fmean(float(r["seca"]["operator_memory_unique"]) for r in runs),
    }


def main() -> None:
    runs = []
    for seed in SEEDS:
        for m in M_LEVELS:
            runs.append(run_seed_m(seed, m))

    fixed_rows = [row for run in runs for row in run["fixed_budget"]]
    consolidation_rows = [run["consolidation"] for run in runs]
    pooled_consolidation = {
        "regm_verified_records": statistics.fmean(float(x["regm_verified_records"]) for x in consolidation_rows),
        "pst_reconstruction_accuracy": statistics.fmean(float(x["pst_reconstruction_accuracy"]) for x in consolidation_rows),
        "structmeans_kind_purity": statistics.fmean(float(x["structmeans_kind_purity"]) for x in consolidation_rows),
        "structmeans_signature_purity": statistics.fmean(float(x["structmeans_signature_purity"]) for x in consolidation_rows),
        "structmeans_compression_ratio": statistics.fmean(float(x["structmeans_compression_ratio"]) for x in consolidation_rows),
        "axon_macros": statistics.fmean(float(x["axon_macros"]) for x in consolidation_rows),
        "axon_bound_reuse_accuracy": statistics.fmean(float(x["axon_bound_reuse_accuracy"]) for x in consolidation_rows),
    }

    result = {
        "protocol": {
            "name": "TACOSM-FUSED-FRONTIER-001",
            "seeds": list(SEEDS),
            "M_levels": list(M_LEVELS),
            "absolute_budgets": list(BUDGETS),
            "train_steps": TRAIN_STEPS,
            "single_eval_steps": EVAL_SINGLE,
            "composite_eval_steps": EVAL_COMPOSITE,
            "train_codebook": TRAIN_KEYS,
            "factor_count": FACTOR_COUNT,
            "factor_size": FACTOR_SIZE,
            "factor_beam": FACTOR_BEAM,
            "coarse_shortlist": COARSE_SHORTLIST,
            "refresh_interval": REFRESH,
            "structmeans_clusters": STRUCT_CLUSTERS,
            "structmeans_cluster_budget": STRUCT_CLUSTER_BUDGET,
            "no_asymptotic_claim_rule": "Product-key factor scoring, cell expansion, and internal exact address scoring are reported separately from fixed final budget.",
        },
        "pooled_fixed": pool_fixed(fixed_rows),
        "pooled_seca": pool_seca(runs),
        "pooled_consolidation": pooled_consolidation,
        "runs": runs,
    }

    Path("artifacts").mkdir(exist_ok=True)
    path = Path("artifacts/TACOSM-FUSED-FRONTIER-001.json")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
