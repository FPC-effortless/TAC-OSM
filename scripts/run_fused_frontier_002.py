#!/usr/bin/env python3
"""TACOSM-FUSED-FRONTIER-002: identifiable routing + structural product-key.

001 showed a useful split: PST/AXON/REGM/SECA executed correctly, while the
learned product-key addressor failed to admit held-out exact operators.
This run changes two things and only two things:

1. single-step evaluation uses unique-target tasks, so exact target selection is
   identifiable and execution success remains the primary outcome;
2. one arm replaces the learned candidate representation with a deterministic
   compositional operator descriptor inside the product-key index.

Arms:
  learned_pkm
  structural_pkm
  structural_pkm_structmeans
  structural_pkm_structmeans_pst

No execution/verifier feedback is available to the router at evaluation time.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import Candidate, Outcome, Query, VerificationResult
from tac_osm.execution_product_key import ExecutionFeedbackProductKeyRouter
from tac_osm.fused_architecture import OperatorDescriptor, FusedOperatorRouter, candidate_operator
from tac_osm.operator_learning import (
    ExperienceStore,
    FixedAXONConsolidator,
    PSTLearner,
    PrimitiveOperator,
    StructMeans,
    TransitionRecord,
    apply_operator,
)
from tac_osm.state import PersistentStore, StateConfig
from tac_osm.structural_product_key import StructuralProductKeyRouter

SEEDS = (10, 11, 12)
M_LEVELS = (128, 256, 512)
BUDGETS = (4, 8, 16)
TRAIN_KEYS = 48
TRAIN_STEPS = 600
EVAL_SINGLE = 80
DIM = 8
ROUTER_INPUT_DIM = 11
FACTOR_COUNT = 3
FACTOR_SIZE = 16
FACTOR_BEAM = 7
COARSE_SHORTLIST = 32
REFRESH = 32
STRUCT_CLUSTERS = 3
STRUCT_CLUSTER_BUDGET = 2


def random_mask(rng: random.Random) -> tuple[int, ...]:
    width = rng.randint(2, 5)
    return tuple(int(i in set(rng.sample(range(DIM), width))) for i in range(DIM))


def make_library(seed: int, m: int) -> tuple[Candidate, ...]:
    rng = random.Random(seed * 100003 + m * 7919)
    seen: set[tuple[str, tuple[int, ...]]] = set()
    candidates: list[Candidate] = []

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
                provenance="fused_frontier_002_operator",
            )
        )
        return True

    # Exactly 16 codebook operators per kind, 48 total.
    for kind in ("toggle", "set1", "set0"):
        while sum(
            int(OperatorDescriptor.decode(c.descriptor).kind == kind)
            for c in candidates
        ) < 16:
            add(kind, random_mask(rng))
    while len(candidates) < m:
        add(rng.choice(("toggle", "set1", "set0")), random_mask(rng))
    return tuple(candidates)


def training_records(candidates: tuple[Candidate, ...], seed: int, repeats: int = 4) -> list[TransitionRecord]:
    rng = random.Random(seed + 7017)
    records = []
    for candidate in candidates[:TRAIN_KEYS]:
        op = candidate_operator(candidate)
        for rep in range(repeats):
            before = tuple(rng.randrange(2) for _ in range(DIM))
            records.append(
                TransitionRecord(
                    before=before,
                    operator=op,
                    after=apply_operator(before, op),
                    verified=True,
                    episode=rep,
                    step=len(records),
                )
            )
    return records


def train_learned_pkm(candidates: tuple[Candidate, ...], seed: int):
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
    keys = tuple(c.key for c in candidates[:TRAIN_KEYS])
    for step in range(TRAIN_STEPS):
        rng = random.Random(seed * 1009 + step * 104729 + len(candidates) * 37)
        target = rng.randrange(TRAIN_KEYS)
        op = candidate_operator(candidates[target])
        before = tuple(rng.randrange(2) for _ in range(DIM))
        goal = apply_operator(before, op)
        query = Query(
            text=" ".join(map(str, before)),
            context=goal,
            step=step,
            provenance="fused_frontier_002_train",
        )
        decision, _ = router.route(query, state, candidates, training_keys=keys)
        selected = decision.selected
        ok = selected >= 0 and apply_operator(before, candidate_operator(candidates[selected])) == goal
        detail = type("D", (), {"gold_index": target, "candidates": candidates})()
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


def build_structural(candidates: tuple[Candidate, ...], seed: int):
    rows = training_records(candidates, seed)
    store = ExperienceStore()
    for row in rows:
        store.append(row)
    sm = StructMeans(STRUCT_CLUSTERS, ("toggle", "set1", "set0"), seed=seed)
    sm.fit(rows)
    pst = PSTLearner(("toggle", "set1", "set0"))
    pst.fit(rows)
    structural = StructuralProductKeyRouter(
        factor_count=FACTOR_COUNT,
        factor_size=FACTOR_SIZE,
        factor_beam=FACTOR_BEAM,
        max_shortlist=COARSE_SHORTLIST,
    )
    return rows, store, sm, pst, structural


def unique_single_tasks(candidates: tuple[Candidate, ...], seed: int) -> list[tuple[tuple[int, ...], tuple[int, ...], int]]:
    rng = random.Random(seed + 901)
    held_toggle = [
        i for i in range(TRAIN_KEYS, len(candidates))
        if OperatorDescriptor.decode(candidates[i].descriptor).kind == "toggle"
    ]
    if not held_toggle:
        raise RuntimeError("no held-out toggle operators")
    tasks = []
    attempts = 0
    while len(tasks) < EVAL_SINGLE and attempts < EVAL_SINGLE * 500:
        attempts += 1
        target = held_toggle[rng.randrange(len(held_toggle))]
        op = candidate_operator(candidates[target])
        state = tuple(rng.randrange(2) for _ in range(DIM))
        # Require both 0->1 and 1->0 changes. This makes the target toggle
        # signature uniquely identifiable among all three operator families.
        changed = [i for i, bit in enumerate(op.mask) if bit]
        if not any(state[i] == 0 for i in changed) or not any(state[i] == 1 for i in changed):
            continue
        goal = apply_operator(state, op)
        solutions = [
            i for i, candidate in enumerate(candidates)
            if apply_operator(state, candidate_operator(candidate)) == goal
        ]
        if solutions == [target]:
            tasks.append((state, goal, target))
    if len(tasks) < EVAL_SINGLE:
        raise RuntimeError(f"could construct only {len(tasks)} unique-target tasks")
    return tasks


def q_for(state: tuple[int, ...], goal: tuple[int, ...], step: int) -> Query:
    return Query(
        text=" ".join(map(str, state)),
        context=goal,
        step=step,
        provenance="fused_frontier_002_eval",
    )


def eval_learned(
    candidates, state, learned, sm, pst, cluster_dummy, budget, mode, seed
):
    router = FusedOperatorRouter(
        learned,
        structmeans=sm,
        cluster_by_index=cluster_dummy,
        pst=pst,
        budget=budget,
        cluster_budget=STRUCT_CLUSTER_BUDGET,
        mode=mode,
    )
    rows = unique_single_tasks(candidates, seed)
    admitted, final_admitted, exact, execution, scored, factor = [], [], [], [], [], []
    for step, (before, goal, target) in enumerate(rows):
        route = router.route(q_for(before, goal, step), state, candidates, training_keys=tuple(c.key for c in candidates[:TRAIN_KEYS]))
        pkm_ids = set(route.pkm_indices)
        final_ids = {candidates.index(c) for c in route.final_candidates}
        admitted.append(int(target in pkm_ids))
        final_admitted.append(int(target in final_ids))
        selected_ok = route.final_selected_candidate is not None and apply_operator(
            before, candidate_operator(route.final_selected_candidate)
        ) == goal
        execution.append(int(selected_ok))
        exact.append(int(route.final_selected_candidate is not None and candidates.index(route.final_selected_candidate) == target))
        scored.append(route.pkm_diag.state_candidates_scored)
        factor.append(route.pkm_diag.factor_score_macs)
    return {
        "M": len(candidates), "budget": budget, "mode": mode, "seed": seed,
        "execution_success": statistics.fmean(execution),
        "exact_target_success": statistics.fmean(exact),
        "pkm_admission_recall": statistics.fmean(admitted),
        "final_admission_recall": statistics.fmean(final_admitted),
        "internal_exact_addresses_scored": statistics.fmean(scored),
        "internal_exact_addresses_scored_over_M": statistics.fmean(scored) / len(candidates),
        "factor_score_macs": statistics.fmean(factor),
    }


def eval_struct(
    candidates, state, structural, rows, sm, pst, budget, seed, use_structmeans, use_pst
):
    route_rows = unique_single_tasks(candidates, seed)
    structural._structmeans = sm if use_structmeans else None
    admitted, final_admitted, execution, exact, exact_scored, factor, pst_ops, gate_ops = [], [], [], [], [], [], [], []
    for step, (before, goal, target) in enumerate(route_rows):
        route = structural.route(
            q_for(before, goal, step),
            state,
            candidates,
            training_records=rows,
            budget=budget,
            cluster_budget=STRUCT_CLUSTER_BUDGET,
            pst=pst if use_pst else None,
        )
        pkm_ids = set(route.admitted_indices)
        final_ids = set(route.final_indices)
        admitted.append(int(target in pkm_ids))
        final_admitted.append(int(target in final_ids))
        candidate = candidates[route.decision.selected] if route.decision.selected >= 0 else None
        execution.append(int(candidate is not None and apply_operator(before, candidate_operator(candidate)) == goal))
        exact.append(int(route.decision.selected == target))
        exact_scored.append(route.diagnostics.internal_exact_addresses_scored)
        factor.append(route.diagnostics.factor_score_macs)
        pst_ops.append(route.diagnostics.pst_prediction_ops)
        gate_ops.append(route.diagnostics.structural_gate_ops)
    return {
        "M": len(candidates),
        "budget": budget,
        "seed": seed,
        "mode": "structural_pkm_structmeans_pst" if use_structmeans and use_pst else (
            "structural_pkm_structmeans" if use_structmeans else "structural_pkm"
        ),
        "execution_success": statistics.fmean(execution),
        "exact_target_success": statistics.fmean(exact),
        "pkm_admission_recall": statistics.fmean(admitted),
        "final_admission_recall": statistics.fmean(final_admitted),
        "internal_exact_addresses_scored": statistics.fmean(exact_scored),
        "internal_exact_addresses_scored_over_M": statistics.fmean(exact_scored) / len(candidates),
        "factor_score_macs": statistics.fmean(factor),
        "pst_prediction_ops": statistics.fmean(pst_ops),
        "structural_gate_ops": statistics.fmean(gate_ops),
    }


def main():
    all_runs = []
    for seed in SEEDS:
        for m in M_LEVELS:
            candidates = make_library(seed, m)
            rows, store, sm, pst, structural = build_structural(candidates, seed)
            learned, state = train_learned_pkm(candidates, seed + m)
            cluster_dummy = tuple(sm.assign(
                TransitionRecord(
                    before=(0,) * DIM,
                    operator=candidate_operator(candidate),
                    after=apply_operator((0,) * DIM, candidate_operator(candidate)),
                    verified=True,
                    episode=-1,
                    step=-1,
                )
            ) for candidate in candidates)

            for budget in BUDGETS:
                all_runs.append(eval_learned(
                    candidates, state, learned, sm, pst, cluster_dummy, budget, "pkm", seed
                ))
                all_runs.append(eval_struct(
                    candidates, state, structural, rows, sm, pst, budget, seed, False, False
                ))
                all_runs.append(eval_struct(
                    candidates, state, structural, rows, sm, pst, budget, seed, True, False
                ))
                all_runs.append(eval_struct(
                    candidates, state, structural, rows, sm, pst, budget, seed, True, True
                ))

    metrics = (
        "execution_success", "exact_target_success",
        "pkm_admission_recall", "final_admission_recall",
        "internal_exact_addresses_scored",
        "internal_exact_addresses_scored_over_M",
        "factor_score_macs", "pst_prediction_ops", "structural_gate_ops"
    )
    grouped = {}
    for row in all_runs:
        key = (int(row["M"]), str(row["mode"]), int(row["budget"]))
        grouped.setdefault(key, []).append(row)

    pooled = []
    for key, members in sorted(grouped.items()):
        m, mode, budget = key
        out = {"M": m, "mode": mode, "budget": budget, "seeds": len(members)}
        for metric in metrics:
            values = [float(x[metric]) for x in members]
            out[metric] = statistics.fmean(values)
        pooled.append(out)

    result = {
        "protocol": {
            "name": "TACOSM-FUSED-FRONTIER-002",
            "seeds": list(SEEDS),
            "M_levels": list(M_LEVELS),
            "absolute_budgets": list(BUDGETS),
            "single_eval_steps": EVAL_SINGLE,
            "train_steps": TRAIN_STEPS,
            "training_codebook": TRAIN_KEYS,
            "evaluation_identifiability": "target is a held-out toggle operator; only tasks with exactly one operator in the full library that reaches the goal are retained",
            "no_rescue_rule": "no execution feedback enters evaluation routing; PST is an explicit ablation",
        },
        "pooled": pooled,
        "runs": all_runs,
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/TACOSM-FUSED-FRONTIER-002.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
