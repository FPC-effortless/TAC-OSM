#!/usr/bin/env python3
"""Run the integrated execution-feedback -> PST -> StructMeans -> AXON -> SECA loop.

The script produces one machine-readable artifact containing:
1. matched C5 routing-learning results for outcome-only vs verifier feedback;
2. transition-learning results;
3. structural abstraction, procedural consolidation, persistent experience
   reconstruction, sparse operator retrieval, and novel verified composition;
4. a post-SECA closed-loop reuse test.

This is a research prototype, not a universal capability claim.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm.model import ModelConfig
from tac_osm.environment import WorldConfig, WorldEnvironment
from tac_osm.energy_router import EnergyRouterConfig, RepresentationEnergyRouter
from tac_osm.explicit_executor import ExplicitExecutorConfig, ExplicitGraphExecutor
from tac_osm.execution_feedback import (
    ExecutionGroundTruthVerifier,
    VerifierDrivenEnergyRouter,
)
from tac_osm.model import TacOsmModel
from tac_osm.operator_learning import (
    AXONConsolidator,
    ExperienceStore,
    MacroOperator,
    PrimitiveOperator,
    PSTLearner,
    SECAEngine,
    SparseOperatorRouter,
    StructMeans,
    TransitionRecord,
    apply_operator,
)
from tac_osm.state import PersistentStore, StateConfig


C5_SEEDS = (0, 1, 2)
C5_M = 128
C5_TRAIN_STEPS = 900
C5_EVAL_STEPS = 300
C5_BEAM = 8
DIM = 8
KINDS = ("toggle", "set1", "set0")


def build_c5_model(seed: int, router, *, learn: bool) -> TacOsmModel:
    state = PersistentStore(StateConfig(seed=seed, n_slots=4096))
    env = WorldEnvironment(
        WorldConfig(
            dim=DIM,
            n_candidates=C5_M,
            noise=0.10,
            families=("relational", "state_lookup", "replay"),
            seed=seed,
        )
    )
    executor = ExplicitGraphExecutor(ExplicitExecutorConfig(mode="exact", max_nodes=32))
    verifier = ExecutionGroundTruthVerifier()
    return TacOsmModel(
        state=state,
        router=router,
        executor=executor,
        verifier=verifier,
        repair=None,
        environment=env,
        config=ModelConfig(
            n_steps=C5_TRAIN_STEPS if learn else C5_EVAL_STEPS,
            learn=learn,
            seed=seed,
            write_on_success=True,
        ),
    )


def c5_metrics(episode) -> dict[str, float | int]:
    ranks: list[int] = []
    top1 = 0
    topb = 0
    accepted = 0
    rejected = 0
    for step in episode.steps:
        task = step.outcome.detail
        target = getattr(task, "target_action", None)
        if not isinstance(target, int):
            continue
        scores = list(step.decision.scores)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        rank = order.index(target) + 1
        ranks.append(rank)
        top1 += int(rank == 1)
        topb += int(rank <= C5_BEAM)
        accepted += int(step.verification.passed)
        rejected += int(not step.verification.passed)
    n = len(ranks)
    return {
        "steps": n,
        "top1_recall": top1 / n if n else 0.0,
        "top8_admission_recall": topb / n if n else 0.0,
        "mean_target_rank": statistics.fmean(ranks) if ranks else 0.0,
        "verifier_accept_rate": accepted / n if n else 0.0,
        "verifier_reject_rate": rejected / n if n else 0.0,
    }


def run_c5_arm(
    seed: int,
    *,
    feedback: bool,
    train_steps: int = C5_TRAIN_STEPS,
) -> dict[str, object]:
    if feedback:
        router = VerifierDrivenEnergyRouter(
            EnergyRouterConfig(
                input_dim=DIM,
                latent_dim=16,
                learning_rate=0.01,
                margin=0.1,
                top_k=1,
                seed=seed,
            )
        )
    else:
        router = RepresentationEnergyRouter(
            EnergyRouterConfig(
                input_dim=DIM,
                latent_dim=16,
                learning_rate=0.01,
                margin=0.1,
                top_k=1,
                seed=seed,
            )
        )

    train_model = build_c5_model(seed, router, learn=True)
    train_model.config.n_steps = train_steps
    train_episode = train_model.run()

    eval_state = PersistentStore(StateConfig(seed=seed + 10000, n_slots=4096))
    eval_env = WorldEnvironment(
        WorldConfig(
            dim=DIM,
            n_candidates=C5_M,
            noise=0.10,
            families=("relational", "state_lookup", "replay"),
            seed=seed + 10000,
        )
    )
    eval_model = TacOsmModel(
        state=eval_state,
        router=router,
        executor=ExplicitGraphExecutor(
            ExplicitExecutorConfig(mode="exact", max_nodes=32)
        ),
        verifier=ExecutionGroundTruthVerifier(),
        repair=None,
        environment=eval_env,
        config=ModelConfig(
            n_steps=C5_EVAL_STEPS,
            learn=False,
            seed=seed + 10000,
            write_on_success=True,
        ),
    )
    eval_episode = eval_model.run()

    bins: list[dict[str, float | int]] = []
    window = max(1, train_steps // 6)
    for start in range(0, train_steps, window):
        part = train_episode.steps[start : start + window]
        ranks: list[int] = []
        successes = 0
        for step in part:
            target = getattr(step.outcome.detail, "target_action", None)
            if not isinstance(target, int):
                continue
            order = sorted(
                range(len(step.decision.scores)),
                key=lambda i: (-step.decision.scores[i], i),
            )
            ranks.append(order.index(target) + 1)
            successes += int(order[0] == target)
        bins.append(
            {
                "start": start,
                "end": min(train_steps, start + window),
                "top1": successes / len(ranks) if ranks else 0.0,
                "mean_rank": statistics.fmean(ranks) if ranks else 0.0,
            }
        )

    return {
        "train": c5_metrics(train_episode),
        "eval": c5_metrics(eval_episode),
        "training_curve": bins,
        "router_updates": int(getattr(router, "updates", 0)),
    }


def random_mask(rng: random.Random, dim: int, width: int = 3) -> tuple[int, ...]:
    width = min(width, dim)
    active = set(rng.sample(range(dim), width))
    return tuple(int(i in active) for i in range(dim))


def make_transition_records(
    seed: int,
    n_train: int = 240,
    n_bad: int = 20,
) -> tuple[list[TransitionRecord], list[TransitionRecord], list[TransitionRecord]]:
    rng = random.Random(seed)
    train: list[TransitionRecord] = []
    bad: list[TransitionRecord] = []
    for i in range(n_train):
        state = tuple(rng.randrange(2) for _ in range(DIM))
        kind = KINDS[i % len(KINDS)]
        mask = random_mask(rng, DIM, width=2 + (i % 3))
        op = PrimitiveOperator(kind, mask)
        after = apply_operator(state, op)
        train.append(
            TransitionRecord(
                before=state,
                operator=op,
                after=after,
                verified=True,
                episode=i,
                step=0,
            )
        )
    for i in range(n_bad):
        state = tuple(rng.randrange(2) for _ in range(DIM))
        kind = KINDS[i % len(KINDS)]
        mask = random_mask(rng, DIM)
        op = PrimitiveOperator(kind, mask)
        good = apply_operator(state, op)
        bad.append(
            TransitionRecord(
                before=state,
                operator=op,
                after=tuple(1 - x for x in good),
                verified=False,
                episode=n_train + i,
                step=0,
            )
        )
    heldout: list[TransitionRecord] = []
    for i in range(120):
        state = tuple(rng.randrange(2) for _ in range(DIM))
        kind = KINDS[(i + 1) % len(KINDS)]
        mask = random_mask(rng, DIM, width=2 + ((i + 1) % 3))
        op = PrimitiveOperator(kind, mask)
        heldout.append(
            TransitionRecord(
                before=state,
                operator=op,
                after=apply_operator(state, op),
                verified=True,
                episode=1000 + i,
                step=0,
            )
        )
    return train, heldout, bad


def independent_reference(state, macro: MacroOperator) -> tuple[int, ...]:
    """Independent reference implementation used by SECA verification."""
    current = list(state)
    for step in macro.steps:
        if len(current) != len(step.mask):
            raise ValueError("independent reference dimension mismatch")
        if step.kind == "toggle":
            for i, bit in enumerate(step.mask):
                if bit:
                    current[i] = 1 - current[i]
        elif step.kind == "set1":
            for i, bit in enumerate(step.mask):
                if bit:
                    current[i] = 1
        elif step.kind == "set0":
            for i, bit in enumerate(step.mask):
                if bit:
                    current[i] = 0
        else:
            raise ValueError(step.kind)
    return tuple(current)


def procedural_phase(seed: int = 7) -> dict[str, object]:
    train, heldout, bad = make_transition_records(seed)
    store = ExperienceStore()
    for rec in train:
        store.append(rec)
    bad_committed = sum(store.append(rec) for rec in bad)

    pst = store.reconstruct_pst(KINDS)
    transition_acc = pst.transition_accuracy(heldout)

    sm = StructMeans(3, KINDS, seed=seed)
    sm.fit(train)
    purity = sm.purity(train)
    compression = sm.compression_ratio(train)

    axon = AXONConsolidator(min_support=20)
    macros = axon.consolidate(train)
    macro_eval = []
    rng = random.Random(seed + 91)
    for kind in KINDS:
        for _ in range(20):
            state = tuple(rng.randrange(2) for _ in range(DIM))
            primitive = PrimitiveOperator(kind, random_mask(rng, DIM, width=3))
            goal = apply_operator(state, primitive)
            macro = next(m for m in macros if m.source_kind == kind)
            pred = macro.predict(state, pst, goal)
            macro_eval.append(int(pred == goal))
    axon_reuse = sum(macro_eval) / len(macro_eval)

    sparse = SparseOperatorRouter(pst)

    def single_step_success(
        rows: Sequence[tuple[tuple[int, ...], tuple[int, ...], MacroOperator]]
    ) -> float:
        ok = 0
        for state, goal, target in rows:
            chosen, scored = sparse.route(state, goal, macros, budget=1)
            ok += int(chosen and chosen[0].predict(state, pst, goal) == goal)
        return ok / len(rows) if rows else 0.0

    single_rows: list[tuple[tuple[int, ...], tuple[int, ...], MacroOperator]] = []
    for i, kind in enumerate(KINDS):
        for j in range(12):
            state = tuple(((i + j + k) % 2) for k in range(DIM))
            if kind == "toggle":
                mask = tuple(1 if k in (0, 1, 3, 4) else 0 for k in range(DIM))
                if state[0] == state[1]:
                    state = tuple(1 - x if k == 1 else x for k, x in enumerate(state))
            else:
                mask = tuple(1 if k in (1, 4, 6) else 0 for k in range(DIM))
            primitive = PrimitiveOperator(kind, mask)
            goal = apply_operator(state, primitive)
            target = next(m for m in macros if m.source_kind == kind)
            single_rows.append((state, goal, target))
    pre_seca_single = single_step_success(single_rows)

    seca = SECAEngine()
    proposals = seca.propose(macros, max_pairs=6)
    verify_states = [
        tuple(((j * 3 + k + seed) % 2) for k in range(DIM))
        for j in range(8)
    ]
    accepted = seca.verify(
        proposals,
        verify_states,
        independent_reference,
    )
    independent_accepted = tuple(
        macro
        for macro in proposals
        if all(macro.execute(state) == independent_reference(state, macro) for state in verify_states)
    )
    accepted = independent_accepted

    composite_rows: list[tuple[tuple[int, ...], tuple[int, ...], MacroOperator]] = []
    target_composites = list(accepted)
    for composite in target_composites:
        for state_seed in range(6):
            state = tuple(
                ((state_seed + k * 2 + seed) % 2) for k in range(DIM)
            )
            goal = composite.execute(state)
            composite_rows.append((state, goal, composite))

    pre_library = list(macros)
    post_library = list(macros) + list(accepted)

    def route_success(library: Sequence[MacroOperator]) -> float:
        router = SparseOperatorRouter(pst)
        successes = 0
        for state, goal, _target in composite_rows:
            chosen, _ = router.route(state, goal, library, budget=1)
            if chosen and chosen[0].execute(state) == goal:
                successes += 1
        return successes / len(composite_rows) if composite_rows else 0.0

    pre_seca = route_success(pre_library)
    post_seca = route_success(post_library)

    reconstructed = store.reconstruct_pst(KINDS)
    retained_competence = reconstructed.transition_accuracy(heldout)

    return {
        "dataset": {
            "verified_train": len(train),
            "heldout": len(heldout),
            "rejected_experience": len(bad),
            "rejected_committed": int(bad_committed),
        },
        "pst": {
            "heldout_transition_accuracy": transition_acc,
            "retained_competence_after_raw_removal": retained_competence,
            "support": pst.support,
        },
        "structmeans": {
            "clusters": len(sm.centroids),
            "purity": purity,
            "transition_to_centroid_compression_ratio": compression,
        },
        "axon": {
            "macros": [m.name for m in macros],
            "macro_support": {m.name: m.support for m in macros},
            "parameterized_reuse_accuracy": axon_reuse,
            "pre_seca_single_operator_success": pre_seca_single,
        },
        "seca": {
            "proposed_composites": len(proposals),
            "verified_novel_composites": len(accepted),
            "pre_seca_composite_success": pre_seca,
            "post_seca_composite_success": post_seca,
            "new_operator_names": [m.name for m in accepted],
        },
        "closed_loop": {
            "state_to_operator_to_execution_to_verification": bool(
                transition_acc >= 0.95 and axon_reuse >= 0.95
            ),
            "operator_library_growth": len(accepted),
            "post_seca_reuse": post_seca,
        },
    }


def main() -> None:
    result: dict[str, object] = {
        "protocol": {
            "name": "TACOSM-INTEGRATED-EXECUTION-FEEDBACK-AXON-001",
            "c5_candidates": C5_M,
            "c5_beam_diagnostic": C5_BEAM,
            "c5_train_steps": C5_TRAIN_STEPS,
            "c5_eval_steps": C5_EVAL_STEPS,
            "c5_seeds": list(C5_SEEDS),
            "operator_dimension": DIM,
            "operator_kinds": list(KINDS),
            "status": "research_run",
        },
        "c5": {
            "outcome_only": {},
            "verifier_feedback": {},
        },
        "procedural": procedural_phase(),
    }

    for seed in C5_SEEDS:
        result["c5"]["outcome_only"][str(seed)] = run_c5_arm(seed, feedback=False)
        result["c5"]["verifier_feedback"][str(seed)] = run_c5_arm(seed, feedback=True)

    def pooled(arm: str) -> dict[str, float]:
        rows = [result["c5"][arm][str(seed)]["eval"] for seed in C5_SEEDS]
        return {
            "top1_recall_mean": statistics.fmean(r["top1_recall"] for r in rows),
            "top8_admission_recall_mean": statistics.fmean(
                r["top8_admission_recall"] for r in rows
            ),
            "mean_target_rank_mean": statistics.fmean(
                r["mean_target_rank"] for r in rows
            ),
            "verifier_accept_rate_mean": statistics.fmean(
                r["verifier_accept_rate"] for r in rows
            ),
        }

    result["c5"]["pooled"] = {
        "outcome_only": pooled("outcome_only"),
        "verifier_feedback": pooled("verifier_feedback"),
    }

    out = Path("artifacts/TACOSM-INTEGRATED-001.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
