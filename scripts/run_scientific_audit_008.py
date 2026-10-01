#!/usr/bin/env python3
"""TACOSM-SCIENTIFIC-AUDIT-008 corrected benchmark.

This runner addresses audit findings without modifying historical artifacts:
- deterministic representability witnesses;
- unique-candidate population scaling;
- true structural support-shift holdout;
- pooled trial-level quantiles;
- normalized all-candidate environment-outcome supervision;
- explicit query-side routing work.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm import Candidate, Query
from tac_osm.c5_persistent_relational_loop import (
    DIM,
    N_OPS,
    OPS,
    CDLPersistentRelationRouter,
    PersistentRelationEnvironment,
    apply_relation,
    build_population,
    make_episode,
    prepare_state,
    PersistentRelationTrial,
    decode_state_value,
)
from tac_osm.temporal import TemporalPersistentState

SEEDS = (0, 1, 2, 3, 4)
CONTROL_STEPS = 800
OUTCOME_BASE_STEPS = 600
OUTCOME_STEPS = 200
TRAIN_M = (64, 128, 256)
EVAL_M = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
TRIALS = 64
RECALL_K = (1, 4, 8, 16, 32, 64)


def pooled_quantile(values, probability: float) -> int:
    ordered = sorted(int(x) for x in values)
    if not ordered:
        raise ValueError("quantile requires values")
    return ordered[max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))]


def deterministic_representability() -> dict[str, object]:
    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rows = {}
    for op_id, op in enumerate(OPS):
        signed = []
        for a, b in combos:
            sa, sb = 2 * a - 1, 2 * b - 1
            p = sa * sb
            if op_id == 0:
                score = -p
            elif op_id == 1:
                score = p
            elif op_id == 2:
                score = sa + sb + p - 1
            else:
                score = sa + sb - p + 1
            y = 1 if apply_relation((a,) * DIM, (b,) * DIM, op_id)[0] else -1
            signed.append((score, y))
        ranking_ok = all(score != 0 and (score > 0) == (y > 0) for score, y in signed)
        rows[op] = {
            "ranking_representable": ranking_ok,
            "scores": [s for s, _ in signed],
            "labels": [y for _, y in signed],
            "witness_is_deterministic": True,
        }
    return rows


def make_unique_trial(seed: int, step: int, m: int, *, mode: str) -> tuple[PersistentRelationTrial, TemporalPersistentState]:
    rng = random.Random(seed * 1_000_003 + step * 7919 + 17)
    while True:
        if mode == "train":
            left = []
            right = []
            for _ in range(DIM):
                a = rng.randrange(2)
                b = rng.randrange(2)
                if a == 1 and b == 1:
                    b = 0
                left.append(a)
                right.append(b)
        elif mode == "ood":
            left = [1] * 4 + [rng.randrange(2) for _ in range(DIM - 4)]
            right = [1] * 4 + [rng.randrange(2) for _ in range(DIM - 4)]
        else:
            left = [rng.randrange(2) for _ in range(DIM)]
            right = [rng.randrange(2) for _ in range(DIM)]
        op = rng.randrange(N_OPS)
        target = apply_relation(left, right, op)
        candidates = build_population(seed + 31, m, target)
        address = f"world:audit008:{seed}:{step:06d}"
        state, query = prepare_state(
            seed=seed,
            step=step,
            address=address,
            left=tuple(left),
            right=tuple(right),
            op=op,
        )
        target_index = next(i for i, c in enumerate(candidates) if c.descriptor == target)
        trial = PersistentRelationTrial(
            candidates=candidates,
            target_descriptor=target,
            target_index=target_index,
            address=address,
            query=query,
            step=step,
        )
        return trial, state


def train_control(seed: int, steps: int = CONTROL_STEPS) -> tuple[CDLPersistentRelationRouter, set[tuple[int, ...]]]:
    router = CDLPersistentRelationRouter(
        seed=seed, learning_rate=0.012, soft_target_epsilon=0.05
    )
    seen_targets: set[tuple[int, ...]] = set()
    for step in range(steps):
        m = TRAIN_M[step % len(TRAIN_M)]
        trial, state = make_unique_trial(seed, step, m, mode="train")
        router.train_exhaustive(trial.query, state, trial.candidates, trial.target_index)
        seen_targets.add(trial.target_descriptor)
    return router, seen_targets


def train_outcome(seed: int) -> tuple[CDLPersistentRelationRouter, int, set[tuple[int, ...]]]:
    router, seen_targets = train_control(seed, steps=OUTCOME_BASE_STEPS)
    env_evaluations = 0
    for step in range(OUTCOME_STEPS):
        m = TRAIN_M[step % len(TRAIN_M)]
        trial, state = make_unique_trial(
            seed, 200_000 + step, m, mode="train"
        )
        qz = list(router.encode_query(trial.query, state))
        zc = [list(router.encode_candidate(c)) for c in trial.candidates]
        labels = []
        env = PersistentRelationEnvironment(trial)
        for i in range(len(trial.candidates)):
            # Environment outcome is the learning signal; target identity is not
            # passed into the router update.
            labels.append(1.0 if env.act(i, casm_output=0.0).success else -1.0)
            env_evaluations += 1
        scores = [router.score_embeddings(qz, z) for z in zc]
        grads_q = [0.0] * router.latent_dim
        grads_c = [[0.0] * router.latent_dim for _ in zc]
        positive = [i for i, y in enumerate(labels) if y > 0]
        negative = [i for i, y in enumerate(labels) if y < 0]
        pos_weight = 0.5 / max(1, len(positive))
        neg_weight = 0.5 / max(1, len(negative))
        for i, (z, y) in enumerate(zip(zc, labels)):
            s = scores[i]
            p = 1.0 / (1.0 + math.exp(max(-40.0, min(40.0, -s))))
            weight = pos_weight if y > 0 else neg_weight
            g = weight * (y - p)
            for r in range(router.latent_dim):
                grads_q[r] += g * z[r]
                grads_c[i][r] += g * qz[r]
        qx = router._query_features(trial.query, state)
        lr = router.learning_rate
        for r in range(router.latent_dim):
            for j in range(len(qx)):
                router.wq[r][j] += lr * grads_q[r] * qx[j]
            router.bq[r] += lr * grads_q[r]
        for i, c in enumerate(trial.candidates):
            cx = router._candidate_features(c)
            for r in range(router.latent_dim):
                g = grads_c[i][r]
                for j in range(DIM):
                    router.wc[r][j] += lr * g * cx[j]
                router.bc[r] += lr * g
        router.updates += 1
        seen_targets.add(trial.target_descriptor)
    return router, env_evaluations, seen_targets


def evaluate(
    router,
    seed: int,
    *,
    mode: str,
    forbidden_targets: set[tuple[int, ...]] | None = None,
) -> dict[str, object]:
    by_m = {}
    for m in EVAL_M:
        ranks = []
        routing_ops = []
        recall_at = {str(k): [] for k in RECALL_K}
        forbidden = forbidden_targets or set()
        for i in range(TRIALS):
            retry = 0
            while True:
                trial, state = make_unique_trial(
                    seed, 400_000 + i + m * 17 + retry * 1_000_003, m, mode=mode
                )
                if trial.target_descriptor not in forbidden:
                    break
                retry += 1
                if retry > 1000:
                    raise RuntimeError("unable to generate target-disjoint evaluation sample")
            scores = router.score_all(trial.query, state, trial.candidates)
            order = sorted(range(len(scores)), key=lambda j: (-scores[j], j))
            rank = order.index(trial.target_index) + 1
            ranks.append(rank)
            # Full dense routing arithmetic: query encoding + candidate encoding + similarities.
            query_encode_ops = router.latent_dim * router.query_dim
            candidate_encode_ops = m * router.latent_dim * DIM
            similarity_ops = m * router.latent_dim
            routing_ops.append(query_encode_ops + candidate_encode_ops + similarity_ops)
            for k in RECALL_K:
                recall_at[str(k)].append(float(rank <= k))
        by_m[str(m)] = {
            "n": len(ranks),
            "trial_ranks": ranks,
            "mean_rank": statistics.fmean(ranks),
            "P90_rank": pooled_quantile(ranks, 0.90),
            "Top1": statistics.fmean(float(r == 1) for r in ranks),
            "RecallAtK": {k: statistics.fmean(v) for k, v in recall_at.items()},
            "mean_dense_routing_ops": statistics.fmean(routing_ops),
        }
    return by_m


def main() -> None:
    assert all(deterministic_representability()[op]["ranking_representable"] for op in OPS)

    control = {}
    outcome = {}
    outcome_eval_cost = {}
    routers = {}
    seen_control = {}
    seen_outcome = {}
    for seed in SEEDS:
        c, seen_c = train_control(seed)
        o, env_evals, seen_o = train_outcome(seed)
        routers[str(seed)] = (c, o)
        seen_control[str(seed)] = seen_c
        seen_outcome[str(seed)] = seen_o
        control[str(seed)] = evaluate(c, seed, mode="eval", forbidden_targets=seen_c)
        outcome[str(seed)] = evaluate(o, seed, mode="eval", forbidden_targets=seen_o)
        outcome_eval_cost[str(seed)] = env_evals

    def pool(arm):
        out = {}
        for m in EVAL_M:
            rows = [arm[str(s)][str(m)] for s in SEEDS]
            pooled = [rank for r in rows for rank in r["trial_ranks"]]
            out[str(m)] = {
                "pooled_n": len(pooled),
                "pooled_mean_rank": statistics.fmean(pooled),
                "pooled_P90_rank": pooled_quantile(pooled, 0.90),
                "pooled_Top1": statistics.fmean(float(x == 1) for x in pooled),
            }
        return out

    result = {
        "experiment_id": "TACOSM-SCIENTIFIC-AUDIT-008",
        "status": "measured",
        "protocol": {
            "seeds": list(SEEDS),
            "control_steps": CONTROL_STEPS,
            "outcome_base_steps": OUTCOME_BASE_STEPS,
            "outcome_steps": OUTCOME_STEPS,
            "train_M": list(TRAIN_M),
            "eval_M": list(EVAL_M),
            "trials_per_seed": TRIALS,
            "recall_at_k": list(RECALL_K),
        },
        "representability": deterministic_representability(),
        "unique_population_scaling": {
            "control": pool(control),
            "outcome_field_environment": pool(outcome),
        },
        "outcome_training_environment_evaluations": outcome_eval_cost,
        "ood_holdout": {
            "control": {str(s): evaluate(routers[str(s)][0], s, mode="ood", forbidden_targets=seen_control[str(s)]) for s in SEEDS},
            "outcome_field_environment": {str(s): evaluate(routers[str(s)][1], s, mode="ood", forbidden_targets=seen_outcome[str(s)]) for s in SEEDS},
        },
        "scope": {
            "semantic_language": False,
            "asymptotic_theorem": False,
            "outcome_labels_are_post_action_environment_only": True,
            "casm_execution_labels": False,
            "ood_is_true_support_shift": True,
        },
    }

    out = ROOT / "artifacts" / "TACOSM-SCIENTIFIC-AUDIT-008.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
